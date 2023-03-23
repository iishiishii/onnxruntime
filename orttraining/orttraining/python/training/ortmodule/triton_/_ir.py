# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import re
from abc import abstractmethod
from collections import OrderedDict, deque
from enum import Enum, auto
from typing import Dict, List, Set, Union

import numpy as np
import onnx
import onnx.numpy_helper
import sympy
from sympy.codegen.rewriting import create_expand_pow_optimization

from ._common import (
    TENSOR_TYPE_TO_NP_TYPE,
    CodeGenContext,
    HardwareContext,
    NodeVisitor,
    SpecialVar,
    parse_onnx_attributes,
)
from ._op_config import is_elementwise_node, is_reduction_node
from ._sympy_utils import FloorDiv, sympy_dot, sympy_symbol


class BufferAttr(Enum):
    GLOBAL = auto()
    ACROSS_SHARED = auto()


class ComputeBuffer(object):
    def __init__(self, name: str, dtype: np.dtype = np.dtype(object), shape: list = None, data: np.ndarray = None):
        self.name = name
        assert isinstance(dtype, np.dtype), "only support numpy dtype"
        self.dtype = dtype
        self.shape = self.parse_shape(shape)
        self.data = self.handle_data(data)
        self.loop_index: List[sympy.Expr] = None
        self.predecessor: IRNode = None
        self.successor: List[IRNode] = []
        self.attributes: Set[BufferAttr] = set()
        self.attr_cross_loop = False

    def handle_data(self, data):
        if data is None:
            return data
        self.dtype = data.dtype
        self.shape = self.parse_shape(data.shape)
        return data

    def parse_shape(self, shape: list):
        if shape is None:
            return None
        else:
            symbol_shapes = []
            for x in shape:
                if isinstance(x, str):
                    rx = re.sub(r"[^a-zA-Z0-9_]+", "_", x)
                    symbol_shape = sympy.Symbol(rx)
                elif isinstance(x, sympy.Symbol):
                    symbol_shape = x
                else:
                    symbol_shape = sympy.Integer(x)
                symbol_shapes.append(symbol_shape)
            return symbol_shapes

    def __eq__(self, o):
        if isinstance(o, ComputeBuffer):
            return self.name == o.name
        elif isinstance(o, str):
            return self.name == o
        else:
            raise Exception("not supported")

    def __hash__(self):
        return hash(self.name)


class IndirectPermuteBuffer(ComputeBuffer):
    def __init__(self, name: str, dtype: np.dtype = np.dtype(object), shape: list = None, data: np.ndarray = None):
        super().__init__(name, dtype, shape, data)
        # only works when the last axis is unchanged
        # [0,2,1,3]?
        self.permute_index: List[int] = None


class IndirectSliceBuffer(ComputeBuffer):
    def __init__(self, name: str, dtype: np.dtype = np.dtype(object), shape: list = None, data: np.ndarray = None):
        super().__init__(name, dtype, shape, data)
        # only works when the last axis is unchanged
        # [axis, start, end, step]
        self.slice_index: List[List] = None


class IRNode:
    def __init__(self):
        self.parent = None
        self.vectorization = False
        self.input: List[ComputeBuffer] = []
        self.output: List[ComputeBuffer] = []

    @abstractmethod
    def code_gen(self, visitor: NodeVisitor, var_context: CodeGenContext, indent: int = 0):
        return visitor.visit(self, var_context, indent)

    @abstractmethod
    def lower(self, visitor: NodeVisitor, context: HardwareContext):
        return visitor.visit(self, context)


class LoopAttr(Enum):
    ScalarLoop = 0
    Parallel = 1
    Reduce = 2
    Vectorization = 3


class Loop(IRNode):
    def __init__(self):
        super().__init__()
        self.var: sympy.Expr = None
        self.reduction_var: OrderedDict = {}
        self.start = sympy.Integer(0)
        self.end = sympy.Integer(0)
        self.step = sympy.Integer(1)
        self.body = None
        self.depth = 0
        self.recompute = False
        self.parallel: bool = False
        self.parallel_nest_loop: Union[Loop, List[Loop]] = None
        self.attributes = LoopAttr.ScalarLoop
        self.forward_var_map: Dict[ComputeBuffer] = OrderedDict()
        self.var_need_post_process: OrderedDict = {}

    def visit(self, var):
        if isinstance(var, sympy.Mul):
            return f"({self.visit(var.args[0])} * {self.visit(var.args[1])})"
        elif isinstance(var, sympy.Add):
            return f"({self.visit(var.args[0])} + {self.visit(var.args[1])})"
        elif isinstance(var, FloorDiv):
            return f"({self.visit(var.args[0])} / {self.visit(var.args[1])})"
        else:
            return str(var)


class IoConnection(object):
    def __init__(self):
        self.users = []
        self.producers = []


# Generally, we use it to handle vectorization type change (vec->scalar)
class PostProcessBlock(Loop):
    def __init__(self, loop: Loop):
        super().__init__()
        self.global_connections: Dict[str, IoConnection] = None
        self.body: List[loop] = [loop]
        self.op_map: Dict[str, str] = {
            "ReduceMax": "hmax",
            "ReduceMin": "hmin",
            "ReduceSum": "hadd",
        }


class ExecutionBlock(IRNode):
    def __init__(self, group: List[IRNode]):
        super().__init__()
        self.input: list[ComputeBuffer] = []
        self.output: list[ComputeBuffer] = []
        self.constant_vars: list[ComputeBuffer] = []
        self.intermediate_var = OrderedDict()
        self.load = OrderedDict()
        self.loop_stack = []
        self.has_reduce = False
        self.recompute = False
        self.reduction_var = OrderedDict()
        # TODO support multiple outputs
        self.dtype = list(group[0].output_with_shapes.values())[0][0]
        self.shape = self.extract_shape(group)
        self.var_map = OrderedDict()
        # A block usually means a fused loop. In triton, it's impossible to have multiple loops in a block.
        # so a function has only one block
        # But A block could have multiple inner loop, some vars may be used across multiple inner loop
        # so we need to maintain a list of vars for forward declaration and initialization and even guide
        # how to do recompute.
        self.forward_var_map_list = [OrderedDict()]
        self.body = None
        self.hw_context = None
        self.connections: Dict[str, IoConnection] = OrderedDict()
        self.fused_groups: List[List[IRNode]] = []

        self.group = group

    def analyze_io_connections(self):
        for group in self.fused_groups:
            for g in group:
                ipt = [g.input] if not isinstance(g.input, (list, dict)) else g.input

                for inp in ipt:
                    in_name = inp if isinstance(inp, str) else inp.name
                    if in_name not in self.connections:
                        self.connections[in_name] = IoConnection()
                    self.connections[in_name].users.append(g)

                for out in g.output:
                    out_name = out if isinstance(out, str) else out.name
                    if out_name not in self.connections:
                        self.connections[out_name] = IoConnection()
                    assert len(self.connections[out_name].producers) == 0, "multiple producers!!"
                    self.connections[out_name].producers.append(g)
        pass

    def extract_shape(self, group: List[IRNode]):
        assert len(group[-1].output_with_shapes) == 1
        if is_reduction_node(group[-1]):
            shape = []
            for i in list(group[-1].input_with_shapes.values())[0][1]:
                ri = re.sub(r"[^a-zA-Z0-9_]+", "_", i) if isinstance(i, str) else i
                shape.append(sympy_symbol(ri))
            self.has_reduce = True
            self.reduction_var[group[-1].current_node.output[0]] = group[-1].op_type
        else:
            shape = []
            for i in list(group[-1].output_with_shapes.values())[0][1]:
                ri = re.sub(r"[^a-zA-Z0-9_]+", "_", i) if isinstance(i, str) else i
                shape.append(sympy_symbol(ri))
        # support scalar
        if len(shape) == 0:
            shape = [sympy_symbol(1)]
        return shape

    def build_inner_most_loop(self):
        self.loop_stack.append(sympy_symbol(f"i_{str(len(self.loop_stack))}"))
        loop = Loop()
        loop.var = self.loop_stack[-1]

        loop.start = sympy.Integer(0)
        loop.end = self.shape[-1]
        loop.body = self.group
        if self.has_reduce:
            loop.reduction_var = self.reduction_var
            loop.attributes = LoopAttr.Reduce
        else:
            loop.attributes = LoopAttr.Vectorization

        loop.depth = 0
        # currently, we have only one map
        loop.forward_var_map = self.forward_var_map_list[0]

        return loop

    def build_loop(self):
        body = self.build_inner_most_loop()
        for i in range(len(self.shape) - 2, -1, -1):
            loop = Loop()
            self.loop_stack.append(f"i_{str(len(self.loop_stack))}")
            loop.var = self.loop_stack[-1]
            loop.start = 0
            loop.end = self.shape[i]
            loop.body = body
            loop.depth = len(self.loop_stack) - 1
            body = loop
        return body

    def gen_var(self, external_var):
        exist_var = set()

        def legal_name(name):
            nonlocal exist_var
            import re

            pos = name.rfind("/")
            if pos != -1:
                name = name[pos + 1 :]
            else:
                name = re.sub(r"[^a-zA-Z0-9]", "_", name)
                if len(name) > 20:
                    name = name[-20:]

            # 1. assure name is legal, not start with digit
            # 2. assure name is different from the original
            name = "aot_" + name
            while name in exist_var:
                name = name + "_1"

            exist_var.add(name)
            return name

        for forward_var_map in self.forward_var_map_list:
            for inp in forward_var_map:
                self.var_map[inp] = legal_name(inp)
        for inp in self.input:
            self.var_map[inp.name] = legal_name(inp.name)
        for out in self.output:
            self.var_map[out.name] = legal_name(out.name)
        for var in self.intermediate_var:
            self.var_map[var] = legal_name(var)
        for lvar in self.load:
            self.var_map[lvar.name] = legal_name(lvar.name)
            v = self.load[lvar].data.reshape(-1) if self.load[lvar].data is not None else None
            if v is not None and v.size == 1:
                v_v = self.var_map[lvar]
                assert v_v not in self.var_map
                self.var_map[v_v] = v

        for out in external_var:
            var = out.name
            self.var_map[var] = legal_name(var)
            if isinstance(out, onnx.NodeProto):
                v = onnx.numpy_helper.to_array(out.attribute[0].t).reshape(-1)
            elif isinstance(out, onnx.TensorProto):
                v = onnx.numpy_helper.to_array(out).reshape(-1)
            else:
                raise NotImplementedError
            # "only support scalar"
            if v.size > 1:
                continue

            v_v = self.var_map[var]
            self.var_map[v_v] = v

        self.var_map[SpecialVar().rbase] = SpecialVar().rbase


class FunctionNode(IRNode):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.input = inputs
        self.output = outputs
        self.name: str = ""
        self.const_var = []
        self.shape_var = []
        self.body: List[ExecutionBlock] = None
        self.hw_context: HardwareContext = None


class ModuleNode(IRNode):
    def __init__(self, modules: Dict[str, onnx.ModelProto]):
        super().__init__()
        self.body: List[FunctionNode] = []
        self.has_vectorization = False
        self.modules = modules


class ComputeNode(IRNode):
    def __init__(self, op_type, inputs, outputs, op_name: str = "", attributes: list = None):
        super().__init__()
        self.op_type_ = op_type
        self.attributes = parse_onnx_attributes(attributes)
        self.input = inputs
        self.output = outputs
        self.op_name = op_name
        self.use_lib_device = True

    @property
    def op_type(self):
        return self.op_type_


class ReduceNode(ComputeNode):
    def __init__(self, body: ComputeNode, axis=-1):
        super().__init__(body.op_type, body.input, body.output, body.op_name)
        self.axis = axis
        self.body: ComputeNode = body
        self.input = body.input
        self.output = body.output
        self.is_final = False


class Indexer:
    def __init__(self):
        self.buf_index = None

    def cal_stride(self, shape, buf: ComputeBuffer):
        stride = [1]
        for i in range(len(shape) - 1, 0, -1):
            stride.append(stride[-1] * shape[i])
        stride = stride[::-1]

        if isinstance(buf, IndirectSliceBuffer):
            for iter_axis in buf.slice_index:
                axis, _, _, step = iter_axis
                stride[axis] *= step

        return stride

    def get_iter_var_by_buffer_type(self, buf: ComputeBuffer, shape: List[sympy.Expr]):
        index: sympy.Expr = buf.loop_index or [sympy_symbol(f"i_{i}") for i in range(len(shape) - 1, -1, -1)]
        if len(index) > len(shape):
            index = index[len(index) - len(shape) :]

        if isinstance(buf, IndirectPermuteBuffer):
            assert len(buf.permute_index) == len(shape)
            index = [index[i] for i in buf.permute_index]
        elif isinstance(buf, IndirectSliceBuffer):
            for iter_axis in buf.slice_index:
                axis, start, _, _ = iter_axis
                index[axis] += start
        elif isinstance(buf, ComputeBuffer):
            pass
        else:
            raise NotImplementedError
        return index

    def gen_index_expr(self, named_var: str, buf: ComputeBuffer):
        shape = buf.shape or (buf.data is not None and buf.data.shape) or [1]
        shape = shape.copy()
        shape = shape[-1:] if buf.attr_cross_loop else shape
        stride = self.cal_stride(shape, buf)
        shape[-1] = 1
        if shape == [1]:
            return ""
        index_of_dim_1 = [i for i in range(len(shape)) if shape[i] == 1]

        # broadcast handling
        br_index = self.get_iter_var_by_buffer_type(
            buf, shape
        )  # [v for idx, v in enumerate(index) if idx not in index_of_dim_1]
        br_stride = [v if idx not in index_of_dim_1 else 0 for idx, v in enumerate(stride)]

        expand_opt = create_expand_pow_optimization(6)
        res = expand_opt(sympy_dot(br_index, br_stride))
        gs = re.findall("([a-zA-Z0-9_]+)\\*\\*(\\d)", str(res))
        assert gs == []  # or gs[0][1] == '2', f"TODO fix me when pow {gs[0][1]} or other"
        # res= re.sub('([a-zA-Z0-9_]+)\*\*(\d)','\g<1>*\g<1>',str(res))
        return res

    def code_gen(self, named_var: str, buf: ComputeBuffer):
        if (buf.data is not None and buf.data.size == 1) or not buf.shape:
            return f"{named_var}"

        index_expr = self.gen_index_expr(named_var, buf)
        if index_expr:
            index_expr = f"+{index_expr}"
        return f"{named_var}{index_expr}"


class LoadNode(IRNode):
    def __init__(self, buf: ComputeBuffer):  # ComputeBuffer
        super().__init__()
        self.input = buf
        self.input.dtype = TENSOR_TYPE_TO_NP_TYPE[buf.dtype] if not isinstance(buf.dtype, np.dtype) else buf.dtype
        self.to_buf = "to"

    @property
    def op_type(self):
        return "Load"


class RangeNode(IRNode):
    def __init__(self, start, end, step, outputs: ComputeBuffer):
        super().__init__()
        self.input = [start, end, step]
        self.output = outputs

    @property
    def op_type(self):
        return "Range"


class MaskNode(IRNode):
    def __init__(self, inputs: List[ComputeBuffer], outputs: ComputeBuffer, shape: list = None):  # ComputeBuffer
        super().__init__()
        # inputs was block_range and boundary
        # mask = block_range < boundary
        self.input = inputs
        self.shape = shape

    @property
    def op_type(self):
        return "Mask"


class MaskLoadNode(IRNode):
    def __init__(self, buf: ComputeBuffer, mask: ComputeBuffer):  # ComputeBuffer
        super().__init__()
        self.input = [buf, mask]
        self.dtype = TENSOR_TYPE_TO_NP_TYPE[buf.dtype] if not isinstance(buf.dtype, np.dtype) else buf.dtype
        self.to_buf = "to"

    @property
    def op_type(self):
        return "MaskLoadNode"


class MaskStoreNode(IRNode):
    def __init__(self, buf: ComputeBuffer, mask: MaskNode):  # ComputeBuffer
        super().__init__()
        self.input = [buf, mask]
        self.dtype = TENSOR_TYPE_TO_NP_TYPE[buf.dtype] if not isinstance(buf.dtype, np.dtype) else buf.dtype
        self.to_buf = "to"

    @property
    def op_type(self):
        return "MaskStoreNode"


class StoreNode(IRNode):
    def __init__(self, buf: ComputeBuffer):  # ComputeBuffer
        super().__init__()
        self.to_var = buf

    @property
    def op_type(self):
        return "Store"


class InterGroupStrategy(object):
    def __init__(self):
        self.count = 0

    def can_fusion(self, node1, node2):
        if is_elementwise_node(node1.op_type):
            return True
        return False

    def do_fusion(self, nodes):
        before_fusion_groups = deque()
        after_fusion_groups = deque()

        for node in nodes:
            before_fusion_groups.append([node])

        while len(before_fusion_groups) > 1:
            node1 = before_fusion_groups.popleft()
            node2 = before_fusion_groups.popleft()
            if self.can_fusion(node1[-1].current_node, node2[-1].current_node):
                node1.extend(node2)
                before_fusion_groups.appendleft(node1)
            else:
                after_fusion_groups.append(node1)
                before_fusion_groups.appendleft(node2)
        after_fusion_groups.extend(before_fusion_groups)

        fusion_blocks = []
        for group in after_fusion_groups:
            fusion_blocks.append(ExecutionBlock(group))
        return fusion_blocks
