// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from '../../../tensor';
import {WebGpuInferenceHandler} from '../inference-handler';

export const shape = async(inferenceHandler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> => {
  validateInputs(inputs);
  return [new Tensor([inputs[0].dims.length], 'int32', undefined, undefined, new Int32Array(inputs[0].dims))];
};

const validateInputs = (inputs: Tensor[]): void => {
  if (!inputs || inputs.length !== 1) {
    throw new Error('Shape requires 1 input.');
  }
};