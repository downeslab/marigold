/*
Copyright (C) 2024–2025 Gregory Teicher

Author: Gregory Teicher

This file is part of Marigold.

Marigold is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Marigold is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Marigold.
If not, see <https://www.gnu.org/licenses/>.
*/

import { channelsRgb, channelsRgba } from "./image.js";

const randomWebAssemblyModule = await WebAssembly.compileStreaming(fetch("../random.wasm"));
const randomWebAssemblyInstance = await WebAssembly.instantiate(
  randomWebAssemblyModule,
  {
    env: {
    }
  }
);
randomWebAssemblyInstance.exports._start();

const neuralNetworkWasmModule = await WebAssembly.compileStreaming(fetch("../neural-network.wasm"));

const instance = await WebAssembly.instantiate(
  neuralNetworkWasmModule,
  {
    env: {
      exp: (x) => { return Math.exp(x); },
      pow: (base, exponent) => { return Math.pow(base, exponent); },
      cos: (x) => { return Math.cos(x); },
      sin: (x) => { return Math.sin(x); }
    }
  }
);
instance.exports._start();

const memoryPageSize = 64 * 1024;
const elementByteSize = 4;


class Layer {
  upstreamLayers = [];
  downstreamLayers = [];
  parameterSizes = [];
  parameterOffsets = [];
  gradientSizes = [];
  gradientOffsets = [];
  bufferSizes = [];
  bufferOffsets = [];

  currentHeight = null;
  currentWidth = null;
  currentChannels = null;

  constructor() { }
  initializeParametersAndGradients() { }
  zeroGradients() { }
  bufferSizesFor(height, width, channels) { }
  outputShapeFor(height, width, channels) { }
  forward() { }
  backward() { }
  currentForwardOutput() { }
  currentBackwardOutput() { }
  setTrainingMode() { }
  setInferenceMode() { }
}


class InputLayer extends Layer {
  cache = null;

  constructor() {
    super();
    delete this.backward;
    delete this.currentBackwardOutput;

    this.bufferSizes.push(null);
    this.bufferOffsets.push(null);
  }

  initializeParametersAndGradients() { }

  zeroGradients() { }

  bufferSizesFor(height, width, channels) {
    const bufferSizes = [];
    bufferSizes.push(height * width * channels);
    return bufferSizes;
  }

  outputShapeFor(height, width, channels) {
    return [height, width, channels];
  }

  forward(image, height, width, channels) {
    this.cache = image;
    this.currentHeight = height;
    this.currentWidth = width;
    this.currentChannels = channels;
  }

  currentForwardOutput() {
    return [
      this.cache,
      this.currentHeight,
      this.currentWidth,
      this.currentChannels
    ];
  }
}


class OutputLayer extends Layer {
  cache = null;

  constructor(upstreamLayer) {
    super();
    upstreamLayer.downstreamLayers.push(this);
    this.upstreamLayers.push(upstreamLayer);

    this.bufferSizes.push(null);
    this.bufferOffsets.push(null);
  }

  initializeParametersAndGradients() { }

  zeroGradients() { }

  bufferSizesFor(height, width, channels) {
    const bufferSizes = [];
    bufferSizes.push(height * width * channels);
    return bufferSizes;
  }

  outputShapeFor(height, width, channels) {
    return [height, width, channels];
  }

  forward() {
    this.currentHeight = this.upstreamLayers[0].currentHeight;
    this.currentWidth = this.upstreamLayers[0].currentWidth;
    this.currentChannels = this.upstreamLayers[0].currentChannels;
  }

  backward(gradient) {
    this.cache = gradient;
  }

  currentForwardOutput() {
    return this.upstreamLayers[0].currentForwardOutput();
  }

  currentBackwardOutput() {
    return [
      this.cache,
      this.currentHeight,
      this.currentWidth,
      this.currentChannels
    ];
  }
}


class AdditionLayer extends Layer {
  constructor(upstreamLayers) {
    super();
    for (const upstreamLayer of upstreamLayers) {
      upstreamLayer.downstreamLayers.push(this);
      this.upstreamLayers.push(upstreamLayer);
    }

    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
  }

  initializeParametersAndGradients() { }

  zeroGradients() { }

  bufferSizesFor(height, width, channels) {
    const bufferSizes = [];
    bufferSizes.push(height * width * channels);
    bufferSizes.push(height * width * channels);
    return bufferSizes;
  }

  outputShapeFor(height, width, channels) {
    return [height, width, channels];
  }

  forward() {
    let [inputOffset1, inputHeight1, inputWidth1, inputChannels1] = this.upstreamLayers[0].currentForwardOutput();
    let [inputOffset2, inputHeight2, inputWidth2, inputChannels2] = this.upstreamLayers[1].currentForwardOutput();

    instance.exports.add_forward(
      inputOffset1,
      inputOffset2,
      this.bufferOffsets[0],
      inputHeight1 * inputWidth1 * inputChannels1
    );

    this.currentHeight = inputHeight1;
    this.currentWidth = inputWidth1;
    this.currentChannels = inputChannels1;
  }

  backward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    instance.exports.zero(this.bufferOffsets[1], this.bufferSizes[1]);

    for (const downstreamLayer of this.downstreamLayers) {
      instance.exports.add_backward(
        downstreamLayer.currentBackwardOutput()[0],
        this.bufferOffsets[1],
        this.bufferSizes[1]
      );
    }
  }

  currentForwardOutput() {
    return [
      this.bufferOffsets[0],
      this.currentHeight,
      this.currentWidth,
      this.currentChannels
    ];
  }

  currentBackwardOutput() {
    return [
      this.bufferOffsets[1],
      this.currentHeight,
      this.currentWidth,
      this.currentChannels
    ];
  }
}


class HardSwishLayer extends Layer {
  constructor(upstreamLayer) {
    super();
    upstreamLayer.downstreamLayers.push(this);
    this.upstreamLayers.push(upstreamLayer);

    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
  }

  initializeParametersAndGradients() { }

  zeroGradients() { }

  bufferSizesFor(height, width, channels) {
    const bufferSizes = [];
    bufferSizes.push(height * width * channels);
    bufferSizes.push(height * width * channels);
    return bufferSizes;
  }

  outputShapeFor(height, width, channels) {
    return [height, width, channels];
  }

  forward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();

    instance.exports.hard_swish_forward(
      inputOffset,
      this.bufferOffsets[0],
      inputHeight * inputWidth * inputChannels
    );

    this.currentHeight = inputHeight;
    this.currentWidth = inputWidth;
    this.currentChannels = inputChannels;
  }

  backward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    instance.exports.zero(this.bufferOffsets[1], this.bufferSizes[1]);

    for (const downstreamLayer of this.downstreamLayers) {
      instance.exports.hard_swish_backward(
        downstreamLayer.currentBackwardOutput()[0],
        this.bufferOffsets[1],
        inputOffset,
        this.bufferSizes[1]
      );
    }
  }

  currentForwardOutput() {
    return [
      this.bufferOffsets[0],
      this.currentHeight,
      this.currentWidth,
      this.currentChannels
    ];
  }

  currentBackwardOutput() {
    return [
      this.bufferOffsets[1],
      this.currentHeight,
      this.currentWidth,
      this.currentChannels
    ];
  }
}


class DropoutLayer extends Layer {
  constructor(upstreamLayer, dropProbability = 0.5) {
    super();
    upstreamLayer.downstreamLayers.push(this);
    this.upstreamLayers.push(upstreamLayer);

    this.dropProbability = dropProbability;

    this.training = false;
    this.cache = null;

    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
  }

  initializeParametersAndGradients() { }

  zeroGradients() { }

  bufferSizesFor(height, width, channels) {
    const bufferSizes = [];
    bufferSizes.push(height * width * channels);
    bufferSizes.push(height * width * channels);
    bufferSizes.push(channels);
    return bufferSizes;
  }

  outputShapeFor(height, width, channels) {
    return [height, width, channels];
  }

  forward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();

    if (this.training) {
      const dropMaskBufferSize = this.bufferSizes[2];
      const dropMaskBufferOffset = this.bufferOffsets[2];
      const dropMaskBufferArray = new Float32Array(
        instance.exports.memory.buffer,
        dropMaskBufferOffset,
        dropMaskBufferSize
      );
      for (let i = 0; i < dropMaskBufferSize; ++i) {
        if (randomWebAssemblyInstance.exports.random_float() < this.dropProbability) {
          dropMaskBufferArray[i] = 0.0;
        }
        else {
          dropMaskBufferArray[i] = 1.0;
        }
      }

      instance.exports.dropout_forward(
        inputOffset,
        this.bufferOffsets[0],
        this.bufferOffsets[2],
        inputHeight,
        inputWidth,
        inputChannels,
        this.dropProbability
      );

      this.currentHeight = inputHeight;
      this.currentWidth = inputWidth;
      this.currentChannels = inputChannels;
    }
    else {
      this.cache = inputOffset;
      this.currentHeight = inputHeight;
      this.currentWidth = inputWidth;
      this.currentChannels = inputChannels;
    }
  }

  backward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    instance.exports.zero(this.bufferOffsets[1], this.bufferSizes[1]);

    for (const downstreamLayer of this.downstreamLayers) {
      instance.exports.dropout_backward(
        downstreamLayer.currentBackwardOutput()[0],
        this.bufferOffsets[1],
        this.bufferOffsets[2],
        inputHeight,
        inputWidth,
        inputChannels
      );
    }
  }

  currentForwardOutput() {
    if (this.training) {
      return [
        this.bufferOffsets[0],
        this.currentHeight,
        this.currentWidth,
        this.currentChannels
      ];
    }
    else {
      return [
        this.cache,
        this.currentHeight,
        this.currentWidth,
        this.currentChannels
      ];
    }
  }

  currentBackwardOutput() {
    if (this.training) {
      return [
        this.bufferOffsets[1],
        this.currentHeight,
        this.currentWidth,
        this.currentChannels
      ];
    }
    else {
      return [
        downstreamLayer.currentBackwardOutput()[0],
        this.currentHeight,
        this.currentWidth,
        this.currentChannels
      ];
    }
  }

  setTrainingMode() {
    this.training = true;
  }

  setInferenceMode() {
    this.training = false;
  }
}


class PixelUnshuffleLayer extends Layer {
  stride = null;

  constructor(upstreamLayer, stride) {
    super();
    upstreamLayer.downstreamLayers.push(this);
    this.upstreamLayers.push(upstreamLayer);

    this.stride = stride;

    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
  }

  initializeParametersAndGradients() { }

  zeroGradients() { }

  bufferSizesFor(height, width, channels) {
    const bufferSizes = [];
    bufferSizes.push(height * width * channels);
    bufferSizes.push(height * width * channels);
    return bufferSizes;
  }

  outputShapeFor(height, width, channels) {
    return [Math.trunc(height / this.stride), Math.trunc(width / this.stride), channels * (this.stride * this.stride)];
  }

  forward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();

    instance.exports.pixel_unshuffle_forward(
      inputOffset,
      this.bufferOffsets[0],
      Math.trunc(inputHeight / this.stride),
      Math.trunc(inputWidth / this.stride),
      inputChannels * (this.stride * this.stride)
    );

    this.currentHeight = Math.trunc(inputHeight / this.stride);
    this.currentWidth = Math.trunc(inputWidth / this.stride);
    this.currentChannels = inputChannels * (this.stride * this.stride);
  }

  backward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    instance.exports.zero(this.bufferOffsets[1], this.bufferSizes[1]);

    for (const downstreamLayer of this.downstreamLayers) {
      instance.exports.pixel_unshuffle_backward(
        downstreamLayer.currentBackwardOutput()[0],
        this.bufferOffsets[1],
        Math.trunc(inputHeight / this.stride),
        Math.trunc(inputWidth / this.stride),
        inputChannels * (this.stride * this.stride)
      );
    }
  }

  currentForwardOutput() {
    return [
      this.bufferOffsets[0],
      this.currentHeight,
      this.currentWidth,
      this.currentChannels
    ];
  }

  currentBackwardOutput() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    return [
      this.bufferOffsets[1],
      inputHeight,
      inputWidth,
      inputChannels
    ];
  }
}


class PixelShuffleLayer extends Layer {
  stride = null;

  constructor(upstreamLayer, stride) {
    super();
    upstreamLayer.downstreamLayers.push(this);
    this.upstreamLayers.push(upstreamLayer);

    this.stride = stride;

    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
  }

  initializeParametersAndGradients() { }

  zeroGradients() { }

  bufferSizesFor(height, width, channels) {
    const bufferSizes = [];
    bufferSizes.push(height * width * channels);
    bufferSizes.push(height * width * channels);
    return bufferSizes;
  }

  outputShapeFor(height, width, channels) {
    return [height * this.stride, width * this.stride, channels / (this.stride * this.stride)];
  }

  forward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();

    instance.exports.pixel_shuffle_forward(
      inputOffset,
      this.bufferOffsets[0],
      inputHeight,
      inputWidth,
      inputChannels
    );

    this.currentHeight = inputHeight * this.stride;
    this.currentWidth = inputWidth * this.stride;
    this.currentChannels = inputChannels / (this.stride * this.stride);
  }

  backward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    instance.exports.zero(this.bufferOffsets[1], this.bufferSizes[1]);

    for (const downstreamLayer of this.downstreamLayers) {
      instance.exports.pixel_shuffle_backward(
        downstreamLayer.currentBackwardOutput()[0],
        this.bufferOffsets[1],
        inputHeight,
        inputWidth,
        inputChannels
      );
    }
  }

  currentForwardOutput() {
    return [
      this.bufferOffsets[0],
      this.currentHeight,
      this.currentWidth,
      this.currentChannels
    ];
  }

  currentBackwardOutput() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    return [
      this.bufferOffsets[1],
      inputHeight,
      inputWidth,
      inputChannels
    ];
  }
}


class InstanceNormalizationLayer extends Layer {
  channels = null;
  epsilon = null;

  constructor(upstreamLayer, channels, epsilon = 1.0e-3) {
    super();
    upstreamLayer.downstreamLayers.push(this);
    this.upstreamLayers.push(upstreamLayer);

    this.channels = channels;
    this.epsilon = epsilon;

    const gammaSize = this.channels;
    const betaSize = this.channels;
    this.parameterSizes.push(gammaSize);
    this.parameterSizes.push(betaSize);
    this.gradientSizes.push(gammaSize);
    this.gradientSizes.push(betaSize);

    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
  }

  initializeParametersAndGradients() {
    const gammaSize = this.parameterSizes[0];
    const gammaOffset = this.parameterOffsets[0];
    const gammaArray = new Float32Array(
      instance.exports.memory.buffer,
      gammaOffset,
      gammaSize
    );
    for (let i = 0; i < gammaSize; ++i) {
      gammaArray[i] = 1.0;
    }

    const betaSize = this.parameterSizes[1];
    const betaOffset = this.parameterOffsets[1];
    const betaArray = new Float32Array(
      instance.exports.memory.buffer,
      betaOffset,
      betaSize
    );
    for (let i = 0; i < betaSize; ++i) {
      betaArray[i] = 0.0;
    }
  }

  zeroGradients() {
    instance.exports.zero(this.gradientOffsets[0], this.gradientSizes[0]);
    instance.exports.zero(this.gradientOffsets[1], this.gradientSizes[1]);
  }

  bufferSizesFor(height, width, channels) {
    const bufferSizes = [];

    bufferSizes.push(height * width * channels);
    bufferSizes.push(height * width * channels);
    bufferSizes.push(channels); // sample_mean
    bufferSizes.push(channels); // sample_std_dev
    bufferSizes.push(channels); // sum_1
    bufferSizes.push(channels); // sum_2

    return bufferSizes;
  }

  outputShapeFor(height, width, channels) {
    return [height, width, channels];
  }

  forward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();

    instance.exports.instance_normalization_forward(
      inputOffset,
      this.bufferOffsets[0],
      this.parameterOffsets[0],
      this.parameterOffsets[1],
      this.bufferOffsets[2],
      this.bufferOffsets[3],
      this.epsilon,
      inputHeight,
      inputWidth,
      inputChannels
    );

    this.currentHeight = inputHeight;
    this.currentWidth = inputWidth;
    this.currentChannels = inputChannels;
  }

  backward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    instance.exports.zero(this.bufferOffsets[1], this.bufferSizes[1]);
    instance.exports.zero(this.bufferOffsets[4], this.bufferSizes[4]);
    instance.exports.zero(this.bufferOffsets[5], this.bufferSizes[5]);

    for (const downstreamLayer of this.downstreamLayers) {
      instance.exports.instance_normalization_backward(
        downstreamLayer.currentBackwardOutput()[0],
        this.bufferOffsets[1],
        this.gradientOffsets[0],
        this.gradientOffsets[1],
        this.parameterOffsets[0],
        this.bufferOffsets[2],
        this.bufferOffsets[3],
        inputOffset,
        this.bufferOffsets[4],
        this.bufferOffsets[5],
        inputHeight,
        inputWidth,
        inputChannels
      );
    }
  }

  currentForwardOutput() {
    return [
      this.bufferOffsets[0],
      this.currentHeight,
      this.currentWidth,
      this.currentChannels
    ];
  }

  currentBackwardOutput() {
    return [
      this.bufferOffsets[1],
      this.currentHeight,
      this.currentWidth,
      this.currentChannels
    ];
  }
}


class PointwiseConvolutionLayer extends Layer {
  channelsIn = null;
  channelsOut = null;
  gain = null;

  constructor(upstreamLayer, channelsIn, channelsOut, gain = 1.0) {
    super();
    upstreamLayer.downstreamLayers.push(this);
    this.upstreamLayers.push(upstreamLayer);

    this.channelsIn = channelsIn;
    this.channelsOut = channelsOut;
    this.gain = gain;

    const kernelSize = this.channelsIn * this.channelsOut;
    const biasSize = this.channelsOut;
    this.parameterSizes.push(kernelSize);
    this.parameterSizes.push(biasSize);
    this.gradientSizes.push(kernelSize);
    this.gradientSizes.push(biasSize);

    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferSizes.push(null); // kernel buffer
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null); // kernel buffer
  }

  initializeParametersAndGradients() {
    const fanIn = this.channelsIn;
    const fanOut = this.channelsOut;
    const glorot_uniform = Math.sqrt(6.0 / (fanIn + fanOut));

    const kernelSize = this.parameterSizes[0];
    const kernelOffset = this.parameterOffsets[0];
    const kernelArray = new Float32Array(
      instance.exports.memory.buffer,
      kernelOffset,
      kernelSize
    );
    for (let i = 0; i < kernelSize; ++i) {
      kernelArray[i] = ((randomWebAssemblyInstance.exports.random_float() - 0.5) * 2.0) * glorot_uniform * this.gain;
    }

    const biasSize = this.parameterSizes[1];
    const biasOffset = this.parameterOffsets[1];
    const biasArray = new Float32Array(
      instance.exports.memory.buffer,
      biasOffset,
      biasSize
    );
    for (let i = 0; i < biasSize; ++i) {
      biasArray[i] = 0.0;
    }
  }

  zeroGradients() {
    instance.exports.zero(this.gradientOffsets[0], this.gradientSizes[0]);
    instance.exports.zero(this.gradientOffsets[1], this.gradientSizes[1]);
  }

  bufferSizesFor(height, width, channels) {
    const bufferSizes = [];

    bufferSizes.push(height * width * this.channelsOut); // y
    bufferSizes.push(height * width * this.channelsIn); // d_x
    bufferSizes.push(this.parameterSizes[0]); // kernel buffer

    return bufferSizes;
  }

  outputShapeFor(height, width, channels) {
    return [height, width, this.channelsOut];
  }

  forward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();

    instance.exports.pointwise_convolution_forward(
      inputOffset,
      this.bufferOffsets[0],
      this.parameterOffsets[0],
      this.parameterOffsets[1],
      inputHeight,
      inputWidth,
      this.channelsIn,
      this.channelsOut
    );

    this.currentHeight = inputHeight;
    this.currentWidth = inputWidth;
    this.currentChannels = this.channelsOut;
  }

  backward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    instance.exports.zero(this.bufferOffsets[1], this.bufferSizes[1]);

    for (const downstreamLayer of this.downstreamLayers) {
      instance.exports.pointwise_convolution_backward(
        downstreamLayer.currentBackwardOutput()[0],
        this.bufferOffsets[1],
        this.gradientOffsets[0],
        this.gradientOffsets[1],
        inputOffset,
        this.parameterOffsets[0],
        this.bufferOffsets[2],
        inputHeight,
        inputWidth,
        this.channelsIn,
        this.channelsOut
      );
    }
  }

  currentForwardOutput() {
    return [
      this.bufferOffsets[0],
      this.currentHeight,
      this.currentWidth,
      this.channelsOut
    ];
  }

  currentBackwardOutput() {
    return [
      this.bufferOffsets[1],
      this.currentHeight,
      this.currentWidth,
      this.channelsIn
    ];
  }
}


class DepthwiseConvolutionLayer extends Layer {
  channels = null;
  filterSize = null;
  gain = null;

  constructor(upstreamLayer, channels, filterSize, gain = 1.0) {
    super();
    upstreamLayer.downstreamLayers.push(this);
    this.upstreamLayers.push(upstreamLayer);

    this.channels = channels;
    this.filterSize = filterSize;
    this.gain = gain;

    const kernelSize = this.channels * this.filterSize * this.filterSize;
    const biasSize = this.channels;
    this.parameterSizes.push(kernelSize);
    this.parameterSizes.push(biasSize);
    this.gradientSizes.push(kernelSize);
    this.gradientSizes.push(biasSize);

    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
  }

  initializeParametersAndGradients() {
    const fanIn = this.channels * this.filterSize * this.filterSize;
    const fanOut = this.channels * this.filterSize * this.filterSize;
    const glorot_uniform = Math.sqrt(6.0 / (fanIn + fanOut));

    const kernelSize = this.parameterSizes[0];
    const kernelOffset = this.parameterOffsets[0];
    const kernelArray = new Float32Array(
      instance.exports.memory.buffer,
      kernelOffset,
      kernelSize
    );
    for (let i = 0; i < kernelSize; ++i) {
      kernelArray[i] = ((randomWebAssemblyInstance.exports.random_float() - 0.5) * 2.0) * glorot_uniform * this.gain;
    }

    const biasSize = this.parameterSizes[1];
    const biasOffset = this.parameterOffsets[1];
    const biasArray = new Float32Array(
      instance.exports.memory.buffer,
      biasOffset,
      biasSize
    );
    for (let i = 0; i < biasSize; ++i) {
      biasArray[i] = 0.0;
    }
  }

  zeroGradients() {
    instance.exports.zero(this.gradientOffsets[0], this.gradientSizes[0]);
    instance.exports.zero(this.gradientOffsets[1], this.gradientSizes[1]);
  }

  bufferSizesFor(height, width, channels) {
    const bufferSizes = [];

    bufferSizes.push(height * width * channels); // y
    bufferSizes.push(height * width * channels); // d_x

    return bufferSizes;
  }

  outputShapeFor(height, width, channels) {
    return [height, width, channels];
  }

  forward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();

    instance.exports.depthwise_convolution_forward(
      inputOffset,
      this.bufferOffsets[0],
      this.parameterOffsets[0],
      this.parameterOffsets[1],
      inputHeight,
      inputWidth,
      inputChannels
    );

    this.currentHeight = inputHeight;
    this.currentWidth = inputWidth;
    this.currentChannels = inputChannels;
  }

  backward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    instance.exports.zero(this.bufferOffsets[1], this.bufferSizes[1]);

    for (const downstreamLayer of this.downstreamLayers) {
      instance.exports.depthwise_convolution_backward(
        downstreamLayer.currentBackwardOutput()[0],
        this.bufferOffsets[1],
        this.gradientOffsets[0],
        this.gradientOffsets[1],
        this.parameterOffsets[0],
        inputOffset,
        inputHeight,
        inputWidth,
        inputChannels
      );
    }
  }

  currentForwardOutput() {
    return [
      this.bufferOffsets[0],
      this.currentHeight,
      this.currentWidth,
      this.currentChannels
    ];
  }

  currentBackwardOutput() {
    return [
      this.bufferOffsets[1],
      this.currentHeight,
      this.currentWidth,
      this.currentChannels
    ];
  }
}


export class NeuralNetwork {
  layers = [];
  layersReversed = [];

  parameterOffset = null;
  gradientOffset = null;
  optimizerOffset = null;
  bufferOffset = null;

  parameterLength = null;
  lastOffset = null;

  optimizerT = 1;

  channelsIn = null;
  channelsMiddle = null;
  channelsOut = null;
  blockCount = null;

  learningRate = null;

  // constructor(channelsIn = 1, channelsMiddle, channelsOut, blockCount, maxImageSize, learningRate) {
  constructor(channelsIn = 3, channelsMiddle, channelsOut, blockCount, maxImageSize, learningRate) {
    this.channelsIn = channelsIn;
    this.channelsMiddle = channelsMiddle;
    this.channelsOut = channelsOut;
    this.blockCount = blockCount;
    this.maxImageSize = maxImageSize;
    this.learningRate = learningRate;

    let expansionRatio = 2;
    let outroExpansionRatio = 2;

    const inputLayer = new InputLayer();
    this.layers.push(inputLayer);

    const introPixelUnshuffle = new PixelUnshuffleLayer(inputLayer, 8);
    this.layers.push(introPixelUnshuffle);

    const introPointwiseConv = new PointwiseConvolutionLayer(introPixelUnshuffle, channelsIn * 8 * 8, this.channelsMiddle, 1.0);
    this.layers.push(introPointwiseConv);

    const introInstanceNorm = new InstanceNormalizationLayer(introPointwiseConv, this.channelsMiddle);
    this.layers.push(introInstanceNorm);

    let previousLayer = introInstanceNorm;

    for (let i = 0; i < this.blockCount; ++i) {
      const expansionConv = new PointwiseConvolutionLayer(previousLayer, this.channelsMiddle, this.channelsMiddle * expansionRatio, Math.sqrt(2.0));
      this.layers.push(expansionConv);

      const hardSwish = new HardSwishLayer(expansionConv);
      this.layers.push(hardSwish);

      const depthwiseConv = new DepthwiseConvolutionLayer(hardSwish, this.channelsMiddle * expansionRatio, 5, 1.0);
      this.layers.push(depthwiseConv);

      const instanceNorm = new InstanceNormalizationLayer(depthwiseConv, this.channelsMiddle * expansionRatio);
      this.layers.push(instanceNorm);

      const reductionConv = new PointwiseConvolutionLayer(instanceNorm, this.channelsMiddle * expansionRatio, this.channelsMiddle, 1.0 / Math.sqrt(this.blockCount));
      this.layers.push(reductionConv);

      const addition = new AdditionLayer([previousLayer, reductionConv]);
      this.layers.push(addition);

      previousLayer = addition;
    }

    const outroExpansionConv = new PointwiseConvolutionLayer(previousLayer, this.channelsMiddle, this.channelsMiddle * outroExpansionRatio, Math.sqrt(2.0));
    this.layers.push(outroExpansionConv);

    const outroHardSwish = new HardSwishLayer(outroExpansionConv, false);
    this.layers.push(outroHardSwish);

    // const outroDropout = new DropoutLayer(outroHardSwish, 0.05);
    const outroDropout = new DropoutLayer(outroHardSwish, 0.1);
    // const outroDropout = new DropoutLayer(outroHardSwish, 0.2);
    this.layers.push(outroDropout);

    const outroLinearConv = new PointwiseConvolutionLayer(outroDropout, this.channelsMiddle * outroExpansionRatio, channelsOut * 4 * 4, 0.0);
    this.layers.push(outroLinearConv);

    const outroPixelShuffle = new PixelShuffleLayer(outroLinearConv, 4);
    this.layers.push(outroPixelShuffle);

    const outputLayer = new OutputLayer(outroPixelShuffle);

    this.layers.push(outputLayer);

    this.layersReversed = this.layers.toReversed();
    this.initializeParametersAndGradients();
  }

  initializeParametersAndGradients() {
    let offset = instance.exports.heap_base();

    this.parameterOffset = offset;
    for (const layer of this.layers) {
      if (layer.parameterSizes.length) {
        for (const parameterSize of layer.parameterSizes) {
          layer.parameterOffsets.push(offset);
          offset += parameterSize * elementByteSize;
        }
      }
    }

    this.gradientOffset = offset;
    for (const layer of this.layers) {
      if (layer.gradientSizes.length) {
        for (const gradientSize of layer.gradientSizes) {
          layer.gradientOffsets.push(offset);
          offset += gradientSize * elementByteSize;
        }
      }
    }

    this.optimizerOffset = offset;
    this.parameterLength = (this.gradientOffset - this.parameterOffset) / elementByteSize;
    offset += 2 * this.parameterLength * elementByteSize;

    this.bufferOffset = offset;

    let tempHeight = this.maxImageSize;
    let tempWidth = this.maxImageSize;
    let tempChannels = this.channelsIn;
    for (const layer of this.layers) {
      const bufferSizes = layer.bufferSizesFor(tempHeight, tempWidth, tempChannels);
      for (let i = 0; i < bufferSizes.length; ++i) {
        layer.bufferSizes[i] = bufferSizes[i];
        layer.bufferOffsets[i] = offset;
        offset += layer.bufferSizes[i] * elementByteSize;
      }
      [tempHeight, tempWidth, tempChannels] = layer.outputShapeFor(tempHeight, tempWidth, tempChannels);
    }

    this.originalOffset = offset;

    offset += 4096 * 4096 * 4 * elementByteSize;
    this.resizedOffset = offset;
    offset += 4096 * 4096 * 4 * elementByteSize;
    this.rotatedOffset = offset;
    offset += 4096 * 4096 * 4 * elementByteSize;
    this.grayOffset = offset;
    offset += 4096 * 4096 * 4 * elementByteSize;
    this.gaussianOffset = offset;
    offset += 4096 * 4096 * 10 * elementByteSize;
    this.gaussianGradientOffset = offset;
    offset += 4096 * 4096 * 10 * elementByteSize;
    this.gaussianCoordinatesOffset = offset;
    offset += 2 * 10 * elementByteSize;

    this.lastOffset = offset;

    const extraPagesNeeded = Math.ceil(this.lastOffset / memoryPageSize) - instance.exports.memory.buffer.byteLength / memoryPageSize;
    if (extraPagesNeeded > 0) {
      const previousPageCount = instance.exports.memory.grow(extraPagesNeeded);
    }

    instance.exports.zero(instance.exports.heap_base(), (this.lastOffset - instance.exports.heap_base()) / elementByteSize);
    for (const layer of this.layers) {
      layer.initializeParametersAndGradients();
    }
  }

  forward(image, height, width, channels) {
    let offset = this.bufferOffset;

    let tempHeight = height;
    let tempWidth = width;
    let tempChannels = channels;

    for (const layer of this.layers) {
      const bufferSizes = layer.bufferSizesFor(tempHeight, tempWidth, tempChannels);
      for (let i = 0; i < bufferSizes.length; ++i) {
        layer.bufferSizes[i] = bufferSizes[i];
        layer.bufferOffsets[i] = offset;
        offset += layer.bufferSizes[i] * elementByteSize;
      }
      [tempHeight, tempWidth, tempChannels] = layer.outputShapeFor(tempHeight, tempWidth, tempChannels);
    }

    let index = 0;
    for (const layer of this.layers) {
      if (index === 0) {
        layer.forward(image, height, width, channels); // Feed data to input layer.
      }
      else {
        layer.forward();
      }
      ++index;
    }
  }

  backward(gradient) {
    let index = 0;
    for (const layer of this.layersReversed) {
      if (index === 0) {
        layer.backward(gradient); // Feed data to output layer.
      }
      else if (index === this.layersReversed.length - 1) {
        // Don't backpropagate through input layer.
      }
      else {
        layer.backward();
      }
      ++index;
    }
  }

  predictions() {
    const predictionsArray = new Float32Array(
      instance.exports.memory.buffer,
      this.layers[this.layers.length - 2].bufferOffsets[0],
      this.layers[this.layers.length - 1].currentHeight * this.layers[this.layers.length - 1].currentWidth * this.layers[this.layers.length - 1].currentChannels
    );
    return predictionsArray;
  }

  zeroGradients() {
    for (const layer of this.layers) {
      layer.zeroGradients();
    }
  }

  updateParameters() {
    const beta1 = 0.9;
    const beta2 = 0.95;
    const epsilon = 1.0e-6;

    const weightDecay = 1.0e-5;

    const scheduleMultiplier = 1.0;

    instance.exports.update_parameters(
      this.gradientOffset,
      this.parameterOffset,
      this.optimizerOffset, // first moment
      this.optimizerOffset + this.parameterLength * elementByteSize, // second moment
      this.parameterLength,
      beta1,
      beta2,
      epsilon,
      scheduleMultiplier,
      this.learningRate,
      weightDecay,
      this.optimizerT
    );
    ++this.optimizerT;
  }

  resize(x, heightIn, widthIn, heightOut, widthOut) {
    const xArray = new Uint8ClampedArray(
      instance.exports.memory.buffer,
      this.originalOffset,
      heightIn * widthIn * channelsRgba
    );

    for (let i = 0; i < heightIn * widthIn * channelsRgba; ++i) {
      xArray[i] = x[i];
    }

    instance.exports.resize_bilinear_rgba_to_rgb(this.originalOffset, this.resizedOffset, heightIn, widthIn, heightOut, widthOut);
  }

  flipHorizontal(height, width) {
    instance.exports.flip_horizontal(this.resizedOffset, height, width);
  }

  flipVertical(height, width) {
    instance.exports.flip_vertical(this.resizedOffset, height, width);
  }

  adjustBrightness(height, width, brightness) {
    instance.exports.adjust_brightness(this.resizedOffset, height, width, brightness);
  }

  adjustGamma(height, width, gamma) {
    instance.exports.adjust_gamma(this.resizedOffset, height, width, gamma);
  }

  rotate(height, width, theta) {
    instance.exports.rotate_bilinear(this.resizedOffset, this.rotatedOffset, height, width, theta);
  }

  rgbToGrayTraining(height, width) {
    instance.exports.rgb_to_gray(this.rotatedOffset, this.grayOffset, height, width);
  }

  rgbToGrayInference(height, width) {
    instance.exports.rgb_to_gray(this.resizedOffset, this.grayOffset, height, width);
  }

  drawGaussians(resizedGaussianHeight, resizedGaussianWidth, keypointCount, coordinates, gaussianStdDev) {
    const coordinatesArray = new Float32Array(
      instance.exports.memory.buffer,
      this.gaussianCoordinatesOffset,
      2 * keypointCount
    );

    for (let i = 0; i < keypointCount; ++i) {
      coordinatesArray[i * 2 + 0] = coordinates[i][0];
      coordinatesArray[i * 2 + 1] = coordinates[i][1];
    }

    instance.exports.draw_gaussians(this.gaussianOffset, resizedGaussianHeight, resizedGaussianWidth, keypointCount, this.gaussianCoordinatesOffset, gaussianStdDev);
  }

  lossForward() {
    return instance.exports.mean_squared_error_forward(
      this.layers[this.layers.length - 2].bufferOffsets[0],
      this.gaussianOffset,
      this.layers[this.layers.length - 1].currentHeight * this.layers[this.layers.length - 1].currentWidth * this.layers[this.layers.length - 1].currentChannels
    );
  }

  lossBackward(loss) {
    instance.exports.mean_squared_error_backward(
      loss,
      this.gaussianGradientOffset,
      this.layers[this.layers.length - 2].bufferOffsets[0],
      this.gaussianOffset,
      this.layers[this.layers.length - 1].currentHeight * this.layers[this.layers.length - 1].currentWidth * this.layers[this.layers.length - 1].currentChannels
    );
  }

  getParameters() {
    const parameterArray = new Float32Array(
      instance.exports.memory.buffer,
      this.parameterOffset,
      this.parameterLength
    );

    const parameterArrayCopy = new Float32Array(this.parameterLength);

    for (let i = 0; i < this.parameterLength; ++i) {
      parameterArrayCopy[i] = parameterArray[i];
    }

    return parameterArrayCopy;
  }

  setParameters(parameterArray) {
    const parameterArrayActual = new Float32Array(
      instance.exports.memory.buffer,
      this.parameterOffset,
      this.parameterLength
    );

    for (let i = 0; i < this.parameterLength; ++i) {
      parameterArrayActual[i] = parameterArray[i];
    }
  }

  seed(seed) {
    randomWebAssemblyInstance.exports.seed(BigInt(seed));
  }

  randomInteger(low, high) {
    return randomWebAssemblyInstance.exports.random_integer(low, high);
  }

  randomFloat() {
    return randomWebAssemblyInstance.exports.random_float();
  }

  setTrainingMode() {
    for (const layer of this.layers) {
      layer.setTrainingMode();
    }
  }

  setInferenceMode() {
    for (const layer of this.layers) {
      layer.setInferenceMode();
    }
  }
}
