/*
Copyright (C) 2024 Gregory Teicher

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

// const instance = await WebAssembly.instantiate(neuralNetworkWasmModule);
const instance = await WebAssembly.instantiate(
  neuralNetworkWasmModule,
  {
    env: {
      exp: (x) => { return Math.exp(x); },
      log: (x) => { return Math.log(x); },
      pow: (x, y) => { return Math.pow(x, y); }
    }
  }
);
instance.exports._start();

// for (let i = 0; i < 10; ++i) {
//   // console.log(instance.exports.b1_vals_table_lookup(i), instance.exports.b2_vals_table_lookup(i));
// }

// for (let i = 0; i < 10; ++i) {
//   // console.log("exp:", Math.exp(-i / 5), instance.exports.fast_approx_exp(-i / 5));
// }

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
    // const imageArray = new Float32Array(
    //   instance.exports.memory.buffer,
    //   this.bufferOffsets[0],
    //   height * width * channels
    // );

    // for (let i = 0; i < height * width * channels; ++i) {
    //   imageArray[i] = image[i];
    // }

    this.cache = image;

    this.currentHeight = height;
    this.currentWidth = width;
    this.currentChannels = channels;
  }

  currentForwardOutput() {
    return [
      // this.bufferOffsets[0],
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
    // const gradientArray = new Float32Array(
    //   instance.exports.memory.buffer,
    //   this.bufferOffsets[0],
    //   this.currentHeight * this.currentWidth * this.currentChannels
    // );

    // for (let i = 0; i < this.currentHeight * this.currentWidth * this.currentChannels; ++i) {
    //   gradientArray[i] = gradient[i];
    // }

    //

    // {
    //   const gradientArray = new Float32Array(
    //     instance.exports.memory.buffer,
    //     this.bufferOffsets[0],
    //     this.currentHeight * this.currentWidth * this.currentChannels
    //   );

    //   const gradientArray_ = new Float32Array(
    //     instance.exports.memory.buffer,
    //     gradient,
    //     this.currentHeight * this.currentWidth * this.currentChannels
    //   );

    //   for (let i = 0; i < this.currentHeight * this.currentWidth * this.currentChannels; ++i) {
    //     gradientArray[i] = gradientArray_[i];
    //     // console.log("values:", gradientArray[i], gradientArray_[i]);
    //   }
    // }

    this.cache = gradient;
  }

  currentForwardOutput() {
    return this.upstreamLayers[0].currentForwardOutput();
  }

  currentBackwardOutput() {
    return [
      // this.bufferOffsets[0],
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

    instance.exports.merge(
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
      instance.exports.accumulate(
        downstreamLayer.currentBackwardOutput()[0],
        this.bufferOffsets[1],
        this.bufferSizes[1]);
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

    {
      // const start = performance.now();
      instance.exports.hard_swish_forward(
        inputOffset,
        this.bufferOffsets[0],
        inputHeight * inputWidth * inputChannels
      );
      // const finish = performance.now();
      // console.log("[hard_swish forward] t =", (finish - start) / 1000, "s");
    }

    this.currentHeight = inputHeight;
    this.currentWidth = inputWidth;
    this.currentChannels = inputChannels;
  }

  backward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    instance.exports.zero(this.bufferOffsets[1], this.bufferSizes[1]);

    {
      // const start = performance.now();
      for (const downstreamLayer of this.downstreamLayers) {
        instance.exports.hard_swish_backward(
          downstreamLayer.currentBackwardOutput()[0],
          this.bufferOffsets[1],
          inputOffset,
          this.bufferSizes[1],
        );
      }
      // const finish = performance.now();
      // console.log("[hard_swish backward] t =", (finish - start) / 1000, "s");
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

    {
      // const start = performance.now();
      instance.exports.pixel_shuffle_forward(
        inputOffset,
        this.bufferOffsets[0],
        inputHeight,
        inputWidth,
        inputChannels
      );
      // const finish = performance.now();
      // console.log("[pixel_shuffle forward] t =", (finish - start) / 1000, "s");
    }

    this.currentHeight = inputHeight * this.stride;
    this.currentWidth = inputWidth * this.stride;
    this.currentChannels = inputChannels / (this.stride * this.stride);
  }

  backward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    instance.exports.zero(this.bufferOffsets[1], this.bufferSizes[1]);

    {
      // const start = performance.now();
      for (const downstreamLayer of this.downstreamLayers) {
        instance.exports.pixel_shuffle_backward(
          downstreamLayer.currentBackwardOutput()[0],
          this.bufferOffsets[1],
          inputHeight,
          inputWidth,
          inputChannels
        );
      }
      // const finish = performance.now();
      // console.log("[pixel_shuffle backward] t =", (finish - start) / 1000, "s");
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

  // constructor(upstreamLayer, channels, epsilon = 1.0e-5) {
  constructor(upstreamLayer, channels, epsilon = 1.0e-4) {
    // constructor(upstreamLayer, channels, epsilon = 1.0e-3) {
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

    {
      // const start = performance.now();
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
      // const finish = performance.now();
      // console.log("[norm forward] t =", (finish - start) / 1000, "s");
    }


    {
      // // const start = performance.now();
      // instance.exports.instance_normalization_forward_template_outer(
      //   inputOffset,
      //   this.bufferOffsets[0],
      //   this.parameterOffsets[0],
      //   this.parameterOffsets[1],
      //   this.bufferOffsets[2],
      //   this.bufferOffsets[3],
      //   this.epsilon,
      //   inputHeight,
      //   inputWidth,
      //   inputChannels
      // );
      // // const finish = performance.now();
      // console.log("[norm new] t =", (finish - start) / 1000, "s");
    }

    this.currentHeight = inputHeight;
    this.currentWidth = inputWidth;
    this.currentChannels = inputChannels;
  }

  backward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    instance.exports.zero(this.bufferOffsets[1], this.bufferSizes[1]);
    instance.exports.zero(this.bufferOffsets[4], this.bufferSizes[4]);
    instance.exports.zero(this.bufferOffsets[5], this.bufferSizes[5]);

    {
      // const start = performance.now();
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
        // const finish = performance.now();
        // console.log("[norm backward] t =", (finish - start) / 1000, "s");
      }
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
  groups = null;

  constructor(upstreamLayer, channelsIn, channelsOut, groups) {
    super();
    upstreamLayer.downstreamLayers.push(this);
    this.upstreamLayers.push(upstreamLayer);

    this.channelsIn = channelsIn;
    this.channelsOut = channelsOut;
    this.groups = groups;

    // const kernelSize = this.channelsIn * this.channelsOut;
    const kernelSize = this.groups * (this.channelsIn / this.groups) * (this.channelsOut / this.groups);
    const biasSize = this.channelsOut;
    this.parameterSizes.push(kernelSize);
    this.parameterSizes.push(biasSize);
    this.gradientSizes.push(kernelSize);
    this.gradientSizes.push(biasSize);

    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferSizes.push(null); // kernel buffer
    this.bufferSizes.push(null); // kernel_ buffer
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null); // kernel buffer
    this.bufferOffsets.push(null); // kernel_ buffer
  }

  initializeParametersAndGradients() {
    const fanIn = this.channelsIn;
    const fanOut = this.channelsOut;
    // const fanIn = this.channelsIn / this.groups;
    // const fanOut = this.channelsOut / this.groups;
    const glorot_uniform = Math.sqrt(6.0 / (fanIn + fanOut));

    const kernelSize = this.parameterSizes[0];
    const kernelOffset = this.parameterOffsets[0];
    const kernelArray = new Float32Array(
      instance.exports.memory.buffer,
      kernelOffset,
      kernelSize
    );
    for (let i = 0; i < kernelSize; ++i) {
      kernelArray[i] = ((randomWebAssemblyInstance.exports.random_float() - 0.5) * 2.0) * glorot_uniform;
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

    // bufferSizes.push(height * width * channels); // y
    // bufferSizes.push(height * width * this.channelsOut); // d_x
    bufferSizes.push(height * width * this.channelsOut); // y
    bufferSizes.push(height * width * this.channelsIn); // d_x
    bufferSizes.push(this.parameterSizes[0]); // kernel buffer

    bufferSizes.push(this.parameterSizes[0]); // kernel_ buffer

    // console.log("bufferSizes:");
    // console.log(bufferSizes[0]);
    // console.log(bufferSizes[1]);
    // console.log(bufferSizes[2]);
    // console.log(bufferSizes[3]);

    return bufferSizes;
  }

  outputShapeFor(height, width, channels) {
    return [height, width, this.channelsOut];
  }

  forward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();

    // console.log("bufferOffsets:");
    // for (const bufferOffset of this.bufferOffsets) {
    //   // console.log(bufferOffset);
    // }
    // console.log("!");

    // {
    //   // const start = performance.now();
    // instance.exports.pointwise_convolution_forward_grouped(
    //   inputOffset,
    //   this.bufferOffsets[0],
    //   this.parameterOffsets[0],
    //   this.parameterOffsets[1],
    //   inputHeight,
    //   inputWidth,
    //   this.channelsIn,
    //   this.channelsOut,
    //   this.groups
    // );
    //   // const finish = performance.now();
    //   // console.log("[pointwise forward] t =", (finish - start) / 1000, "s");
    // }

    // instance.exports.pointwise_convolution_forward_grouped(
    //   inputOffset,
    //   this.bufferOffsets[0],
    //   this.parameterOffsets[0],
    //   this.parameterOffsets[1],
    //   inputHeight,
    //   inputWidth,
    //   this.channelsIn,
    //   this.channelsOut,
    //   this.groups
    // );

    // console.log("this.channelsIn:", this.channelsIn, "this.channelsOut:", this.channelsOut);

    let n = 10;

    // {
    //   // const start = performance.now();
    //   for (let i = 0; i < n; ++i) {
    //     instance.exports.pointwise_convolution_forward(
    //       inputOffset,
    //       this.bufferOffsets[0],
    //       this.parameterOffsets[0],
    //       this.parameterOffsets[1],
    //       inputHeight,
    //       inputWidth,
    //       this.channelsIn,
    //       this.channelsOut
    //     );
    //   }
    //   // const finish = performance.now();
    //   // console.log("[pointwise forward] t =", (finish - start) / 1000, "s");
    // }

    // {
    //   // const start = performance.now();
    //   for (let i = 0; i < n; ++i) {
    //     instance.exports.pointwise_convolution_forward_test_1(
    //       inputOffset,
    //       this.bufferOffsets[0],
    //       this.parameterOffsets[0],
    //       this.parameterOffsets[1],
    //       this.bufferOffsets[3],
    //       inputHeight,
    //       inputWidth,
    //       this.channelsIn,
    //       this.channelsOut
    //     );
    //   }
    //   // const finish = performance.now();
    //   // console.log("[pointwise forward test 1] t =", (finish - start) / 1000, "s");
    // }

    // {
    //   // const start = performance.now();
    //   for (let i = 0; i < n; ++i) {
    //     instance.exports.pointwise_convolution_forward_test_2(
    //       inputOffset,
    //       this.bufferOffsets[0],
    //       this.parameterOffsets[0],
    //       this.parameterOffsets[1],
    //       this.bufferOffsets[3],
    //       inputHeight,
    //       inputWidth,
    //       this.channelsIn,
    //       this.channelsOut
    //     );
    //   }
    //   // const finish = performance.now();
    //   // console.log("[pointwise forward test 2] t =", (finish - start) / 1000, "s");
    // }

    {
      // const start = performance.now();
      // for (let i = 0; i < n; ++i) {
      instance.exports.pointwise_convolution_forward_test_3(
        inputOffset,
        this.bufferOffsets[0],
        this.parameterOffsets[0],
        this.parameterOffsets[1],
        this.bufferOffsets[3],
        inputHeight,
        inputWidth,
        this.channelsIn,
        this.channelsOut
      );
      // }
      // const finish = performance.now();
      // console.log("[pointwise forward test 3] t =", (finish - start) / 1000, "s");
    }

    // {
    //   // const start = performance.now();
    //   for (let i = 0; i < n; ++i) {
    //     instance.exports.pointwise_convolution_forward_test_4(
    //       inputOffset,
    //       this.bufferOffsets[0],
    //       this.parameterOffsets[0],
    //       this.parameterOffsets[1],
    //       this.bufferOffsets[3],
    //       inputHeight,
    //       inputWidth,
    //       this.channelsIn,
    //       this.channelsOut
    //     );
    //   }
    //   // const finish = performance.now();
    //   // console.log("[pointwise forward test 4] t =", (finish - start) / 1000, "s");
    // }

    // {
    //   // const start = performance.now();
    //   for (let i = 0; i < n; ++i) {
    //     instance.exports.pointwise_convolution_forward_test_5(
    //       inputOffset,
    //       this.bufferOffsets[0],
    //       this.parameterOffsets[0],
    //       this.parameterOffsets[1],
    //       this.bufferOffsets[3],
    //       inputHeight,
    //       inputWidth,
    //       this.channelsIn,
    //       this.channelsOut
    //     );
    //   }
    //   // const finish = performance.now();
    //   // console.log("[pointwise forward test 5] t =", (finish - start) / 1000, "s");
    // }

    // if (this.channelsIn == 4 && this.channelsOut == 8) {
    //   const k = new Float32Array(
    //     instance.exports.memory.buffer,
    //     this.parameterOffsets[0],
    //     this.parameterSizes[0]
    //   );

    //   const k_ = new Float32Array(
    //     instance.exports.memory.buffer,
    //     this.bufferOffsets[3],
    //     this.bufferSizes[3]
    //   );

    //   for (let i = 0; i < this.channelsIn; ++i) {
    //     for (let j = 0; j < this.channelsOut; ++j) {
    //       // console.log("k[", i, ", ", j, "]: ", k[i * this.channelsOut + j], " ; k_[", j, ", ", i, "]: ", k_[i * this.channelsOut + j]);
    //     }
    //   }
    // }

    this.currentHeight = inputHeight;
    this.currentWidth = inputWidth;
    this.currentChannels = this.channelsOut;
  }

  backward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    instance.exports.zero(this.bufferOffsets[1], this.bufferSizes[1]);
    // instance.exports.zero(this.bufferOffsets[2], this.bufferSizes[2]);

    // {
    //   // const start = performance.now();
    // for (const downstreamLayer of this.downstreamLayers) {
    //   instance.exports.pointwise_convolution_backward_grouped(
    //     downstreamLayer.currentBackwardOutput()[0],
    //     this.bufferOffsets[1],
    //     this.gradientOffsets[0],
    //     this.gradientOffsets[1],
    //     inputOffset,
    //     this.parameterOffsets[0],
    //     this.bufferOffsets[2],
    //     inputHeight,
    //     inputWidth,
    //     this.channelsIn,
    //     this.channelsOut,
    //     this.groups
    //   );
    //   // const finish = performance.now();
    //   // console.log("[pointwise backward] t =", (finish - start) / 1000, "s");
    // }

    // console.log("this.channelsIn:", this.channelsIn, "this.channelsOut:", this.channelsOut);

    let n = 10;

    // {
    //   // const start = performance.now();
    //   for (let i = 0; i < n; ++i) {
    //     for (const downstreamLayer of this.downstreamLayers) {
    //       instance.exports.pointwise_convolution_backward_test_1(
    //         downstreamLayer.currentBackwardOutput()[0],
    //         this.bufferOffsets[1],
    //         this.gradientOffsets[0],
    //         this.gradientOffsets[1],
    //         inputOffset,
    //         this.parameterOffsets[0],
    //         this.bufferOffsets[2],
    //         inputHeight,
    //         inputWidth,
    //         this.channelsIn,
    //         this.channelsOut
    //       );
    //     }
    //   }
    //   // const finish = performance.now();
    //   // console.log("[pointwise backward test 1] t =", (finish - start) / 1000, "s");
    // }

    {
      // const start = performance.now();
      // for (let i = 0; i < n; ++i) {
      for (const downstreamLayer of this.downstreamLayers) {
        instance.exports.pointwise_convolution_backward_test_2(
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
      // }
      // const finish = performance.now();
      // console.log("[pointwise backward test 2] t =", (finish - start) / 1000, "s");
    }

    // {
    //   // const start = performance.now();
    //   for (let i = 0; i < n; ++i) {
    //     for (const downstreamLayer of this.downstreamLayers) {
    //       instance.exports.pointwise_convolution_backward_test_3(
    //         downstreamLayer.currentBackwardOutput()[0],
    //         this.bufferOffsets[1],
    //         this.gradientOffsets[0],
    //         this.gradientOffsets[1],
    //         inputOffset,
    //         this.parameterOffsets[0],
    //         this.bufferOffsets[2],
    //         inputHeight,
    //         inputWidth,
    //         this.channelsIn,
    //         this.channelsOut
    //       );
    //     }
    //   }
    //   // const finish = performance.now();
    //   // console.log("[pointwise backward test 3] t =", (finish - start) / 1000, "s");
    // }

    // {
    //   // const start = performance.now();
    //   for (let i = 0; i < n; ++i) {
    //     for (const downstreamLayer of this.downstreamLayers) {
    //       instance.exports.pointwise_convolution_backward_d_out_test_1(
    //         downstreamLayer.currentBackwardOutput()[0],
    //         this.bufferOffsets[1],
    //         this.gradientOffsets[0],
    //         this.gradientOffsets[1],
    //         inputOffset,
    //         this.parameterOffsets[0],
    //         this.bufferOffsets[2],
    //         inputHeight,
    //         inputWidth,
    //         this.channelsIn,
    //         this.channelsOut
    //       );
    //     }
    //   }
    //   // const finish = performance.now();
    //   // console.log("[pointwise backward test 1] t =", (finish - start) / 1000, "s");
    // }

    // {
    //   // const start = performance.now();
    //   for (let i = 0; i < n; ++i) {
    //     for (const downstreamLayer of this.downstreamLayers) {
    //       instance.exports.pointwise_convolution_backward_d_out_test_2(
    //         downstreamLayer.currentBackwardOutput()[0],
    //         this.bufferOffsets[1],
    //         this.gradientOffsets[0],
    //         this.gradientOffsets[1],
    //         inputOffset,
    //         this.parameterOffsets[0],
    //         this.bufferOffsets[2],
    //         inputHeight,
    //         inputWidth,
    //         this.channelsIn,
    //         this.channelsOut
    //       );
    //     }
    //   }
    //   // const finish = performance.now();
    //   // console.log("[pointwise backward test 2] t =", (finish - start) / 1000, "s");
    // }
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

  constructor(upstreamLayer, channels, filterSize) {
    super();
    upstreamLayer.downstreamLayers.push(this);
    this.upstreamLayers.push(upstreamLayer);

    this.channels = channels;
    this.filterSize = filterSize;

    const kernelSize = this.channels * this.filterSize * this.filterSize;
    const biasSize = this.channels;
    this.parameterSizes.push(kernelSize);
    this.parameterSizes.push(biasSize);
    this.gradientSizes.push(kernelSize);
    this.gradientSizes.push(biasSize);

    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
  }

  initializeParametersAndGradients() {
    const fanIn = this.channels * this.filterSize * this.filterSize;
    const fanOut = this.channels * this.filterSize * this.filterSize;
    // const fanIn = this.filterSize * this.filterSize;
    // const fanOut = this.filterSize * this.filterSize;
    const glorot_uniform = Math.sqrt(6.0 / (fanIn + fanOut));

    const kernelSize = this.parameterSizes[0];
    const kernelOffset = this.parameterOffsets[0];
    const kernelArray = new Float32Array(
      instance.exports.memory.buffer,
      kernelOffset,
      kernelSize
    );
    for (let i = 0; i < kernelSize; ++i) {
      kernelArray[i] = ((randomWebAssemblyInstance.exports.random_float() - 0.5) * 2.0) * glorot_uniform;
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
    // bufferSizes.push(height * width * this.channelsOut); // d_x
    bufferSizes.push(height * width * channels); // d_x
    // bufferSizes.push((height + 2 * 3) * (width + 2 * 3) * channels); // x_pad
    // bufferSizes.push((height + 2 * 3) * (width + 2 * 3) * channels); // d_x_pad
    // bufferSizes.push((height + 2 * 2) * (width + 2 * 2) * channels); // x_pad
    // // bufferSizes.push((height + 2 * 2) * (width + 2 * 2) * channels); // d_x_pad
    bufferSizes.push((height + 2 * 1) * (width + 2 * 1) * channels); // x_pad
    bufferSizes.push((height + 2 * 1) * (width + 2 * 1) * channels); // d_x_pad

    return bufferSizes;
  }

  outputShapeFor(height, width, channels) {
    return [height, width, channels];
  }

  forward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();

    // {
    //   // const start = performance.now();
    //   for (let i = 0; i < 10; ++i) {
    //     instance.exports.depthwise_convolution_forward(
    //       inputOffset,
    //       this.bufferOffsets[0],
    //       this.parameterOffsets[0],
    //       this.parameterOffsets[1],
    //       this.bufferOffsets[2],
    //       inputHeight,
    //       inputWidth,
    //       inputChannels
    //     );
    //   }
    //   // const finish = performance.now();
    //   // console.log("[depthwise forward] t =", (finish - start) / 1000, "s");
    // }

    // {
    //   // const start = performance.now();
    //   for (let i = 0; i < 10; ++i) {
    //     instance.exports.depthwise_convolution_forward_unpadded(
    //       inputOffset,
    //       this.bufferOffsets[0],
    //       this.parameterOffsets[0],
    //       this.parameterOffsets[1],
    //       // this.bufferOffsets[2],
    //       inputHeight,
    //       inputWidth,
    //       inputChannels
    //     );
    //   }
    //   // const finish = performance.now();
    //   // console.log("[depthwise forward unpadded] t =", (finish - start) / 1000, "s");
    // }

    {
      // const start = performance.now();
      // for (let i = 0; i < 10; ++i) {
      instance.exports.depthwise_convolution_forward_unpadded_(
        inputOffset,
        this.bufferOffsets[0],
        this.parameterOffsets[0],
        this.parameterOffsets[1],
        // this.bufferOffsets[2],
        inputHeight,
        inputWidth,
        inputChannels
      );
      // }
      // const finish = performance.now();
      // console.log("[depthwise forward unpadded boop] t =", (finish - start) / 1000, "s");
    }

    // {
    //   // const start = performance.now();
    //   instance.exports.depthwise_convolution_forward_template(
    //     inputOffset,
    //     this.bufferOffsets[0],
    //     this.parameterOffsets[0],
    //     this.parameterOffsets[1],
    //     this.bufferOffsets[2],
    //     inputHeight,
    //     inputWidth
    //   );
    //   // const finish = performance.now();
    //   // console.log("[depthwise forward new] t =", (finish - start) / 1000, "s");
    // }

    this.currentHeight = inputHeight;
    this.currentWidth = inputWidth;
    this.currentChannels = inputChannels;
  }

  backward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    instance.exports.zero(this.bufferOffsets[1], this.bufferSizes[1]);
    instance.exports.zero(this.bufferOffsets[3], this.bufferSizes[3]);

    for (const downstreamLayer of this.downstreamLayers) {
      // {
      //   // const start = performance.now();
      //   // for (let i = 0; i < 10; ++i) {
      //   instance.exports.depthwise_convolution_backward(
      //     downstreamLayer.currentBackwardOutput()[0],
      //     this.bufferOffsets[1],
      //     this.gradientOffsets[0],
      //     this.gradientOffsets[1],
      //     this.parameterOffsets[0],
      //     this.bufferOffsets[2],
      //     this.bufferOffsets[3],
      //     inputHeight,
      //     inputWidth,
      //     inputChannels
      //   );
      //   // }
      //   // const finish = performance.now();
      //   // console.log("[depthwise backward] t =", (finish - start) / 1000, "s");
      // }

      // {
      //   // const start = performance.now();
      //   for (let i = 0; i < 10; ++i) {
      //     instance.exports.depthwise_convolution_backward_unpadded_(
      //       downstreamLayer.currentBackwardOutput()[0],
      //       this.bufferOffsets[1],
      //       this.gradientOffsets[0],
      //       this.gradientOffsets[1],
      //       this.parameterOffsets[0],
      //       inputOffset,
      //       inputHeight,
      //       inputWidth,
      //       inputChannels
      //     );
      //   }
      //   // const finish = performance.now();
      //   // console.log("[depthwise backward unpadded boop] t =", (finish - start) / 1000, "s");
      // }

      {
        // const start = performance.now();
        // for (let i = 0; i < 10; ++i) {
        instance.exports.depthwise_convolution_backward_unpadded(
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
        // }
        // const finish = performance.now();
        // console.log("[depthwise backward unpadded] t =", (finish - start) / 1000, "s");
      }

      // {
      //   // const start = performance.now();
      //   instance.exports.depthwise_convolution_backward_template(
      //     downstreamLayer.currentBackwardOutput()[0],
      //     this.bufferOffsets[1],
      //     this.gradientOffsets[0],
      //     this.gradientOffsets[1],
      //     this.parameterOffsets[0],
      //     this.bufferOffsets[2],
      //     this.bufferOffsets[3],
      //     inputHeight,
      //     inputWidth
      //   );
      //   // const finish = performance.now();
      //   // console.log("[depthwise backward new] t =", (finish - start) / 1000, "s");
      // }
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


class PatchifiedConvolutionLayer extends Layer {
  channelsIn = null;
  channelsOut = null;
  filterSize = null;
  stride = null;

  constructor(upstreamLayer, channelsIn, channelsOut, filterSize) {
    super();
    upstreamLayer.downstreamLayers.push(this);
    this.upstreamLayers.push(upstreamLayer);

    this.channelsIn = channelsIn;
    this.channelsOut = channelsOut;
    this.filterSize = filterSize;
    this.stride = this.filterSize;

    const kernelSize = this.channelsOut * this.filterSize * this.filterSize * this.channelsIn;
    const biasSize = this.channelsOut;
    this.parameterSizes.push(kernelSize);
    this.parameterSizes.push(biasSize);
    this.gradientSizes.push(kernelSize);
    this.gradientSizes.push(biasSize);

    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferSizes.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
    this.bufferOffsets.push(null);
  }

  initializeParametersAndGradients() {
    const fanIn = this.channelsIn * this.filterSize * this.filterSize;
    const fanOut = this.channelsOut * this.filterSize * this.filterSize;
    const glorot_uniform = Math.sqrt(6.0 / (fanIn + fanOut));

    const kernelSize = this.parameterSizes[0];
    const kernelOffset = this.parameterOffsets[0];
    const kernelArray = new Float32Array(
      instance.exports.memory.buffer,
      kernelOffset,
      kernelSize
    );
    for (let i = 0; i < kernelSize; ++i) {
      kernelArray[i] = ((randomWebAssemblyInstance.exports.random_float() - 0.5) * 2.0) * glorot_uniform;
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

    bufferSizes.push((height / this.stride) * (width / this.stride) * this.channelsOut); // y
    bufferSizes.push(height * width * channels); // d_x
    bufferSizes.push((height / this.stride) * (width / this.stride) * this.filterSize * this.filterSize * this.channelsIn); // rows
    // bufferSizes.push(this.channelsIn * this.filterSize * this.filterSize * width * height); // d_rows

    return bufferSizes;
  }

  outputShapeFor(height, width, channels) {
    return [height / this.stride, width / this.stride, this.channelsOut];
  }

  forward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();

    let n = 10;

    // {
    //   // const start = performance.now();
    //   // for (let i = 0; i < n; ++i) {
    //   instance.exports.patchified_convolution_forward(
    //     inputOffset,
    //     this.bufferOffsets[0],
    //     this.parameterOffsets[0],
    //     this.parameterOffsets[1],
    //     inputHeight,
    //     inputWidth,
    //     this.channelsOut
    //   );
    //   // }
    //   // const finish = performance.now();
    //   // console.log("patchified forward direct t =", (finish - start) / 1000, "s");
    // }

    // {
    //   // const start = performance.now();
    //   // for (let i = 0; i < n; ++i) {
    //   instance.exports.patchified_convolution_im2row_forward(
    //     inputOffset,
    //     this.bufferOffsets[0],
    //     this.parameterOffsets[0],
    //     this.parameterOffsets[1],
    //     this.bufferOffsets[2],
    //     inputHeight,
    //     inputWidth,
    //     this.channelsOut
    //   );
    //   // }
    //   // const finish = performance.now();
    //   // console.log("patchified forward im2row old t =", (finish - start) / 1000, "s");
    // }

    {
      // const start = performance.now();
      // for (let i = 0; i < n; ++i) {
      instance.exports.patchified_convolution_im2row_forward_(
        inputOffset,
        this.bufferOffsets[0],
        this.parameterOffsets[0],
        this.parameterOffsets[1],
        this.bufferOffsets[2],
        inputHeight,
        inputWidth,
        this.channelsOut
      );
      // }
      // const finish = performance.now();
      // console.log("patchified forward im2row new t =", (finish - start) / 1000, "s");
    }

    this.currentHeight = inputHeight / this.stride;
    this.currentWidth = inputWidth / this.stride;
    this.currentChannels = this.channelsOut;
  }

  backward() {
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    instance.exports.zero(this.bufferOffsets[1], this.bufferSizes[1]);
    // instance.exports.zero(this.bufferOffsets[2], this.bufferSizes[2]);
    // instance.exports.zero(this.bufferOffsets[3], this.bufferSizes[3]);

    let n = 10;

    for (const downstreamLayer of this.downstreamLayers) {
      // {
      //   // const start = performance.now();
      //   // for (let i = 0; i < n; ++i) {
      //   instance.exports.patchified_convolution_backward(
      //     downstreamLayer.currentBackwardOutput()[0],
      //     this.bufferOffsets[1],
      //     this.gradientOffsets[0],
      //     this.gradientOffsets[1],
      //     inputOffset,
      //     this.parameterOffsets[0],
      //     inputHeight,
      //     inputWidth,
      //     this.channelsOut
      //   );
      //   // }
      //   // const finish = performance.now();
      //   // console.log("patchified backward direct t =", (finish - start) / 1000, "s");
      // }

      {
        // const start = performance.now();
        // for (let i = 0; i < n; ++i) {
        instance.exports.patchified_convolution_im2row_backward(
          downstreamLayer.currentBackwardOutput()[0],
          this.bufferOffsets[1],
          this.gradientOffsets[0],
          this.gradientOffsets[1],
          inputOffset,
          this.parameterOffsets[0],
          this.bufferOffsets[2],
          inputHeight,
          inputWidth,
          this.channelsOut
        );
        // }
        // const finish = performance.now();
        // console.log("patchified backward im2row t =", (finish - start) / 1000, "s");
      }
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
    let [inputOffset, inputHeight, inputWidth, inputChannels] = this.upstreamLayers[0].currentForwardOutput();
    return [
      this.bufferOffsets[1],
      inputHeight,
      inputWidth,
      inputChannels
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

  constructor(channelsIn = 3, channelsMiddle, channelsOut, blockCount, maxImageSize, learningRate) {
    this.channelsIn = channelsIn;
    this.channelsMiddle = channelsMiddle;
    this.channelsOut = channelsOut;
    this.blockCount = blockCount;
    this.maxImageSize = maxImageSize;
    this.learningRate = learningRate;

    // console.log(`this.channelsIn: ${this.channelsIn}`);
    // console.log(`this.channelsMiddle: ${this.channelsMiddle}`);
    // console.log(`this.channelsOut: ${this.channelsOut}`);
    // console.log(`this.blockCount: ${this.blockCount}`);
    // console.log(`this.maxImageSize: ${this.maxImageSize}`);
    // console.log(`this.learningRate: ${this.learningRate}`);

    const inputLayer = new InputLayer();
    this.layers.push(inputLayer);

    //

    // const introPatchifiedConv = new PatchifiedConvolutionLayer(inputLayer, channelsIn, this.channelsMiddle * 4, 8);
    // this.layers.push(introPatchifiedConv);

    // const introInstanceNorm = new InstanceNormalizationLayer(introPatchifiedConv, this.channelsMiddle * 4);
    // this.layers.push(introInstanceNorm);

    // const introHardSwish = new HardSwishLayer(introInstanceNorm);
    // this.layers.push(introHardSwish);

    // const introPointwiseConv = new PointwiseConvolutionLayer(introHardSwish, this.channelsMiddle * 4, this.channelsMiddle, 1);
    // this.layers.push(introPointwiseConv);

    // const introInstanceNorm_ = new InstanceNormalizationLayer(introPointwiseConv, this.channelsMiddle);
    // this.layers.push(introInstanceNorm_);

    //

    const introPatchifiedConv = new PatchifiedConvolutionLayer(inputLayer, channelsIn, this.channelsMiddle, 8);
    this.layers.push(introPatchifiedConv);

    const introInstanceNorm = new InstanceNormalizationLayer(introPatchifiedConv, this.channelsMiddle);
    this.layers.push(introInstanceNorm);

    //

    // let previousLayer = introInstanceNorm_;
    let previousLayer = introInstanceNorm;

    // console.log("previousLayer before:", previousLayer);
    for (let i = 0; i < blockCount; ++i) {
      const expansionConv = new PointwiseConvolutionLayer(previousLayer, this.channelsMiddle, this.channelsMiddle * 3, 1);
      this.layers.push(expansionConv);

      const hardSwish = new HardSwishLayer(expansionConv);
      this.layers.push(hardSwish);

      const depthwiseConv = new DepthwiseConvolutionLayer(hardSwish, this.channelsMiddle * 3, 3);
      this.layers.push(depthwiseConv);

      const InstanceNorm = new InstanceNormalizationLayer(depthwiseConv, this.channelsMiddle * 3);
      this.layers.push(InstanceNorm);

      const reductionConv = new PointwiseConvolutionLayer(InstanceNorm, this.channelsMiddle * 3, this.channelsMiddle, 1);
      this.layers.push(reductionConv);

      const addition = new AdditionLayer([previousLayer, reductionConv]);
      this.layers.push(addition);

      previousLayer = addition;
      // console.log("previousLayer during:", previousLayer);
    }
    // console.log("previousLayer after:", previousLayer);

    const outroInstanceNorm = new InstanceNormalizationLayer(previousLayer, this.channelsMiddle);
    this.layers.push(outroInstanceNorm);

    const outroExpansionConv = new PointwiseConvolutionLayer(outroInstanceNorm, this.channelsMiddle, this.channelsMiddle * 8, 1);
    this.layers.push(outroExpansionConv);

    const outroHardSwish = new HardSwishLayer(outroExpansionConv);
    this.layers.push(outroHardSwish);

    const outroReductionConv = new PointwiseConvolutionLayer(outroHardSwish, this.channelsMiddle * 8, channelsOut * 4 * 4, 1);
    this.layers.push(outroReductionConv);

    const outroPixelShuffle = new PixelShuffleLayer(outroReductionConv, 4);
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


    //

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

    this.lastOffset = offset;

    // console.log("[old] MEMORY:", instance.exports.memory);
    // console.log("[old] instance.exports.memory.buffer.byteLength: ", instance.exports.memory.buffer.byteLength);

    // console.log("this.bufferOffset:", this.bufferOffset);
    // console.log("this.lastOffset:", this.lastOffset);

    this.originalOffset = offset;
    offset += 4096 * 4096 * 4 * elementByteSize;
    this.resizedOffset = offset;
    offset += 4096 * 4096 * 4 * elementByteSize;
    this.rotatedOffset = offset;
    offset += 4096 * 4096 * 4 * elementByteSize;
    this.gaussianOffset = offset;
    offset += 4096 * 4096 * 10 * elementByteSize;
    this.gaussianGradientOffset = offset;
    offset += 4096 * 4096 * 10 * elementByteSize;
    this.gaussianCoordinatesOffset = offset;
    offset += 2 * 10 * elementByteSize;

    // const extraBytesForPreprocessing = 2 * 4 * 4096 * 4096 * elementByteSize;
    // const extraBytesForPreprocessing = 3 * 4 * 4096 * 4096 * elementByteSize;
    const extraBytesForPreprocessing = 50 * 4096 * 4096 * elementByteSize;
    const extraPagesNeeded = Math.ceil((this.lastOffset + extraBytesForPreprocessing) / memoryPageSize) - instance.exports.memory.buffer.byteLength / memoryPageSize;
    // console.log("extraPagesNeeded:", extraPagesNeeded);
    if (extraPagesNeeded > 0) {
      const previousPageCount = instance.exports.memory.grow(extraPagesNeeded);
      // console.log(`previousPageCount: ${previousPageCount}`);
    }

    // console.log("[new] MEMORY:", instance.exports.memory);
    // console.log("[new] instance.exports.memory.buffer.byteLength: ", instance.exports.memory.buffer.byteLength);

    //

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

    // this.lastOffset = offset;

    let index = 0;
    for (const layer of this.layers) {
      if (index === 0) {
        // layer.forward(image, height, width, channels); // Feed data to input layer.
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
    // const parametersArray = new Float32Array(
    //   instance.exports.memory.buffer,
    //   this.parameterOffset,
    //   this.parameterLength
    // );

    // const gradientsArray = new Float32Array(
    //   instance.exports.memory.buffer,
    //   this.gradientOffset,
    //   this.parameterLength
    // );

    // const mArray = new Float32Array(
    //   instance.exports.memory.buffer,
    //   this.optimizerOffset,
    //   this.parameterLength
    // );

    // const vArray = new Float32Array(
    //   instance.exports.memory.buffer,
    //   this.optimizerOffset + this.parameterLength * elementByteSize,
    //   this.parameterLength
    // );

    // const beta1 = 0.9;
    // const beta2 = 0.999;
    // const epsilon = 1.0e-8;

    const beta1 = 0.9;
    const beta2 = 0.999;
    const epsilon = 1.0e-6;

    const weightDecay = 0.0;

    const scheduleMultiplier = 1.0;

    //

    // {
    //   // const start = performance.now();

    //   for (let i = 0; i < this.parameterLength; ++i) {
    //     mArray[i] = beta1 * mArray[i] + (1.0 - beta1) * gradientsArray[i];
    //     vArray[i] = beta2 * vArray[i] + (1.0 - beta2) * gradientsArray[i] ** 2;

    //     const m = mArray[i] / (1.0 - beta1 ** this.optimizerT);
    //     const v = vArray[i] / (1.0 - beta2 ** this.optimizerT);

    //     parametersArray[i] = parametersArray[i] - scheduleMultiplier * (this.learningRate * m / (Math.sqrt(v) + epsilon) + weightDecay * parametersArray[i]);
    //   }
    //   ++this.optimizerT;

    //   // const finish = performance.now();
    //   // console.log("[ecma] update t =", (finish - start) / 1000, "s");
    // }

    //

    // {
    //   // const start = performance.now();

    // console.log("table lookups:", instance.exports.b1_vals_table_lookup(this.optimizerT), instance.exports.b2_vals_table_lookup(this.optimizerT));

    instance.exports.update_parameters(
      this.gradientOffset,
      this.parameterOffset,
      this.optimizerOffset, // reminder: use "first moment" terminology
      this.optimizerOffset + this.parameterLength * elementByteSize, // reminder: use "second moment" terminology
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

    //   // const finish = performance.now();
    //   // console.log("[wasm] update t =", (finish - start) / 1000, "s");
    // }

    //

    // {
    //   // const start = performance.now();

    //   instance.exports.update_parameters_alt(
    //     this.gradientOffset,
    //     this.parameterOffset,
    //     this.optimizerOffset,
    //     this.optimizerOffset + this.parameterLength * elementByteSize,
    //     this.parameterLength,
    //     beta1,
    //     beta2,
    //     epsilon,
    //     scheduleMultiplier,
    //     this.learningRate,
    //     weightDecay,
    //     this.optimizerT
    //   );
    //   ++this.optimizerT;

    //   // const finish = performance.now();
    //   // console.log("[Math] update t =", (finish - start) / 1000, "s");
    // }
  }

  resize(x, heightIn, widthIn, heightOut, widthOut) {
    // console.log("instance.exports.memory.buffer.byteLength: ", instance.exports.memory.buffer.byteLength);
    // console.log("xArray: ", this.lastOffset, heightIn * widthIn * channelsRgba);
    // console.log("yArray: ", this.lastOffset + heightIn * widthIn * channelsRgba, heightOut * widthOut * channelsRgb * elementByteSize);

    const xArray = new Uint8ClampedArray(
      instance.exports.memory.buffer,
      this.originalOffset,
      heightIn * widthIn * channelsRgba
    );

    for (let i = 0; i < heightIn * widthIn * channelsRgba; ++i) {
      xArray[i] = x[i];
    }

    // const yArray = new Float32Array(
    //   instance.exports.memory.buffer,
    //   this.lastOffset + heightIn * widthIn * channelsRgba,
    //   heightOut * widthOut * channelsRgb
    // );

    // {
    //   // const start = performance.now();
    instance.exports.resize_bilinear_rgba_to_rgb(this.originalOffset, this.resizedOffset, heightIn, widthIn, heightOut, widthOut);
    //   // const finish = performance.now();
    //   // console.log("[resize wasm inner] t =", (finish - start) / 1000, "s");
    // }

    // for (let i = 0; i < heightOut * widthOut * channelsRgb; ++i) {
    //   y[i] = yArray[i];
    // }
  }

  flipHorizontal(height, width) {
    instance.exports.flip_horizontal(this.resizedOffset, height, width);
  }

  flipVertical(height, width) {
    instance.exports.flip_vertical(this.resizedOffset, height, width);
  }

  brightnessAdjustment(height, width, brightnessAdjustment) {
    instance.exports.brightness_adjustment(this.resizedOffset, height, width, brightnessAdjustment);
  }

  rotate(height, width, cosTheta, sinTheta, padValue) {
    // console.log("instance.exports.memory.buffer.byteLength: ", instance.exports.memory.buffer.byteLength);
    // console.log("xArray: ", this.lastOffset, height * width * channelsRgb * elementByteSize);
    // console.log("yArray: ", this.lastOffset + height * width * channelsRgb * elementByteSize, height * width * channelsRgb * elementByteSize);

    // const xArray = new Float32Array(
    //   instance.exports.memory.buffer,
    //   this.lastOffset,
    //   height * width * channelsRgb
    // );

    // for (let i = 0; i < height * width * channelsRgb; ++i) {
    //   xArray[i] = x[i];
    // }

    // const yArray = new Float32Array(
    //   instance.exports.memory.buffer,
    //   this.lastOffset + height * width * channelsRgb * elementByteSize,
    //   height * width * channelsRgb
    // );

    // {
    // // const start = performance.now();
    // const resizedArray = new Float32Array(
    //   instance.exports.memory.buffer,
    //   this.resizedOffset,
    //   height * width * channelsRgb
    // );

    // const sortedResizedArray = resizedArray.toSorted((a, b) => { return a - b; });
    // const half = Math.floor(sortedResizedArray.length / 2);
    // const median = (sortedResizedArray.length % 2 ? sortedResizedArray[half] : (sortedResizedArray[half - 1] + sortedResizedArray[half]) / 2);
    // padValue = median;
    // // const finish = performance.now();
    // console.log("[rotate wasm inner median] t =", (finish - start) / 1000, "s");
    // }

    // {
    //   // const start = performance.now();
    instance.exports.rotate_bilinear(this.resizedOffset, this.rotatedOffset, height, width, cosTheta, sinTheta, padValue);
    //   // const finish = performance.now();
    //   // console.log("[rotate wasm inner] t =", (finish - start) / 1000, "s");
    // }

    // for (let i = 0; i < height * width * channelsRgb; ++i) {
    //   y[i] = yArray[i];
    // }
  }

  drawGaussians(resizedGaussianHeight, resizedGaussianWidth, keypointCount, coordinates, gaussianStdDev) {
    // const gaussiansArray = new Float32Array(
    //   instance.exports.memory.buffer,
    //   this.gaussianOffset,
    //   resizedGaussianHeight * resizedGaussianWidth * keypointCount
    // );

    const coordinatesArray = new Float32Array(
      instance.exports.memory.buffer,
      this.gaussianCoordinatesOffset,
      2 * keypointCount
    );

    for (let i = 0; i < keypointCount; ++i) {
      coordinatesArray[i * 2 + 0] = coordinates[i][0];
      coordinatesArray[i * 2 + 1] = coordinates[i][1];
    }

    // {
    //   // const start = performance.now();
    instance.exports.draw_gaussians(this.gaussianOffset, resizedGaussianHeight, resizedGaussianWidth, keypointCount, this.gaussianCoordinatesOffset, gaussianStdDev);
    //   // const finish = performance.now();
    //   // console.log("[gaussians old] t =", (finish - start) / 1000, "s");
    // }

    // {
    //   // const start = performance.now();
    //   instance.exports.draw_gaussians_fast(this.gaussianOffset, resizedGaussianHeight, resizedGaussianWidth, keypointCount, this.gaussianCoordinatesOffset, gaussianStdDev);
    //   // const finish = performance.now();
    //   // console.log("[gaussians new] t =", (finish - start) / 1000, "s");
    // }

    // for (let i = 0; i < resizedGaussianHeight * resizedGaussianWidth * keypointCount; ++i) {
    //   gaussians[i] = gaussiansArray[i];
    // }
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

    // const gradientsArray = new Float32Array(
    //   instance.exports.memory.buffer,
    //   this.gaussianGradientOffset,
    //   this.layers[this.layers.length - 1].currentHeight * this.layers[this.layers.length - 1].currentWidth * this.layers[this.layers.length - 1].currentChannels
    // );

    // for (let i = 0; i < this.layers[this.layers.length - 1].currentHeight * this.layers[this.layers.length - 1].currentWidth * this.layers[this.layers.length - 1].currentChannels; ++i) {
    //   gradients[i] = gradientsArray[i];
    //   // console.log("values:", gradients[i], gradientsArray[i]);
    // }
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
      // console.log("copying", i, parameterArray[i]);
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
}
