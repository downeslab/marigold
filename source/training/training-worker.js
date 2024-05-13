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

import { SectionWorker } from "../section-worker.js";

import { channelsRgb, channelsRgba, nearestValidImageSize, argmax } from "../image.js";
import { NeuralNetwork } from "../neural-network.js";
import { degreesToRadians } from "../core/math.js";


class TrainingWorker extends SectionWorker {
  neuralNetwork = null;

  trainingValidationSplit = 0.75;
  // trainingIndices = null;
  // validationIndices = null;

  epochs = null; // maxEpochs
  batchSize = null;
  gradientAccumulationSize = 1;

  // gaussianStdDev = 1.0;
  // gaussianStdDev = 1.5;
  gaussianStdDev = 2.0;

  horizontalFlip = false;
  verticalFlip = false;

  learningRate = null;

  blobs = null;

  epoch = null;

  constructor() {
    super("model");

    self.addEventListener(
      "message",
      (message) => {
        if (message.data.type === "importDataset") {
          this.importDataset(message.data.fileHandle);
        }
        else if (message.data.type === "trainingSeed") {
          this.trainingSeed = message.data.trainingSeed;
          if (this.data.labels) {
            this.shuffleDataset();
          }
        }
        // else if (message.data.type === "shuffleDataset") {
        //   this.shuffleDataset();
        // }
        else if (message.data.type === "horizontalFlip") {
          this.horizontalFlip = message.data.horizontalFlip;
          // console.log("horizontalFlip:", message.data.horizontalFlip);
        }
        else if (message.data.type === "verticalFlip") {
          this.verticalFlip = message.data.verticalFlip;
          // console.log("verticalFlip:", message.data.verticalFlip);
        }
        else if (message.data.type === "maxImageSize") {
          this.data.maxImageSize = +message.data.maxImageSize;
          this.data.maxGaussianSize = +this.data.maxImageSize / 2;
          // console.log(this.data.maxImageSize, this.data.maxGaussianSize);
        }
        else if (message.data.type === "channelCount") {
          this.data.channelCount = +message.data.channelCount;
          // console.log(this.data.channelCount);
        }
        else if (message.data.type === "blockCount") {
          this.data.blockCount = +message.data.blockCount;
          // console.log(this.data.blockCount);
        }
        else if (message.data.type === "batchSize") {
          this.batchSize = +message.data.batchSize;
          // console.log(this.batchSize);
        }
        else if (message.data.type === "learningRate") {
          this.learningRate = +message.data.learningRate;
          // console.log(this.learningRate);
        }
        else if (message.data.type === "epochs") {
          this.epochs = +message.data.epochs;
          // console.log(this.epochs);
        }
        else if (message.data.type === "startTraining") {
          this.startTraining();
        }
      }
    );

    this.reset();
  }


  reset() {
    super.reset();
    this.neuralNetwork = null;
  }


  initializeData() {
    super.initializeData();

    this.data.trainingIndices = null;
    this.data.validationIndices = null;

    this.data.keypointCount = null;
    this.data.channelCount = null;
    this.data.blockCount = null;
    this.data.maxImageSize = null;
    this.data.maxGaussianSize = null; // not currently using?

    this.data.meanTrainingLosses = null;
    this.data.meanValidationLosses = null;
    this.data.bestTrainingLoss = null;
    this.data.bestValidationLoss = null;
    this.data.cachedPredictions = null;

    this.data.currentWeights = null;
    this.data.bestWeights = null;
  }

  sanityCheckData() {
    super.sanityCheckData();
  }

  //
  //
  //

  async maybeOpenExistingFile(fileHandle) {
    try {
      this.fileHandle = fileHandle;

      const file = await fileHandle.getFile();
      this.data = JSON.parse(await file.text());
      this.sanityCheckData();

      self.postMessage({ type: "openExistingFileSuccess" });
    }
    catch {
      this.fileHandle = null;
      this.data = null;
      self.postMessage({ type: "openExistingFileFailure" });
    }
  }

  //
  //
  //

  async importDataset(fileHandle) {
    // console.log("importDataset");

    const file = await fileHandle.getFile();

    try {
      const data = JSON.parse(await file.text());
      // console.log("data:", data);
      this.data.labels = data.labels;
      // reminder: get name of dataset from fileHandle and add to this.data.labels.metadata
    }
    catch {
    }

    // console.log(this.data.labels);

    // self.postMessage({ type: "importDatasetSuccess", trainingIndices: this.data.trainingIndices, validationIndices: this.data.validationIndices });
    this.shuffleDataset();

    //

    this.blobs = [];
    for (const label of this.data.labels) {
      const imageData = label.image;
      const imageResult = await fetch(imageData);
      const blob = await imageResult.blob();
      this.blobs.push(blob);
    }

    //

    this.data.keypointCount = this.data.labels[0].label.length;

    //

    // console.log("importDataset (done)");
  }

  async shuffleDataset() {
    this.data.trainingIndices = [];
    this.data.validationIndices = [];

    const dataPoints = this.data.labels.length;
    const trainingDataPoints = Math.round(this.trainingValidationSplit * dataPoints);
    const validationDataPoints = dataPoints - trainingDataPoints;

    //

    // if (this.neuralNetwork === null) {
    //   this.neuralNetwork = new NeuralNetwork(channelsRgb, this.data.channelCount, this.data.keypointCount, this.data.blockCount, this.data.maxImageSize, this.learningRate);
    // }

    const neuralNetwork = new NeuralNetwork(channelsRgb, 8, 1, 4, 96, this.learningRate);

    neuralNetwork.seed(this.trainingSeed);
    for (let i = 0; i < 100; ++i) {
      const randomInteger = neuralNetwork.randomInteger(0, 6);
      // console.log(`randomInteger: ${randomInteger}`);
    }

    neuralNetwork.seed(this.trainingSeed);
    for (let i = 0; i < 100; ++i) {
      const randomFloat = neuralNetwork.randomFloat();
      // console.log(`randomFloat: ${randomFloat}`);
    }

    //

    let i = 0;
    {
      while (this.data.trainingIndices.length < trainingDataPoints || this.data.validationIndices.length < validationDataPoints) {
        const random = neuralNetwork.randomFloat();
        if (random < this.trainingValidationSplit && this.data.trainingIndices.length < trainingDataPoints) {
          this.data.trainingIndices.push(i);
          ++i;
        }
        else if (random >= this.trainingValidationSplit && this.data.validationIndices.length < validationDataPoints) {
          this.data.validationIndices.push(i);
          ++i;
        }
      }
    }
    self.postMessage({ type: "importDatasetSuccess", trainingIndices: this.data.trainingIndices, validationIndices: this.data.validationIndices });
  }

  async startTraining() {
    // console.log("startTraining");
    const epochStart = performance.now();

    // console.log("things");
    // console.log(this.data.keypointCount);
    // console.log(this.data.channelCount);
    // console.log(this.data.blockCount);
    // console.log(this.data.maxImageSize);
    // console.log(this.data.maxGaussianSize);
    // console.log(this.horizontalFlip);
    // console.log(this.verticalFlip);
    // console.log("/things");

    if (this.neuralNetwork === null) {
      this.neuralNetwork = new NeuralNetwork(channelsRgb, this.data.channelCount, this.data.keypointCount, this.data.blockCount, this.data.maxImageSize, this.learningRate);
    }

    if (this.data.meanTrainingLosses === null) {
      this.data.meanTrainingLosses = [];
    }
    if (this.data.meanValidationLosses === null) {
      this.data.meanValidationLosses = [];
    }

    if (this.epoch === null) {
      this.epoch = 0;
    }

    let meanTrainingLoss = 0.0;
    let meanValidationLoss = 0.0;

    // console.log("this.data.trainingIndices:", this.data.trainingIndices);
    let shuffledIndices = [];
    while (shuffledIndices.length < this.data.trainingIndices.length) {
      const random = this.neuralNetwork.randomInteger(0, this.data.trainingIndices.length);
      if (!shuffledIndices.includes(random)) {
        shuffledIndices.push(random);
      }
    }
    // console.log("shuffledIndices:", shuffledIndices);

    let batchIndex = 0;
    for (let trainingIndex = 0; trainingIndex < this.data.trainingIndices.length; ++trainingIndex) {
      // const image = this.data.labels[this.data.trainingIndices[trainingIndex]].image;
      const image = this.data.labels[this.data.trainingIndices[shuffledIndices[trainingIndex]]].image;
      const imageResult = await fetch(image);
      const imageBlob = await imageResult.blob();
      const imageBitmap = await createImageBitmap(imageBlob, { colorSpaceConversion: "none" });

      const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
      const context = canvas.getContext("2d");
      context.drawImage(imageBitmap, 0, 0, imageBitmap.width, imageBitmap.height);
      const imageData = context.getImageData(0, 0, canvas.width, canvas.height);

      // const resizedHeight = this.data.maxImageSize;
      // const resizedWidth = this.data.maxImageSize;
      const [resizedHeight, resizedWidth] = nearestValidImageSize(imageBitmap.height, imageBitmap.width, this.data.maxImageSize, 8);
      // console.log("image sizes:", imageBitmap.height, imageBitmap.width, resizedHeight, resizedWidth);

      const resizedGaussianHeight = resizedHeight / 2;
      const resizedGaussianWidth = resizedWidth / 2;

      // const resized = new Float32Array(resizedHeight * resizedWidth * channelsRgb);

      // {
      //   const start = performance.now();
      //   resizeBilinearRgbaToRgb(imageData.data, resized, imageBitmap.height, imageBitmap.width, resizedHeight, resizedWidth);
      //   const finish = performance.now();
      //   // console.log("[resize ecma] t =", (finish - start) / 1000, "s");
      // }
      // {
      //   const start = performance.now();
      this.neuralNetwork.resize(imageData.data, imageBitmap.height, imageBitmap.width, resizedHeight, resizedWidth);
      //   const finish = performance.now();
      //   // console.log("[resize wasm] t =", (finish - start) / 1000, "s");
      // }

      // const label = this.data.labels[this.data.trainingIndices[trainingIndex]].label;
      const label = this.data.labels[this.data.trainingIndices[shuffledIndices[trainingIndex]]].label;

      const coordinates = [];
      for (const coordinate of label) {
        coordinates.push([coordinate.y, coordinate.x]);
      }
      for (let i = 0; i < coordinates.length; ++i) {
        coordinates[i][0] *= resizedHeight / imageBitmap.height;
        coordinates[i][1] *= resizedWidth / imageBitmap.width;
      }
      for (let i = 0; i < coordinates.length; ++i) {
        coordinates[i][0] *= resizedGaussianHeight / resizedHeight;
        coordinates[i][1] *= resizedGaussianWidth / resizedWidth;
      }

      // if (this.verticalFlip) {
      //   if (Math.random() < 0.5) {
      //     flipVertical(resized, resizedHeight, resizedWidth, channelsRgb);

      //     for (let i = 0; i < coordinates.length; ++i) {
      //       coordinates[i][0] = resizedGaussianHeight - coordinates[i][0];
      //     }
      //   }
      // }
      // if (this.horizontalFlip) {
      //   if (Math.random() < 0.5) {
      //     flipHorizontal(resized, resizedHeight, resizedWidth, channelsRgb);

      //     for (let i = 0; i < coordinates.length; ++i) {
      //       coordinates[i][1] = resizedGaussianWidth - coordinates[i][1];
      //     }
      //   }
      // }

      // const brightnessAdjustment = (Math.random() - 0.5) / 10.0; // uniform(0.05, 0.05)
      // for (let i = 0; i < resized.length; ++i) {
      //   resized[i] += brightnessAdjustment;
      // }

      if (this.verticalFlip) {
        if (this.neuralNetwork.randomFloat() < 0.5) {
          // flipVertical(resized, resizedHeight, resizedWidth, channelsRgb);
          this.neuralNetwork.flipVertical(resizedHeight, resizedWidth);

          for (let i = 0; i < coordinates.length; ++i) {
            coordinates[i][0] = resizedGaussianHeight - coordinates[i][0];
          }
        }
      }
      if (this.horizontalFlip) {
        if (this.neuralNetwork.randomFloat() < 0.5) {
          // flipHorizontal(resized, resizedHeight, resizedWidth, channelsRgb);
          this.neuralNetwork.flipHorizontal(resizedHeight, resizedWidth);

          for (let i = 0; i < coordinates.length; ++i) {
            coordinates[i][1] = resizedGaussianWidth - coordinates[i][1];
          }
        }
      }

      const brightnessAdjustment = (this.neuralNetwork.randomFloat() - 0.5) / 10.0; // uniform(0.05, 0.05)
      // for (let i = 0; i < resized.length; ++i) {
      //   resized[i] += brightnessAdjustment;
      // }
      this.neuralNetwork.brightnessAdjustment(resizedHeight, resizedWidth, brightnessAdjustment);

      const angle = (this.neuralNetwork.randomFloat() - 0.5) * 2 * 15; // uniform(-15, 15);
      const theta = degreesToRadians(angle);
      // const rotated = new Float32Array(resizedHeight * resizedWidth * channelsRgb);

      // {
      //   const start = performance.now();
      //   rotateBilinear(resized, rotated, resizedHeight, resizedWidth, theta, null);
      //   const finish = performance.now();
      //   // console.log("[rotate ecma] t =", (finish - start) / 1000, "s");
      // }
      // {
      //   const start = performance.now();
      this.neuralNetwork.rotate(resizedHeight, resizedWidth, Math.cos(theta), Math.sin(theta), null);
      //   const finish = performance.now();
      //   // console.log("[rotate wasm] t =", (finish - start) / 1000, "s");
      // }

      for (let i = 0; i < coordinates.length; ++i) {
        coordinates[i][0] -= resizedGaussianHeight / 2.0;
        coordinates[i][1] -= resizedGaussianWidth / 2.0;

        const cosTheta = Math.cos(-theta);
        const sinTheta = Math.sin(-theta);

        const rotationMatrix = [cosTheta, sinTheta, -sinTheta, cosTheta];

        let yPrime = rotationMatrix[0] * coordinates[i][0] + rotationMatrix[1] * coordinates[i][1];
        let xPrime = rotationMatrix[2] * coordinates[i][0] + rotationMatrix[3] * coordinates[i][1];

        yPrime += resizedGaussianHeight / 2.0;
        xPrime += resizedGaussianWidth / 2.0;

        coordinates[i][0] = yPrime;
        coordinates[i][1] = xPrime;
      }

      // const gaussians = new Float32Array(resizedGaussianHeight * resizedGaussianWidth * this.data.keypointCount);
      // drawGaussians(gaussians, resizedGaussianHeight, resizedGaussianWidth, this.data.keypointCount, coordinates, this.gaussianStdDev);
      this.neuralNetwork.drawGaussians(resizedGaussianHeight, resizedGaussianWidth, this.data.keypointCount, coordinates, this.gaussianStdDev);

      {
        const start = performance.now();
        // this.neuralNetwork.forward(resized, resizedHeight, resizedWidth, channelsRgb);
        this.neuralNetwork.forward(this.neuralNetwork.rotatedOffset, resizedHeight, resizedWidth, channelsRgb);
        const finish = performance.now();
        // if (trainingIndex < 4) {
        // console.log("[train] neuralNetwork.forward t =", (finish - start) / 1000, "s");
        // }
      }

      // const predictions = this.neuralNetwork.predictions();

      // const trainingLoss = meanSquaredErrorForward(predictions, gaussians, resizedGaussianHeight * resizedGaussianWidth * this.data.keypointCount);
      const trainingLoss = this.neuralNetwork.lossForward();
      // trainingLoss /= this.batchSize;
      meanTrainingLoss += trainingLoss;

      // const gradients = new Float32Array(resizedGaussianHeight * resizedGaussianWidth * this.data.keypointCount);
      // meanSquaredErrorBackward(trainingLoss, gradients, predictions, gaussians, resizedGaussianHeight * resizedGaussianWidth * this.data.keypointCount);
      // meanSquaredErrorBackward(trainingLoss / this.batchSize, gradients, predictions, gaussians, resizedGaussianHeight * resizedGaussianWidth * this.data.keypointCount);
      // not sure which of these makes more sense; i think the one that divides by batch size seems to work a bit better in practice and probably makes more theoretical sense
      // this.neuralNetwork.lossBackward(trainingLoss);
      this.neuralNetwork.lossBackward(trainingLoss / this.batchSize);

      {
        const start = performance.now();
        // this.neuralNetwork.backward(gradients);
        this.neuralNetwork.backward(this.neuralNetwork.gaussianGradientOffset);
        const finish = performance.now();
        // if (trainingIndex < 4) {
        // console.log("[train] neuralNetwork.backward t =", (finish - start) / 1000, "s");
        // }
      }

      ++batchIndex;
      // reminder: what if the last batch is not a full batch? (need to handle partial batches)
      if (batchIndex === this.batchSize) {
        this.neuralNetwork.updateParameters();
        this.neuralNetwork.zeroGradients();
        batchIndex = 0;
      }
    }
    meanTrainingLoss /= this.data.trainingIndices.length;

    //
    //
    //

    let cachedPredictions = [];
    for (let validationIndex = 0; validationIndex < this.data.validationIndices.length; ++validationIndex) {
      const image = this.data.labels[this.data.validationIndices[validationIndex]].image;
      const imageResult = await fetch(image);
      const imageBlob = await imageResult.blob();
      const imageBitmap = await createImageBitmap(imageBlob, { colorSpaceConversion: "none" });

      const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
      const context = canvas.getContext("2d");
      context.drawImage(imageBitmap, 0, 0, imageBitmap.width, imageBitmap.height);
      const imageData = context.getImageData(0, 0, canvas.width, canvas.height);

      // const resizedHeight = this.data.maxImageSize;
      // const resizedWidth = this.data.maxImageSize;
      const [resizedHeight, resizedWidth] = nearestValidImageSize(imageBitmap.height, imageBitmap.width, this.data.maxImageSize, 8);

      const resizedGaussianHeight = resizedHeight / 2;
      const resizedGaussianWidth = resizedWidth / 2;

      // const resized = new Float32Array(resizedHeight * resizedWidth * channelsRgb);
      // resizeBilinearRgbaToRgb(imageData.data, resized, imageBitmap.height, imageBitmap.width, resizedHeight, resizedWidth);
      this.neuralNetwork.resize(imageData.data, imageBitmap.height, imageBitmap.width, resizedHeight, resizedWidth);

      const label = this.data.labels[this.data.validationIndices[validationIndex]].label;

      const coordinates = [];
      for (const coordinate of label) {
        coordinates.push([coordinate.y, coordinate.x]);
      }
      for (let i = 0; i < coordinates.length; ++i) {
        coordinates[i][0] *= resizedHeight / imageBitmap.height;
        coordinates[i][1] *= resizedWidth / imageBitmap.width;
      }
      for (let i = 0; i < coordinates.length; ++i) {
        coordinates[i][0] *= resizedGaussianHeight / resizedHeight;
        coordinates[i][1] *= resizedGaussianWidth / resizedWidth;
      }

      // const gaussians = new Float32Array(resizedGaussianHeight * resizedGaussianWidth * this.data.keypointCount);
      // drawGaussians(gaussians, resizedGaussianHeight, resizedGaussianWidth, this.data.keypointCount, coordinates, this.gaussianStdDev);
      this.neuralNetwork.drawGaussians(resizedGaussianHeight, resizedGaussianWidth, this.data.keypointCount, coordinates, this.gaussianStdDev);

      {
        const start = performance.now();
        // this.neuralNetwork.forward(resized, resizedHeight, resizedWidth, channelsRgb);
        this.neuralNetwork.forward(this.neuralNetwork.resizedOffset, resizedHeight, resizedWidth, channelsRgb);
        const finish = performance.now();
        // if (validationIndex < 4) {
        // console.log("[valid] neuralNetwork.forward t =", (finish - start) / 1000, "s");
        // }
      }

      const predictions = this.neuralNetwork.predictions();
      const predictionCoordinates = argmax(predictions, resizedGaussianHeight, resizedGaussianWidth, this.data.keypointCount);
      // {
      //   const start = performance.now();
      //   const predictionCoordinates = argmax(predictions, resizedGaussianHeight, resizedGaussianWidth, this.data.keypointCount);
      //   const finish = performance.now();
      //   // console.log("[argmax old] t =", (finish - start) / 1000, "s");
      // }
      // {
      //   const start = performance.now();
      //   const predictionCoordinates = argmax(predictions, resizedGaussianHeight, resizedGaussianWidth, this.data.keypointCount);
      //   const finish = performance.now();
      //   // console.log("[argmax new] t =", (finish - start) / 1000, "s");
      // }
      for (let i = 0; i < predictionCoordinates.length; ++i) {
        predictionCoordinates[i].y *= resizedHeight / resizedGaussianHeight;
        predictionCoordinates[i].x *= resizedWidth / resizedGaussianWidth;
      }
      for (let i = 0; i < predictionCoordinates.length; ++i) {
        predictionCoordinates[i].y *= imageBitmap.height / resizedHeight;
        predictionCoordinates[i].x *= imageBitmap.width / resizedWidth;
      }
      cachedPredictions.push(predictionCoordinates);

      // const validationLoss = meanSquaredErrorForward(predictions, gaussians, resizedGaussianHeight * resizedGaussianWidth * this.data.keypointCount);
      const validationLoss = this.neuralNetwork.lossForward();
      meanValidationLoss += validationLoss;
    }
    meanValidationLoss /= this.data.validationIndices.length;

    //
    //
    //

    this.data.meanTrainingLosses.push(meanTrainingLoss);
    this.data.meanValidationLosses.push(meanValidationLoss);
    if (this.epoch === 0) {
      this.data.bestTrainingLoss = meanTrainingLoss;
      this.data.bestValidationLoss = meanValidationLoss;
      this.data.cachedPredictions = cachedPredictions;

      // console.log("boop?");
      this.data.bestWeights = this.neuralNetwork.getParameters();
      // console.log("boop!");
    }
    else { // reminder: maybe this can be a single else if? lol
      if (meanValidationLoss < this.data.bestValidationLoss) // use <= ?
      {
        this.data.bestValidationLoss = meanValidationLoss;
        this.data.bestTrainingLoss = meanTrainingLoss; // rename to this.data.trainingLossForBestValidationLoss?
        this.data.cachedPredictions = cachedPredictions;

        // console.log("boop?");
        this.data.bestWeights = this.neuralNetwork.getParameters();
        // console.log("boop!");
      }
    }

    // console.log(`[epoch: ${this.epoch}] meanTrainingLoss: ${meanTrainingLoss}`);
    // console.log(`[epoch: ${this.epoch}] meanValidationLoss: ${meanValidationLoss}`);

    self.postMessage(
      {
        type: "epochDone",
        epoch: this.epoch,
        bestTrainingLoss: this.data.bestTrainingLoss,
        bestValidationLoss: this.data.bestValidationLoss,
        meanTrainingLosses: this.data.meanTrainingLosses,
        meanValidationLosses: this.data.meanValidationLosses,
        labels: this.data.labels,
        trainingIndices: this.data.trainingIndices,
        validationIndices: this.data.validationIndices,
        cachedPredictions: this.data.cachedPredictions,
        blobs: this.blobs
      }
    );

    ++this.epoch;


    const epochFinish = performance.now();
    // console.log("[epoch] t =", (epochFinish - epochStart) / 1000, "s");

    // console.log("startTraining (done)");
  }
}


new TrainingWorker();
