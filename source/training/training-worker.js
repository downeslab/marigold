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


function degreesToRadians(deg) {
  return deg * (Math.PI / 180);
}


class TrainingWorker extends SectionWorker {
  neuralNetwork = null;

  trainingValidationSplit = 0.75;

  epochs = null;
  batchSize = null;
  gradientAccumulationSize = 1;

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
        else if (message.data.type === "horizontalFlip") {
          this.horizontalFlip = message.data.horizontalFlip;
        }
        else if (message.data.type === "verticalFlip") {
          this.verticalFlip = message.data.verticalFlip;
        }
        else if (message.data.type === "maxImageSize") {
          this.data.maxImageSize = +message.data.maxImageSize;
          this.data.maxGaussianSize = +this.data.maxImageSize / 2;
        }
        else if (message.data.type === "channelCount") {
          this.data.channelCount = +message.data.channelCount;
        }
        else if (message.data.type === "blockCount") {
          this.data.blockCount = +message.data.blockCount;
        }
        else if (message.data.type === "batchSize") {
          this.batchSize = +message.data.batchSize;
        }
        else if (message.data.type === "learningRate") {
          this.learningRate = +message.data.learningRate;
        }
        else if (message.data.type === "epochs") {
          this.epochs = +message.data.epochs;
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
    this.data.maxGaussianSize = null;

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

  async importDataset(fileHandle) {
    const file = await fileHandle.getFile();

    try {
      const data = JSON.parse(await file.text());
      this.data.labels = data.labels;
    }
    catch {
    }

    this.shuffleDataset();

    this.blobs = [];
    for (const label of this.data.labels) {
      const imageData = label.image;
      const imageResult = await fetch(imageData);
      const blob = await imageResult.blob();
      this.blobs.push(blob);
    }

    this.data.keypointCount = this.data.labels[0].label.length;
  }

  async shuffleDataset() {
    this.data.trainingIndices = [];
    this.data.validationIndices = [];

    const dataPoints = this.data.labels.length;
    const trainingDataPoints = Math.round(this.trainingValidationSplit * dataPoints);
    const validationDataPoints = dataPoints - trainingDataPoints;

    const neuralNetwork = new NeuralNetwork(channelsRgb, 8, 1, 4, 96, this.learningRate);

    neuralNetwork.seed(this.trainingSeed);
    for (let i = 0; i < 100; ++i) {
      const randomInteger = neuralNetwork.randomInteger(0, 6);
    }

    neuralNetwork.seed(this.trainingSeed);
    for (let i = 0; i < 100; ++i) {
      const randomFloat = neuralNetwork.randomFloat();
    }

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
    const epochStart = performance.now();

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

    let shuffledIndices = [];
    while (shuffledIndices.length < this.data.trainingIndices.length) {
      const random = this.neuralNetwork.randomInteger(0, this.data.trainingIndices.length);
      if (!shuffledIndices.includes(random)) {
        shuffledIndices.push(random);
      }
    }

    let batchIndex = 0;
    for (let trainingIndex = 0; trainingIndex < this.data.trainingIndices.length; ++trainingIndex) {
      const image = this.data.labels[this.data.trainingIndices[shuffledIndices[trainingIndex]]].image;
      const imageResult = await fetch(image);
      const imageBlob = await imageResult.blob();
      const imageBitmap = await createImageBitmap(imageBlob, { colorSpaceConversion: "none" });

      const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
      const context = canvas.getContext("2d");
      context.drawImage(imageBitmap, 0, 0, imageBitmap.width, imageBitmap.height);
      const imageData = context.getImageData(0, 0, canvas.width, canvas.height);

      const [resizedHeight, resizedWidth] = nearestValidImageSize(imageBitmap.height, imageBitmap.width, this.data.maxImageSize, 8);

      const resizedGaussianHeight = resizedHeight / 2;
      const resizedGaussianWidth = resizedWidth / 2;

      this.neuralNetwork.resize(imageData.data, imageBitmap.height, imageBitmap.width, resizedHeight, resizedWidth);

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

      if (this.verticalFlip) {
        if (this.neuralNetwork.randomFloat() < 0.5) {
          this.neuralNetwork.flipVertical(resizedHeight, resizedWidth);

          for (let i = 0; i < coordinates.length; ++i) {
            coordinates[i][0] = resizedGaussianHeight - coordinates[i][0];
          }
        }
      }
      if (this.horizontalFlip) {
        if (this.neuralNetwork.randomFloat() < 0.5) {
          this.neuralNetwork.flipHorizontal(resizedHeight, resizedWidth);

          for (let i = 0; i < coordinates.length; ++i) {
            coordinates[i][1] = resizedGaussianWidth - coordinates[i][1];
          }
        }
      }

      const brightnessAdjustment = (this.neuralNetwork.randomFloat() - 0.5) / 10.0;
      this.neuralNetwork.brightnessAdjustment(resizedHeight, resizedWidth, brightnessAdjustment);

      const angle = (this.neuralNetwork.randomFloat() - 0.5) * 2 * 15;
      const theta = degreesToRadians(angle);

      this.neuralNetwork.rotate(resizedHeight, resizedWidth, Math.cos(theta), Math.sin(theta), null);

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

      this.neuralNetwork.drawGaussians(resizedGaussianHeight, resizedGaussianWidth, this.data.keypointCount, coordinates, this.gaussianStdDev);

      this.neuralNetwork.forward(this.neuralNetwork.rotatedOffset, resizedHeight, resizedWidth, channelsRgb);

      const trainingLoss = this.neuralNetwork.lossForward();
      meanTrainingLoss += trainingLoss;

      this.neuralNetwork.lossBackward(trainingLoss / this.batchSize);

      this.neuralNetwork.backward(this.neuralNetwork.gaussianGradientOffset);

      ++batchIndex;
      if (batchIndex === this.batchSize) {
        this.neuralNetwork.updateParameters();
        this.neuralNetwork.zeroGradients();
        batchIndex = 0;
      }
    }
    meanTrainingLoss /= this.data.trainingIndices.length;

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

      const [resizedHeight, resizedWidth] = nearestValidImageSize(imageBitmap.height, imageBitmap.width, this.data.maxImageSize, 8);

      const resizedGaussianHeight = resizedHeight / 2;
      const resizedGaussianWidth = resizedWidth / 2;

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

      this.neuralNetwork.drawGaussians(resizedGaussianHeight, resizedGaussianWidth, this.data.keypointCount, coordinates, this.gaussianStdDev);

      this.neuralNetwork.forward(this.neuralNetwork.resizedOffset, resizedHeight, resizedWidth, channelsRgb);

      const predictions = this.neuralNetwork.predictions();
      const predictionCoordinates = argmax(predictions, resizedGaussianHeight, resizedGaussianWidth, this.data.keypointCount);
      for (let i = 0; i < predictionCoordinates.length; ++i) {
        predictionCoordinates[i].y *= resizedHeight / resizedGaussianHeight;
        predictionCoordinates[i].x *= resizedWidth / resizedGaussianWidth;
      }
      for (let i = 0; i < predictionCoordinates.length; ++i) {
        predictionCoordinates[i].y *= imageBitmap.height / resizedHeight;
        predictionCoordinates[i].x *= imageBitmap.width / resizedWidth;
      }
      cachedPredictions.push(predictionCoordinates);

      const validationLoss = this.neuralNetwork.lossForward();
      meanValidationLoss += validationLoss;
    }
    meanValidationLoss /= this.data.validationIndices.length;

    this.data.meanTrainingLosses.push(meanTrainingLoss);
    this.data.meanValidationLosses.push(meanValidationLoss);
    if (this.epoch === 0) {
      this.data.bestTrainingLoss = meanTrainingLoss;
      this.data.bestValidationLoss = meanValidationLoss;
      this.data.cachedPredictions = cachedPredictions;

      this.data.bestWeights = this.neuralNetwork.getParameters();
    }
    else {
      if (meanValidationLoss <= this.data.bestValidationLoss) {
        this.data.bestValidationLoss = meanValidationLoss;
        this.data.bestTrainingLoss = meanTrainingLoss;
        this.data.cachedPredictions = cachedPredictions;

        this.data.bestWeights = this.neuralNetwork.getParameters();
      }
    }

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
  }
}


new TrainingWorker();
