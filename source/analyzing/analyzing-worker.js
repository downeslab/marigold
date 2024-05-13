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

import { MovieReader } from "../movie-reader.js";
import { channelsRgb, channelsRgba, nearestValidImageSize, argmax, argmaxWithinCircle } from "../image.js";
import { NeuralNetwork } from "../neural-network.js";
import { testZip } from "../zip.js";


let temp = null;


class AnalyzingWorker extends SectionWorker {
  movieReader = null;
  calibrationMovieReader = null;

  arenas = null;
  frameSelection = null;

  neuralNetwork = null;
  // neuralNetworkMovieReader = null;
  neuralNetworkResults = [];

  analyzing = false;

  canvas = null;
  context = null;

  arenaShape = null;

  constructor() {
    super("analysis");

    temp = this; // weird

    self.addEventListener(
      "message",
      (message) => {
        //
        //
        //
        // console.log("worker received message:", message.data.type);
        if (message.data.type === "keypointCount") {
          this.data.keypointCount = Number(message.data.keypointCount);
        }
        else if (message.data.type === "maybeLoadMovie") {
          this.maybeLoadMovie(message.data.fileHandle);
        }
        else if (message.data.type === "frameRequest") {
          this.movieReader.seekFrame(message.data.index);
        }
        else if (message.data.type === "maybeLoadModel") {
          this.maybeLoadModel(message.data.fileHandle);
        }
        else if (message.data.type === "frameSelection") {
          this.setFrameSelection(message.data.firstFrame, message.data.lastFrame);
        }
        else if (message.data.type === "arenaShape") {
          this.arenaShape = message.data.arenaShape;
        }
        else if (message.data.type === "arenas") {
          this.setArenas(message.data.arenas);
        }
        else if (message.data.type === "startAnalysis") {
          this.startAnalysis();
        }
        else if (message.data.type === "processFrame") {
          this.processFrame(message.data.index);
        }
        else if (message.data.type === "manualCorrection") {
          this.acceptManualCorrection(message.data.frameIndex, message.data.row, message.data.arenaColumns, message.data.column, message.data.coordinates);
        }
        else if (message.data.type === "exportResults") {
          this.testZip(message.data.svgFilenames, message.data.svgFileContents, message.data.csvFilenames, message.data.csvFileContents, message.data.summaryFilename, message.data.summaryFileContents);
        }
        //
        //
        //
      }
    );
  }

  initializeData() {
    super.initializeData();
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

  onCalibrationFrameReady(data) {
    // console.log("onCalibrationFrameReady");
    self.postMessage({ type: "calibrationFrameReady", frame: data.frame, frameNumber: data.frameNumber });
  }

  async maybeLoadCalibrationMovie(fileHandle) {
    const file = await fileHandle.getFile();
    const data = await file.arrayBuffer();
    const buffer = new ArrayBuffer(data.byteLength);
    new Uint8Array(buffer).set(new Uint8Array(data));

    let calibrationMovieReader = null;
    try {
      // console.log("movieReader?");
      calibrationMovieReader = new MovieReader(buffer, this.onFrameReady);
      // console.log("movieReader!");
    }
    catch (error) {
      // console.log(error);
      self.postMessage({ type: "loadCalibrationMovieFailure", filename: fileHandle.name });
    }

    if (movieReader) {
      this.calibrationMovieReader = calibrationMovieReader;
      self.postMessage({ type: "loadCalibrationMovieSuccess", filename: fileHandle.name, numFrames: this.calibrationMovieReader.mp4Parser.numFrames, frameWidth: this.calibrationMovieReader.mp4Parser.frameWidth, frameHeight: this.calibrationMovieReader.mp4Parser.frameHeight });
    }
  }

  //
  //
  //

  setFrameSelection(firstFrame, lastFrame) {
    this.frameSelection = { firstFrame: firstFrame, lastFrame: lastFrame };
  }

  setArenas(arenas) {
    this.arenas = arenas;
  }

  //
  //
  //

  onFrameReady(data) {
    // console.log("onFrameReady");

    // const x = new Uint8ClampedArray(temp.movieReader.mp4Parser.frameHeight * temp.movieReader.mp4Parser.frameWidth * 4);

    // for (let i = 0; i < temp.movieReader.mp4Parser.frameHeight * temp.movieReader.mp4Parser.frameWidth * 4; ++i) {
    //   if (i < 100) {
    //     // console.log("data COPYING:", data.frame[i]);
    //   }
    //   x[i] = data.frame[i];
    // }

    if (temp.analyzing) {
      temp.onNeuralNetworkFrameReady(data);
    }
    else {
      self.postMessage({ type: "frameReady", frame: data.frame, frameNumber: data.frameNumber });
    }

    // // temp.onNeuralNetworkFrameReady(data.frame, data.frameNumber);
    // temp.onNeuralNetworkFrameReady(data);
  }

  async maybeLoadMovie(fileHandle) {
    const file = await fileHandle.getFile();
    const data = await file.arrayBuffer();
    const buffer = new ArrayBuffer(data.byteLength);
    new Uint8Array(buffer).set(new Uint8Array(data));

    let movieReader = null;
    try {
      // console.log("movieReader?");
      movieReader = new MovieReader(buffer, this.onFrameReady);
      // this.neuralNetworkBuffer = structuredClone(buffer);
      // this.neuralNetworkMovieReader = new MovieReader(this.neuralNetworkBuffer, this.onNeuralNetworkFrameReady);
      // console.log("movieReader!");
    }
    catch (error) {
      // console.log(error);
      self.postMessage({ type: "loadMovieFailure", filename: fileHandle.name });
    }

    if (movieReader) {
      this.movieReader = movieReader;
      self.postMessage({ type: "loadMovieSuccess", filename: fileHandle.name, numFrames: this.movieReader.mp4Parser.numFrames, frameWidth: this.movieReader.mp4Parser.frameWidth, frameHeight: this.movieReader.mp4Parser.frameHeight });
    }
    else {
      // console.log("hmmmmm");
    }
  }


  async maybeLoadModel(fileHandle) {
    // console.log("maybeLoadModel");

    const file = await fileHandle.getFile();
    const json = JSON.parse(await file.text());

    // console.log("json:", channelsRgb, json.channelCount, json.keypointCount, json.blockCount, json.maxImageSize, null, json.bestWeights);

    this.data.channelCount = json.channelCount;
    this.data.keypointCount = json.keypointCount;
    this.data.blockCount = json.blockCount;
    this.data.maxImageSize = json.maxImageSize;

    this.neuralNetwork = new NeuralNetwork(channelsRgb, json.channelCount, json.keypointCount, json.blockCount, json.maxImageSize, null);
    this.neuralNetwork.setParameters(json.bestWeights);

    // this.neuralNetworkMovieReader.seekFrame(0);
    // this.processFrame(0);

    // for (let i = 0; i < temp.neuralNetworkMovieReader.mp4Parser.numFrames - 1; ++i) {
    //   await this.neuralNetworkMovieReader.seekFrame(i);
    // }

    self.postMessage({ type: "loadModelSuccess", filename: fileHandle.name });
    // self.postMessage({ type: "loadModelFailure", filename: fileHandle.name });
  }


  async startAnalysis() {
    // this.neuralNetworkMovieReader.seekFrame(0);

    if (this.frameSelection === null) {
      // this.frameSelection = { firstFrame: 0, lastFrame: this.neuralNetworkMovieReader.mp4Parser.numFrames - 1 };
      this.frameSelection = { firstFrame: 0, lastFrame: this.movieReader.mp4Parser.numFrames - 1 };
    }
    if (this.arenas === null) {
      this.arenas = [{ x: 0, y: 0, width: this.movieReader.mp4Parser.frameWidth, height: temp.movieReader.mp4Parser.frameHeight }];
    }

    this.analyzing = true;

    await this.processFrame(0);

    // await this.processFrame(this.frameSelection.firstFrame);

    // for (let i = 0; i < temp.neuralNetworkMovieReader.mp4Parser.numFrames - 1; ++i) {
    //   await this.neuralNetworkMovieReader.seekFrame(i);
    // }
  }

  lastT = null;
  async onNeuralNetworkFrameReady(data) {
    // console.log("onNeuralNetworkFrameReady");

    // reminder: can i use scheduleTask to avoid having to cram this all into this function?
    const t = performance.now();
    if (temp.lastT === null) {
      temp.lastT = t;
    }
    // console.log("t =", (t - temp.lastT) / 1000);
    temp.lastT = t;

    //
    //
    //

    // if (temp.frameSelection === null || (data.frameNumber >= temp.frameSelection.firstFrame && data.frameNumber <= temp.frameSelection.lastFrame)) {

    // const canvas = new OffscreenCanvas(data.frame.width, data.frame.height);
    // // const context = canvas.getContext("2d");
    // const context = canvas.getContext("2d", { willReadFrequently: true });
    // context.drawImage(data.frame, 0, 0, data.frame.width, data.frame.height);

    if (!temp.canvas) {
      temp.canvas = new OffscreenCanvas(data.frame.width, data.frame.height);
      // temp.context = temp.canvas.getContext("2d");
      temp.context = temp.canvas.getContext("2d", { willReadFrequently: true });
    }
    temp.context.drawImage(data.frame, 0, 0, data.frame.width, data.frame.height);

    // const resized = new Float32Array(resizedHeight * resizedWidth * channelsRgb);
    // const resized = new Float32Array(temp.data.maxImageSize * temp.data.maxImageSize * channelsRgb);

    const frameCoordinates = [];
    // console.log("temp.arenas:", temp.arenas);
    for (let i = 0; i < temp.arenas.length; ++i) {
      const arena = temp.arenas[i];

      //

      const start = performance.now();

      // const canvas = new OffscreenCanvas(data.frame.width, data.frame.height);
      // const context = canvas.getContext("2d");
      // context.drawImage(data.frame, 0, 0, data.frame.width, data.frame.height);
      const imageData = temp.context.getImageData(arena.x, arena.y, arena.width, arena.height);

      const [resizedHeight, resizedWidth] = nearestValidImageSize(arena.height, arena.width, temp.data.maxImageSize, 8);

      const resizedGaussianHeight = resizedHeight / 2;
      const resizedGaussianWidth = resizedWidth / 2;

      // const resized = new Float32Array(resizedHeight * resizedWidth * channelsRgb);

      // resizeBilinearRgbaToRgb(imageData.data, resized, imageBitmap.height, imageBitmap.width, resizedHeight, resizedWidth);
      // temp.neuralNetwork.resize(imageData.data, resized, arena.height, arena.width, resizedHeight, resizedWidth);
      temp.neuralNetwork.resize(imageData.data, arena.height, arena.width, resizedHeight, resizedWidth);

      const finish = performance.now();
      // console.log("stuff t =", (finish - start) / 1000, "s");

      {
        const start = performance.now();
        // temp.neuralNetwork.forward(resized, resizedHeight, resizedWidth, channelsRgb);
        temp.neuralNetwork.forward(temp.neuralNetwork.resizedOffset, resizedHeight, resizedWidth, channelsRgb);
        const finish = performance.now();
        // console.log("neuralNetwork.forward t =", (finish - start) / 1000, "s");
      }

      const start_ = performance.now();

      const predictions = temp.neuralNetwork.predictions();
      let predictionCoordinates = null;
      if (temp.arenaShape === "circle") {
        predictionCoordinates = argmaxWithinCircle(predictions, resizedGaussianHeight, temp.data.keypointCount);
      }
      else {
        predictionCoordinates = argmax(predictions, resizedGaussianHeight, resizedGaussianWidth, temp.data.keypointCount);
      }
      for (let i = 0; i < predictionCoordinates.length; ++i) {
        predictionCoordinates[i].y *= resizedHeight / resizedGaussianHeight;
        predictionCoordinates[i].x *= resizedWidth / resizedGaussianWidth;
      }
      for (let i = 0; i < predictionCoordinates.length; ++i) {
        predictionCoordinates[i].y *= arena.height / resizedHeight;
        predictionCoordinates[i].x *= arena.width / resizedWidth;
      }
      for (let i = 0; i < predictionCoordinates.length; ++i) {
        predictionCoordinates[i].y += arena.y;
        predictionCoordinates[i].x += arena.x;
      }

      const finish_ = performance.now();
      // console.log("other stuff t =", (finish_ - start_) / 1000, "s");

      // // console.log("predictionCoordinates:", predictionCoordinates);

      // //
      // //
      // //

      const frameArenaCoordinates = predictionCoordinates;
      frameCoordinates.push(frameArenaCoordinates);
    }
    temp.neuralNetworkResults.push(frameCoordinates);

    // // console.log("noice");

    // //

    // if (data.frameNumber < temp.neuralNetworkMovieReader.mp4Parser.numFrames - 1) {
    //   // console.log("seeking frame:", data.frameNumber + 1);
    //   await temp.neuralNetworkMovieReader.seekFrame(data.frameNumber + 1);
    // }

    // if (data.frameNumber === temp.neuralNetworkMovieReader.mp4Parser.numFrames - 1) {
    // self.postMessage({ type: "resultsReady", results: temp.neuralNetworkResults });
    self.postMessage({ type: "resultsReady", results: temp.neuralNetworkResults, frame: data.frame, frameNumber: data.frameNumber });
    // }
    // }
    // else {
    //   temp.neuralNetworkResults.push(null);
    //   self.postMessage({ type: "resultsReady", results: temp.neuralNetworkResults, frame: data.frame, frameNumber: data.frameNumber });
    // }

    if (data.frameNumber === this.frameSelection.lastFrame) {
      temp.analyzing = false;
    }
  }


  async processFrame(index) {
    if (index >= this.frameSelection.firstFrame && index <= this.frameSelection.lastFrame) {
      // this.neuralNetworkMovieReader.seekFrame(index);
      this.movieReader.seekFrame(index);
    }
    else {
      temp.neuralNetworkResults.push(null);
      self.postMessage({ type: "resultsReady", results: this.neuralNetworkResults, frame: null, frameNumber: index });
    }
  }


  acceptManualCorrection(frameIndex, row, arenaColumns, column, coordinates) {
    this.neuralNetworkResults[frameIndex][row * arenaColumns + column] = coordinates;
  }


  async testZip(svgFilenames, svgFileContents, csvFilenames, csvFileContents, summaryFilename, summaryFileContents) {
    const blob = testZip(svgFilenames, svgFileContents, csvFilenames, csvFileContents, summaryFilename, summaryFileContents);

    self.postMessage({ type: "exportReady", blob: blob });
  }
}

new AnalyzingWorker();
