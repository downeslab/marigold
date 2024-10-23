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

import { Section } from "../section.js";

import { interpolateColormap } from "../colormap.js";
import { maxImageSizeOptions, maxImageSizeDefault, batchSizeOptions, batchSizeDefault, learningRateOptions, learningRateDefault } from "../constants.js";


function beforeUnloadListener(event) {
  event.preventDefault();
  return (event.returnValue = "");
}


function round(number, digits) {
  return number.toFixed(digits);
}


export class TrainingSection extends Section {
  cachedLabels = null;
  cachedTrainingIndices = null;
  cachedValidationIndices = null;
  cachedPredictions = null;
  previewIndex = null;
  blobs = null;

  meanTrainingLosses = null;
  meanValidationLosses = null;

  trainingPaused = null;

  constructor() {
    super(
      "model",
      "training"
    );

    this.worker.addEventListener(
      "message",
      (message) => {
        if (message.data.type === "importDatasetSuccess") {
          this.onImportDatasetSuccess(message.data.trainingIndices, message.data.validationIndices);
        }
        else if (message.data.type === "importDatasetFailure") {
          this.onImportDatasetFailure();
        }
        else if (message.data.type === "epochDone") {
          this.onEpochDone(
            message.data.epoch,
            message.data.bestTrainingLoss,
            message.data.bestValidationLoss,
            message.data.meanTrainingLosses,
            message.data.meanValidationLosses,
            message.data.labels,
            message.data.trainingIndices,
            message.data.validationIndices,
            message.data.cachedPredictions,
            message.data.blobs
          );
        }
      }
    );

    document.querySelector("#load-dataset-button").addEventListener(
      "click",
      (event) => {
        this.maybeImportDataset();
      }
    );

    document.querySelector("#start-training-button").addEventListener(
      "click",
      (event) => {
        this.maybeStartTraining();
      }
    );

    document.querySelector("#pause-training-button").addEventListener(
      "click",
      (event) => {
        this.maybePauseTraining();
      }
    );

    document.querySelector("#training-seed-input").addEventListener(
      "change",
      (event) => {
        let value = document.querySelector("#training-seed-input").value;
        this.worker.postMessage({ type: "trainingSeed", trainingSeed: +value });

        this.unsavedChanges = true;
        this.showStatus(Section.unsavedMessage);
        addEventListener("beforeunload", beforeUnloadListener);
      }
    );
    {
      let value = document.querySelector("#training-seed-input").value;
      this.worker.postMessage({ type: "trainingSeed", trainingSeed: +value });
    }

    document.querySelector("#horizontal-flip-checkbox").addEventListener(
      "input",
      (event) => {
        this.worker.postMessage({ type: "horizontalFlip", horizontalFlip: document.querySelector("#horizontal-flip-checkbox").checked });

        this.unsavedChanges = true;
        this.showStatus(Section.unsavedMessage);
        addEventListener("beforeunload", beforeUnloadListener);
      }
    );
    this.worker.postMessage({ type: "horizontalFlip", horizontalFlip: document.querySelector("#horizontal-flip-checkbox").checked });

    document.querySelector("#vertical-flip-checkbox").addEventListener(
      "input",
      (event) => {
        this.worker.postMessage({ type: "verticalFlip", verticalFlip: document.querySelector("#vertical-flip-checkbox").checked });

        this.unsavedChanges = true;
        this.showStatus(Section.unsavedMessage);
        addEventListener("beforeunload", beforeUnloadListener);
      }
    );
    this.worker.postMessage({ type: "verticalFlip", verticalFlip: document.querySelector("#vertical-flip-checkbox").checked });

    //

    for (const input of document.querySelectorAll("input[name=max-input-size]")) {
      input.addEventListener(
        "change",
        (event) => {
          const value = document.querySelector("input[name=max-input-size]:checked").value;
          this.worker.postMessage({ type: "maxImageSize", maxImageSize: +value });

          this.unsavedChanges = true;
          this.showStatus(Section.unsavedMessage);
          addEventListener("beforeunload", beforeUnloadListener);
        }
      );
    }
    {
      const value = document.querySelector("input[name=max-input-size]:checked").value;
      this.worker.postMessage({ type: "maxImageSize", maxImageSize: +value });
    }

    for (const input of document.querySelectorAll("input[name=channels-per-keypoint]")) {
      input.addEventListener(
        "change",
        (event) => {
          const value = document.querySelector("input[name=channels-per-keypoint]:checked").value;
          this.worker.postMessage({ type: "channelCount", channelCount: +value });

          this.unsavedChanges = true;
          this.showStatus(Section.unsavedMessage);
          addEventListener("beforeunload", beforeUnloadListener);
        }
      );
    }
    {
      const value = document.querySelector("input[name=channels-per-keypoint]:checked").value;
      this.worker.postMessage({ type: "channelCount", channelCount: +value });
    }

    for (const input of document.querySelectorAll("input[name=block-count]")) {
      input.addEventListener(
        "change",
        (event) => {
          const value = document.querySelector("input[name=block-count]:checked").value;
          this.worker.postMessage({ type: "blockCount", blockCount: +value });

          this.unsavedChanges = true;
          this.showStatus(Section.unsavedMessage);
          addEventListener("beforeunload", beforeUnloadListener);
        }
      );
    }
    {
      const value = document.querySelector("input[name=block-count]:checked").value;
      this.worker.postMessage({ type: "blockCount", blockCount: +value });
    }

    for (const input of document.querySelectorAll("input[name=batch-size]")) {
      input.addEventListener(
        "change",
        (event) => {
          const value = document.querySelector("input[name=batch-size]:checked").value;
          this.worker.postMessage({ type: "batchSize", batchSize: +value });

          this.unsavedChanges = true;
          this.showStatus(Section.unsavedMessage);
          addEventListener("beforeunload", beforeUnloadListener);
        }
      );
    }
    {
      const value = document.querySelector("input[name=batch-size]:checked").value;
      this.worker.postMessage({ type: "batchSize", batchSize: +value });
    }

    for (const input of document.querySelectorAll("input[name=learning-rate]")) {
      input.addEventListener(
        "change",
        (event) => {
          const value = document.querySelector("input[name=learning-rate]:checked").value;
          this.worker.postMessage({ type: "learningRate", learningRate: +value });

          this.unsavedChanges = true;
          this.showStatus(Section.unsavedMessage);
          addEventListener("beforeunload", beforeUnloadListener);
        }
      );
    }
    {
      const value = document.querySelector("input[name=learning-rate]:checked").value;
      this.worker.postMessage({ type: "learningRate", learningRate: +value });
    }

    document.querySelector("#epochs-input").addEventListener(
      "change",
      (event) => {
        let value = document.querySelector("#epochs-input").value;
        this.worker.postMessage({ type: "epochs", epochs: +value });

        this.unsavedChanges = true;
        this.showStatus(Section.unsavedMessage);
        addEventListener("beforeunload", beforeUnloadListener);

        document.querySelector("#start-training-button").removeAttribute("disabled"); // temp
      }
    );
    {
      let value = document.querySelector("#epochs-input").value;
      this.worker.postMessage({ type: "epochs", epochs: +value });
    }

    document.querySelector("#training-dataset-setup-button").addEventListener(
      "click",
      (event) => {
        if (document.querySelector("#training-dataset-setup-button").classList.contains("activated")) {
          document.querySelector("#training-dataset-setup-button-divider").style.display = "flex";
          document.querySelector("#training-neural-network-setup-button-divider").style.display = "flex";
          document.querySelector("#training-training-process-button-divider").style.display = "flex";

          document.querySelector("#training-dataset-setup-button").style.display = "flex";
          document.querySelector("#training-neural-network-setup-button").style.display = "flex";
          document.querySelector("#training-training-process-button").style.display = "flex";

          document.querySelector("#training-dataset-setup-button").classList.remove("activated");
          document.querySelector("#training-dataset-setup-button").classList.remove("panel-button-bottom");

          document.querySelector("#training-dataset-setup-inner-section-left-side-1").style.display = "none";
          document.querySelector("#training-dataset-setup-inner-section-left-side-2").style.display = "none";
          document.querySelector("#training-dataset-setup-inner-section-right-side").style.display = "none";
        }
        else {
          document.querySelector("#training-neural-network-setup-button-divider").style.display = "none";
          document.querySelector("#training-training-process-button-divider").style.display = "none";

          document.querySelector("#training-neural-network-setup-button").style.display = "none";
          document.querySelector("#training-training-process-button").style.display = "none";

          document.querySelector("#training-dataset-setup-button").classList.add("activated");
          document.querySelector("#training-dataset-setup-button").classList.add("panel-button-bottom");

          document.querySelector("#training-dataset-setup-inner-section-left-side-1").style.display = "flex";
          // document.querySelector("#training-dataset-setup-inner-section-left-side-2").style.display = "flex"; // hide this for now
          document.querySelector("#training-dataset-setup-inner-section-right-side").style.display = "flex";
        }

        window.scrollTo(0, 0);
      }
    );

    document.querySelector("#training-neural-network-setup-button").addEventListener(
      "click",
      (event) => {
        if (document.querySelector("#training-neural-network-setup-button").classList.contains("activated")) {
          document.querySelector("#training-dataset-setup-button-divider").style.display = "flex";
          document.querySelector("#training-neural-network-setup-button-divider").style.display = "flex";
          document.querySelector("#training-training-process-button-divider").style.display = "flex";

          document.querySelector("#training-dataset-setup-button").style.display = "flex";
          document.querySelector("#training-neural-network-setup-button").style.display = "flex";
          document.querySelector("#training-training-process-button").style.display = "flex";

          document.querySelector("#training-neural-network-setup-button").classList.remove("activated");
          document.querySelector("#training-neural-network-setup-button").classList.remove("panel-button-bottom");

          document.querySelector("#training-neural-network-setup-inner-section-left-side-1").style.display = "none";
          document.querySelector("#training-neural-network-setup-inner-section-left-side-2").style.display = "none";
          document.querySelector("#training-neural-network-setup-inner-section-left-side-3").style.display = "none";
          document.querySelector("#training-neural-network-setup-inner-section-left-side-4").style.display = "none";
          document.querySelector("#training-neural-network-setup-inner-section-left-side-5").style.display = "none";
        }
        else {
          document.querySelector("#training-dataset-setup-button-divider").style.display = "none";
          document.querySelector("#training-training-process-button-divider").style.display = "none";

          document.querySelector("#training-dataset-setup-button").style.display = "none";
          document.querySelector("#training-training-process-button").style.display = "none";

          document.querySelector("#training-neural-network-setup-button").classList.add("activated");
          document.querySelector("#training-neural-network-setup-button").classList.add("panel-button-bottom");

          document.querySelector("#training-neural-network-setup-inner-section-left-side-1").style.display = "flex";
          document.querySelector("#training-neural-network-setup-inner-section-left-side-2").style.display = "flex";
          document.querySelector("#training-neural-network-setup-inner-section-left-side-3").style.display = "flex";
          document.querySelector("#training-neural-network-setup-inner-section-left-side-4").style.display = "flex";
          document.querySelector("#training-neural-network-setup-inner-section-left-side-5").style.display = "flex";
        }

        window.scrollTo(0, 0);
      }
    );

    document.querySelector("#training-training-process-button").addEventListener(
      "click",
      (event) => {
        if (document.querySelector("#training-training-process-button").classList.contains("activated")) {

          document.querySelector("#training-dataset-setup-button-divider").style.display = "flex";
          document.querySelector("#training-neural-network-setup-button-divider").style.display = "flex";
          document.querySelector("#training-training-process-button-divider").style.display = "flex";

          document.querySelector("#training-dataset-setup-button").style.display = "flex";
          document.querySelector("#training-neural-network-setup-button").style.display = "flex";
          document.querySelector("#training-training-process-button").style.display = "flex";

          document.querySelector("#training-training-process-button").classList.remove("activated");
          document.querySelector("#training-training-process-button").classList.remove("panel-button-bottom");

          document.querySelector("#training-training-process-inner-section-left-side").style.display = "none";
          document.querySelector("#training-training-process-inner-section-right-side").style.display = "none";
        }
        else {
          document.querySelector("#training-dataset-setup-button-divider").style.display = "none";
          document.querySelector("#training-neural-network-setup-button-divider").style.display = "none";

          document.querySelector("#training-dataset-setup-button").style.display = "none";
          document.querySelector("#training-neural-network-setup-button").style.display = "none";

          document.querySelector("#training-training-process-button").classList.add("activated");
          document.querySelector("#training-training-process-button").classList.add("panel-button-bottom");

          document.querySelector("#training-training-process-inner-section-left-side").style.display = "flex";
          document.querySelector("#training-training-process-inner-section-right-side").style.display = "flex";
        }

        window.scrollTo(0, 0);
      }
    );

    this.reset();
  }


  reset() {
    super.reset();
    document.querySelector("#start-training-button").setAttribute("disabled", "");

    document.querySelector("#training-dataset-setup-inner-section-left-side-1").style.display = "none";
    document.querySelector("#training-dataset-setup-inner-section-left-side-2").style.display = "none";
    document.querySelector("#training-neural-network-setup-inner-section-left-side-1").style.display = "none";
    document.querySelector("#training-neural-network-setup-inner-section-left-side-2").style.display = "none";
    document.querySelector("#training-neural-network-setup-inner-section-left-side-3").style.display = "none";
    document.querySelector("#training-neural-network-setup-inner-section-left-side-4").style.display = "none";
    document.querySelector("#training-neural-network-setup-inner-section-left-side-5").style.display = "none";
    document.querySelector("#training-training-process-inner-section-left-side").style.display = "none";

    document.querySelector("#training-dataset-setup-inner-section-right-side").style.display = "none";
    document.querySelector("#training-neural-network-setup-inner-section-right-side").style.display = "none";
    document.querySelector("#training-training-process-inner-section-right-side").style.display = "none";
  }


  enter() {
    this.worker.postMessage({ type: "horizontalFlip", horizontalFlip: document.querySelector("#horizontal-flip-checkbox").checked });
    this.worker.postMessage({ type: "verticalFlip", verticalFlip: document.querySelector("#vertical-flip-checkbox").checked });

    {
      let value = document.querySelector("#training-seed-input").value;
      this.worker.postMessage({ type: "trainingSeed", trainingSeed: +value });
    }
    {
      const value = document.querySelector("input[name=max-input-size]:checked").value;
      this.worker.postMessage({ type: "maxImageSize", maxImageSize: +value });
    }
    {
      const value = document.querySelector("input[name=channels-per-keypoint]:checked").value;
      this.worker.postMessage({ type: "channelCount", channelCount: +value });
    }
    {
      const value = document.querySelector("input[name=block-count]:checked").value;
      this.worker.postMessage({ type: "blockCount", blockCount: +value });
    }
    {
      const value = document.querySelector("input[name=batch-size]:checked").value;
      this.worker.postMessage({ type: "batchSize", batchSize: +value });
    }
    {
      const value = document.querySelector("input[name=learning-rate]:checked").value;
      this.worker.postMessage({ type: "learningRate", learningRate: +value });
    }
    {
      let value = document.querySelector("#epochs-input").value;
      this.worker.postMessage({ type: "epochs", epochs: +value });
    }

    super.enter();
  }

  leave() {
    super.leave();
    window.location.reload();
  }


  onSaveFileSuccess() {
    super.onSaveFileSuccess();
    removeEventListener("beforeunload", beforeUnloadListener);
  }


  async maybeImportDataset() {
    let fileHandle = null;
    try {
      [fileHandle] = await window.showOpenFilePicker(
        {
          id: "loadDataset",
          startIn: "documents",
          types: [
            {
              description: "Marigold files",
              accept: {
                "application/json": [".marigold"]
              }
            }
          ],
          mode: "read"
        }
      );
    }
    catch {
    }

    if (fileHandle) {
      this.worker.postMessage({ type: "importDataset", fileHandle, fileHandle });
    }
  }

  async onImportDatasetSuccess(trainingIndices, validationIndices) {
    document.querySelector("#start-training-button").removeAttribute("disabled");

    this.unsavedChanges = true;
    this.showStatus(Section.unsavedMessage);
    addEventListener("beforeunload", beforeUnloadListener);

    let trainingIndicesList = document.querySelector("#training-indices");
    trainingIndicesList.innerHTML = "";
    for (const trainingIndex of trainingIndices) {
      let element = document.createElement("li");
      element.textContent = `Image ${trainingIndex + 1}`;
      element.classList.add("training-status");
      trainingIndicesList.appendChild(element);
    }
    let validationIndicesList = document.querySelector("#validation-indices");
    validationIndicesList.innerHTML = "";
    for (const validationIndex of validationIndices) {
      let element = document.createElement("li");
      element.textContent = `Image ${validationIndex + 1}`;
      element.classList.add("training-status");
      validationIndicesList.appendChild(element);
    }
  }

  onImportDatasetFailure() {
    document.querySelector("#generic-dialog-heading").textContent = "[Error heading]";
    document.querySelector("#generic-dialog-blurb").textContent = "[Error blurb]";
    document.querySelector("#generic-dialog").showModal();
  }


  maybeStartTraining() {
    this.trainingPaused = false;

    document.querySelector("#start-training-button").setAttribute("disabled", "");
    document.querySelector("#pause-training-button").removeAttribute("disabled");

    this.worker.postMessage({ type: "startTraining" });
  }

  maybePauseTraining() {
    this.trainingPaused = true;

    document.querySelector("#start-training-button").setAttribute("disabled", "");
    document.querySelector("#pause-training-button").setAttribute("disabled", "");
  }


  async onEpochDone(
    epoch,
    bestTrainingLoss,
    bestValidationLoss,
    meanTrainingLosses,
    meanValidationLosses,
    labels,
    trainingIndices,
    validationIndices,
    cachedPredictions,
    blobs
  ) {
    if (epoch === 0) {
      const input = document.querySelector("#training-preview-index-input");
      input.removeAttribute("disabled");

      input.value = 1;

      this.cachedLabels = structuredClone(labels);
      this.cachedPredictions = structuredClone(cachedPredictions);
      this.cachedTrainingIndices = structuredClone(trainingIndices);
      this.cachedValidationIndices = structuredClone(validationIndices);

      input.addEventListener(
        "input",
        (event) => {
          let min = 1;
          let max = this.cachedValidationIndices.length;
          const input = document.querySelector("#training-preview-index-input");
          input.value = Math.max(min, Math.min(max, input.value));
          this.updateTrainingPreview();
        }
      );

      this.blobs = blobs;
    }
    else {
      this.cachedPredictions = structuredClone(cachedPredictions);
    }

    document.querySelector("#epoch-status").textContent = `Epoch: ${epoch + 1}`;
    document.querySelector("#training-loss-status").textContent = `Current training loss: ${round(meanTrainingLosses[meanTrainingLosses.length - 1], 5)}`;
    document.querySelector("#validation-loss-status").textContent = `Current validation loss: ${round(meanValidationLosses[meanValidationLosses.length - 1], 5)}`;
    document.querySelector("#best-training-loss-status").textContent = `Best training loss: ${round(bestTrainingLoss, 5)}`;
    document.querySelector("#best-validation-loss-status").textContent = `Best validation loss: ${round(bestValidationLoss, 5)}`;

    this.meanTrainingLosses = structuredClone(meanTrainingLosses);
    this.meanValidationLosses = structuredClone(meanValidationLosses);

    this.updateTrainingPreview();

    this.unsavedChanges = true;
    this.showStatus(Section.unsavedMessage);
    addEventListener("beforeunload", beforeUnloadListener);

    if (epoch < document.querySelector("#epochs-input").value - 1 && !this.trainingPaused) {
      this.maybeStartTraining();
    }
    else {
      this.maybePauseTraining();
    }

    if (epoch < document.querySelector("#epochs-input").value - 1 && this.trainingPaused) {
      document.querySelector("#start-training-button").removeAttribute("disabled");
    }
  }

  async updateTrainingPreview() {
    const input = document.querySelector("#training-preview-index-input");
    this.previewIndex = input.value - 1;

    const label = this.cachedPredictions[this.previewIndex];

    const image = this.blobs[this.cachedValidationIndices[this.previewIndex]];
    const imageBitmap = await createImageBitmap(image, { colorSpaceConversion: "none" });

    let canvas = document.querySelector("#training-preview-canvas");
    const context = canvas.getContext("2d");

    const ratio = window.devicePixelRatio || 1;
    const boundingRect = canvas.getBoundingClientRect();

    canvas.width = imageBitmap.width;
    canvas.height = imageBitmap.height;

    context.save();

    const zoom = 1.0;

    const panOffsetX = 0.0;
    const panOffsetY = 0.0;

    const scale = Math.min(
      (boundingRect.width / boundingRect.height) * (canvas.height / canvas.width),
      (boundingRect.height / boundingRect.width) * (canvas.width / canvas.height)
    );

    const panCenterX = canvas.width / 2 + panOffsetX;
    const panCenterY = canvas.height / 2 + panOffsetY;

    context.setTransform(
      scale * zoom,
      0,
      0,
      scale * zoom,
      (canvas.width / 2) - (panCenterX * scale * zoom),
      (canvas.height / 2) - (panCenterY * scale * zoom)
    );

    context.drawImage(imageBitmap, 0, 0, imageBitmap.width, imageBitmap.height);

    const colormap = interpolateColormap("plasma", label.length, true);

    let i = 0;
    for (const coordinate of label) {
      const radius = 3.75 * Math.min(imageBitmap.width, imageBitmap.height) / 512;

      context.beginPath();
      context.arc(coordinate.x, coordinate.y, radius, 0, 2 * Math.PI, false);
      context.fillStyle = `rgb(${Math.round(255 * colormap[i][0])}, ${Math.round(255 * colormap[i][1])}, ${Math.round(255 * colormap[i][2])}, 0.75)`;
      context.fill();
      ++i;
    }

    context.restore();

    const plotWidth = 32;
    const plotHeight = 24;

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("id", "training-plot");
    svg.setAttribute("viewBox", `0 0 ${plotWidth} ${plotHeight}`);

    let maxLoss = 0.0;
    for (const loss of this.meanTrainingLosses) {
      if (loss > maxLoss) {
        maxLoss = loss;
      }
    }
    for (const loss of this.meanValidationLosses) {
      if (loss > maxLoss) {
        maxLoss = loss;
      }
    }

    maxLoss = 1.125;

    const axisLabelMargin = 2;
    const circleRadius = 0.5;

    let j = 0;
    for (const loss of this.meanTrainingLosses) {
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", axisLabelMargin + circleRadius + j * ((plotWidth - 2 * circleRadius - axisLabelMargin) / this.meanTrainingLosses.length));
      circle.setAttribute("cy", plotHeight - (axisLabelMargin + circleRadius + loss * ((plotHeight - 2 * circleRadius - axisLabelMargin) / maxLoss)));
      circle.setAttribute("r", circleRadius);
      circle.setAttribute("fill", "hsl(195deg, 50%, 50%, 0.625)");
      circle.setAttribute("id", `training-${j}`);
      circle.setAttribute("title", `Epoch: ${j}; training loss: ${loss}`);
      svg.appendChild(circle);
      ++j;
    }

    j = 0;
    for (const loss of this.meanValidationLosses) {
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", axisLabelMargin + circleRadius + j * ((plotWidth - 2 * circleRadius - axisLabelMargin) / this.meanValidationLosses.length));
      circle.setAttribute("cy", plotHeight - (axisLabelMargin + circleRadius + loss * ((plotHeight - 2 * circleRadius - axisLabelMargin) / maxLoss)));
      circle.setAttribute("r", circleRadius);
      circle.setAttribute("fill", "hsl(15deg, 50%, 50%, 0.625)");
      circle.setAttribute("id", `training-${j}`);
      circle.setAttribute("title", `Epoch: ${j}; validation loss: ${loss}`);
      svg.appendChild(circle);
      ++j;
    }

    const yAxisLabelGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    yAxisLabelGroup.setAttribute("transform", "rotate(-90 16 12) translate(0 -15)");
    svg.appendChild(yAxisLabelGroup);

    const yAxisLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    yAxisLabel.setAttribute("x", 16);
    yAxisLabel.setAttribute("y", 12);
    yAxisLabel.setAttribute("fill", "hsl(0deg, 0%, 100%, 0.875)");
    yAxisLabel.classList.add("axis-label");
    yAxisLabel.textContent = "Loss (mean squared error)";
    yAxisLabelGroup.appendChild(yAxisLabel);

    const xAxisLabelGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    xAxisLabelGroup.setAttribute("transform", "rotate(0) translate(0 0)");
    svg.appendChild(xAxisLabelGroup);

    const xAxisLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    xAxisLabel.setAttribute("x", 16);
    xAxisLabel.setAttribute("y", 23);
    xAxisLabel.setAttribute("fill", "hsl(0deg, 0%, 100%, 0.875)");
    xAxisLabel.classList.add("axis-label");
    xAxisLabel.textContent = "Epoch";
    xAxisLabelGroup.appendChild(xAxisLabel);

    document.querySelector("#training-plot").replaceWith(svg);
  }
}
