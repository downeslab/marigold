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
import { maxKeypointCount } from "../constants.js";
import { getCsvData } from "./kinematics.js";
import { initializePlots, updatePlots, updatePlotData, clearPlotData } from "./trajectory-plot.js";


function beforeUnloadListener(event) {
  event.preventDefault();
  return (event.returnValue = "");
}


export class AnalyzingSection extends Section {
  numFrames = null;
  frameWidth = null;
  frameHeight = null;

  currentFrame = null;
  currentFrameIndex = null;

  zoom = 1;
  panOffsetX = 0;
  panOffsetY = 0;

  currentlyPlaying = false;

  calibrationNumFrames = null;
  calibrationFrameWidth = null;
  calibrationFrameHeight = null;

  calibrationCurrentFrame = null;
  calibrationCurrentFrameIndex = null;

  calibrationZoom = 1;
  calibrationPanOffsetX = 0;
  calibrationPanOffsetY = 0;

  calibrationCurrentlyPlaying = false;

  arenas = null;

  cachedResults = null;

  keypointCount = null;

  manuallyCorrecting = false;
  manuallyCorrectingRow = null;
  manuallyCorrectingColumn = null;
  manualCorrection = null;
  manualCorrectionIndex = null;

  mouseDown = false;

  constructor() {
    super(
      "analysis",
      "analyzing"
    );

    this.worker.addEventListener(
      "message",
      (message) => {
        if (message.data.type === "loadCalibrationMovieSuccess") {
          this.onCalibrationLoadMovieSuccess(message.data.filename, message.data.numFrames, message.data.frameWidth, message.data.frameHeight);
        }
        else if (message.data.type === "loadCalibrationMovieFailure") {
          this.onCalibrationLoadMovieFailure(message.data.filename);
        }
        else if (message.data.type === "calibrationFrameReady") {
          this.onCalibrationFrameReady(message.data.frame, message.data.frameNumber);
        }

        else if (message.data.type === "loadMovieSuccess") {
          this.onLoadMovieSuccess(message.data.filename, message.data.numFrames, message.data.frameWidth, message.data.frameHeight);
        }
        else if (message.data.type === "loadMovieFailure") {
          this.onLoadMovieFailure(message.data.filename);
        }
        else if (message.data.type === "frameReady") {
          this.onFrameReady(message.data.frame, message.data.frameNumber);
        }

        else if (message.data.type === "loadModelSuccess") {
          this.onLoadModelSuccess(message.data.filename);
        }
        else if (message.data.type === "loadModelFailure") {
          this.onLoadModelFailure(message.data.filename);
        }
        else if (message.data.type === "resultsReady") {
          this.onResultsReady(message.data.results, message.data.frame, message.data.frameNumber);
        }
        else if (message.data.type === "exportReady") {
          this.onExportReady(message.data.blob);
        }
      }
    );

    initializePlots();

    document.querySelector("#analyzing-load-model-button").addEventListener(
      "click",
      (event) => {
        this.maybeLoadModel();
      }
    );

    document.querySelector("#analyzing-calibration-load-movie-button").addEventListener(
      "click",
      (event) => {
        document.querySelector("#generic-dialog-close-button").addEventListener(
          "click",
          (event) => { document.querySelector("#generic-dialog").close(); },
          { once: true }
        );
        document.querySelector("#generic-dialog-heading").textContent = "Not enabled";
        document.querySelector("#generic-dialog-blurb").textContent = "Sorry, that feature is still under construction!";
        document.querySelector("#generic-dialog").showModal();
      }
    );

    document.querySelector("#analyzing-calibration-load-movie-button").addEventListener(
      "click",
      (event) => {
      }
    );

    document.querySelector("#analyzing-export-results-button").addEventListener(
      "click",
      (event) => {
        this.prepareForExport();
      }
    );

    document.querySelector("#analyzing-trajectory-plot-preview-keypoint-input").addEventListener(
      "change",
      (event) => {
        this.updateTrajectoryPlots();
      }
    );

    document.querySelector("#analyzing-load-movie-button").addEventListener(
      "click",
      (event) => {
        this.maybeLoadMovie();
      }
    );

    const resizeObserver = new ResizeObserver(
      (entries) => {
        for (const entry of entries) {
          this.draw();
        }
      }
    );
    resizeObserver.observe(document.querySelector("#analyzing-movie-player-canvas"));

    document.querySelector("#analyzing-movie-player-canvas").addEventListener(
      "wheel",
      (event) => { this.onCanvasScroll(event); },
      { passive: false }
    );

    document.querySelector("#analyzing-zoom-in-button").addEventListener("click", (event) => { this.onZoomInButtonClick(); }
    );
    document.querySelector("#analyzing-zoom-out-button").addEventListener("click", (event) => { this.onZoomOutButtonClick(); }
    );
    document.querySelector("#analyzing-zoom-reset-button").addEventListener("click", (event) => { this.onZoomResetButtonClick(); }
    );

    document.querySelector("#analyzing-first-frame-button").addEventListener(
      "click",
      (event) => {
        this.worker.postMessage({ type: "frameRequest", index: 0 });
      }
    );
    document.querySelector("#analyzing-previous-frame-button").addEventListener(
      "click",
      (event) => {
        this.worker.postMessage({ type: "frameRequest", index: Math.max(0, this.currentFrameIndex - 1) });
      }
    );
    document.querySelector("#analyzing-play-or-pause-button").addEventListener(
      "click",
      (event) => {
        this.currentlyPlaying = !this.currentlyPlaying;
        if (this.currentlyPlaying) {
          document.querySelector("#analyzing-play-or-pause-button").classList.add("activated");
          this.worker.postMessage({ type: "frameRequest", index: Math.min(this.numFrames - 1, this.currentFrameIndex + 1) });
        }
        else {
          document.querySelector("#analyzing-play-or-pause-button").classList.remove("activated");
        }
      }
    );
    document.querySelector("#analyzing-next-frame-button").addEventListener(
      "click",
      (event) => {
        this.worker.postMessage({ type: "frameRequest", index: Math.min(this.numFrames - 1, this.currentFrameIndex + 1) });
      }
    );
    document.querySelector("#analyzing-last-frame-button").addEventListener(
      "click",
      (event) => {
        this.worker.postMessage({ type: "frameRequest", index: this.numFrames - 1 });
      }
    );

    document.querySelector("#analyzing-frame-range-input").addEventListener(
      "change",
      (event) => {
        this.worker.postMessage({ type: "frameRequest", index: document.querySelector("#analyzing-frame-range-input").value - 1 });
      }
    );
    document.querySelector("#analyzing-frame-number-input").addEventListener(
      "change",
      (event) => {
        this.worker.postMessage({ type: "frameRequest", index: document.querySelector("#analyzing-frame-number-input").value - 1 });
      }
    );

    document.querySelector("#analyzing-start-analysis-button").addEventListener(
      "click",
      (event) => {
        const progressElement = document.querySelector("#analysis-progress");
        progressElement.value = 0;

        this.worker.postMessage({ type: "startAnalysis" });
        document.querySelector("#analyzing-start-analysis-button").setAttribute("disabled", "");
      }
    );

    document.querySelector("#analyzing-movie-player-canvas").addEventListener(
      "mousedown",
      (event) => {
        if (this.manuallyCorrecting) {
          const canvas = document.querySelector("#analyzing-movie-player-canvas");
          const context = canvas.getContext("2d");

          const boundingRect = canvas.getBoundingClientRect();

          context.save();

          const scale = Math.min((boundingRect.width / boundingRect.height) * (canvas.height / canvas.width), (boundingRect.height / boundingRect.width) * (canvas.width / canvas.height));

          const panCenterX = canvas.width / 2 + this.panOffsetX;
          const panCenterY = canvas.height / 2 + this.panOffsetY;

          context.setTransform(
            scale * this.zoom,
            0,
            0,
            scale * this.zoom,
            (canvas.width / 2) - (panCenterX * scale * this.zoom),
            (canvas.height / 2) - (panCenterY * scale * this.zoom)
          );
          const scale_ = Math.max(canvas.width / boundingRect.width, canvas.height / boundingRect.height) / this.zoom;

          let diffX = (boundingRect.width * scale_ - canvas.width) / 2;
          let diffY = (boundingRect.height * scale_ - canvas.height) / 2;

          const x = (event.clientX - boundingRect.left) * scale_ - diffX;
          const y = (event.clientY - boundingRect.top) * scale_ - diffY;

          const x_ = x + this.panOffsetX;
          const y_ = y + this.panOffsetY;

          context.restore();

          this.mouseDown = true;
          this.manualCorrection[this.manualCorrectionIndex].x = x_;
          this.manualCorrection[this.manualCorrectionIndex].y = y_;
          document.querySelector(`#analyzing-keypoint-${this.manualCorrectionIndex}-x-input`).value = x_;
          document.querySelector(`#analyzing-keypoint-${this.manualCorrectionIndex}-y-input`).value = y_;

          this.draw();
        }
      }
    );
    document.querySelector("#analyzing-movie-player-canvas").addEventListener(
      "mousemove",
      (event) => {
        if (this.mouseDown) {
          if (this.manuallyCorrecting) {
            const canvas = document.querySelector("#analyzing-movie-player-canvas");
            const context = canvas.getContext("2d");

            const boundingRect = canvas.getBoundingClientRect();

            context.save();

            const scale = Math.min((boundingRect.width / boundingRect.height) * (canvas.height / canvas.width), (boundingRect.height / boundingRect.width) * (canvas.width / canvas.height));

            const panCenterX = canvas.width / 2 + this.panOffsetX;
            const panCenterY = canvas.height / 2 + this.panOffsetY;

            context.setTransform(
              scale * this.zoom,
              0,
              0,
              scale * this.zoom,
              (canvas.width / 2) - (panCenterX * scale * this.zoom),
              (canvas.height / 2) - (panCenterY * scale * this.zoom)
            );

            const scale_ = Math.max(canvas.width / boundingRect.width, canvas.height / boundingRect.height) / this.zoom;

            let diffX = (boundingRect.width * scale_ - canvas.width) / 2;
            let diffY = (boundingRect.height * scale_ - canvas.height) / 2;

            const x = (event.clientX - boundingRect.left) * scale_ - diffX;
            const y = (event.clientY - boundingRect.top) * scale_ - diffY;

            const x_ = x + this.panOffsetX;
            const y_ = y + this.panOffsetY;

            context.restore();

            this.manualCorrection[this.manualCorrectionIndex].x = x_;
            this.manualCorrection[this.manualCorrectionIndex].y = y_;
            document.querySelector(`#analyzing-keypoint-${this.manualCorrectionIndex}-x-input`).value = x_;
            document.querySelector(`#analyzing-keypoint-${this.manualCorrectionIndex}-y-input`).value = y_;
            // }

            this.draw();
          }
        }
      }
    );
    document.querySelector("#analyzing-movie-player-canvas").addEventListener(
      "mouseup",
      (event) => {
        if (this.manuallyCorrecting) {
          const canvas = document.querySelector("#analyzing-movie-player-canvas");
          const context = canvas.getContext("2d");

          const boundingRect = canvas.getBoundingClientRect();

          context.save();

          const scale = Math.min((boundingRect.width / boundingRect.height) * (canvas.height / canvas.width), (boundingRect.height / boundingRect.width) * (canvas.width / canvas.height));

          const panCenterX = canvas.width / 2 + this.panOffsetX;
          const panCenterY = canvas.height / 2 + this.panOffsetY;

          context.setTransform(
            scale * this.zoom,
            0,
            0,
            scale * this.zoom,
            (canvas.width / 2) - (panCenterX * scale * this.zoom),
            (canvas.height / 2) - (panCenterY * scale * this.zoom)
          );

          const scale_ = Math.max(canvas.width / boundingRect.width, canvas.height / boundingRect.height) / this.zoom;

          let diffX = (boundingRect.width * scale_ - canvas.width) / 2;
          let diffY = (boundingRect.height * scale_ - canvas.height) / 2;

          const x = (event.clientX - boundingRect.left) * scale_ - diffX;
          const y = (event.clientY - boundingRect.top) * scale_ - diffY;

          const x_ = x + this.panOffsetX;
          const y_ = y + this.panOffsetY;

          context.restore();

          this.mouseDown = false;
          this.manualCorrection[this.manualCorrectionIndex].x = x_;
          this.manualCorrection[this.manualCorrectionIndex].y = y_;
          document.querySelector(`#analyzing-keypoint-${this.manualCorrectionIndex}-x-input`).value = x_;
          document.querySelector(`#analyzing-keypoint-${this.manualCorrectionIndex}-y-input`).value = y_;
          this.manualCorrectionIndex = (this.manualCorrectionIndex + 1) % this.keypointCount;

          this.draw();
        }
      }
    );

    document.querySelector("#analyzing-start-manual-correction-button").addEventListener(
      "click",
      (event) => {
        document.querySelector("#analyzing-start-manual-correction-button").style.display = "none";
        document.querySelector("#analyzing-manual-correction-button-group").style.display = "flex";
        document.querySelector("#analyzing-manual-correction-arena-details").style.display = "grid";
        document.querySelector("#analyzing-manual-correction-keypoint-details").style.display = "none";
        document.querySelector("#analyzing-manual-correction-buttons").style.display = "flex";

        document.querySelector("#analyzing-set-arena-button").classList.add("button-group-button-activated");
        document.querySelector("#analyzing-set-keypoints-button").classList.remove("button-group-button-activated");

        const manualCorrectionColumnInput = document.querySelector("#analyzing-manual-correction-column-input");
        manualCorrectionColumnInput.value = 1;
        const manualCorrectionRowInput = document.querySelector("#analyzing-manual-correction-row-input");
        manualCorrectionRowInput.value = 1;
      }
    );

    document.querySelector("#analyzing-set-arena-button").addEventListener(
      "click",
      (event) => {
        document.querySelector("#analyzing-manual-correction-arena-details").style.display = "grid";
        document.querySelector("#analyzing-manual-correction-keypoint-details").style.display = "none";

        document.querySelector("#analyzing-set-arena-button").classList.add("button-group-button-activated");
        document.querySelector("#analyzing-set-keypoints-button").classList.remove("button-group-button-activated");
      }
    );

    document.querySelector("#analyzing-set-keypoints-button").addEventListener(
      "click",
      (event) => {
        const firstFrameInput = document.querySelector("#analyzing-first-frame-input");
        const firstFrame = firstFrameInput.value;

        const keypointCount = this.cachedResults[firstFrame][0].length;

        const column = document.querySelector("#analyzing-manual-correction-column-input").value - 1;
        const row = document.querySelector("#analyzing-manual-correction-row-input").value - 1;

        this.manuallyCorrecting = true;
        this.manuallyCorrectingRow = row;
        this.manuallyCorrectingColumn = column;
        this.manualCorrection = [];
        this.manualCorrectionIndex = 0;
        this.keypointCount = keypointCount;

        for (let i = 0; i < keypointCount; ++i) {
          document.querySelector(`#analyzing-keypoint-${i}-x-input`).value = "";
          document.querySelector(`#analyzing-keypoint-${i}-y-input`).value = "";
          this.manualCorrection.push({ x: NaN, y: NaN });
        }

        this.draw();

        document.querySelector("#analyzing-manual-correction-arena-details").style.display = "none";
        document.querySelector("#analyzing-manual-correction-keypoint-details").style.display = "grid";

        document.querySelector("#analyzing-set-arena-button").classList.remove("button-group-button-activated");
        document.querySelector("#analyzing-set-keypoints-button").classList.add("button-group-button-activated");
      }
    );

    document.querySelector("#analyzing-cancel-manual-correction-button").addEventListener(
      "click",
      (event) => {
        document.querySelector("#analyzing-start-manual-correction-button").style.display = "block";
        document.querySelector("#analyzing-manual-correction-button-group").style.display = "none";
        document.querySelector("#analyzing-manual-correction-arena-details").style.display = "none";
        document.querySelector("#analyzing-manual-correction-keypoint-details").style.display = "none";
        document.querySelector("#analyzing-manual-correction-buttons").style.display = "none";

        this.manuallyCorrecting = false;
        this.manuallyCorrectingRow = null;
        this.manuallyCorrectingColumn = null;
        this.manualCorrection = null;
        this.manualCorrectionIndex = null;
      }
    );

    document.querySelector("#analyzing-done-manual-correction-button").addEventListener(
      "click",
      (event) => {
        document.querySelector("#analyzing-start-manual-correction-button").style.display = "block";
        document.querySelector("#analyzing-manual-correction-button-group").style.display = "none";
        document.querySelector("#analyzing-manual-correction-arena-details").style.display = "none";
        document.querySelector("#analyzing-manual-correction-keypoint-details").style.display = "none";
        document.querySelector("#analyzing-manual-correction-buttons").style.display = "none";

        const arenaColumnsInput = document.querySelector("#analyzing-arena-columns-input");
        const arenaColumns = +arenaColumnsInput.value;

        this.worker.postMessage(
          {
            type: "manualCorrection",
            frameIndex: this.currentFrameIndex,
            row: this.manuallyCorrectingRow,
            arenacolumns: arenaColumns,
            column: this.manuallyCorrectingColumn,
            coordinates: this.manualCorrection
          }
        );
        this.cachedResults[this.currentFrameIndex][this.manuallyCorrectingRow * arenaColumns + this.manuallyCorrectingColumn] = this.manualCorrection;
        this.updateTrajectoryPlots();

        this.manuallyCorrecting = false;
        this.manuallyCorrectingRow = null;
        this.manuallyCorrectingColumn = null;
        this.manualCorrection = null;
        this.manualCorrectionIndex = null;
      }
    );

    const manualCorrectionColumnInput = document.querySelector("#analyzing-manual-correction-column-input");
    manualCorrectionColumnInput.addEventListener(
      "change",
      (event) => {
      }
    );
    const manualCorrectionRowInput = document.querySelector("#analyzing-manual-correction-row-input");
    manualCorrectionRowInput.addEventListener(
      "change",
      (event) => {
      }
    );

    //
    //
    //

    document.querySelector("#analyzing-calibration-button").addEventListener(
      "click",
      (event) => {
        if (document.querySelector("#analyzing-calibration-button").classList.contains("activated")) {
          document.querySelector("#analyzing-calibration-button-divider").style.display = "flex";
          document.querySelector("#analyzing-analysis-button-divider").style.display = "flex";
          document.querySelector("#analyzing-kinematics-button-divider").style.display = "flex";

          document.querySelector("#analyzing-calibration-button").style.display = "flex";
          document.querySelector("#analyzing-analysis-button").style.display = "flex";
          document.querySelector("#analyzing-kinematics-button").style.display = "flex";

          document.querySelector("#analyzing-calibration-button").classList.remove("activated");
          document.querySelector("#analyzing-calibration-button").classList.remove("panel-button-bottom");

          document.querySelector("#analyzing-calibration-inner-section-left-side-1").style.display = "none";
          document.querySelector("#analyzing-calibration-inner-section-left-side-2").style.display = "none";
          document.querySelector("#analyzing-calibration-inner-section-right-side").style.display = "none";
        }
        else {
          document.querySelector("#analyzing-analysis-button-divider").style.display = "none";
          document.querySelector("#analyzing-kinematics-button-divider").style.display = "none";

          document.querySelector("#analyzing-analysis-button").style.display = "none";
          document.querySelector("#analyzing-kinematics-button").style.display = "none";

          document.querySelector("#analyzing-calibration-button").classList.add("activated");
          document.querySelector("#analyzing-calibration-button").classList.add("panel-button-bottom");

          document.querySelector("#analyzing-calibration-inner-section-left-side-1").style.display = "flex";
          document.querySelector("#analyzing-calibration-inner-section-left-side-2").style.display = "flex";
          document.querySelector("#analyzing-calibration-inner-section-right-side").style.display = "flex";
        }

        window.scrollTo(0, 0);
      }
    );

    document.querySelector("#analyzing-analysis-button").addEventListener(
      "click",
      (event) => {
        if (document.querySelector("#analyzing-analysis-button").classList.contains("activated")) {
          document.querySelector("#analyzing-calibration-button-divider").style.display = "flex";
          document.querySelector("#analyzing-analysis-button-divider").style.display = "flex";
          document.querySelector("#analyzing-kinematics-button-divider").style.display = "flex";

          document.querySelector("#analyzing-calibration-button").style.display = "flex";
          document.querySelector("#analyzing-analysis-button").style.display = "flex";
          document.querySelector("#analyzing-kinematics-button").style.display = "flex";

          document.querySelector("#analyzing-analysis-button").classList.remove("activated");
          document.querySelector("#analyzing-analysis-button").classList.remove("panel-button-bottom");

          document.querySelector("#analyzing-analysis-inner-section-left-side-1").style.display = "none";
          document.querySelector("#analyzing-analysis-inner-section-left-side-2").style.display = "none";
          document.querySelector("#analyzing-analysis-inner-section-left-side-3").style.display = "none";
          document.querySelector("#analyzing-analysis-inner-section-left-side-4").style.display = "none";
          document.querySelector("#analyzing-analysis-inner-section-left-side-5").style.display = "none";
          document.querySelector("#analyzing-analysis-inner-section-right-side").style.display = "none";
        }
        else {
          document.querySelector("#analyzing-calibration-button-divider").style.display = "none";
          document.querySelector("#analyzing-kinematics-button-divider").style.display = "none";

          document.querySelector("#analyzing-calibration-button").style.display = "none";
          document.querySelector("#analyzing-kinematics-button").style.display = "none";

          document.querySelector("#analyzing-analysis-button").classList.add("activated");
          document.querySelector("#analyzing-analysis-button").classList.add("panel-button-bottom");

          document.querySelector("#analyzing-analysis-inner-section-left-side-1").style.display = "flex";
          document.querySelector("#analyzing-analysis-inner-section-left-side-2").style.display = "flex";
          document.querySelector("#analyzing-analysis-inner-section-left-side-3").style.display = "flex";
          document.querySelector("#analyzing-analysis-inner-section-left-side-4").style.display = "flex";
          document.querySelector("#analyzing-analysis-inner-section-left-side-5").style.display = "flex";
          document.querySelector("#analyzing-analysis-inner-section-right-side").style.display = "flex";
        }

        window.scrollTo(0, 0);
      }
    );

    document.querySelector("#analyzing-kinematics-button").addEventListener(
      "click",
      (event) => {
        if (document.querySelector("#analyzing-kinematics-button").classList.contains("activated")) {
          document.querySelector("#analyzing-calibration-button-divider").style.display = "flex";
          document.querySelector("#analyzing-analysis-button-divider").style.display = "flex";
          document.querySelector("#analyzing-kinematics-button-divider").style.display = "flex";

          document.querySelector("#analyzing-calibration-button").style.display = "flex";
          document.querySelector("#analyzing-analysis-button").style.display = "flex";
          document.querySelector("#analyzing-kinematics-button").style.display = "flex";

          document.querySelector("#analyzing-kinematics-button").classList.remove("activated");
          document.querySelector("#analyzing-kinematics-button").classList.remove("panel-button-bottom");

          document.querySelector("#analyzing-kinematics-inner-section-left-side-1").style.display = "none";
          document.querySelector("#analyzing-kinematics-inner-section-left-side-2").style.display = "none";
          document.querySelector("#analyzing-kinematics-inner-section-right-side").style.display = "none";
        }
        else {
          document.querySelector("#analyzing-calibration-button-divider").style.display = "none";
          document.querySelector("#analyzing-analysis-button-divider").style.display = "none";

          document.querySelector("#analyzing-calibration-button").style.display = "none";
          document.querySelector("#analyzing-analysis-button").style.display = "none";

          document.querySelector("#analyzing-kinematics-button").classList.add("activated");
          document.querySelector("#analyzing-kinematics-button").classList.add("panel-button-bottom");

          document.querySelector("#analyzing-kinematics-inner-section-left-side-1").style.display = "none";
          document.querySelector("#analyzing-kinematics-inner-section-left-side-2").style.display = "flex";
          document.querySelector("#analyzing-kinematics-inner-section-right-side").style.display = "flex";
        }

        window.scrollTo(0, 0);
      }
    );

    for (const input of document.querySelectorAll("input[name=arena-shape]")) {
      input.addEventListener(
        "change",
        (event) => {
          const value = document.querySelector("input[name=arena-shape]:checked").value;
          this.worker.postMessage({ type: "arenaShape", arenaShape: value });

          if (value === "circle") {
            document.querySelector("#analyzing-arena-diameter-input").style.display = "block";
            document.querySelector("#analyzing-arena-length-input").style.display = "none";
            document.querySelector("#analyzing-arena-width-input").style.display = "none";
            document.querySelector("#analyzing-arena-height-input").style.display = "none";

            document.querySelector("#analyzing-arena-diameter-input-label").style.display = "block";
            document.querySelector("#analyzing-arena-length-input-label").style.display = "none";
            document.querySelector("#analyzing-arena-width-input-label").style.display = "none";
            document.querySelector("#analyzing-arena-height-input-label").style.display = "none";

            this.onArenaGridChanged();
            this.draw();
          }
          else if (value === "square") {
            document.querySelector("#analyzing-arena-diameter-input").style.display = "none";
            document.querySelector("#analyzing-arena-length-input").style.display = "block";
            document.querySelector("#analyzing-arena-width-input").style.display = "none";
            document.querySelector("#analyzing-arena-height-input").style.display = "none";

            document.querySelector("#analyzing-arena-diameter-input-label").style.display = "none";
            document.querySelector("#analyzing-arena-length-input-label").style.display = "block";
            document.querySelector("#analyzing-arena-width-input-label").style.display = "none";
            document.querySelector("#analyzing-arena-height-input-label").style.display = "none";

            this.onArenaGridChanged();
            this.draw();
          }
          else if (value === "rectangle") {
            document.querySelector("#analyzing-arena-diameter-input").style.display = "none";
            document.querySelector("#analyzing-arena-length-input").style.display = "none";
            document.querySelector("#analyzing-arena-width-input").style.display = "block";
            document.querySelector("#analyzing-arena-height-input").style.display = "block";

            document.querySelector("#analyzing-arena-diameter-input-label").style.display = "none";
            document.querySelector("#analyzing-arena-length-input-label").style.display = "none";
            document.querySelector("#analyzing-arena-width-input-label").style.display = "block";
            document.querySelector("#analyzing-arena-height-input-label").style.display = "block";

            this.onArenaGridChanged();
            this.draw();
          }
        }
      );
    }
    {
      const value = document.querySelector("input[name=arena-shape]:checked").value;
      this.worker.postMessage({ type: "arenaShape", arenaShape: value });

      if (value === "circle") {
        document.querySelector("#analyzing-arena-diameter-input").style.display = "block";
        document.querySelector("#analyzing-arena-length-input").style.display = "none";
        document.querySelector("#analyzing-arena-width-input").style.display = "none";
        document.querySelector("#analyzing-arena-height-input").style.display = "none";

        document.querySelector("#analyzing-arena-diameter-input-label").style.display = "block";
        document.querySelector("#analyzing-arena-length-input-label").style.display = "none";
        document.querySelector("#analyzing-arena-width-input-label").style.display = "none";
        document.querySelector("#analyzing-arena-height-input-label").style.display = "none";
      }
      else if (value === "square") {
        document.querySelector("#analyzing-arena-diameter-input").style.display = "none";
        document.querySelector("#analyzing-arena-length-input").style.display = "block";
        document.querySelector("#analyzing-arena-width-input").style.display = "none";
        document.querySelector("#analyzing-arena-height-input").style.display = "none";

        document.querySelector("#analyzing-arena-diameter-input-label").style.display = "none";
        document.querySelector("#analyzing-arena-length-input-label").style.display = "block";
        document.querySelector("#analyzing-arena-width-input-label").style.display = "none";
        document.querySelector("#analyzing-arena-height-input-label").style.display = "none";
      }
      else if (value === "rectangle") {
        document.querySelector("#analyzing-arena-diameter-input").style.display = "none";
        document.querySelector("#analyzing-arena-length-input").style.display = "none";
        document.querySelector("#analyzing-arena-width-input").style.display = "block";
        document.querySelector("#analyzing-arena-height-input").style.display = "block";

        document.querySelector("#analyzing-arena-diameter-input-label").style.display = "none";
        document.querySelector("#analyzing-arena-length-input-label").style.display = "none";
        document.querySelector("#analyzing-arena-width-input-label").style.display = "block";
        document.querySelector("#analyzing-arena-height-input-label").style.display = "block";
      }
    }

    const arenaXInput = document.querySelector("#analyzing-arena-x-input");
    arenaXInput.addEventListener(
      "change",
      (event) => { this.onArenaGridChanged(); }
    );
    const arenaYInput = document.querySelector("#analyzing-arena-y-input");
    arenaYInput.addEventListener(
      "change",
      (event) => { this.onArenaGridChanged(); }
    );
    const arenaDiameterInput = document.querySelector("#analyzing-arena-diameter-input");
    arenaDiameterInput.addEventListener(
      "change",
      (event) => { this.onArenaGridChanged(); }
    );
    const arenaLengthInput = document.querySelector("#analyzing-arena-length-input");
    arenaLengthInput.addEventListener(
      "change",
      (event) => { this.onArenaGridChanged(); }
    );
    const arenaWidthInput = document.querySelector("#analyzing-arena-width-input");
    arenaWidthInput.addEventListener(
      "change",
      (event) => { this.onArenaGridChanged(); }
    );
    const arenaHeightInput = document.querySelector("#analyzing-arena-height-input");
    arenaHeightInput.addEventListener(
      "change",
      (event) => { this.onArenaGridChanged(); }
    );
    const arenaRowsInput = document.querySelector("#analyzing-arena-rows-input");
    arenaRowsInput.addEventListener(
      "change",
      (event) => { this.onArenaGridChanged(); }
    );
    const arenaColumnsInput = document.querySelector("#analyzing-arena-columns-input");
    arenaColumnsInput.addEventListener(
      "change",
      (event) => { this.onArenaGridChanged(); }
    );
    const arenaSpacingInput = document.querySelector("#analyzing-arena-spacing-input");
    arenaSpacingInput.addEventListener(
      "change",
      (event) => { this.onArenaGridChanged(); }
    );

    const firstFrameInput = document.querySelector("#analyzing-first-frame-input");
    firstFrameInput.addEventListener(
      "change",
      (event) => { this.onFrameSelectionChanged(); }
    );
    const lastFrameInput = document.querySelector("#analyzing-last-frame-input");
    lastFrameInput.addEventListener(
      "change",
      (event) => { this.onFrameSelectionChanged(); }
    );

    this.reset();
  }


  reset() {
    super.reset();

    document.querySelector("#analyzing-calibration-inner-section-left-side-1").style.display = "none";
    document.querySelector("#analyzing-calibration-inner-section-left-side-2").style.display = "none";
    document.querySelector("#analyzing-analysis-inner-section-left-side-1").style.display = "none";
    document.querySelector("#analyzing-analysis-inner-section-left-side-2").style.display = "none";
    document.querySelector("#analyzing-analysis-inner-section-left-side-3").style.display = "none";
    document.querySelector("#analyzing-analysis-inner-section-left-side-4").style.display = "none";
    document.querySelector("#analyzing-analysis-inner-section-left-side-5").style.display = "none";
    document.querySelector("#analyzing-kinematics-inner-section-left-side-1").style.display = "none";
    document.querySelector("#analyzing-kinematics-inner-section-left-side-2").style.display = "none";

    document.querySelector("#analyzing-calibration-inner-section-right-side").style.display = "none";
    document.querySelector("#analyzing-analysis-inner-section-right-side").style.display = "none";
    document.querySelector("#analyzing-kinematics-inner-section-right-side").style.display = "none";

    document.querySelector("#analyzing-manual-correction-button-group").style.display = "none";
    document.querySelector("#analyzing-manual-correction-arena-details").style.display = "none";
    document.querySelector("#analyzing-manual-correction-keypoint-details").style.display = "none";
    document.querySelector("#analyzing-manual-correction-buttons").style.display = "none";
  }


  enter() {
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


  async maybeLoadMovie() {
    let fileHandle = null;
    try {
      [fileHandle] = await window.showOpenFilePicker(
        {
          id: "openAnalyzingMovie",
          startIn: "documents",
          types: [
            {
            }
          ],
          mode: "read"
        }
      );
    }
    catch {
    }

    if (fileHandle) {
      this.worker.postMessage({ type: "maybeLoadMovie", fileHandle: fileHandle });
    }
  }

  onLoadMovieSuccess(filename, numFrames, frameWidth, frameHeight) {
    document.querySelector("#analyzing-load-movie-button-label").textContent = filename;

    this.numFrames = numFrames;
    this.frameWidth = frameWidth;
    this.frameHeight = frameHeight;

    this.currentFrame = null;
    this.currentFrameIndex = null;

    this.zoom = 1;
    this.panOffsetX = 0;
    this.panOffsetY = 0;

    this.currentlyPlaying = false;

    const rangeInput = document.querySelector("#analyzing-frame-range-input");
    rangeInput.min = 1;
    rangeInput.value = 1;
    rangeInput.max = this.numFrames;

    const numberInput = document.querySelector("#analyzing-frame-number-input");
    numberInput.min = 1;
    numberInput.value = 1;
    numberInput.max = this.numFrames;

    this.worker.postMessage({ type: "frameRequest", index: 0 });

    const firstFrameInput = document.querySelector("#analyzing-first-frame-input");
    firstFrameInput.min = 1;
    firstFrameInput.value = 1;
    firstFrameInput.max = this.numFrames;

    const lastFrameInput = document.querySelector("#analyzing-last-frame-input");
    lastFrameInput.min = 1;
    lastFrameInput.value = this.numFrames;
    lastFrameInput.max = this.numFrames;

    const arenaXInput = document.querySelector("#analyzing-arena-x-input");
    arenaXInput.min = 0;
    arenaXInput.value = 0;
    arenaXInput.max = this.frameWidth;

    const arenaYInput = document.querySelector("#analyzing-arena-y-input");
    arenaYInput.min = 0;
    arenaYInput.value = 0;
    arenaYInput.max = this.frameHeight;

    const arenaDiameterInput = document.querySelector("#analyzing-arena-diameter-input");
    arenaDiameterInput.min = 0;
    arenaDiameterInput.value = Math.min(this.frameWidth, this.frameHeight);
    arenaDiameterInput.max = Math.min(this.frameWidth, this.frameHeight);

    const arenaLengthInput = document.querySelector("#analyzing-arena-length-input");
    arenaLengthInput.min = 0;
    arenaLengthInput.value = Math.min(this.frameWidth, this.frameHeight);
    arenaLengthInput.max = Math.min(this.frameWidth, this.frameHeight);

    const arenaWidthInput = document.querySelector("#analyzing-arena-width-input");
    arenaWidthInput.min = 0;
    arenaWidthInput.value = this.frameWidth;
    arenaWidthInput.max = this.frameWidth;

    const arenaHeightInput = document.querySelector("#analyzing-arena-height-input");
    arenaHeightInput.min = 0;
    arenaHeightInput.value = this.frameHeight;
    arenaHeightInput.max = this.frameHeight;

    const arenaRowsInput = document.querySelector("#analyzing-arena-rows-input");
    arenaRowsInput.min = 1;
    arenaRowsInput.value = 1;
    arenaRowsInput.max = 1000;

    const arenaColumnsInput = document.querySelector("#analyzing-arena-columns-input");
    arenaColumnsInput.min = 1;
    arenaColumnsInput.value = 1;
    arenaColumnsInput.max = 1000;

    const arenaSpacingInput = document.querySelector("#analyzing-arena-spacing-input");
    arenaSpacingInput.min = 0;
    arenaSpacingInput.value = 0;
    arenaSpacingInput.max = 1000;

    this.onArenaGridChanged();
  }

  onLoadMovieFailure(filename) {
    document.querySelector("#generic-dialog-heading").textContent = "Error opening movie file";
    document.querySelector("#generic-dialog-blurb").textContent = `Couldn't open \"${filename}.\" It might be in a movie file format that Marigold doesn't recognize.`;

    document.querySelector("#generic-dialog-close-button").addEventListener(
      "click",
      (event) => { document.querySelector("#generic-dialog").close(); },
      { once: true }
    );

    document.querySelector("#generic-dialog").showModal();
  }


  draw() {
    if (this.currentFrame) {
      const canvas = document.querySelector("#analyzing-movie-player-canvas");
      const context = canvas.getContext("2d");

      const boundingRect = canvas.getBoundingClientRect();

      canvas.width = this.frameWidth;
      canvas.height = this.frameHeight;

      context.save();

      const scale = Math.min((boundingRect.width / boundingRect.height) * (canvas.height / canvas.width), (boundingRect.height / boundingRect.width) * (canvas.width / canvas.height));

      const panCenterX = canvas.width / 2 + this.panOffsetX;
      const panCenterY = canvas.height / 2 + this.panOffsetY;

      context.setTransform(
        scale * this.zoom,
        0,
        0,
        scale * this.zoom,
        (canvas.width / 2) - (panCenterX * scale * this.zoom),
        (canvas.height / 2) - (panCenterY * scale * this.zoom)
      );

      context.drawImage(this.currentFrame, 0, 0, this.frameWidth, this.frameHeight, 0, 0, canvas.width, canvas.height);

      if (this.arenas) {
        const arenaRowsInput = document.querySelector("#analyzing-arena-rows-input");
        const arenaRows = arenaRowsInput.value;

        const arenaColumnsInput = document.querySelector("#analyzing-arena-columns-input");
        const arenaColumns = arenaColumnsInput.value;

        for (let row = 0; row < arenaRows; ++row) {
          for (let column = 0; column < arenaColumns; ++column) {
            const arena = this.arenas[row * arenaColumns + column];
            context.lineWidth = 5;
            context.strokeStyle = "hsl(15, 50%, 50%, 50%)";
            context.beginPath();
            if (arena.shape === "circle") {
              context.arc(arena.x + arena.width / 2, arena.y + arena.width / 2, arena.width / 2, 0, 2 * Math.PI, false);
            }
            else {
              context.rect(arena.x, arena.y, arena.width, arena.height);
            }
            context.stroke();
          }
        }
      }

      if (this.cachedResults && this.cachedResults[this.currentFrameIndex] && this.currentFrameIndex <= this.cachedResults.length - 1) {
        const arenaRowsInput = document.querySelector("#analyzing-arena-rows-input");
        const arenaRows = arenaRowsInput.value;

        const arenaColumnsInput = document.querySelector("#analyzing-arena-columns-input");
        const arenaColumns = arenaColumnsInput.value;

        const frameResult = this.cachedResults[this.currentFrameIndex];
        for (let row = 0; row < arenaRows; ++row) {
          for (let column = 0; column < arenaColumns; ++column) {
            const frameArenaResult = frameResult[row * arenaColumns + column];
            const colormap = interpolateColormap("plasma", frameArenaResult.length, true);

            let keypoints = frameArenaResult;
            if (this.manuallyCorrecting && this.manuallyCorrectingRow === row && this.manuallyCorrectingColumn === column) {
              keypoints = this.manualCorrection;
            }

            for (let i = 0; i < keypoints.length; ++i) {
              const radius = 2.5 * Math.min(this.frameWidth, this.frameHeight) / 512;
              context.fillStyle = `rgb(${colormap[i][0] * 255}, ${colormap[i][1] * 255}, ${colormap[i][2] * 255}, 0.75)`;
              context.beginPath();
              context.arc(keypoints[i].x, keypoints[i].y, radius, 0, 2 * Math.PI, false);
              context.fill();
            }
          }
        }
      }

      context.restore();
    }
  }

  onFrameReady(frame, frameNumber) {
    this.currentFrame = frame;
    this.currentFrameIndex = frameNumber;

    const rangeInput = document.querySelector("#analyzing-frame-range-input");
    rangeInput.value = this.currentFrameIndex + 1;

    const numberInput = document.querySelector("#analyzing-frame-number-input");
    numberInput.value = this.currentFrameIndex + 1;

    this.draw();

    if (this.currentlyPlaying) {
      if (this.currentFrameIndex === this.numFrames - 1) {
        this.currentlyPlaying = false;
        document.querySelector("#analyzing-play-or-pause-button").classList.remove("activated");
      }
      else {
        this.worker.postMessage({ type: "frameRequest", index: this.currentFrameIndex + 1 });
      }
    }
  }


  onCanvasScroll(event) {
    if (event.ctrlKey) {
      this.zoom *= 1.0 + 0.001 * event.deltaY;
    }
    else {
      this.panOffsetX += 0.05 * event.deltaX;
      this.panOffsetY += 0.05 * event.deltaY;
    }
    this.draw();
    event.preventDefault();
  }

  onZoomInButtonClick() {
    this.zoom *= 1.25;
    this.draw();
  }

  onZoomOutButtonClick() {
    this.zoom /= 1.25;
    this.draw();
  }

  onZoomResetButtonClick() {
    this.zoom = 1;
    this.panOffsetX = 0;
    this.panOffsetY = 0;
    this.draw();
  }


  async maybeLoadModel() {
    let fileHandle = null;
    try {
      [fileHandle] = await window.showOpenFilePicker(
        {
          id: "openAnalyzingModel",
          startIn: "documents",
          types: [
            {
            }
          ],
          mode: "read"
        }
      );
    }
    catch {
    }

    if (fileHandle) {
      this.worker.postMessage({ type: "maybeLoadModel", fileHandle: fileHandle });
    }
  }

  onLoadModelSuccess(filename) {
    document.querySelector("#analyzing-load-model-button-label").textContent = filename;
  }

  onLoadModelFailure(filename) {
  }


  onArenaGridChanged() {
    const arenaXInput = document.querySelector("#analyzing-arena-x-input");
    const arenaX = +arenaXInput.value;

    const arenaYInput = document.querySelector("#analyzing-arena-y-input");
    const arenaY = +arenaYInput.value;

    const arenaDiameterInput = document.querySelector("#analyzing-arena-diameter-input");
    const arenaLengthInput = document.querySelector("#analyzing-arena-length-input");
    const arenaWidthInput = document.querySelector("#analyzing-arena-width-input");
    const arenaHeightInput = document.querySelector("#analyzing-arena-height-input");
    let arenaWidth = null;
    let arenaHeight = null;
    const arenaShape = document.querySelector("input[name=arena-shape]:checked").value;
    if (arenaShape === "circle") {
      arenaWidth = +arenaDiameterInput.value;
      arenaHeight = +arenaDiameterInput.value;
    }
    else if (arenaShape === "square") {
      arenaWidth = +arenaLengthInput.value;
      arenaHeight = +arenaLengthInput.value;
    }
    else if (arenaShape === "rectangle") {
      arenaWidth = +arenaWidthInput.value;
      arenaHeight = +arenaHeightInput.value;
    }

    const arenaRowsInput = document.querySelector("#analyzing-arena-rows-input");
    const arenaRows = +arenaRowsInput.value;

    const arenaColumnsInput = document.querySelector("#analyzing-arena-columns-input");
    const arenaColumns = +arenaColumnsInput.value;

    const arenaSpacingInput = document.querySelector("#analyzing-arena-spacing-input");
    const arenaSpacing = +arenaSpacingInput.value;

    this.arenas = [];
    let currentY = arenaY;
    for (let row = 0; row < arenaRows; ++row) {
      let currentX = arenaX;
      for (let column = 0; column < arenaColumns; ++column) {
        const arena = {
          x: currentX,
          y: currentY,
          width: arenaWidth,
          height: arenaHeight,
          shape: arenaShape
        };
        this.arenas.push(arena);

        currentX += arenaWidth + arenaSpacing;
      }
      currentY += arenaHeight + arenaSpacing;
    }

    this.draw();

    this.worker.postMessage({ type: "arenas", arenas: this.arenas });
  }

  onFrameSelectionChanged() {
    const firstFrameInput = document.querySelector("#analyzing-first-frame-input");
    const firstFrame = +firstFrameInput.value - 1;

    const lastFrameInput = document.querySelector("#analyzing-last-frame-input");
    const lastFrame = +lastFrameInput.value - 1;

    this.worker.postMessage({ type: "frameSelection", firstFrame: firstFrame, lastFrame: lastFrame });
  }


  onResultsReady(stuff, frame, frameNumber) {
    this.cachedResults = stuff;

    if (frame) {
      this.onFrameReady(frame, frameNumber);
    }

    const firstFrameInput = document.querySelector("#analyzing-first-frame-input");
    const firstFrame = +firstFrameInput.value - 1;

    const lastFrameInput = document.querySelector("#analyzing-last-frame-input");
    const lastFrame = +lastFrameInput.value - 1;

    if (this.cachedResults.length < this.numFrames) {
      this.worker.postMessage({ type: "processFrame", index: this.cachedResults.length });
    }
    else {
      document.querySelector("#analyzing-start-analysis-button").removeAttribute("disabled");
    }

    if (frameNumber >= firstFrame && frameNumber <= lastFrame) {
      const progressElement = document.querySelector("#analysis-progress");
      const value = (frameNumber - firstFrame) / (lastFrame - firstFrame);
      progressElement.value = value;
    }

    if (frameNumber === lastFrame) {
      const arenaRowsInput = document.querySelector("#analyzing-arena-rows-input");
      const arenaColumnsInput = document.querySelector("#analyzing-arena-columns-input");
      const arenaRows = arenaRowsInput.value;
      const arenaColumns = arenaColumnsInput.value;
      const keypointCount = this.cachedResults[firstFrame][0].length;

      document.querySelector("#analyzing-trajectory-plot-preview-keypoint-input").value = 1;

      document.querySelector("#analyzing-trajectory-plot-preview-keypoint-input").max = keypointCount;

      this.updateTrajectoryPlots();
    }

    const keypointCount = this.cachedResults[firstFrame][0].length;
    for (let i = 0; i < keypointCount; ++i) {
      document.querySelector(`#analyzing-keypoint-${i}-x-input-label`).style.display = "block";
      document.querySelector(`#analyzing-keypoint-${i}-x-input`).style.display = "block";
      document.querySelector(`#analyzing-keypoint-${i}-y-input-label`).style.display = "block";
      document.querySelector(`#analyzing-keypoint-${i}-y-input`).style.display = "block";
    }
    for (let i = keypointCount; i < maxKeypointCount; ++i) {
      document.querySelector(`#analyzing-keypoint-${i}-x-input-label`).style.display = "none";
      document.querySelector(`#analyzing-keypoint-${i}-x-input`).style.display = "none";
      document.querySelector(`#analyzing-keypoint-${i}-y-input-label`).style.display = "none";
      document.querySelector(`#analyzing-keypoint-${i}-y-input`).style.display = "none";
    }

    const arenaRowsInput = document.querySelector("#analyzing-arena-rows-input");
    const arenaColumnsInput = document.querySelector("#analyzing-arena-columns-input");
    const arenaRows = arenaRowsInput.value;
    const arenaColumns = arenaColumnsInput.value;

    const manualCorrectionColumnInput = document.querySelector("#analyzing-manual-correction-column-input");
    manualCorrectionColumnInput.max = arenaColumns;
    const manualCorrectionRowInput = document.querySelector("#analyzing-manual-correction-row-input");
    manualCorrectionRowInput.max = arenaRows;

    document.querySelector("#analyzing-angle-a").max = keypointCount;
    document.querySelector("#analyzing-angle-b").max = keypointCount;
    document.querySelector("#analyzing-angle-c").max = keypointCount;
  }

  updateTrajectoryPlots() {
    const firstFrameInput = document.querySelector("#analyzing-first-frame-input");
    const firstFrame = +firstFrameInput.value - 1;

    const lastFrameInput = document.querySelector("#analyzing-last-frame-input");
    const lastFrame = +lastFrameInput.value - 1;

    const arenaRowsInput = document.querySelector("#analyzing-arena-rows-input");
    const arenaColumnsInput = document.querySelector("#analyzing-arena-columns-input");
    const arenaRows = arenaRowsInput.value;
    const arenaColumns = arenaColumnsInput.value;
    const keypointCount = this.cachedResults[firstFrame][0].length;

    const currentColumn = null;
    const currentRow = null;
    const currentKeypoint = document.querySelector("#analyzing-trajectory-plot-preview-keypoint-input").value - 1;

    const arenaShape = document.querySelector("input[name=arena-shape]:checked").value;

    updatePlotData(this.cachedResults.slice(firstFrame, lastFrame + 1), this.arenas, arenaRows, arenaColumns, keypointCount, currentRow, currentColumn, currentKeypoint, firstFrame, lastFrame, arenaShape);
  }

  prepareForExport() {
    const svgFilenames = [];
    const svgFileContents = [];

    const svgElements = document.querySelectorAll(".trajectory-plot-hidden");
    for (const svgElement of svgElements) {
      svgFilenames.push(`${svgElement.id}.svg`);

      const serializer = new XMLSerializer();
      const string = serializer.serializeToString(svgElement);
      string.replace(`<svg`, `<svg xmlns="http://www.w3.org/2000/svg"`);
      svgFileContents.push(string);
    }

    const firstFrameInput = document.querySelector("#analyzing-first-frame-input");
    const firstFrame = +firstFrameInput.value - 1;

    const lastFrameInput = document.querySelector("#analyzing-last-frame-input");
    const lastFrame = +lastFrameInput.value - 1;

    const arenaRowsInput = document.querySelector("#analyzing-arena-rows-input");
    const arenaColumnsInput = document.querySelector("#analyzing-arena-columns-input");
    const arenaRows = arenaRowsInput.value;
    const arenaColumns = arenaColumnsInput.value;
    const keypointCount = this.cachedResults[firstFrame][0].length;

    const currentColumn = null;
    const currentRow = null;
    const currentKeypoint = document.querySelector("#analyzing-trajectory-plot-preview-keypoint-input").value - 1;

    const angleKeypoint1 = document.querySelector("#analyzing-angle-a").value - 1;
    const angleKeypoint2 = document.querySelector("#analyzing-angle-b").value - 1;
    const angleKeypoint3 = document.querySelector("#analyzing-angle-c").value - 1;

    const { csvFilenames, csvFileContents, summaryFilename, summaryFileContents } = getCsvData(
      this.cachedResults.slice(firstFrame, lastFrame + 1), this.arenas, arenaRows, arenaColumns, keypointCount, currentRow, currentColumn, currentKeypoint, firstFrame, lastFrame, angleKeypoint1, angleKeypoint2, angleKeypoint3
    );

    this.worker.postMessage({ type: "exportResults", svgFilenames: svgFilenames, svgFileContents: svgFileContents, csvFilenames: csvFilenames, csvFileContents: csvFileContents, summaryFilename: summaryFilename, summaryFileContents: summaryFileContents });
  }

  onExportReady(blob) {
    const url = URL.createObjectURL(blob);
    const filename = "results.zip";

    const a = document.createElement("a");
    a.setAttribute("href", url);
    a.setAttribute("download", filename);
    a.click();

    URL.revokeObjectURL(url);
  }
}
