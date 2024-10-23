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
import { minKeypointCount, maxKeypointCount, defaultKeypointCount } from "../constants.js";
import { LabelingThumbnails } from "./labeling-thumbnails.js";


function beforeUnloadListener(event) {
  event.preventDefault();
  return (event.returnValue = "");
}


export class LabelingSection extends Section {
  keypointCount = 1;

  filename = null;

  numFrames = null;
  frameWidth = null;
  frameHeight = null;

  currentFrame = null;
  currentFrameIndex = null;

  zoom = 1;
  panOffsetX = 0;
  panOffsetY = 0;

  currentlyPlaying = false;

  currentlyLabeling = false;
  currentlyLabelingArena = false;
  currentlyLabelingKeypoints = false;

  arenaX = 0;
  arenaY = 0;
  arenaWidth = 0;
  arenaHeight = 0;

  currentLabels = [];
  currentLabelIndex = 0;

  mouseDown = false;

  labelingThumbnails = new LabelingThumbnails();

  constructor() {
    super(
      "dataset",
      "labeling"
    );

    this.worker.addEventListener(
      "message",
      (message) => {
        if (message.data.type === "loadMovieSuccess") {
          this.onLoadMovieSuccess(message.data.filename, message.data.numFrames, message.data.frameWidth, message.data.frameHeight);
        }
        else if (message.data.type === "loadMovieFailure") {
          this.onLoadMovieFailure(message.data.filename);
        }
        else if (message.data.type === "frameReady") {
          this.onFrameReady(message.data.frame, message.data.frameNumber);
        }
      }
    );

    document.querySelector("#labeling-thumbnails").addEventListener(
      "labelRemoved",
      (event) => {
        const index = event.detail;
        this.worker.postMessage({ type: "removeLabel", index: index });

        this.unsavedChanges = true;
        this.showStatus(Section.unsavedMessage);
        addEventListener("beforeunload", beforeUnloadListener);
      }
    );

    document.querySelector("#keypoint-count-input").addEventListener(
      "change",
      (event) => {
        let value = document.querySelector("#keypoint-count-input").value;
        this.worker.postMessage({ type: "keypointCount", keypointCount: +value });

        this.keypointCount = value;

        for (let i = 0; i < this.keypointCount; ++i) {
          document.querySelector(`#labeling-keypoint-${i}-x-input`).style.display = "block";
          document.querySelector(`#labeling-keypoint-${i}-y-input`).style.display = "block";

          document.querySelector(`#labeling-keypoint-${i}-x-input-label`).style.display = "block";
          document.querySelector(`#labeling-keypoint-${i}-y-input-label`).style.display = "block";
        }
        for (let i = this.keypointCount; i < maxKeypointCount; ++i) {
          document.querySelector(`#labeling-keypoint-${i}-x-input`).style.display = "none";
          document.querySelector(`#labeling-keypoint-${i}-y-input`).style.display = "none";

          document.querySelector(`#labeling-keypoint-${i}-x-input-label`).style.display = "none";
          document.querySelector(`#labeling-keypoint-${i}-y-input-label`).style.display = "none";
        }

        this.unsavedChanges = true;
        this.showStatus(Section.unsavedMessage);
        addEventListener("beforeunload", beforeUnloadListener);
      }
    );
    {
      let value = document.querySelector("#keypoint-count-input").value;
      this.worker.postMessage({ type: "keypointCount", keypointCount: +value });
    }

    document.querySelector("#labeling-load-movie-button").addEventListener(
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
    resizeObserver.observe(document.querySelector("#labeling-movie-player-canvas"));

    document.querySelector("#labeling-movie-player-canvas").addEventListener(
      "wheel",
      (event) => { this.onCanvasScroll(event); },
      { passive: false }
    );

    document.querySelector("#labeling-zoom-in-button").addEventListener("click", (event) => { this.onZoomInButtonClick(); }
    );
    document.querySelector("#labeling-zoom-out-button").addEventListener("click", (event) => { this.onZoomOutButtonClick(); }
    );
    document.querySelector("#labeling-zoom-reset-button").addEventListener("click", (event) => { this.onZoomResetButtonClick(); }
    );

    document.querySelector("#labeling-first-frame-button").addEventListener(
      "click",
      (event) => {
        this.worker.postMessage({ type: "frameRequest", index: 0 });
      }
    );
    document.querySelector("#labeling-previous-frame-button").addEventListener(
      "click",
      (event) => {
        this.worker.postMessage({ type: "frameRequest", index: Math.max(0, this.currentFrameIndex - 1) });
      }
    );
    document.querySelector("#labeling-play-or-pause-button").addEventListener(
      "click",
      (event) => {
        this.currentlyPlaying = !this.currentlyPlaying;
        if (this.currentlyPlaying) {
          document.querySelector("#labeling-play-or-pause-button").classList.add("activated");
          this.worker.postMessage({ type: "frameRequest", index: Math.min(this.numFrames - 1, this.currentFrameIndex + 1) });
        }
        else {
          document.querySelector("#labeling-play-or-pause-button").classList.remove("activated");
        }
      }
    );
    document.querySelector("#labeling-next-frame-button").addEventListener(
      "click",
      (event) => {
        this.worker.postMessage({ type: "frameRequest", index: Math.min(this.numFrames - 1, this.currentFrameIndex + 1) });
      }
    );
    document.querySelector("#labeling-last-frame-button").addEventListener(
      "click",
      (event) => {
        this.worker.postMessage({ type: "frameRequest", index: this.numFrames - 1 });
      }
    );

    document.querySelector("#labeling-frame-range-input").addEventListener(
      "change",
      (event) => {
        this.worker.postMessage({ type: "frameRequest", index: document.querySelector("#labeling-frame-range-input").value - 1 });
      }
    );
    document.querySelector("#labeling-frame-number-input").addEventListener(
      "change",
      (event) => {
        this.worker.postMessage({ type: "frameRequest", index: document.querySelector("#labeling-frame-number-input").value - 1 });
      }
    );

    document.querySelector("#labeling-label-current-frame-button").addEventListener(
      "click",
      (event) => {
        this.startLabeling();
      }
    );

    document.querySelector("#labeling-set-arena-button").addEventListener(
      "click",
      (event) => {
        this.setArena();
      }
    );

    document.querySelector("#labeling-set-keypoints-button").addEventListener(
      "click",
      (event) => {
        this.setKeypoints();
      }
    );

    document.querySelector("#labeling-cancel-labeling-button").addEventListener(
      "click",
      (event) => {
        this.cancelLabeling();
      }
    );

    document.querySelector("#labeling-done-labeling-button").addEventListener(
      "click",
      (event) => {
        this.doneLabeling();
      }
    );

    document.querySelector("#labeling-arena-x-input").addEventListener(
      "change",
      (event) => {
        this.arenaX = document.querySelector("#labeling-arena-x-input").value;
        this.draw();
      }
    );
    document.querySelector("#labeling-arena-y-input").addEventListener(
      "change",
      (event) => {
        this.arenaY = document.querySelector("#labeling-arena-y-input").value;
        this.draw();
      }
    );
    document.querySelector("#labeling-arena-width-input").addEventListener(
      "change",
      (event) => {
        this.arenaWidth = document.querySelector("#labeling-arena-width-input").value;
        this.draw();
      }
    );
    document.querySelector("#labeling-arena-height-input").addEventListener(
      "change",
      (event) => {
        this.arenaHeight = document.querySelector("#labeling-arena-height-input").value;
        this.draw();
      }
    );

    document.querySelector("#labeling-movie-player-canvas").addEventListener(
      "mousedown",
      (event) => {
        if (this.currentlyLabeling) {
          const canvas = document.querySelector("#labeling-movie-player-canvas");
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
          if (this.currentlyLabelingArena) {
            this.arenaX = x_;
            this.arenaY = y_;
            this.arenaWidth = 0;
            this.arenaHeight = 0;
            document.querySelector("#labeling-arena-x-input").value = this.arenaX;
            document.querySelector("#labeling-arena-y-input").value = this.arenaY;
            document.querySelector("#labeling-arena-width-input").value = this.arenaWidth;
            document.querySelector("#labeling-arena-height-input").value = this.arenaHeight;
          }
          else if (this.currentlyLabelingKeypoints) {
            this.currentLabels[this.currentLabelIndex].x = x_;
            this.currentLabels[this.currentLabelIndex].y = y_;
            document.querySelector(`#labeling-keypoint-${this.currentLabelIndex}-x-input`).value = x_;
            document.querySelector(`#labeling-keypoint-${this.currentLabelIndex}-y-input`).value = y_;
          }

          this.draw();
        }
      }
    );
    document.querySelector("#labeling-movie-player-canvas").addEventListener(
      "mousemove",
      (event) => {
        if (this.mouseDown) {
          if (this.currentlyLabeling) {
            const canvas = document.querySelector("#labeling-movie-player-canvas");
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

            if (this.currentlyLabelingArena) {
              this.arenaWidth = x_ - this.arenaX;
              this.arenaHeight = y_ - this.arenaY;
              document.querySelector("#labeling-arena-width-input").value = this.arenaWidth;
              document.querySelector("#labeling-arena-height-input").value = this.arenaHeight;
              this.draw();
            }
            else if (this.currentlyLabelingKeypoints) {
              this.currentLabels[this.currentLabelIndex].x = x_;
              this.currentLabels[this.currentLabelIndex].y = y_;
              document.querySelector(`#labeling-keypoint-${this.currentLabelIndex}-x-input`).value = x_;
              document.querySelector(`#labeling-keypoint-${this.currentLabelIndex}-y-input`).value = y_;
            }

            this.draw();
          }
        }
      }
    );
    document.querySelector("#labeling-movie-player-canvas").addEventListener(
      "mouseup",
      (event) => {
        if (this.currentlyLabeling) {
          const canvas = document.querySelector("#labeling-movie-player-canvas");
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
          if (this.currentlyLabelingArena) {
            this.arenaWidth = x_ - this.arenaX;
            this.arenaHeight = y_ - this.arenaY;
            document.querySelector("#labeling-arena-width-input").value = this.arenaWidth;
            document.querySelector("#labeling-arena-height-input").value = this.arenaHeight;
            this.draw();
          }
          else if (this.currentlyLabelingKeypoints) {
            this.currentLabels[this.currentLabelIndex].x = x_;
            this.currentLabels[this.currentLabelIndex].y = y_;
            document.querySelector(`#labeling-keypoint-${this.currentLabelIndex}-x-input`).value = x_;
            document.querySelector(`#labeling-keypoint-${this.currentLabelIndex}-y-input`).value = y_;
            this.currentLabelIndex = (this.currentLabelIndex + 1) % this.keypointCount;
          }

          this.draw();
        }
      }
    );

    document.querySelector("#labeling-configure-dataset-button").addEventListener(
      "click",
      (event) => {
        if (document.querySelector("#labeling-configure-dataset-button").classList.contains("activated")) {

          document.querySelector("#labeling-configure-dataset-button-divider").style.display = "flex";
          document.querySelector("#labeling-extract-and-label-frames-button-divider").style.display = "flex";
          document.querySelector("#labeling-review-labeled-frames-button-divider").style.display = "flex";

          document.querySelector("#labeling-configure-dataset-button").style.display = "flex";
          document.querySelector("#labeling-extract-and-label-frames-button").style.display = "flex";
          document.querySelector("#labeling-review-labeled-frames-button").style.display = "flex";

          document.querySelector("#labeling-configure-dataset-button").classList.remove("activated");
          document.querySelector("#labeling-configure-dataset-button").classList.remove("panel-button-bottom");

          document.querySelector("#labeling-configure-dataset-inner-section-left-side").style.display = "none";
        }
        else {
          document.querySelector("#labeling-extract-and-label-frames-button-divider").style.display = "none";
          document.querySelector("#labeling-review-labeled-frames-button-divider").style.display = "none";

          document.querySelector("#labeling-extract-and-label-frames-button").style.display = "none";
          document.querySelector("#labeling-review-labeled-frames-button").style.display = "none";

          document.querySelector("#labeling-configure-dataset-button").classList.add("activated");
          document.querySelector("#labeling-configure-dataset-button").classList.add("panel-button-bottom");

          document.querySelector("#labeling-configure-dataset-inner-section-left-side").style.display = "flex";
        }

        window.scrollTo(0, 0);
      }
    );

    document.querySelector("#labeling-extract-and-label-frames-button").addEventListener(
      "click",
      (event) => {
        if (document.querySelector("#labeling-extract-and-label-frames-button").classList.contains("activated")) {

          document.querySelector("#labeling-configure-dataset-button-divider").style.display = "flex";
          document.querySelector("#labeling-extract-and-label-frames-button-divider").style.display = "flex";
          document.querySelector("#labeling-review-labeled-frames-button-divider").style.display = "flex";

          document.querySelector("#labeling-configure-dataset-button").style.display = "flex";
          document.querySelector("#labeling-extract-and-label-frames-button").style.display = "flex";
          document.querySelector("#labeling-review-labeled-frames-button").style.display = "flex";

          document.querySelector("#labeling-extract-and-label-frames-button").classList.remove("activated");
          document.querySelector("#labeling-extract-and-label-frames-button").classList.remove("panel-button-bottom");

          document.querySelector("#labeling-extract-and-label-frames-inner-section-left-side").style.display = "none";
          document.querySelector("#labeling-extract-and-label-frames-inner-section-left-side-2").style.display = "none";
          document.querySelector("#labeling-extract-and-label-frames-inner-section-right-side").style.display = "none";
        }
        else {

          document.querySelector("#labeling-configure-dataset-button-divider").style.display = "none";
          document.querySelector("#labeling-review-labeled-frames-button-divider").style.display = "none";

          document.querySelector("#labeling-configure-dataset-button").style.display = "none";
          document.querySelector("#labeling-review-labeled-frames-button").style.display = "none";

          document.querySelector("#labeling-extract-and-label-frames-button").classList.add("activated");
          document.querySelector("#labeling-extract-and-label-frames-button").classList.add("panel-button-bottom");

          document.querySelector("#labeling-extract-and-label-frames-inner-section-left-side").style.display = "flex";
          document.querySelector("#labeling-extract-and-label-frames-inner-section-left-side-2").style.display = "flex";
          document.querySelector("#labeling-extract-and-label-frames-inner-section-right-side").style.display = "flex";
        }

        window.scrollTo(0, 0);
      }
    );

    document.querySelector("#labeling-review-labeled-frames-button").addEventListener(
      "click",
      (event) => {
        if (document.querySelector("#labeling-review-labeled-frames-button").classList.contains("activated")) {

          document.querySelector("#labeling-configure-dataset-button-divider").style.display = "flex";
          document.querySelector("#labeling-extract-and-label-frames-button-divider").style.display = "flex";
          document.querySelector("#labeling-review-labeled-frames-button-divider").style.display = "flex";

          document.querySelector("#labeling-configure-dataset-button").style.display = "flex";
          document.querySelector("#labeling-extract-and-label-frames-button").style.display = "flex";
          document.querySelector("#labeling-review-labeled-frames-button").style.display = "flex";

          document.querySelector("#labeling-review-labeled-frames-button").classList.remove("activated");
          document.querySelector("#labeling-review-labeled-frames-button").classList.remove("panel-button-bottom");

          document.querySelector("#labeling-review-labeled-frames-inner-section-left-side").style.display = "none";
          document.querySelector("#labeling-review-labeled-frames-inner-section-right-side").style.display = "none";
        }
        else {

          document.querySelector("#labeling-configure-dataset-button-divider").style.display = "none";
          document.querySelector("#labeling-extract-and-label-frames-button-divider").style.display = "none";

          document.querySelector("#labeling-configure-dataset-button").style.display = "none";
          document.querySelector("#labeling-extract-and-label-frames-button").style.display = "none";

          document.querySelector("#labeling-review-labeled-frames-button").classList.add("activated");
          document.querySelector("#labeling-review-labeled-frames-button").classList.add("panel-button-bottom");

          document.querySelector("#labeling-review-labeled-frames-inner-section-left-side").style.display = "flex";
          document.querySelector("#labeling-review-labeled-frames-inner-section-right-side").style.display = "flex";
        }

        window.scrollTo(0, 0);
      }
    );

    this.reset();
  }


  reset() {
    super.reset();

    document.querySelector("#labeling-configure-dataset-inner-section-left-side").style.display = "none";
    document.querySelector("#labeling-extract-and-label-frames-inner-section-left-side").style.display = "none";
    document.querySelector("#labeling-review-labeled-frames-inner-section-left-side").style.display = "none";

    document.querySelector("#labeling-configure-dataset-inner-section-right-side").style.display = "none";
    document.querySelector("#labeling-extract-and-label-frames-inner-section-right-side").style.display = "none";
    document.querySelector("#labeling-review-labeled-frames-inner-section-right-side").style.display = "none";

    document.querySelector("#keypoint-count-input").value = defaultKeypointCount;

    const canvas = document.querySelector("#labeling-movie-player-canvas");
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);

    this.labelingThumbnails.clear();
  }


  enter() {
    {
      let value = document.querySelector("#keypoint-count-input").value;
      this.worker.postMessage({ type: "keypointCount", keypointCount: +value });
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


  async onOpenExistingFileSuccess(message) {
    this.showStatus(Section.savedMessage);
    this.unsavedChanges = false;

    const keypointCount = message.data.keypointCount;
    document.querySelector("#keypoint-count-input").value = keypointCount;

    const metadata = message.data.metadata;

    const coordinates = message.data.coordinates;

    const blobs = message.data.blobs;

    let i = 0;
    for (const blob of blobs) {
      await this.labelingThumbnails.push(blob, metadata[i].arena.width, metadata[i].arena.height, keypointCount, coordinates[i], metadata[i].filename, metadata[i].frameNumber);
      ++i;
    }

    this.enter();
  }


  async maybeLoadMovie() {
    let fileHandle = null;
    try {
      [fileHandle] = await window.showOpenFilePicker(
        {
          id: "openLabelingMovie",
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
    else {
      Section.enableWorkflowButtons();
    }
  }

  onLoadMovieSuccess(filename, numFrames, frameWidth, frameHeight) {
    document.querySelector("#labeling-load-movie-button-label").textContent = filename;

    this.filename = filename;

    this.numFrames = numFrames;
    this.frameWidth = frameWidth;
    this.frameHeight = frameHeight;

    this.currentFrame = null;
    this.currentFrameIndex = null;

    this.zoom = 1;
    this.panOffsetX = 0;
    this.panOffsetY = 0;

    this.currentlyPlaying = false;

    const rangeInput = document.querySelector("#labeling-frame-range-input");
    rangeInput.min = 1;
    rangeInput.value = 1;
    rangeInput.max = this.numFrames;

    const numberInput = document.querySelector("#labeling-frame-number-input");
    numberInput.min = 1;
    numberInput.value = 1;
    numberInput.max = this.numFrames;

    this.worker.postMessage({ type: "frameRequest", index: 0 });

    document.querySelector("#labeling-label-current-frame-button").removeAttribute("disabled");
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


  async draw() {
    if (this.currentFrame) {
      const canvas = document.querySelector("#labeling-movie-player-canvas");
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

      if (this.currentlyLabeling) {
        context.lineWidth = 5;
        context.strokeStyle = "hsl(15, 50%, 50%, 50%)";
        context.beginPath();
        context.rect(this.arenaX, this.arenaY, this.arenaWidth, this.arenaHeight);
        context.stroke();

        const colormap = interpolateColormap("plasma", this.keypointCount, true);
        let i = 0;
        for (const label of this.currentLabels) {
          const radius = 3.75 * Math.min(this.frameWidth, this.frameHeight) / 512;
          context.fillStyle = `rgb(${colormap[i][0] * 255}, ${colormap[i][1] * 255}, ${colormap[i][2] * 255}, 0.75)`;
          context.beginPath();
          context.arc(label.x, label.y, radius, 0, 2 * Math.PI, false);
          context.fill();
          ++i;
        }
      }

      context.restore();
    }
  }

  onFrameReady(frame, frameNumber) {
    this.currentFrame = frame;
    this.currentFrameIndex = frameNumber;

    const rangeInput = document.querySelector("#labeling-frame-range-input");
    rangeInput.value = this.currentFrameIndex + 1;

    const numberInput = document.querySelector("#labeling-frame-number-input");
    numberInput.value = this.currentFrameIndex + 1;

    this.draw();

    if (this.currentlyPlaying) {
      if (this.currentFrameIndex === this.numFrames - 1) {
        this.currentlyPlaying = false;
        document.querySelector("#labeling-play-or-pause-button").classList.remove("activated");
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







  startLabeling() {
    this.currentlyLabeling = true;
    this.currentlyLabelingArena = true;
    this.currentlyLabelingKeypoints = false;

    this.setArena();

    document.querySelector("#labeling-arena-x-input").min = 0;
    document.querySelector("#labeling-arena-x-input").max = this.frameWidth;
    document.querySelector("#labeling-arena-x-input").value = 0;

    document.querySelector("#labeling-arena-y-input").min = 0;
    document.querySelector("#labeling-arena-y-input").max = this.frameHeight;
    document.querySelector("#labeling-arena-y-input").value = 0;

    document.querySelector("#labeling-arena-width-input").min = 0;
    document.querySelector("#labeling-arena-width-input").max = this.frameWidth;
    document.querySelector("#labeling-arena-width-input").value = this.frameWidth;

    document.querySelector("#labeling-arena-height-input").min = 0;
    document.querySelector("#labeling-arena-height-input").max = this.frameHeight;
    document.querySelector("#labeling-arena-height-input").value = this.frameHeight;

    this.arenaX = 0;
    this.arenaY = 0;
    this.arenaWidth = this.frameWidth;
    this.arenaHeight = this.frameHeight;

    for (let i = 0; i < this.keypointCount; ++i) {
      document.querySelector(`#labeling-keypoint-${i}-x-input`).style.display = "block";
      document.querySelector(`#labeling-keypoint-${i}-y-input`).style.display = "block";

      document.querySelector(`#labeling-keypoint-${i}-x-input-label`).style.display = "block";
      document.querySelector(`#labeling-keypoint-${i}-y-input-label`).style.display = "block";
    }
    for (let i = this.keypointCount; i < maxKeypointCount; ++i) {
      document.querySelector(`#labeling-keypoint-${i}-x-input`).style.display = "none";
      document.querySelector(`#labeling-keypoint-${i}-y-input`).style.display = "none";

      document.querySelector(`#labeling-keypoint-${i}-x-input-label`).style.display = "none";
      document.querySelector(`#labeling-keypoint-${i}-y-input-label`).style.display = "none";
    }

    this.currentLabels = [];
    for (let i = 0; i < this.keypointCount; ++i) {
      this.currentLabels.push({ x: 0, y: 0 });
      document.querySelector(`#labeling-keypoint-${i}-x-input`).value = "";
      document.querySelector(`#labeling-keypoint-${i}-y-input`).value = "";
    }
    this.currentLabelIndex = 0;

    this.draw();

    document.querySelector("#labeling-label-current-frame-button").style.display = "none";
    document.querySelector("#labeling-details").style.display = "flex";
  }

  setArena() {
    this.currentlyLabeling = true;
    this.currentlyLabelingArena = true;
    this.currentlyLabelingKeypoints = false;

    document.querySelector("#labeling-set-arena-button").classList.add("button-group-button-activated");
    document.querySelector("#labeling-set-keypoints-button").classList.remove("button-group-button-activated");

    document.querySelector("#labeling-arena-details").style.display = "grid";
    document.querySelector("#labeling-keypoint-details").style.display = "none";
  }

  setKeypoints() {
    this.currentlyLabeling = true;
    this.currentlyLabelingArena = false;
    this.currentlyLabelingKeypoints = true;

    document.querySelector("#labeling-set-arena-button").classList.remove("button-group-button-activated");
    document.querySelector("#labeling-set-keypoints-button").classList.add("button-group-button-activated");

    document.querySelector("#labeling-arena-details").style.display = "none";
    document.querySelector("#labeling-keypoint-details").style.display = "grid";
  }

  cancelLabeling() {
    this.currentlyLabeling = false;
    this.currentlyLabelingArena = false;
    this.currentlyLabelingKeypoints = false;

    document.querySelector("#labeling-label-current-frame-button").style.display = "flex";
    document.querySelector("#labeling-details").style.display = "none";

    this.draw();
  }

  async doneLabeling() {
    this.currentlyLabeling = false;
    this.currentlyLabelingArena = false;
    this.currentlyLabelingKeypoints = false;

    document.querySelector("#labeling-label-current-frame-button").style.display = "flex";
    document.querySelector("#labeling-details").style.display = "none";

    this.arenaX = Math.round(this.arenaX);
    this.arenaY = Math.round(this.arenaY);
    this.arenaWidth = Math.round(this.arenaWidth);
    this.arenaHeight = Math.round(this.arenaHeight);

    const coordinates = this.currentLabels;
    for (let i = 0; i < coordinates.length; ++i) {
      coordinates[i].x -= this.arenaX;
      coordinates[i].y -= this.arenaY;
    }

    const canvas = new OffscreenCanvas(this.arenaWidth, this.arenaHeight);
    const context = canvas.getContext("2d");
    context.drawImage(this.currentFrame, this.arenaX, this.arenaY, this.arenaWidth, this.arenaHeight, 0, 0, canvas.width, canvas.height);

    const blob = await canvas.convertToBlob({ type: "image/png" });

    await this.labelingThumbnails.push(blob, this.arenaWidth, this.arenaHeight, this.keypointCount, coordinates, this.filename, this.currentFrameIndex);

    this.worker.postMessage({ type: "addLabel", filename: this.filename, frameNumber: this.currentFrameIndex, arena: { x: this.arenaX, y: this.arenaY, width: this.arenaWidth, height: this.arenaHeight }, coordinates: this.currentLabels, image: blob });

    this.draw();

    this.unsavedChanges = true;
    this.showStatus(Section.unsavedMessage);
    addEventListener("beforeunload", beforeUnloadListener);
  }
}
