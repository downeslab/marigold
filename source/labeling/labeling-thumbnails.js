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

import { interpolateColormap } from "../colormap.js";


export class LabelingThumbnails {
  currentLength = 0;
  currentElements = [];

  constructor() {
  }

  async push(frameBlob, arenaWidth, arenaHeight, keypointCount, coordinates, filename, frameNumber) {
    let divElement = document.createElement("div");
    divElement.classList.add("labeling-thumbnail-container");
    divElement.id = `labeling-thumbnail-${this.currentLength}`;

    let canvasElement = document.createElement("canvas");
    canvasElement.classList.add("labeling-thumbnail-canvas");
    canvasElement.width = Math.max(arenaWidth, arenaHeight);
    canvasElement.height = Math.max(arenaWidth, arenaHeight);
    let canvasContext = canvasElement.getContext("2d");
    const imageBitmap = await createImageBitmap(frameBlob, { colorSpaceConversion: "none" });
    canvasContext.drawImage(imageBitmap, 0, 0, arenaWidth, arenaHeight, Math.round((Math.max(arenaWidth, arenaHeight) - arenaWidth) / 2), Math.round((Math.max(arenaWidth, arenaHeight) - arenaHeight) / 2), arenaWidth, arenaHeight);

    const xOffset = Math.round((Math.max(arenaWidth, arenaHeight) - arenaWidth) / 2);
    const yOffset = Math.round((Math.max(arenaWidth, arenaHeight) - arenaHeight) / 2);

    const colormap = interpolateColormap("plasma", keypointCount, true);
    let i = 0;
    for (const label of coordinates) {
      const radius = 3.75 * Math.min(imageBitmap.width, imageBitmap.height) / 512;
      canvasContext.fillStyle = `rgb(${colormap[i][0] * 255}, ${colormap[i][1] * 255}, ${colormap[i][2] * 255}, 0.75)`;
      canvasContext.beginPath();
      canvasContext.arc(xOffset + label.x, yOffset + label.y, radius, 0, 2 * Math.PI, false);
      canvasContext.fill();
      ++i;
    }

    divElement.appendChild(canvasElement);

    let dividerElement = document.createElement("div");
    dividerElement.classList.add("generic-divider");
    divElement.appendChild(dividerElement);

    let metadataDivElement = document.createElement("div");
    metadataDivElement.classList.add("labeling-thumbnail-metadata-container");
    divElement.appendChild(metadataDivElement);

    let dataIndexLabelElement = document.createElement("p");
    dataIndexLabelElement.classList.add("generic-blurb-small");
    dataIndexLabelElement.textContent = "Label number:";
    metadataDivElement.appendChild(dataIndexLabelElement);

    let dataIndexElement = document.createElement("p");
    dataIndexElement.classList.add("generic-blurb-small");
    dataIndexElement.textContent = `${this.currentLength + 1}`;
    metadataDivElement.appendChild(dataIndexElement);

    let movieNameLabelElement = document.createElement("p");
    movieNameLabelElement.classList.add("generic-blurb-small");
    movieNameLabelElement.textContent = "Source movie:";
    metadataDivElement.appendChild(movieNameLabelElement);

    let movieNameElement = document.createElement("p");
    movieNameElement.classList.add("generic-blurb-small");
    movieNameElement.textContent = `${filename}`;
    metadataDivElement.appendChild(movieNameElement);

    let frameNumberLabelNameElement = document.createElement("p");
    frameNumberLabelNameElement.classList.add("generic-blurb-small");
    frameNumberLabelNameElement.textContent = "Source frame:";
    metadataDivElement.appendChild(frameNumberLabelNameElement);

    let frameNumberNameElement = document.createElement("p");
    frameNumberNameElement.classList.add("generic-blurb-small");
    frameNumberNameElement.textContent = `${frameNumber + 1}`;
    metadataDivElement.appendChild(frameNumberNameElement);

    let buttonElement = document.createElement("button");
    buttonElement.classList.add("generic-button");
    buttonElement.textContent = "Delete";

    divElement.appendChild(buttonElement);

    this.currentElements.push(divElement);

    document.querySelector("#labeling-thumbnails").appendChild(divElement);

    buttonElement.addEventListener(
      "click",
      (event) => {
        this.remove(divElement);
      }
    );

    //

    ++this.currentLength;
  }

  remove(divElement) {
    let index = 0;
    for (const element of this.currentElements) {
      if (element === divElement) {
        break;
      }
      else {
        ++index;
      }
    }

    while (divElement.firstChild) {
      divElement.removeChild(divElement.firstChild);
    }

    divElement.remove();

    this.currentElements.splice(index, 1);

    for (let i = index + 1; i < this.currentLength; ++i) {
      const element = document.querySelector(`#labeling-thumbnail-${i}`);

      element.id = `labeling-thumbnail-${i - 1}`;

      for (const child of document.querySelector(`#labeling-thumbnail-${i - 1} .labeling-thumbnail-metadata-container`).children) {
        if (child.textContent === "Label number:") {
          child.nextElementSibling.textContent = `${i}`;
        }
      }
    }

    document.querySelector("#labeling-thumbnails").dispatchEvent(new CustomEvent("labelRemoved", { detail: index }));

    --this.currentLength;
  }

  clear() {
    for (const element of this.currentElements) {
      this.remove(element);
    }
  }
}
