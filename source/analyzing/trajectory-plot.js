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

let plotData = null;
let arenas = null;
let arenaRows = null;
let arenaColumns = null;
let keypointCount = null;
let currentRow = 0;
let currentColumn = 0;
let currentKeypoint = 0;
let firstFrame = null;
let lastFrame = null;
let arenaShape = null;

// let plotWidth = null;
// let plotHeight = null;


export function updatePlots() {
  if (plotData) {
    // const plotWidth = 32;
    // const plotWidth = 24;
    // const plotHeight = 24;

    // const arenaShape = "rectangle";

    const arenaWidth = arenas[0].width;
    const arenaHeight = arenas[0].height;

    const pointRadius = document.querySelector("#radius-input").value * (Math.min(arenaWidth, arenaHeight) / 24);

    const plotXOffset = Math.ceil(pointRadius);
    const plotYOffset = Math.ceil(pointRadius);

    // const plotWidth = arenas[0].width;
    // const plotHeight = arenas[0].height;
    const plotWidth = arenaWidth + 2 * plotXOffset;
    const plotHeight = arenaHeight + 2 * plotYOffset;

    // const elements = document.querySelectorAll(".trajectory-plot-hidden");
    // console.log("num:", elements);
    for (const element of document.querySelectorAll(".trajectory-plot-hidden")) {
      element.remove();
    }
    // const elements = document.querySelectorAll(".trajectory-plot-test");
    // console.log("num:", elements);
    for (const element of document.querySelectorAll(".trajectory-plot-test")) {
      element.remove();
    }
    if (document.querySelector("#big-svg")) {
      document.querySelector("#big-svg").remove();
    }
    // while (elements.length) {
    //   elements[0].remove();
    // }
    // const element = document.querySelector(".trajectory-plot-selected");
    // if (element) {
    //   element.remove();
    // }

    const bigSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    bigSvg.setAttribute("viewBox", `0 0 ${plotWidth * arenaColumns} ${plotHeight * arenaRows}`);
    bigSvg.id = "big-svg";

    for (let row = 0; row < arenaRows; ++row) {
      for (let column = 0; column < arenaColumns; ++column) {
        for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
          // console.log("status:", row, column, keypoint);

          const trajectory = [];
          for (let frame = 0; frame < plotData.length; ++frame) {
            trajectory.push(plotData[frame][row * arenaColumns + column]);
          }
          const arena = arenas[row * arenaColumns + column];
          // console.log(trajectory);
          // console.log(arena);

          const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
          svg.setAttribute("viewBox", `0 0 ${plotWidth} ${plotHeight}`);

          const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
          // title.textContent = "Trajectory plot";
          title.textContent = `Column ${column + 1}, Row ${row + 1}`;
          svg.appendChild(title);

          // const borderCircle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          // borderCircle.setAttribute("id", "border-circle");
          // borderCircle.setAttribute("cx", "12");
          // borderCircle.setAttribute("cy", "12");
          // borderCircle.setAttribute("r", 10 + 0.5 * document.querySelector("#border-thickness").value);
          // borderCircle.setAttribute("fill", "none");
          // borderCircle.setAttribute("stroke", document.querySelector("input[name=background-color]:checked").value === "black" ? "white" : "black");
          // borderCircle.setAttribute("stroke-width", document.querySelector("#border-thickness").value);
          // svg.appendChild(borderCircle);

          // let colormap = interpolateColormap(document.querySelector("#colormap-select").value, trajectory.length, document.querySelector("#reverse-colormap-input-checkbox").checked);
          let colormap = interpolateColormap(document.querySelector("#colormap-select").value, trajectory.length, true);

          // if (row === currentRow && column === currentColumn && keypoint === currentKeypoint) {
          //   const cloned = svg.cloneNode(true);
          //   cloned.setAttribute("id", `trajectory-plot-selected`);
          //   cloned.classList.add("trajectory-plot-selected");
          //   document.querySelector(".trajectory-plot-selected").replaceWith(cloned);

          //   const background = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          //   background.setAttribute("id", "background-circle");
          //   background.setAttribute("cx", "12");
          //   background.setAttribute("cy", "12");
          //   background.setAttribute("r", "10");
          //   // background.setAttribute("r", "12");
          //   background.setAttribute("fill", "hsl(0, 0%, 100%, 0.05)");
          //   cloned.appendChild(background);

          //   for (let i = 0; i < trajectory.length; ++i) {
          //     const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          //     circle.setAttribute("id", `circle-${i}`);
          //     circle.setAttribute("cx", 2 + (trajectory[i][keypoint].x - arena.x) * ((plotWidth - 4) / arena.width));
          //     circle.setAttribute("cy", 2 + (trajectory[i][keypoint].y - arena.y) * ((plotHeight - 4) / arena.height));
          //     circle.setAttribute("r", document.querySelector("#radius-input").value);
          //     circle.setAttribute("fill", `rgb(${colormap[i][0] * 255}, ${colormap[i][1] * 255}, ${colormap[i][2] * 255})`);
          //     cloned.appendChild(circle);

          //     const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
          //     title.textContent = `Frame ${firstFrame + i + 1}`;
          //     circle.appendChild(title);
          //   }
          // }
          // else {
          if (arenaShape === "circle") {
            // console.log("plot circle");
            // const background = document.createElementNS("http://www.w3.org/2000/svg", "ellipse"); // reminder: temporary (replace with circle when square aspect ratio is enforced)
            const background = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            // background.setAttribute("id", "background");
            // background.setAttribute("cx", "12");
            // background.setAttribute("cy", "12");
            // background.setAttribute("r", "10");
            background.setAttribute("cx", plotWidth / 2);
            background.setAttribute("cy", plotHeight / 2);
            // background.setAttribute("rx", arenaWidth / 2);
            // background.setAttribute("ry", arenaHeight / 2);
            background.setAttribute("r", arenaWidth / 2);
            background.setAttribute("fill", "black");
            svg.appendChild(background);
          }
          else if (arenaShape === "square" || arenaShape === "rectangle") {
            // console.log("plot square/rect");
            const background = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            // background.setAttribute("id", "background");
            background.setAttribute("x", plotXOffset);
            background.setAttribute("y", plotYOffset);
            background.setAttribute("width", arenaWidth);
            background.setAttribute("height", arenaHeight);
            background.setAttribute("fill", "black");
            svg.appendChild(background);
          }

          for (let i = 0; i < trajectory.length; ++i) {
            const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            // circle.setAttribute("id", `circle-${i}`);
            // circle.setAttribute("cx", 2 + (trajectory[i][keypoint].x - arena.x) * ((plotWidth - 4) / arena.width));
            // circle.setAttribute("cy", 2 + (trajectory[i][keypoint].y - arena.y) * ((plotHeight - 4) / arena.height));
            circle.setAttribute("cx", plotXOffset + trajectory[i][keypoint].x - arena.x);
            circle.setAttribute("cy", plotYOffset + trajectory[i][keypoint].y - arena.y);
            circle.setAttribute("r", pointRadius);
            circle.setAttribute("fill", `rgb(${colormap[i][0] * 255}, ${colormap[i][1] * 255}, ${colormap[i][2] * 255})`);
            svg.appendChild(circle);

            const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
            title.textContent = `Frame ${firstFrame + i + 1}`;
            circle.appendChild(title);
          }
          // svg.classList.add("trajectory-plot-hidden");
          // svg.classList.add("trajectory-plot-test");
          // svg.setAttribute("id", `trajectory-plot-column-${column + 1}-row-${row + 1}-keypoint-${keypoint + 1}`);
          // document.querySelector("#trajectory-plot-actual-plot-area").appendChild(svg);
          // }
          // document.querySelector("#trajectory-plot-actual-plot-area").style.gridTemplateColumns = "auto ".repeat(arenaColumns - 1) + "auto";
          // document.querySelector("#trajectory-plot-actual-plot-area").style.gridTemplateColumns = "minmax(0, 1fr) ".repeat(arenaColumns - 1) + "minmax(0, 1fr)";

          const cloned = svg.cloneNode(true);
          cloned.classList.add("trajectory-plot-hidden");
          cloned.setAttribute("id", `trajectory-plot-column-${column + 1}-row-${row + 1}-keypoint-${keypoint + 1}`);
          document.body.appendChild(cloned);

          if (keypoint === currentKeypoint) {
            svg.classList.add("trajectory-plot-test");
            svg.setAttribute("x", column * plotWidth);
            svg.setAttribute("y", row * plotHeight);
            svg.setAttribute("width", plotWidth);
            svg.setAttribute("height", plotHeight);
            bigSvg.appendChild(svg);
          }
        }
      }
    }
    // document.querySelector(".inner-section-right-side-strict-trajectory-plot").appendChild(bigSvg);
    document.querySelector(".inner-section-right-side-strict-trajectory-plot").insertBefore(bigSvg, document.querySelector("#analyzing-trajectory-plot-details"));
    bigSvg.classList.add("trajectory-plot-big");

  }
  else {
    // reminder: remove everything
  }
}


export function initializePlots() {
  for (const input of document.querySelectorAll("input[name=background-color]")) {
    input.addEventListener(
      "change",
      (event) => {
        updatePlots();
      }
    );
  }

  document.querySelector("#colormap-select").addEventListener(
    "change",
    (event) => {
      updatePlots();
    }
  );

  document.querySelector("#reverse-colormap-input-checkbox").addEventListener(
    "change",
    (event) => {
      updatePlots();
    }
  );

  document.querySelector("#radius-input").addEventListener(
    "change",
    (event) => {
      updatePlots();
    }
  );

  document.querySelector("#border-thickness").addEventListener(
    "change",
    (event) => {
      updatePlots();
    }
  );

  updatePlots();
}


export function updatePlotData(data, arenas_, arenaRows_, arenaColumns_, keypointCount_, currentRow_, currentColumn_, currentKeypoint_, firstFrame_, lastFrame_, arenaShape_) {
  plotData = data;
  arenas = arenas_;
  arenaRows = arenaRows_;
  arenaColumns = arenaColumns_;
  keypointCount = keypointCount_;
  currentRow = currentRow_;
  currentColumn = currentColumn_;
  currentKeypoint = currentKeypoint_;
  firstFrame = firstFrame_;
  lastFrame = lastFrame_;
  arenaShape = arenaShape_;
  updatePlots();
}


export function clearPlotData(data) {
  plotData = null;
  updatePlots();
}
