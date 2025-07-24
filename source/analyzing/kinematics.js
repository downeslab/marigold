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

let kinematicsData = null;

let arenas = null;
let arenaRows = null;
let arenaColumns = null;
let keypointCount = null;
let firstFrame = null;
let lastFrame = null;

let framesPerSecond = null;
let pixelsPerMillimeter = null;


function radiansToDegrees(rad) {
  return rad / (Math.PI / 180);
}


function calculateFrameNumbers() {
  const frameNumbers = [];

  for (let i = firstFrame; i < lastFrame + 1; ++i) {
    frameNumbers.push(i + 1);
  }

  return frameNumbers;
}


function calculateFrameNumbersNormalized() {
  const frameNumbers = [];

  for (let i = 0; i < lastFrame + 1 - firstFrame; ++i) {
    frameNumbers.push(i + 1);
  }

  return frameNumbers;
}


function calculateTimestamps() {
  const timestamps = [];

  for (let i = firstFrame; i < lastFrame + 1; ++i) {
    timestamps.push(i / framesPerSecond);
  }

  return timestamps;
}


function calculateTimestampsNormalized() {
  const timestamps = [];

  for (let i = 0; i < lastFrame + 1 - firstFrame; ++i) {
    timestamps.push(i / framesPerSecond);
  }

  return timestamps;
}


function calculateXYPixels(kinematicsData, arenas, row, column, keypointCount) {
  const xYPixels = [];

  for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
    xYPixels.push([]);
  }

  const arena = arenas[row * arenaColumns + column];
  for (let frame = 0; frame < lastFrame + 1 - firstFrame; ++frame) {
    for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
      xYPixels[keypoint].push(
        {
          x: kinematicsData[frame][row * arenaColumns + column][keypoint].x - arena.x,
          y: kinematicsData[frame][row * arenaColumns + column][keypoint].y - arena.y
        }
      );
    }
  }

  return xYPixels;
}


function calculateXYMm(kinematicsData, arenas, row, column, keypointCount) {
  const xYMm = [];

  for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
    xYMm.push([]);
  }

  const arena = arenas[row * arenaColumns + column];
  for (let frame = 0; frame < lastFrame + 1 - firstFrame; ++frame) {
    for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
      xYMm[keypoint].push(
        {
          x: (kinematicsData[frame][row * arenaColumns + column][keypoint].x - arena.x) / pixelsPerMillimeter,
          y: (kinematicsData[frame][row * arenaColumns + column][keypoint].y - arena.y) / pixelsPerMillimeter
        }
      );
    }
  }

  return xYMm;
}


export function getCsvData(data, arenas_, arenaRows_, arenaColumns_, keypointCount_, currentRow_, currentColumn_, currentKeypoint_, firstFrame_, lastFrame_, angleKeypoint1_, angleKeypoint2_, angleKeypoint3_) {
  kinematicsData = data;

  framesPerSecond = document.querySelector("#framerate-input").value;
  pixelsPerMillimeter = document.querySelector("#scale-input").value;

  arenas = arenas_;
  arenaRows = arenaRows_;
  arenaColumns = arenaColumns_;
  keypointCount = keypointCount_;
  firstFrame = firstFrame_;
  lastFrame = lastFrame_;

  const summaryFilename = "";
  const summaryFileContents = "";

  const csvFilenames = [];
  const csvFileContents = [];

  const frameNumbers = calculateFrameNumbers();
  const timestamps = calculateTimestamps();
  const frameNumbersNormalized = calculateFrameNumbersNormalized();
  const timestampsNormalized = calculateTimestampsNormalized();

  for (let row = 0; row < arenaRows; ++row) {
    for (let column = 0; column < arenaColumns; ++column) {
      const filename = `vector-parameters-column-${column + 1}-row-${row + 1}.csv`;
      let fileContents = "";

      const xYPixels = calculateXYPixels(kinematicsData, arenas, row, column, keypointCount);
      const xYMm = calculateXYMm(kinematicsData, arenas, row, column, keypointCount);

      let header = "";

      header += "frame_number,timestamp_in_seconds,";
      if (firstFrame != 0) {
        header += "frame_number_normalized,timestamp_normalized_in_seconds,";
      }
      for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
        header += `x_${keypoint + 1}_in_pixels,y_${keypoint + 1}_in_pixels,`;
      }
      for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
        header += `x_${keypoint + 1}_in_millimeters,y_${keypoint + 1}_in_millimeters,`;
      }

      fileContents += header.slice(0, header.length - 1);
      fileContents += "\n";

      for (let i = 0; i < frameNumbers.length; ++i) {
        let line = "";

        line += `${frameNumbers[i]},`;
        line += `${timestamps[i]},`;
        if (firstFrame != 0) {
          line += `${frameNumbersNormalized[i]},`;
          line += `${timestampsNormalized[i]},`;
        }

        for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
          line += `${xYPixels[keypoint][i].x},`;
          line += `${xYPixels[keypoint][i].y},`;
        }
        for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
          line += `${xYMm[keypoint][i].x},`;
          line += `${xYMm[keypoint][i].y},`;
        }

        fileContents += line.slice(0, line.length - 1);
        fileContents += "\n";
      }

      csvFilenames.push(filename);
      csvFileContents.push(fileContents);
    }
  }

  return { csvFilenames, csvFileContents, summaryFilename, summaryFileContents };
}
