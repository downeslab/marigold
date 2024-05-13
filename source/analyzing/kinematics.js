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

import { radiansToDegrees } from "../core/math.js";

let kinematicsData = null;

let arenas = null;
let arenaRows = null;
let arenaColumns = null;
let keypointCount = null;
let firstFrame = null;
let lastFrame = null;

// let currentRow = 0;
// let currentColumn = 0;
// let currentKeypoint = 0;

let angleKeypoint1 = null;
let angleKeypoint2 = null;
let angleKeypoint3 = null;

let framesPerSecond = null;
let pixelsPerMillimeter = null;


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


function calculateDistancePixels(kinematicsData, arenas, row, column, keypointCount) {
  const distancePixels = [];

  for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
    distancePixels.push(["nan"]);
  }

  const arena = arenas[row * arenaColumns + column];
  for (let frame = 1; frame < lastFrame + 1 - firstFrame; ++frame) {
    for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
      distancePixels[keypoint].push(
        Math.hypot(
          kinematicsData[frame][row * arenaColumns + column][keypoint].x - arena.x - kinematicsData[frame - 1][row * arenaColumns + column][keypoint].x - arena.x,
          kinematicsData[frame][row * arenaColumns + column][keypoint].y - arena.y - kinematicsData[frame - 1][row * arenaColumns + column][keypoint].y - arena.y,
        )
      );
    }
  }

  return distancePixels;
}


function calculateDistanceMm(kinematicsData, arenas, row, column, keypointCount) {
  const distanceMm = [];

  for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
    distanceMm.push(["nan"]);
  }

  const arena = arenas[row * arenaColumns + column];
  for (let frame = 1; frame < lastFrame + 1 - firstFrame; ++frame) {
    for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
      distanceMm[keypoint].push(
        Math.hypot(
          (kinematicsData[frame][row * arenaColumns + column][keypoint].x - arena.x - kinematicsData[frame - 1][row * arenaColumns + column][keypoint].x - arena.x) / pixelsPerMillimeter,
          (kinematicsData[frame][row * arenaColumns + column][keypoint].y - arena.y - kinematicsData[frame - 1][row * arenaColumns + column][keypoint].y - arena.y) / pixelsPerMillimeter,
        )
      );
    }
  }

  return distanceMm;
}


function calculateSpeedPixelsPerFrame(kinematicsData, arenas, row, column, keypointCount) {
  const speedPixelsPerFrame = [];

  for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
    speedPixelsPerFrame.push(["nan"]);
  }

  const arena = arenas[row * arenaColumns + column];
  for (let frame = 1; frame < lastFrame + 1 - firstFrame; ++frame) {
    for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
      speedPixelsPerFrame[keypoint].push(
        Math.hypot(
          kinematicsData[frame][row * arenaColumns + column][keypoint].x - arena.x - kinematicsData[frame - 1][row * arenaColumns + column][keypoint].x - arena.x,
          kinematicsData[frame][row * arenaColumns + column][keypoint].y - arena.y - kinematicsData[frame - 1][row * arenaColumns + column][keypoint].y - arena.y,
        )
      );
    }
  }

  return speedPixelsPerFrame;
}


function calculateSpeedMillimetersPerSecond(kinematicsData, arenas, row, column, keypointCount) {
  const speedMillimetersPerSecond = [];

  for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
    speedMillimetersPerSecond.push(["nan"]);
  }

  const arena = arenas[row * arenaColumns + column];
  for (let frame = 1; frame < lastFrame + 1 - firstFrame; ++frame) {
    for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
      speedMillimetersPerSecond[keypoint].push(
        Math.hypot(
          (kinematicsData[frame][row * arenaColumns + column][keypoint].x - arena.x - kinematicsData[frame - 1][row * arenaColumns + column][keypoint].x - arena.x) / pixelsPerMillimeter,
          (kinematicsData[frame][row * arenaColumns + column][keypoint].y - arena.y - kinematicsData[frame - 1][row * arenaColumns + column][keypoint].y - arena.y) / pixelsPerMillimeter,
        )
      );
    }
  }

  return speedMillimetersPerSecond;
}


function calculateAngle(kinematicsData, arenas, row, column, angleKeypoint1, angleKeypoint2, angleKeypoint3) {
  const angle = [];

  const arena = arenas[row * arenaColumns + column];
  for (let frame = 0; frame < lastFrame + 1 - firstFrame; ++frame) {
    const a = {
      x: kinematicsData[frame][row * arenaColumns + column][angleKeypoint1].x - arena.x,
      y: kinematicsData[frame][row * arenaColumns + column][angleKeypoint1].y - arena.y
    };
    const b = {
      x: kinematicsData[frame][row * arenaColumns + column][angleKeypoint2].x - arena.x,
      y: kinematicsData[frame][row * arenaColumns + column][angleKeypoint2].y - arena.y
    };
    const c = {
      x: kinematicsData[frame][row * arenaColumns + column][angleKeypoint3].x - arena.x,
      y: kinematicsData[frame][row * arenaColumns + column][angleKeypoint3].y - arena.y
    };

    const ab = Math.atan2(a.y - b.y, a.x - b.x);
    const cb = Math.atan2(c.y - b.y, c.x - b.x);

    let rad = cb - ab;
    if (rad < 0) {
      rad += 2 * Math.PI;
    }
    const deg = radiansToDegrees(Math.PI - rad);

    angle.push(deg);
  }

  return angle;
}


function makeEducatedGuessCorrectionsForAnglesWithAbsoluteMagnitudeGreaterThan180Degrees(originalAngles) {
  // let newAngles = [];

  for (let i = 1; i < originalAngles.length; ++i) {
    const diff = Math.abs(originalAngles[i] - originalAngles[i - 1]);

    if (diff >= 180) {
      if (originalAngles[i] > 0) {
        originalAngles[i] -= 360;
      }
      else {
        originalAngles[i] += 360;
      }
    }

    if (Math.abs(originalAngles[i]) > 270) {
      if (originalAngles[i] > 0) {
        originalAngles[i] -= 360;
      }
      else {
        originalAngles[i] += 360;
      }
    }
  }

  return originalAngles;
}


function calculateNetDistance(kinematicsData, arenas, row, column, keypointCount) {
  let netDistance = [];

  // for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
  //   netDistance.push([]);
  // }

  const arena = arenas[row * arenaColumns + column];
  for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
    netDistance.push(
      Math.hypot(
        (kinematicsData[lastFrame - firstFrame][row * arenaColumns + column][keypoint].x - arena.x) / pixelsPerMillimeter -
        (kinematicsData[0][row * arenaColumns + column][keypoint].x - arena.x) / pixelsPerMillimeter,
        (kinematicsData[lastFrame - firstFrame][row * arenaColumns + column][keypoint].y - arena.y) / pixelsPerMillimeter -
        (kinematicsData[0][row * arenaColumns + column][keypoint].y - arena.y) / pixelsPerMillimeter,
      )
    );
  }

  // console.log("net:", netDistance);

  return netDistance;
}


function calculateCumulativeDistance(distance) {
  let cumulativeDistance = [];

  // for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
  //   cumulativeDistance.push([0.0]);
  // }

  for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
    let sum = 0.0;
    for (let i = 1; i < distance[0].length; ++i) {
      sum += distance[keypoint][i];
    }
    cumulativeDistance.push(sum);
  }

  // console.log("distance:", distance);
  // console.log("cumulativeDistance:", cumulativeDistance);

  return cumulativeDistance;
}


function minAbsolute(array) {
  let result = Math.abs(array[0]);
  for (let i = 1; i < array.length; ++i) {
    const abs = Math.abs(array[i]);
    if (abs < result) {
      result = abs;
    }
  }
  return result;
}


function maxAbsolute(array) {
  let result = Math.abs(array[0]);
  for (let i = 1; i < array.length; ++i) {
    const abs = Math.abs(array[i]);
    if (abs > result) {
      result = abs;
    }
  }
  return result;
}


function mean(array) {
  const n = array.length;
  const result = array.reduce((a, b) => { return a + b; }) / n;
  return result;
}


function median(array) {
  const sorted = array.toSorted((a, b) => { return a - b; });
  let result = null;
  if (array.length % 2) {
    result = sorted[Math.floor(array.length / 2)];
  }
  else {
    result = (sorted[Math.floor(array.length / 2) - 1] + sorted[Math.floor(array.length / 2)]) / 2;
  }
  return result;
}


// export function getCsvData(data, arenas_, arenaRows_, arenaColumns_, keypointCount_, currentRow_, currentColumn_, currentKeypoint_, firstFrame_, lastFrame_, angleKeypoint1_, angleKeypoint2_, angleKeypoint3_) {
//   kinematicsData = data;

//   framesPerSecond = document.querySelector("#framerate-input").value;
//   pixelsPerMillimeter = document.querySelector("#scale-input").value;

//   arenas = arenas_;
//   arenaRows = arenaRows_;
//   arenaColumns = arenaColumns_;
//   keypointCount = keypointCount_;
//   firstFrame = firstFrame_;
//   lastFrame = lastFrame_;

//   // console.log("raw:", kinematicsData, arenas);

//   // currentRow = currentRow_;
//   // currentColumn = currentColumn_;
//   // currentKeypoint = currentKeypoint_;

//   const csvFilenames = [];
//   const csvFileContents = [];

//   const frameNumbers = calculateFrameNumbers();
//   const timestamps = calculateTimestamps();
//   const frameNumbersNormalized = calculateFrameNumbersNormalized();
//   const timestampsNormalized = calculateTimestampsNormalized();

//   //

//   angleKeypoint1 = angleKeypoint1_;
//   angleKeypoint2 = angleKeypoint2_;
//   angleKeypoint3 = angleKeypoint3_;

//   let calculatingAngle = false;
//   if (angleKeypoint1 >= 0 && angleKeypoint1 < keypointCount && angleKeypoint2 >= 0 && angleKeypoint2 < keypointCount && angleKeypoint3 >= 0 && angleKeypoint3 < keypointCount) {
//     calculatingAngle = true;
//   }

//   //

//   const summaryFilename = "scalar_parameters.csv";
//   let summaryFileContents = "";

//   let summaryFileHeader = "";
//   summaryFileHeader += "column,row,";
//   summaryFileHeader += "duration,";

//   for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//     summaryFileHeader += `net_distance_${keypoint + 1}_in_millimeters,`;
//   }
//   for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//     summaryFileHeader += `cumulative_distance_${keypoint + 1}_in_millimeters,`;
//   }

//   for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//     summaryFileHeader += `min_speed_${keypoint + 1}_in_millimeters_per_second,`;
//   }
//   for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//     summaryFileHeader += `max_speed_${keypoint + 1}_in_millimeters_per_second,`;
//   }
//   for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//     summaryFileHeader += `mean_speed_${keypoint + 1}_in_millimeters_per_second,`;
//   }
//   for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//     summaryFileHeader += `median_speed_${keypoint + 1}_in_millimeters_per_second,`;
//   }

//   if (calculatingAngle) {
//     summaryFileHeader += `min_absolute_angle_${angleKeypoint1 + 1}_${angleKeypoint2 + 1}_${angleKeypoint3 + 1}_in_degrees,`;
//     summaryFileHeader += `max_absolute_angle_${angleKeypoint1 + 1}_${angleKeypoint2 + 1}_${angleKeypoint3 + 1}_in_degrees,`;
//     summaryFileHeader += `mean_absolute_angle_${angleKeypoint1 + 1}_${angleKeypoint2 + 1}_${angleKeypoint3 + 1}_in_degrees,`;
//     summaryFileHeader += `median_absolute_angle_${angleKeypoint1 + 1}_${angleKeypoint2 + 1}_${angleKeypoint3 + 1}_in_degrees,`;
//   }

//   summaryFileContents += summaryFileHeader.slice(0, summaryFileHeader.length - 1);
//   summaryFileContents += "\n";

//   //

//   for (let row = 0; row < arenaRows; ++row) {
//     for (let column = 0; column < arenaColumns; ++column) {
//       const filename = `vector-parameters-column-${column + 1}-row-${row + 1}.csv`;
//       let fileContents = "";

//       //

//       const xYPixels = calculateXYPixels(kinematicsData, arenas, row, column, keypointCount);
//       const xYMm = calculateXYMm(kinematicsData, arenas, row, column, keypointCount);

//       //

//       const distancePixels = calculateDistancePixels(kinematicsData, arenas, row, column, keypointCount);
//       const distanceMm = calculateDistanceMm(kinematicsData, arenas, row, column, keypointCount);
//       const speedPixelsPerFrame = calculateSpeedPixelsPerFrame(kinematicsData, arenas, row, column, keypointCount);
//       const speedMillimetersPerSecond = calculateSpeedMillimetersPerSecond(kinematicsData, arenas, row, column, keypointCount);

//       //

//       let angle = null;
//       if (calculatingAngle) {
//         angle = calculateAngle(kinematicsData, arenas, row, column, angleKeypoint1, angleKeypoint2, angleKeypoint3);
//         angle = makeEducatedGuessCorrectionsForAnglesWithAbsoluteMagnitudeGreaterThan180Degrees(angle);
//       }

//       //

//       let header = "";

//       header += "frame_number,timestamp_in_seconds,";
//       if (firstFrame != 0) {
//         header += "frame_number_normalized,timestamp_normalized_in_seconds,";
//       }
//       for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//         header += `x_${keypoint + 1}_in_pixels,y_${keypoint + 1}_in_pixels,`;
//       }
//       for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//         header += `x_${keypoint + 1}_in_millimeters,y_${keypoint + 1}_in_millimeters,`;
//       }
//       // for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//       //   header += `distance_${keypoint + 1}_in_pixels,`;
//       // }
//       for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//         header += `distance_${keypoint + 1}_in_millimeters,`;
//       } distanceMm;
//       // for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//       //   header += `speed_${keypoint + 1}_in_pixels_per_frame,`;
//       // }
//       for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//         header += `speed_${keypoint + 1}_in_millimeters_per_second,`;
//       }
//       if (calculatingAngle) {
//         header += `angle_${angleKeypoint1 + 1}_${angleKeypoint2 + 1}_${angleKeypoint3 + 1}_in_degrees,`;
//       }
//       // reminder: angles

//       fileContents += header.slice(0, header.length - 1);
//       fileContents += "\n";

//       //

//       for (let i = 0; i < frameNumbers.length; ++i) {
//         let line = "";

//         line += `${frameNumbers[i]},`;
//         line += `${timestamps[i]},`;
//         if (firstFrame != 0) {
//           line += `${frameNumbersNormalized[i]},`;
//           line += `${timestampsNormalized[i]},`;
//         }

//         for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//           line += `${xYPixels[keypoint][i].x},`;
//           line += `${xYPixels[keypoint][i].y},`;
//         }
//         for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//           line += `${xYMm[keypoint][i].x},`;
//           line += `${xYMm[keypoint][i].y},`;
//         }
//         // for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//         //   line += `${distancePixels[keypoint][i]},`;
//         // }
//         for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//           line += `${distanceMm[keypoint][i]},`;
//         }
//         // for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//         //   line += `${speedPixelsPerFrame[keypoint][i]},`;
//         // }
//         for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//           line += `${speedMillimetersPerSecond[keypoint][i]},`;
//         }
//         if (calculatingAngle) {
//           line += `${angle[i]},`;
//         }

//         fileContents += line.slice(0, line.length - 1);
//         fileContents += "\n";
//       }

//       csvFilenames.push(filename);
//       csvFileContents.push(fileContents);

//       //

//       const netDistance = calculateNetDistance(kinematicsData, arenas, row, column, keypointCount);
//       const cumulativeDistance = calculateCumulativeDistance(distanceMm);

//       let summaryFileLine = "";
//       summaryFileLine += `${column + 1},${row + 1},`;

//       summaryFileLine += `${timestamps[timestamps.length - 1]},`;

//       for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//         summaryFileLine += `${netDistance[keypoint]},`;
//       }
//       for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//         summaryFileLine += `${cumulativeDistance[keypoint]},`;
//       }

//       for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//         summaryFileLine += `${Math.min(...speedMillimetersPerSecond[keypoint].slice(1))},`;
//       }
//       for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//         summaryFileLine += `${Math.max(...speedMillimetersPerSecond[keypoint].slice(1))},`;
//       }
//       for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//         summaryFileLine += `${mean(speedMillimetersPerSecond[keypoint].slice(1))},`;
//       }
//       for (let keypoint = 0; keypoint < keypointCount; ++keypoint) {
//         summaryFileLine += `${median(speedMillimetersPerSecond[keypoint].slice(1))},`;
//       }
//       // console.log("speedMillimetersPerSecond[keypoint].slice(1, -1):", speedMillimetersPerSecond[0].slice(1, 0));

//       if (calculatingAngle) {
//         summaryFileLine += `${minAbsolute(angle)},`;
//         summaryFileLine += `${maxAbsolute(angle)},`;
//         summaryFileLine += `${mean(angle)},`;
//         summaryFileLine += `${median(angle)},`;
//       }

//       summaryFileContents += summaryFileLine.slice(0, summaryFileLine.length - 1);
//       summaryFileContents += "\n";
//     }
//   }

//   return { csvFilenames, csvFileContents, summaryFilename, summaryFileContents };
// }


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

  console.log("raw:", kinematicsData, arenas);

  const summaryFilename = "";
  const summaryFileContents = "";

  const csvFilenames = [];
  const csvFileContents = [];

  const frameNumbers = calculateFrameNumbers();
  const timestamps = calculateTimestamps();
  const frameNumbersNormalized = calculateFrameNumbersNormalized();
  const timestampsNormalized = calculateTimestampsNormalized();

  //

  for (let row = 0; row < arenaRows; ++row) {
    for (let column = 0; column < arenaColumns; ++column) {
      const filename = `vector-parameters-column-${column + 1}-row-${row + 1}.csv`;
      let fileContents = "";

      //

      const xYPixels = calculateXYPixels(kinematicsData, arenas, row, column, keypointCount);
      const xYMm = calculateXYMm(kinematicsData, arenas, row, column, keypointCount);

      //

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

      //

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
