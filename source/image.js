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

export const channelsRgb = 3;
export const channelsRgba = 4;


export function nearestValidImageSize(originalHeight, originalWidth, maxImageSize, mustBeMultipleOf = 8) {
  if (maxImageSize % mustBeMultipleOf !== 0) {
    throw new Error();
  }

  let newHeight = null;
  let newWidth = null;

  if (originalHeight < originalWidth) {
    newWidth = maxImageSize;
    newHeight = Math.round(originalHeight * (newWidth / originalWidth));
    newHeight += mustBeMultipleOf - (newHeight % mustBeMultipleOf);
  }
  else {
    newHeight = maxImageSize;
    newWidth = Math.round(originalWidth * (newHeight / originalHeight));
    newWidth += mustBeMultipleOf - (newWidth % mustBeMultipleOf);
  }

  return [newHeight, newWidth];
}


export function rgbToRgba(x, y, width, height) {
  for (let h = 0; h < height; ++h) {
    for (let w = 0; w < width; ++w) {
      y.data[h * width * channelsRgba + w * channelsRgba + 0] = Math.round(x[h * width * channelsRgb + w * channelsRgb + 0] * 255);
      y.data[h * width * channelsRgba + w * channelsRgba + 1] = Math.round(x[h * width * channelsRgb + w * channelsRgb + 1] * 255);
      y.data[h * width * channelsRgba + w * channelsRgba + 2] = Math.round(x[h * width * channelsRgb + w * channelsRgb + 2] * 255);
      y.data[h * width * channelsRgba + w * channelsRgba + 3] = 255;
    }
  }
}


export function rgbaUint8ToRgbFloat32(x, y, width, height) {
  for (let h = 0; h < height; ++h) {
    for (let w = 0; w < width; ++w) {
      y[h * width * channelsRgb + w * channelsRgb + 0] = x[h * width * channelsRgba + w * channelsRgba + 0] / 255;
      y[h * width * channelsRgb + w * channelsRgb + 1] = x[h * width * channelsRgba + w * channelsRgba + 1] / 255;
      y[h * width * channelsRgb + w * channelsRgb + 2] = x[h * width * channelsRgba + w * channelsRgba + 2] / 255;
    }
  }
}


export function argmax(heatmaps, height, width, channels) {
  let maxY = Array(channels); // const?
  let maxX = Array(channels); // const?
  let maxValues = Array(channels); // const?

  for (let c = 0; c < channels; ++c) {
    maxValues[c] = heatmaps[c];
    // maxValues[c] = Number.NEGATIVE_INFINITY;
  }

  for (let h = 0; h < height; ++h) {
    for (let w = 0; w < width; ++w) {
      for (let c = 0; c < channels; ++c) {
        if (heatmaps[h * width * channels + w * channels + c] > maxValues[c]) {
          maxValues[c] = heatmaps[h * width * channels + w * channels + c];
          maxY[c] = h;
          maxX[c] = w;
        }
      }
    }
  }

  const result = [];
  for (let c = 0; c < channels; ++c) {
    result.push({ y: maxY[c], x: maxX[c] });
  }

  return result;
}

export function argmaxWithinCircle(heatmaps, length, channels) {
  let maxY = Array(channels); // const?
  let maxX = Array(channels); // const?
  let maxValues = Array(channels); // const?

  for (let c = 0; c < channels; ++c) {
    // maxValues[c] = heatmaps[c];
    maxValues[c] = Number.NEGATIVE_INFINITY;
  }

  // reminder: need to think about this more carefully and make sure it's correct

  let centerY = length / 2;
  let centerX = length / 2;
  let radius = Math.ceil(length / 2);

  for (let h = 0; h < length; ++h) {
    for (let w = 0; w < length; ++w) {
      for (let c = 0; c < channels; ++c) {
        if (Math.hypot(Math.abs(h - centerY), Math.abs(w - centerX)) <= radius) {
          if (heatmaps[h * length * channels + w * channels + c] > maxValues[c]) {
            maxValues[c] = heatmaps[h * length * channels + w * channels + c];
            maxY[c] = h;
            maxX[c] = w;
          }
        }
      }
    }
  }

  const result = [];
  for (let c = 0; c < channels; ++c) {
    result.push({ y: maxY[c], x: maxX[c] });
  }

  return result;
}
