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

export const minDatasetSize = 10; // not sure about these
export const maxMovieSize = 4096; // not sure about these

export const minKeypointCount = 1;
export const maxKeypointCount = 10;
export const defaultKeypointCount = minKeypointCount;

export const colormapOptions = [
  "plasma",
  "viridis"
];
export const colormapDefault = "plasma";

export const datasetSplitOptions = [
  0.6,
  0.75,
  0.9
];
export const datasetSplitDefault = 0.75;

export const maxImageSizeOptions = [
  96,
  128,
  160,
  192,
  224,
  320,
  384,
  448,
  512,
  640,
  768,
  896,
  1024
];
export const maxImageSizeDefault = 384;

export const channelCountOptions = [
  8,
  12,
  16,
  20,
  24,
  28,
  32,
  36,
  40,
  44,
  48,
  52,
  56,
  60,
  64
];
export const channelCountDefault = 24;

export const blockCountOptions = [
  4,
  6,
  8,
  10,
  12
];
export const blockCountDefault = 8;

export const learningRateOptions = [
  0.00001,
  0.0005,
  0.0001,
  0.0005,
  0.001,
  0.005,
  0.01
];
export const learningRateDefault = 0.001;

export const batchSizeOptions = [
  2,
  4,
  8,
  16,
  32
];
export const batchSizeDefault = 8;
