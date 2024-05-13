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

export function round(number, digits) {
  // return +Number.parseFloat(number).toFixed(digits);
  // return Number.parseFloat(number).toFixed(digits);
  return number.toFixed(digits);
}


export function degreesToRadians(deg) {
  return deg * (Math.PI / 180);
}


export function radiansToDegrees(rad) {
  return rad / (Math.PI / 180);
}
