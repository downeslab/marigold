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

export class BrowserCompatibility {
  #compatible = null;

  constructor() {
    this.#compatible = true;
    if (!window.Worker) {
      this.#compatible = false;
    }
    else if (!window.VideoDecoder) {
      this.#compatible = false;
    }
    else if (!window.showSaveFilePicker) {
      this.#compatible = false;
    }
    else if (!window.showOpenFilePicker) {
      this.#compatible = false;
    }
  }

  seemsOkay() {
    return this.#compatible;
  }

  showDialog() {
    const dialogCloseButton = document.querySelector("#home-browser-compatibility-dialog-close-button");
    dialogCloseButton.addEventListener(
      "click",
      (event) => {
        const dialog = document.querySelector("#home-browser-compatibility-dialog");
        dialog.close();
      },
      { once: true }
    );

    const dialog = document.querySelector("#home-browser-compatibility-dialog");
    dialog.showModal();
  }
}
