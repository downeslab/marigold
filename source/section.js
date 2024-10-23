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

export class Section {
  static unsavedMessage = "Some changes since last save.";
  static savingMessage = "Saving…";
  static savedMessage = "No changes since last save.";

  fileType = null;
  worker = null;

  unsavedChanges = null;

  constructor(fileType, prefix, worker) {
    this.fileType = fileType;
    this.prefix = prefix;
    this.worker = new Worker(`./${this.prefix}/${this.prefix}-worker.js`, { type: "module" });

    this.worker.addEventListener(
      "message",
      (message) => {
        if (message.data.type === "startNewFileSuccess") {
          this.onStartNewFileSuccess();
        }
        else if (message.data.type === "startNewFileFailure") {
          this.onStartNewFileFailure();
        }
        else if (message.data.type === "openExistingFileSuccess") {
          this.onOpenExistingFileSuccess(message);
        }
        else if (message.data.type === "openExistingFileFailure") {
          this.onOpenExistingFileFailure();
        }
        else if (message.data.type === "saveFileSuccess") {
          this.onSaveFileSuccess();
        }
        else if (message.data.type === "saveFileFailure") {
          this.onSaveFileFailure(message.data.reason, message.data.fileHandle);
        }
        else if (message.data.type === "closeFileSuccess") {
          this.onCloseFileSuccess();
        }
        else if (message.data.type === "closeFileFailure") {
          this.onCloseFileFailure();
        }
      }
    );

    document.querySelector(`#save-${this.fileType}-button`).addEventListener(
      "click",
      (event) => {
        this.maybeSaveFile();
      }
    );

    document.querySelector(`#close-${this.fileType}-button`).addEventListener(
      "click",
      (event) => {
        this.showCloseFileDialog();
      }
    );
  }


  static disableWorkflowButtons() {
    document.querySelector("#home-start-new-dataset-button").setAttribute("disabled", "");
    document.querySelector("#home-open-existing-dataset-button").setAttribute("disabled", "");
    document.querySelector("#home-start-new-model-button").setAttribute("disabled", "");
    document.querySelector("#home-open-existing-model-button").setAttribute("disabled", "");
    document.querySelector("#home-start-new-analysis-button").setAttribute("disabled", "");
    document.querySelector("#home-open-existing-analysis-button").setAttribute("disabled", "");
  }

  static enableWorkflowButtons() {
    document.querySelector("#home-start-new-dataset-button").removeAttribute("disabled");
    document.querySelector("#home-open-existing-dataset-button").removeAttribute("disabled");
    document.querySelector("#home-start-new-model-button").removeAttribute("disabled");
    document.querySelector("#home-open-existing-model-button").removeAttribute("disabled");
    document.querySelector("#home-start-new-analysis-button").removeAttribute("disabled");
    document.querySelector("#home-open-existing-analysis-button").removeAttribute("disabled");
  }


  reset() {
    document.querySelector(`#${this.fileType}-header-heading`).textContent = "";
    document.querySelector(`#${this.fileType}-header-status`).textContent = "";

    this.unsavedChanges = null;
  }


  enter() {
    document.querySelector("#home").style.display = "none";
    document.querySelector(`#${this.prefix}`).style.display = "flex";
    window.scrollTo(0, 0);
  }

  leave() {
    this.reset();
    document.querySelector(`#${this.prefix}`).style.display = "none";
    document.querySelector("#home").style.display = "flex";
    window.scrollTo(0, 0);

    Section.enableWorkflowButtons();
  }


  async maybeStartNewFile() {
    Section.disableWorkflowButtons();

    let fileHandle = null;
    try {
      fileHandle = await window.showSaveFilePicker(
        {
          id: "startNewFile",
          startIn: "documents",
          types: [
            {
              description: "Marigold files",
              accept: {
                "application/json": [".marigold"]
              }
            }
          ],
          suggestedName: `Untitled ${this.fileType[0].toUpperCase()}${this.fileType.slice(1)}.marigold`
        }
      );
    }
    catch {
    }

    if (fileHandle) {
      this.worker.postMessage({ type: "maybeStartNewFile", fileHandle: fileHandle });
      document.querySelector(`#${this.fileType}-header-heading`).textContent = fileHandle.name;
    }
    else {
      Section.enableWorkflowButtons();
    }
  }

  onStartNewFileSuccess() {
    this.showStatus(Section.savedMessage);
    this.unsavedChanges = false;
    this.enter();
  }

  onStartNewFileFailure() {
    Section.enableWorkflowButtons();
  }


  async maybeOpenExistingFile() {
    Section.disableWorkflowButtons();

    let fileHandle = null;
    try {
      [fileHandle] = await window.showOpenFilePicker(
        {
          id: "openExistingFile",
          startIn: "documents",
          types: [
            {
              description: "Marigold files",
              accept: {
                "application/json": [".marigold"]
              }
            }
          ],
          mode: "readWrite"
        }
      );
    }
    catch {
    }

    if (fileHandle) {
      this.worker.postMessage({ type: "maybeOpenExistingFile", fileHandle: fileHandle });
      document.querySelector(`#${this.fileType}-header-heading`).textContent = fileHandle.name;
    }
    else {
      Section.enableWorkflowButtons();
    }
  }

  async onOpenExistingFileSuccess(message) {
    this.showStatus(Section.savedMessage);
    this.unsavedChanges = false;
    this.enter();
  }

  onOpenExistingFileFailure() {
    Section.enableWorkflowButtons();

    document.querySelector("#generic-dialog-heading").textContent = "Error opening file";
    document.querySelector("#generic-dialog-blurb").textContent = `The file could not be opened or is not a valid ${this.fileType} file.`;

    document.querySelector("#generic-dialog-close-button").addEventListener(
      "click",
      (event) => { document.querySelector("#generic-dialog").close(); },
      { once: true }
    );

    document.querySelector("#generic-dialog").showModal();
  }


  maybeSaveFile() {
    document.querySelector(`#close-${this.fileType}-button`).setAttribute("disabled", "");
    document.querySelector(`#save-${this.fileType}-button`).setAttribute("disabled", "");

    this.showStatus(Section.savingMessage);
    this.worker.postMessage({ type: "maybeSaveFile" });
  }

  onSaveFileSuccess() {
    document.querySelector(`#close-${this.fileType}-button`).removeAttribute("disabled");
    document.querySelector(`#save-${this.fileType}-button`).removeAttribute("disabled");

    this.showStatus(Section.savedMessage);
    this.unsavedChanges = false;
  }

  async onSaveFileFailure(reason, fileHandle) {
    if (reason === "permission") {
      const options = {};
      options.mode = "readwrite";
      await fileHandle.requestPermission(options);
      if (await fileHandle.queryPermission(options) === "granted") {
        this.maybeSaveFile();
      }
      else {
        document.querySelector(`#close-${this.fileType}-button`).removeAttribute("disabled");
        document.querySelector(`#save-${this.fileType}-button`).removeAttribute("disabled");

        document.querySelector("#generic-dialog-heading").textContent = "Error saving file";
        document.querySelector("#generic-dialog-blurb").textContent = "Please try again, being sure to grant Marigold permission to edit the file.";

        document.querySelector("#generic-dialog-close-button").addEventListener(
          "click",
          (event) => { document.querySelector("#generic-dialog").close(); },
          { once: true }
        );

        document.querySelector("#generic-dialog").showModal();
      }
    }
    else {
      document.querySelector(`#close-${this.fileType}-button`).removeAttribute("disabled");
      document.querySelector(`#save-${this.fileType}-button`).removeAttribute("disabled");

      document.querySelector("#generic-dialog-heading").textContent = "Error saving file";
      document.querySelector("#generic-dialog-blurb").textContent = "The file could not be saved. Maybe try again?";

      document.querySelector("#generic-dialog-close-button").addEventListener(
        "click",
        (event) => { document.querySelector("#generic-dialog").close(); },
        { once: true }
      );

      document.querySelector("#generic-dialog").showModal();
    }

    this.showStatus(this.unsavedChanges ? Section.unsavedMessage : Section.savedMessage);
  }

  showCloseFileDialog() {
    const cancelButton = document.querySelector(`#close-file-dialog-cancel-button`);
    const confirmButton = document.querySelector(`#close-file-dialog-confirm-button`);

    let cancel = (event) => {
      cancelButton.setAttribute("disabled", "");
      confirmButton.setAttribute("disabled", "");

      cancelButton.removeEventListener("click", cancel);
      confirmButton.removeEventListener("click", confirm);

      document.querySelector("#close-file-dialog").close();

      cancelButton.removeAttribute("disabled");
      confirmButton.removeAttribute("disabled");
    };

    let confirm = (event) => {
      cancelButton.setAttribute("disabled", "");
      confirmButton.setAttribute("disabled", "");

      cancelButton.removeEventListener("click", cancel);
      confirmButton.removeEventListener("click", confirm);

      this.maybeCloseFile();
      document.querySelector("#close-file-dialog").close();

      cancelButton.removeAttribute("disabled");
      confirmButton.removeAttribute("disabled");
    };

    cancelButton.addEventListener(
      "click",
      cancel
    );

    confirmButton.addEventListener(
      "click",
      confirm
    );

    document.querySelector("#close-file-dialog").showModal();
  }


  maybeCloseFile() {
    this.worker.postMessage({ type: "maybeCloseFile" });
  }

  onCloseFileSuccess() {
    this.leave();
    this.reset();
  }

  onCloseFileFailure() {
  }


  showStatus(message) {
    document.querySelector(`#${this.fileType}-header-status`).textContent = message;
  }
}
