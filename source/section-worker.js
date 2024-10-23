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

export class SectionWorker {
  fileType = null;

  fileHandle = null;
  data = null;


  constructor(fileType) {
    this.fileType = fileType;

    this.initializeData();

    self.addEventListener(
      "message",
      (message) => {
        if (message.data.type === "maybeStartNewFile") {
          this.maybeStartNewFile(message.data.fileHandle);
        }
        else if (message.data.type === "maybeOpenExistingFile") {
          this.maybeOpenExistingFile(message.data.fileHandle);
        }
        else if (message.data.type === "maybeSaveFile") {
          this.maybeSaveFile();
        }
        else if (message.data.type === "maybeCloseFile") {
          this.maybeCloseFile();
        }
        else if (message.data.type === "reset") {
          this.reset();
        }
      }
    );
  }


  async reset() {
    this.fileHandle = null;
    this.data = null;

    this.initializeData();
  }


  initializeData() {
    this.data = {};
    this.data.marigoldFileType = this.fileType;
  }

  sanityCheckData() {
    if (!this.data) {
      throw new Error("");
    }
    if (this.data.marigoldFileType !== `${this.fileType}`) {
      throw new Error("");
    }
  }


  async maybeStartNewFile(fileHandle) {
    try {
      this.fileHandle = fileHandle;

      this.initializeData();
      await this.saveFileUnchecked();

      self.postMessage({ type: "startNewFileSuccess" });
    }
    catch {
      this.fileHandle = null;
      this.data = null;
      self.postMessage({ type: "startNewFileFailure" });
    }
  }

  async maybeOpenExistingFile(fileHandle) {
    try {
      this.fileHandle = fileHandle;

      const file = await fileHandle.getFile();
      this.data = JSON.parse(await file.text());
      this.sanityCheckData();

      self.postMessage({ type: "openExistingFileSuccess" });
    }
    catch {
      this.fileHandle = null;
      this.data = null;
      self.postMessage({ type: "openExistingFileFailure" });
    }
  }

  async maybeSaveFile() {
    if (await this.fileHandle.queryPermission({ mode: "readwrite" }) !== "granted") {
      self.postMessage({ type: "saveFileFailure", reason: "permission", fileHandle: this.fileHandle });
    }
    else {
      try {
        await this.saveFileUnchecked();
        self.postMessage({ type: "saveFileSuccess" });
      }
      catch {
        self.postMessage({ type: "saveFileFailure", reason: "unknown", fileHandle: this.fileHandle });
      }
    }
  }

  async saveFileUnchecked() {
    const writable = await this.fileHandle.createWritable();
    await writable.write(JSON.stringify(this.data));
    await writable.close();
  }


  maybeCloseFile() {
    try {
      this.reset();
      self.postMessage({ type: "closeFileSuccess" });
    }
    catch {
      self.postMessage({ type: "closeFileFailure" });
    }
  }
}
