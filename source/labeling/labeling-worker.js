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

import { SectionWorker } from "../section-worker.js";

import { minKeypointCount, maxKeypointCount, defaultKeypointCount } from "../constants.js";
import { MovieReader } from "../movie-reader.js";


class LabelingWorker extends SectionWorker {
  movieReader = null;

  constructor() {
    super("dataset");

    self.addEventListener(
      "message",
      (message) => {
        if (message.data.type === "keypointCount") {
          this.data.keypointCount = message.data.keypointCount;
        }
        else if (message.data.type === "maybeLoadMovie") {
          this.maybeLoadMovie(message.data.fileHandle);
        }
        else if (message.data.type === "frameRequest") {
          this.movieReader.seekFrame(message.data.index);
        }
        else if (message.data.type === "addLabel") {
          const fileReader = new FileReaderSync();
          const imageData = fileReader.readAsDataURL(message.data.image);

          this.data.labels.push(
            {
              metadata: {
                filename: message.data.filename,
                frameNumber: message.data.frameNumber,
                arena: message.data.arena,
              },
              label: message.data.coordinates,
              image: imageData
            }
          );
        }
        else if (message.data.type === "removeLabel") {
          const index = message.data.index;
          this.data.labels.splice(index, 1);
        }
        else if (message.data.type === "boop") {
          this.boop(message.data.directoryHandle);
        }
      }
    );
  }

  initializeData() {
    super.initializeData();
    this.data.labels = [];
    this.data.keypointCount = defaultKeypointCount;
  }

  sanityCheckData() {
    super.sanityCheckData();
    if (this.data.keypointCount < minKeypointCount || this.data.keypointCount > maxKeypointCount) {
      throw new Error();
    }
  }

  async maybeOpenExistingFile(fileHandle) {
    try {
      this.fileHandle = fileHandle;

      const file = await fileHandle.getFile();
      this.data = JSON.parse(await file.text());
      this.sanityCheckData();

      const keypointCount = this.data.keypointCount;

      const blobs = [];
      for (const label of this.data.labels) {
        const imageData = label.image;
        const imageResult = await fetch(imageData);
        const blob = await imageResult.blob();
        blobs.push(blob);
      }

      const coordinates = [];
      for (const label of this.data.labels) {
        coordinates.push(label.label);
      }

      const metadata = [];
      for (const label of this.data.labels) {
        metadata.push(
          {
            filename: label.metadata.filename,
            frameNumber: label.metadata.frameNumber,
            arena: label.metadata.arena
          }
        );
      }

      self.postMessage(
        {
          type: "openExistingFileSuccess",
          keypointCount: keypointCount,
          blobs: blobs,
          coordinates: coordinates,
          metadata: metadata
        }
      );
    }
    catch {
      this.fileHandle = null;
      this.data = null;
      self.postMessage({ type: "openExistingFileFailure" });
    }
  }

  onFrameReady(data) {
    self.postMessage({ type: "frameReady", frame: data.frame, frameNumber: data.frameNumber });
  }

  async maybeLoadMovie(fileHandle) {
    const file = await fileHandle.getFile();
    const data = await file.arrayBuffer();
    const buffer = new ArrayBuffer(data.byteLength);
    new Uint8Array(buffer).set(new Uint8Array(data));

    let movieReader = null;
    try {
      movieReader = new MovieReader(buffer, this.onFrameReady);
    }
    catch (error) {
      self.postMessage({ type: "loadMovieFailure", filename: fileHandle.name });
    }

    if (movieReader) {
      this.movieReader = movieReader;
      self.postMessage({ type: "loadMovieSuccess", filename: fileHandle.name, numFrames: this.movieReader.mp4Parser.numFrames, frameWidth: this.movieReader.mp4Parser.frameWidth, frameHeight: this.movieReader.mp4Parser.frameHeight });
    }
  }
}


new LabelingWorker();
