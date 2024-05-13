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

import { minKeypointCount, maxKeypointCount, defaultKeypointCount } from "../core/constants.js";
import { MovieReader } from "../movie-reader.js";


class LabelingWorker extends SectionWorker {
  movieReader = null;

  constructor() {
    super("dataset");

    self.addEventListener(
      "message",
      (message) => {
        //
        //
        //
        if (message.data.type === "keypointCount") {
          this.data.keypointCount = message.data.keypointCount;
        }
        else if (message.data.type === "maybeLoadMovie") {
          this.maybeLoadMovie(message.data.fileHandle);
          // this.maybeLoadMovie(message.data.uri);
        }
        else if (message.data.type === "frameRequest") {
          this.movieReader.seekFrame(message.data.index);
        }
        else if (message.data.type === "addLabel") {
          // console.log(message.data.image);
          const fileReader = new FileReaderSync();
          // console.log(fileReader);
          const imageData = fileReader.readAsDataURL(message.data.image);
          // console.log(imageData);

          // this.data.labels.push({ coordinates: message.data.coordinates, image: URL.createObjectURL(message.data.image) });
          this.data.labels.push(
            {
              metadata: {
                filename: message.data.filename,
                frameNumber: message.data.frameNumber,
                arena: message.data.arena,
              },
              label: message.data.coordinates, // "keypointCoordinates"?
              image: imageData
            }
          );
          // console.log(this.data.labels);
        }
        else if (message.data.type === "removeLabel") {
          const index = message.data.index;
          this.data.labels.splice(index, 1);
        }
        //
        //
        //
        else if (message.data.type === "boop") {
          this.boop(message.data.directoryHandle);
        }
        //
        //
        //
      }
    );
  }

  initializeData() {
    // console.log("initializing data");
    super.initializeData();
    this.data.labels = [];
    // this.data.keypointCount = null;
    this.data.keypointCount = defaultKeypointCount;
  }

  sanityCheckData() {
    super.sanityCheckData();
    if (this.data.keypointCount < minKeypointCount || this.data.keypointCount > maxKeypointCount) {
      throw new Error();
    }
  }

  //
  //
  //

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

  //
  //
  //


  onFrameReady(data) {
    // console.log("onFrameReady");
    self.postMessage({ type: "frameReady", frame: data.frame, frameNumber: data.frameNumber });
  }

  async maybeLoadMovie(fileHandle) {
    const file = await fileHandle.getFile();
    const data = await file.arrayBuffer();
    const buffer = new ArrayBuffer(data.byteLength);
    new Uint8Array(buffer).set(new Uint8Array(data));

    let movieReader = null;
    try {
      // console.log("movieReader?");
      movieReader = new MovieReader(buffer, this.onFrameReady);
      // console.log("movieReader!");
    }
    catch (error) {
      // console.log(error);
      self.postMessage({ type: "loadMovieFailure", filename: fileHandle.name });
    }

    if (movieReader) {
      this.movieReader = movieReader;
      self.postMessage({ type: "loadMovieSuccess", filename: fileHandle.name, numFrames: this.movieReader.mp4Parser.numFrames, frameWidth: this.movieReader.mp4Parser.frameWidth, frameHeight: this.movieReader.mp4Parser.frameHeight });
    }
  }

  // async maybeLoadMovie(uri) {
  //   const response = await fetch(uri);
  //   // console.log(response);
  //   const data = await response.arrayBuffer();
  //   // console.log(data);
  //   const buffer = new ArrayBuffer(data.byteLength);
  //   new Uint8Array(buffer).set(new Uint8Array(data));

  //   let movieReader = null;
  //   try {
  //     // console.log("movieReader?");
  //     movieReader = new MovieReader(buffer, this.onFrameReady);
  //     // console.log("movieReader!");
  //   }
  //   catch (error) {
  //     // console.log(error);
  //     self.postMessage({ type: "loadMovieFailure", filename: "boop" });
  //   }

  //   if (movieReader) {
  //     this.movieReader = movieReader;
  //     self.postMessage({ type: "loadMovieSuccess", filename: "boop", numFrames: this.movieReader.mp4Parser.numFrames, frameWidth: this.movieReader.mp4Parser.frameWidth, frameHeight: this.movieReader.mp4Parser.frameHeight });
  //   }
  // }

  //
  //
  //

  async boop(directoryHandle) {
    for await (const entry of directoryHandle.values()) {
      // console.log(entry.name);

      if (entry.name.includes(".png")) {
        // console.log("png");
        const imageFileHandle = await directoryHandle.getFileHandle(entry.name, { create: false });
        const imageFile = await imageFileHandle.getFile();
        const imageBitmap = await createImageBitmap(imageFile, { colorSpaceConversion: "none" });

        const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
        const context = canvas.getContext("2d");
        context.drawImage(imageBitmap, 0, 0, imageBitmap.width, imageBitmap.height);
        // const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
        // const imageData = canvas.toDataURL("image/png");
        const imageDataBlob = await canvas.convertToBlob();
        // const imageData = await URL.createObjectURL(imageDataBlob);
        // console.log(imageDataBlob);
        const fileReader = new FileReaderSync();
        // console.log(fileReader);
        const imageData = fileReader.readAsDataURL(imageDataBlob);
        // console.log(imageData);

        const labelFileHandle = await directoryHandle.getFileHandle(entry.name.replace(".png", ".json"), { create: false });
        const labelFile = await labelFileHandle.getFile();
        const labelText = await labelFile.text();
        const labelData = JSON.parse(labelText);

        const metadata = {};
        metadata.filename = entry.name.split("_frame")[0];
        metadata.frameNumber = parseInt(entry.name.split("_frame_")[1].split(".png")[0]);
        metadata.arena = { "x": 0, "y": 0, "width": imageBitmap.width, "height": imageBitmap.height };

        this.data.labels.push({ image: imageData, label: labelData, metadata: metadata });

        const keypointCount = labelData.length;
        this.data.keypointCount = keypointCount;
      }
    }

    // console.log("booped!");
  }
}


new LabelingWorker();
