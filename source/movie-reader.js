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

class MP4Parser {
  buffer = null;
  dataView = null;

  ftpyOffset = null;
  mdatOffset = null;
  moovOffset = null;
  mvhdOffset = null;
  trakOffset = null;
  tkhdOffset = null;
  mdiaOffset = null;
  mdhdOffset = null;
  hdlrOffset = null;
  minfOffset = null;
  stblOffset = null;
  stsdOffset = null;
  sttsOffset = null;
  cttsOffset = null;
  stssOffset = null;
  stscOffset = null;
  stszOffset = null;
  stcoOffset = null;

  numFrames = null;
  frameWidth = null;
  frameHeight = null;
  syncFrames = new Set(); // null?
  entrySizes = []; // null?

  codecProfileInfo = null;
  infoNeededForDecoder = null;
  samplesPerChunk = [];
  chunkOffsets = [];
  frameOffsets = [];

  constructor(buffer) {
    this.buffer = buffer;
    this.dataView = new DataView(buffer);

    this.parseFtyp();
    this.parseMdat();

    this.parseMoov();
    this.parseMvhd();

    this.parseTrak();
    this.parseTkhd();
    this.parseMdia();
    this.parseMdhd();
    this.parseHdlr();

    this.parseMinf();
    this.parseStbl();
    this.parseStsd();
    this.parseStts();
    this.parseCtts();
    this.parseStss();
    this.parseStco();
    this.parseStsc();
    this.parseStsz();

    let frameIndex = 0;
    let chunkOffset = 0;
    for (let i = 0; i < this.samplesPerChunk.length; ++i) {
      // chunkOffset = this.chunkOffsets[i];
      chunkOffset = Number(this.chunkOffsets[i]);
      let sampleOffset = 0;
      for (let j = 0; j < this.samplesPerChunk[i]; ++j) {
        this.frameOffsets.push(chunkOffset + sampleOffset);
        sampleOffset += this.entrySizes[frameIndex];
        ++frameIndex;
      }
    }
  }

  findOffset(query, startFrom = 0) {
    // const test = new Uint8Array(this.buffer);

    // let offset = null;
    // for (let i = 0; i < test.byteLength - 3; ++i) {
    //   if (test[i + 0] === query.charCodeAt(0) && test[i + 1] === query.charCodeAt(1)
    //     && test[i + 2] === query.charCodeAt(2) && test[i + 3] === query.charCodeAt(3)) {
    //     offset = i + 4;
    //     break;
    //   }
    // }

    // return offset;

    const test = new Uint8Array(this.buffer);

    let offset = null;
    // reminder: should i be using query.charCodeAt or query.codePointAt?
    for (let i = startFrom; i < test.byteLength - 3; ++i) {
      if (test[i + 0] === query.charCodeAt(0) && test[i + 1] === query.charCodeAt(1)
        && test[i + 2] === query.charCodeAt(2) && test[i + 3] === query.charCodeAt(3)) {
        offset = i + 4;
        break;
      }
    }

    return offset;
  }

  parseFtyp() {
    this.ftypOffset = this.findOffset("ftyp");

    let offset = this.ftypOffset;

    const majorBrand = this.dataView.getUint32(offset, false);
    offset += 4;

    const minorBrand = this.dataView.getUint32(offset, false);
    offset += 4;
  }

  parseMdat() {
    // this.mdatOffset = this.findOffset("mdat");
    // // reminder: there can be more than one (or even zero) mdat boxes per file

    let offset = 0;
    while (offset < this.buffer.byteLength - 3 && offset !== null) {
      // console.log("looping");
      offset = this.findOffset("mdat", offset);
      if (offset !== null) {
        this.mdatOffset = offset;
        // console.log("mdat:", this.mdatOffset);
      }
    }
  }

  parseMoov() {
    this.moovOffset = this.findOffset("moov");
  }

  parseMvhd() {
    this.mvhdOffset = this.findOffset("mvhd");

    let offset = this.ftypOffset;

    const version = this.dataView.getUint8(offset, false);
    offset += 4;

    let timescale = null;
    let duration = null;

    if (version === 0) {
      offset += 4; // creation_time
      offset += 4; // modification_time
      timescale = this.dataView.getUint32(offset, false);
      offset += 4;
      duration = this.dataView.getUint32(offset, false);
      offset += 4;
    }
    else if (version === 1) {
      offset += 8; // creation_time
      offset += 8; // modification_time
      timescale = this.dataView.getUint32(offset, false);
      offset += 4;
      duration = this.dataView.getBigUint64(offset, false);
      offset += 8;
    }

    const rate = this.dataView.getInt32(offset, false) / 65536.0;
    offset += 4;

    offset += 2; // volume
    offset += 2; // reserved
    offset += 2 * 4; // reserved

    let matrix = [];
    for (let i = 0; i < 9; ++i) {
      matrix.push(this.dataView.getInt32(offset, false) / 65536.0);
      offset += 4;
    }
  }

  parseTrak() {
    // this.trakOffset = this.findOffset("trak");

    // reminder: this will probably fail if there are no video tracks, possibly by looping forever
    let offset = 0;
    while (offset < this.buffer.byteLength - 3) {
      // console.log("looping");
      offset = this.findOffset("trak", offset);

      //

      let hdlrOffset = this.findOffset("hdlr", offset);

      let offset_ = hdlrOffset;

      offset_ += 4; // version
      offset_ += 4; // pre_defined

      const handlrType = this.dataView.getUint32(offset_, false);
      offset_ += 4;

      let isVideo = false;
      if (handlrType === 1986618469) {
        // console.log("isVideo: true");
        isVideo = true;
        this.trakOffset = offset;
        // console.log("breaking");
        break;
      }
      else {
        // console.log("isVideo: false");
        isVideo = false;
      }
    }
  }

  parseTkhd() {
    this.tkhdOffset = this.findOffset("tkhd", this.trakOffset);

    let offset = this.tkhdOffset;

    const version = this.dataView.getUint8(offset, false);
    offset += 4;

    let trackId = null;
    let duration = null;

    if (version === 0) {
      offset += 4; // creation_time
      offset += 4; // modification_time
      trackId = this.dataView.getUint32(offset, false);
      offset += 4;
      offset += 4; // reserved
      duration = this.dataView.getUint32(offset, false);
      offset += 4;
    }
    else if (version === 1) {
      offset += 8; // creation_time
      offset += 8; // modification_time
      trackId = this.dataView.getUint32(offset, false);
      offset += 4;
      offset += 4; // reserved
      duration = this.dataView.getBigUint64(offset, false);
      offset += 8;
    }

    offset += 2 * 4; // reserved
    offset += 2; // layer
    offset += 2; // alternate_group
    offset += 2; // volume
    offset += 2; // reserved

    let matrix = [];
    for (let i = 0; i < 9; ++i) {
      matrix.push(this.dataView.getInt32(offset, false) / 65536.0);
      offset += 4;
    }

    const width = this.dataView.getInt32(offset, false) / 65536.0;
    offset += 4;

    const height = this.dataView.getInt32(offset, false) / 65536.0;
    offset += 4;

    this.frameWidth = width;
    this.frameHeight = height;
  }

  parseMdia() {
    this.mdiaOffset = this.findOffset("mdia", this.trakOffset);
  }

  parseMdhd() {
    this.mdhdOffset = this.findOffset("mdhd", this.trakOffset);
  }

  parseHdlr() {
    this.hdlrOffset = this.findOffset("hdlr", this.trakOffset);

    let offset = this.hdlrOffset;

    offset += 4; // version
    offset += 4; // pre_defined

    const handlrType = this.dataView.getUint32(offset, false);
    offset += 4;

    if (handlrType === 1986618469) {
      // console.log("isVideo: true");
    }
    else {
      // console.log("isVideo: false");
    }

    //

    // let isVideo = false;
    // this.hdlrOffset = this.findOffset("hdlr");

    // let offset = this.hdlrOffset;

    // offset += 4; // version
    // offset += 4; // pre_defined

    // let handlrType = this.dataView.getUint32(offset, false);
    // offset += 4;

    // if (handlrType === 1986618469) {
    //   isVideo = true;
    //   // console.log("isVideo: true");
    // }
    // else {
    //   isVideo = false;
    //   // console.log("isVideo: false");
    // }

    // // reminder: this will loop forever if no video is found; should check when offset reaches end of file and break
    // while (!isVideo) {
    //   this.hdlrOffset = this.findOffset("hdlr", this.hdlrOffset);

    //   offset = this.hdlrOffset;

    //   offset += 4; // version
    //   offset += 4; // pre_defined

    //   handlrType = this.dataView.getUint32(offset, false);
    //   offset += 4;

    //   if (handlrType === 1986618469) {
    //     isVideo = true;
    //     // console.log("isVideo: true");
    //   }
    //   else {
    //     isVideo = false;
    //     // console.log("isVideo: false");
    //   }
    // }
  }

  parseMinf() {
    // this.minfOffset = this.findOffset("minf");
    this.minfOffset = this.findOffset("minf", this.trakOffset);
  }

  parseStbl() {
    // this.stblOffset = this.findOffset("stbl");
    this.stblOffset = this.findOffset("stbl", this.trakOffset);
  }

  parseStsd() {
    // this.stsdOffset = this.findOffset("stsd");
    this.stsdOffset = this.findOffset("stsd", this.trakOffset);

    const test = new Uint8Array(this.buffer);

    let offset = this.stsdOffset;

    const version = this.dataView.getUint8(offset, false);
    offset += 4;

    const entryCount = this.dataView.getUint32(offset, false);
    offset += 4;

    offset += 4; // box size
    offset += 4; // box name

    for (let i = 0; i < entryCount; ++i) {
      offset += 6 * 1; // reserved
      offset += 2; // data_reference_index

      offset += 2; // pre_defined
      offset += 2; // reserved
      offset += 3 * 4; // pre_defined

      const width = this.dataView.getUint16(offset, false);
      offset += 2;

      const height = this.dataView.getUint16(offset, false);
      offset += 2;

      offset += 4; // horizresolution
      offset += 4; // vertresolution
      offset += 4; // reserved

      const frameCount = this.dataView.getUint16(offset, false);
      offset += 2;

      const compressorNameLength = this.dataView.getUint8(offset, false);
      offset += 1;

      const compressorName = new TextDecoder().decode(test.subarray(offset, offset + compressorNameLength));
      offset += compressorNameLength;
      offset += 32 - (compressorNameLength + 1);

      const depth = this.dataView.getUint16(offset, false);
      offset += 2;

      offset += 2; // pre_defined

      const decoderInfoSize = this.dataView.getUint32(offset, false);
      offset += 4;

      const decoderInfoName = this.dataView.getUint32(offset, false);
      offset += 4;
      // console.log("decoderInfoName:", decoderInfoName);

      const decoderInfo = test.subarray(offset, offset + decoderInfoSize - 2 * 4);
      offset += decoderInfoSize;

      this.infoNeededForDecoder = decoderInfo;
      // console.log("decoderInfo:", decoderInfo);

      if (decoderInfoName === 1635148611) {
        this.codecProfileInfo = "avc1.";
        this.codecProfileInfo += parseInt(decoderInfo.subarray(1, 2)).toString(16).padStart(2, "0");
        this.codecProfileInfo += parseInt(decoderInfo.subarray(2, 3)).toString(16).padStart(2, "0");
        this.codecProfileInfo += parseInt(decoderInfo.subarray(3, 4)).toString(16).padStart(2, "0");
      }
    }
  }

  parseStts() {
    // this.sttsOffset = this.findOffset("stts");
    this.sttsOffset = this.findOffset("stts", this.trakOffset);

    let offset = this.sttsOffset;

    offset += 4; // version

    const entryCount = this.dataView.getUint32(offset, false);
    offset += 4;

    for (let i = 0; i < entryCount; ++i) {
      const sampleCount = this.dataView.getUint32(offset, false);
      offset += 4;

      const sampleDelta = this.dataView.getUint32(offset, false);
      offset += 4;
    }
  }

  parseCtts() {
    // // this.cttsOffset = this.findOffset("ctts");
    // this.cttsOffset = this.findOffset("ctts", this.trakOffset);
    // console.log(this.cttsOffset);

    // let offset = this.cttsOffset;

    // const version = this.dataView.getUint8(offset, false);
    // offset += 4;

    // const entryCount = this.dataView.getUint32(offset, false);
    // offset += 4;

    // for (let i = 0; i < entryCount; ++i) {
    //   if (version === 0) {
    //     const sampleCount = this.dataView.getUint32(offset, false);
    //     offset += 4;

    //     const sampleOffset = this.dataView.getUint32(offset, false);
    //     offset += 4;
    //   }
    //   else if (version === 1) {
    //     const sampleCount = this.dataView.getUint32(offset, false);
    //     offset += 4;

    //     const sampleOffset = this.dataView.getInt32(offset, false);
    //     offset += 4;
    //   }
    // }
  }

  parseStss() {
    // this.stssOffset = this.findOffset("stss");
    this.stssOffset = this.findOffset("stss", this.trakOffset);

    let offset = this.stssOffset;

    offset += 4; // version

    const entryCount = this.dataView.getUint32(offset, false);
    offset += 4;

    for (let i = 0; i < entryCount; ++i) {
      const sampleNumber = this.dataView.getUint32(offset, false);
      offset += 4;

      this.syncFrames.add(sampleNumber - 1);
    }
  }

  parseStco() {
    this.stcoOffset = this.findOffset("stco", this.trakOffset);
    // console.log(this.stcoOffset);
    if (this.stcoOffset !== null) {
      let offset = this.stcoOffset;

      offset += 4; // version

      const entryCount = this.dataView.getUint32(offset, false);
      offset += 4;

      for (let i = 0; i < entryCount; ++i) {
        const chunkOffset = this.dataView.getUint32(offset, false);
        offset += 4;

        this.chunkOffsets.push(chunkOffset);
      }
    }
    else {
      this.stcoOffset = this.findOffset("co64", this.trakOffset);
      // console.log(this.stcoOffset);

      let offset = this.stcoOffset;

      offset += 4; // version

      const entryCount = this.dataView.getUint32(offset, false);
      offset += 4;

      // console.log("entryCount:", entryCount);

      for (let i = 0; i < entryCount; ++i) {
        const chunkOffset = this.dataView.getBigUint64(offset, false);
        // console.log("chunkOffset:", chunkOffset);
        offset += 8;

        this.chunkOffsets.push(chunkOffset);
      }
      // console.log("this.chunkOffsets:", this.chunkOffsets);
    }
  }

  parseStsc() {
    // this.stscOffset = this.findOffset("stsc");
    this.stscOffset = this.findOffset("stsc", this.trakOffset);

    let offset = this.stscOffset;

    const version = this.dataView.getUint8(offset, false);
    offset += 4;

    const entryCount = this.dataView.getUint32(offset, false);
    offset += 4;

    if (entryCount === 1) {
      for (let i = 0; i < entryCount; ++i) {
        const firstChunk = this.dataView.getUint32(offset, false);
        offset += 4;

        const samplesPerChunk = this.dataView.getUint32(offset, false);
        offset += 4;

        const sampleDescriptionIndex = this.dataView.getUint32(offset, false);
        offset += 4;

        for (let j = 0; j < this.chunkOffsets.length; ++j) {
          this.samplesPerChunk.push(samplesPerChunk);
        }
      }
    }
    else {
      for (let i = 0; i < entryCount - 1; ++i) {
        const firstChunk = this.dataView.getUint32(offset, false);
        offset += 4;

        const samplesPerChunk = this.dataView.getUint32(offset, false);
        offset += 4;

        const sampleDescriptionIndex = this.dataView.getUint32(offset, false);
        offset += 4;

        const nextFirstChunk = this.dataView.getUint32(offset, false);
        // reminder: don't add to offset here

        for (let j = 0; j < nextFirstChunk - firstChunk; ++j) {
          this.samplesPerChunk.push(samplesPerChunk);
        }
      }

      const firstChunk = this.dataView.getUint32(offset, false);
      offset += 4;

      const samplesPerChunk = this.dataView.getUint32(offset, false);
      offset += 4;

      const sampleDescriptionIndex = this.dataView.getUint32(offset, false);
      offset += 4;

      for (let j = firstChunk - 1; j < this.chunkOffsets.length; ++j) {
        this.samplesPerChunk.push(samplesPerChunk);
      }
    }
  }

  parseStsz() {
    // this.stszOffset = this.findOffset("stsz");
    this.stszOffset = this.findOffset("stsz", this.trakOffset);

    let offset = this.stszOffset;

    offset += 4; // version

    const sampleSize = this.dataView.getUint32(offset, false);
    offset += 4;

    const sampleCount = this.dataView.getUint32(offset, false);
    offset += 4;

    this.numFrames = sampleCount;

    if (sampleSize === 0) {
      for (let i = 0; i < sampleCount; ++i) {
        const entrySize = this.dataView.getUint32(offset, false);
        offset += 4;

        this.entrySizes.push(entrySize);
      }
    }
  }
}


export class MovieReader {
  // reminder: need to detect when codec isn't available in browser

  decoderConfig = null;
  decoder = null;
  buffer = null;
  mp4Parser = null;

  samplesPushedToDecoder = 0;
  samplesPulledFromDecoder = 0;
  lastFrameRendered = -1;
  cachedFrames = {};
  targetFrame = -1;

  callback = null;

  constructor(buffer, callback) {
    this.callback = callback;
    this.buffer = buffer;

    this.mp4Parser = new MP4Parser(this.buffer);

    this.decoder = new VideoDecoder(
      {
        output: async (frame) => {
          await this.processFrame(frame);
        },
        error: (e) => {
          // console.log("Oops!");
        }
      }
    );

    this.decoderConfig = {
      codec: this.mp4Parser.codecProfileInfo,
      codedWidth: this.mp4Parser.frameWidth,
      codedHeight: this.mp4Parser.frameHeight,
      description: this.mp4Parser.infoNeededForDecoder,
      // hardwareAcceleration: "prefer-hardware",
      // hardwareAcceleration: "prefer-software",
      // optimizeForLatency: true
      optimizeForLatency: false
    };

    // try {
    this.decoder.configure(this.decoderConfig);
    // }
    // catch (error) {
    // console.log(error);
    // throw error;
    // }

    this.decoder.addEventListener(
      "dequeue",
      async (dequeue) => {
        await this.onDequeue();
      }
    );
  }

  async renderFrame(index) {
    this.callback(
      {
        frame: this.cachedFrames[index],
        frameNumber: index
      }
    );

    delete this.cachedFrames[index];
    this.lastFrameRendered = index;
  }

  async processFrame(frame) {
    if (this.samplesPulledFromDecoder === this.targetFrame) {
      // const start = performance.now();
      this.cachedFrames[this.samplesPulledFromDecoder] = await createImageBitmap(frame);
      // const finish = performance.now();
      // console.log("createImageBitmap t =", (finish - start) / 1000, "s");
      frame.close();
      await this.renderFrame(this.targetFrame);
    }
    else if (this.samplesPulledFromDecoder > this.targetFrame) {
      this.cachedFrames[this.samplesPulledFromDecoder] = await createImageBitmap(frame);
      frame.close();
    }
    else {
      frame.close();
    }

    ++this.samplesPulledFromDecoder;
  }

  async onDequeue() {
    if (this.samplesPulledFromDecoder < this.targetFrame + 1) {
      await this.pushSample();
    }
  }

  async pushSample() {
    if (this.samplesPushedToDecoder < this.mp4Parser.numFrames) {
      const data = new ArrayBuffer(this.mp4Parser.entrySizes[this.samplesPushedToDecoder]);
      new Uint8Array(data).set(
        new Uint8Array(
          this.buffer,
          this.mp4Parser.frameOffsets[this.samplesPushedToDecoder],
          this.mp4Parser.entrySizes[this.samplesPushedToDecoder]
        )
      );

      const type = this.mp4Parser.syncFrames.has(this.samplesPushedToDecoder) ? "key" : "delta";
      const chunk = new EncodedVideoChunk(
        {
          type: type,
          data: data,
          timestamp: this.samplesPushedToDecoder
        }
      );
      this.decoder.decode(chunk);
      ++this.samplesPushedToDecoder;
    }
    else {
      const data = new ArrayBuffer(this.mp4Parser.entrySizes[this.mp4Parser.numFrames - 1]);
      new Uint8Array(data).set(
        new Uint8Array(
          this.buffer,
          this.mp4Parser.frameOffsets[this.mp4Parser.numFrames - 1],
          this.mp4Parser.entrySizes[this.mp4Parser.numFrames - 1]
        )
      );

      const type = this.mp4Parser.syncFrames.has(this.mp4Parser.numFrames - 1) ? "key" : "delta";
      const chunk = new EncodedVideoChunk(
        {
          type: type,
          data: data,
          timestamp: this.mp4Parser.numFrames - 1
        }
      );
      this.decoder.decode(chunk);
      // ++this.samplesPushedToDecoder;

      // await this.decoder.flush();
    }
  }

  async seekFrame(index) {
    this.targetFrame = index;

    if (this.lastFrameRendered === -1 || this.targetFrame === this.lastFrameRendered + 1) {
      if (this.cachedFrames[this.targetFrame]) {
        await this.renderFrame(this.targetFrame);
      }
      else {
        await this.pushSample();
      }
    }
    else {
      for (const frame in this.cachedFrames) {
        this.cachedFrames[frame].close();
        delete this.cachedFrames[frame];
      }

      let syncFrameIndexBefore = -1;
      for (const syncFrame of this.mp4Parser.syncFrames) {
        if (syncFrame <= this.targetFrame) {
          syncFrameIndexBefore = syncFrame;
        }
      }

      this.decoder.reset();
      this.decoder.configure(this.decoderConfig);

      this.samplesPushedToDecoder = syncFrameIndexBefore;
      this.samplesPulledFromDecoder = syncFrameIndexBefore;
      this.lastFrameRendered = syncFrameIndexBefore - 1;
      await this.pushSample();
    }
  }

  // close() {
  //   this.decoder.close();
  // }
}
