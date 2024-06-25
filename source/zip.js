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

const zipWebAssemblyModule = await WebAssembly.compileStreaming(fetch("../zip.wasm"));
const zipWebAssemblyInstance = await WebAssembly.instantiate(
  zipWebAssemblyModule,
  {
    env: {}
  }
);
zipWebAssemblyInstance.exports._start();


function CRC32(str, crc = 0) {
  let n = 0;
  let x = 0;

  crc = crc ^ (-1);
  for (let i = 0; i < str.length; ++i) {
    n = (crc ^ str.charCodeAt(i)) & 0xff;
    x = "0x" + BigInt(zipWebAssemblyInstance.exports.crc_32_table_lookup(n) >>> 0).toString(16);
    crc = (crc >>> 8) ^ x;
  }

  return crc ^ (-1);
}


export function testZip(svgFilenames, svgFileContents, csvFilenames, csvFileContents, summaryFilename, summaryFileContents) {
  let blobContents = [];

  let centralDirectoryOffset = 0;

  let filenamesArray = svgFilenames.concat(csvFilenames);
  let fileContentsArray = svgFileContents.concat(csvFileContents);
  // filenamesArray.push(summaryFilename);
  // fileContentsArray.push(summaryFileContents);

  let offsets = [];
  for (let i = 0; i < filenamesArray.length; ++i) {
    const filename = filenamesArray[i];
    const fileContents = fileContentsArray[i];
    const fileCRC32 = CRC32(fileContents);

    //
    //
    //

    // 4 bytes: Signature
    blobContents.push(new Uint32Array([0x04034b50]));
    centralDirectoryOffset += 4;

    // 2 bytes: Minimum version needed to extract
    blobContents.push(new Uint16Array([0]));
    centralDirectoryOffset += 2;

    // 2 bytes: Bit flag
    blobContents.push(new Uint16Array([0]));
    centralDirectoryOffset += 2;

    // 2 bytes: Compression method
    blobContents.push(new Uint16Array([0]));
    centralDirectoryOffset += 2;

    // 2 bytes: File last modification time (MS-DOS format)
    blobContents.push(
      new Uint16Array([zipWebAssemblyInstance.exports.ms_dos_time(
        new Date().getSeconds(),
        new Date().getMinutes(),
        new Date().getHours(),
      )])
    );
    centralDirectoryOffset += 2;

    // 2 bytes: File last modification date (MS-DOS format)
    blobContents.push(
      new Uint16Array([zipWebAssemblyInstance.exports.ms_dos_date(
        new Date().getDate(),
        new Date().getMonth() + 1,
        new Date().getFullYear() - 1980
      )])
    );
    centralDirectoryOffset += 2;

    // 4 bytes: CRC-32 of uncompressed data
    blobContents.push(new Uint32Array([fileCRC32]));
    centralDirectoryOffset += 4;

    // 4 bytes: Compressed size
    blobContents.push(new Uint32Array([fileContents.length]));
    centralDirectoryOffset += 4;

    // 4 bytes: Uncompressed size
    blobContents.push(new Uint32Array([fileContents.length]));
    centralDirectoryOffset += 4;

    // 2 bytes: File name length (n)
    blobContents.push(new Uint16Array([filename.length]));
    centralDirectoryOffset += 2;

    // 2 bytes: Extra field length (m)
    blobContents.push(new Uint16Array([0]));
    centralDirectoryOffset += 2;

    // n bytes: File name
    blobContents.push(filename);
    centralDirectoryOffset += filename.length;

    // m bytes: Extra field

    // length: 30 + n -> 32

    //
    //
    //

    // file
    blobContents.push(fileContents);
    centralDirectoryOffset += fileContents.length;
    offsets += fileContents.length;

    // length: variable

  }

  //
  //
  //

  let centralDirectorySize = 0;

  for (let i = 0; i < filenamesArray.length; ++i) {
    const filename = filenamesArray[i];
    const fileContents = fileContentsArray[i];
    const fileCRC32 = CRC32(fileContents);

    // 4 bytes: Signature
    blobContents.push(new Uint32Array([0x02014b50]));
    centralDirectorySize += 4;

    // 2 bytes: Version made by
    blobContents.push(new Uint16Array([0]));
    centralDirectorySize += 2;

    // 2 bytes: Minimum version needed to extract
    blobContents.push(new Uint16Array([0]));
    centralDirectorySize += 2;

    // 2 bytes: Bit flag
    blobContents.push(new Uint16Array([0]));
    centralDirectorySize += 2;

    // 2 bytes: Compression method
    blobContents.push(new Uint16Array([0]));
    centralDirectorySize += 2;

    // 2 bytes: File last modification time (MS-DOS format)
    blobContents.push(new Uint16Array([0]));
    centralDirectorySize += 2;

    // 2 bytes: File last modification date (MS-DOS format)
    blobContents.push(new Uint16Array([0]));
    centralDirectorySize += 2;

    // 4 bytes: CRC-32 of uncompressed data
    // blobContents.push(new Uint32Array([0]));
    blobContents.push(new Uint32Array([fileCRC32]));
    centralDirectorySize += 4;

    // 4 bytes: Compressed size
    blobContents.push(new Uint32Array([fileContents.length]));
    centralDirectorySize += 4;

    // 4 bytes: Uncompressed size
    blobContents.push(new Uint32Array([fileContents.length]));
    centralDirectorySize += 4;

    // 2 bytes: File name length (n)
    blobContents.push(new Uint16Array([filename.length]));
    centralDirectorySize += 2;

    // 2 bytes: Extra field length (m)
    blobContents.push(new Uint16Array([0]));
    centralDirectorySize += 2;

    // 2 bytes: File comment length (k)
    blobContents.push(new Uint16Array([0]));
    centralDirectorySize += 2;

    // 2 bytes: Disk number where file starts
    blobContents.push(new Uint16Array([0]));
    centralDirectorySize += 2;

    // 2 bytes: Internal file attributes
    blobContents.push(new Uint16Array([0]));
    centralDirectorySize += 2;

    // 4 bytes: External file attributes
    blobContents.push(new Uint32Array([0]));
    centralDirectorySize += 4;

    // 4 bytes: Offset of local file header (from start of disk)
    blobContents.push(new Uint32Array([offsets[i]]));
    centralDirectorySize += 4;

    // n bytes: File name
    blobContents.push(filename);
    centralDirectorySize += filename.length;

    // m bytes: Extra field

    // k bytes: File comment

    // length: 46 + n -> 48
  }

  //
  //
  //

  // 4 bytes: Signature
  blobContents.push(new Uint32Array([0x06054b50]));

  // 2 bytes: Number of this disk
  blobContents.push(new Uint16Array([0]));

  // 2 bytes: Disk where central directory starts
  blobContents.push(new Uint16Array([0]));

  // 2 bytes: Number of central directory records on this disk
  // blobContents.push(new Uint16Array([0]));
  blobContents.push(new Uint16Array([filenamesArray.length]));

  // 2 bytes: Total number of central directory records
  // blobContents.push(new Uint16Array([0]));
  blobContents.push(new Uint16Array([filenamesArray.length]));

  // 4 bytes: Size of central directory in bytes
  // blobContents.push(new Uint32Array([0]));
  blobContents.push(new Uint32Array([centralDirectorySize + 22]));

  // 4 bytes: Offset to start of central directory
  // blobContents.push(new Uint32Array([0]));
  blobContents.push(new Uint32Array([centralDirectoryOffset]));

  // 2 bytes: Comment length (n)
  blobContents.push(new Uint16Array([0]));

  // n bytes: Comment

  // length: 22

  //
  //
  //

  // await writable.close();

  const blob = new Blob(blobContents);
  return blob;
}
