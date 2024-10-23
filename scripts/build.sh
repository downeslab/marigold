#!/bin/bash

set -Eeuxo pipefail

mkdir -p build
cp source/browser-compatibility.js build
cp source/colormap.js build
cp source/constants.js build
cp source/deferred.css build
cp source/image.js build
cp source/index.html build
cp source/main.css build
cp source/main.js build
cp source/manifest.json build
cp source/movie-reader.js build
cp source/neural-network.js build
cp source/section.js build
cp source/section-worker.js build
cp source/zip.js build

mkdir -p build/analyzing
cp source/analyzing/analyzing.html build/analyzing
cp source/analyzing/analyzing-section.js build/analyzing
cp source/analyzing/analyzing-worker.js build/analyzing
cp source/analyzing/kinematics.js build/analyzing
cp source/analyzing/trajectory-plot.js build/analyzing

mkdir -p build/labeling
cp source/labeling/labeling.html build/labeling
cp source/labeling/labeling-section.js build/labeling
cp source/labeling/labeling-thumbnails.js build/labeling
cp source/labeling/labeling-worker.js build/labeling

mkdir -p build/training
cp source/training/training.html build/training
cp source/training/training-section.js build/training
cp source/training/training-worker.js build/training

clang++ \
--target=wasm32 \
-nostdlib \
-std=c++23 \
-O3 \
-mbulk-memory \
-msimd128 \
-flto \
-Isource \
-Wall \
-Wextra \
-Wpedantic \
-Wl,--allow-undefined \
-Wl,--import-undefined \
-Wl,--export-all \
-Wl,--lto-O3 \
-Wl,--initial-memory=$[16 * 1024 * 1024] \
-Wl,--max-memory=$[4 * 1024 * 1024 * 1024] \
-Wl,-z,stack-size=$[8 * 1024 * 1024] \
-o build/neural-network.wasm \
source/neural-network.cxx
# -Wl,--no-entry \
# -Wl,--entry=_start \
# -mextendend-const \
# -matomics \

clang++ \
--target=wasm32 \
-nostdlib \
-std=c++23 \
-O3 \
-matomics \
-mbulk-memory \
-msimd128 \
-flto \
-Isource \
-Wall \
-Wextra \
-Wpedantic \
-Wl,--allow-undefined \
-Wl,--import-undefined \
-Wl,--export-all \
-Wl,--lto-O3 \
-Wl,--initial-memory=$[16 * 1024 * 1024] \
-Wl,--max-memory=$[4 * 1024 * 1024 * 1024] \
-Wl,-z,stack-size=$[8 * 1024 * 1024] \
-o build/random.wasm \
source/random.cxx
# -Wl,--no-entry \
# -Wl,--entry=_start \

clang++ \
--target=wasm32 \
-nostdlib \
-std=c++23 \
-O3 \
-matomics \
-mbulk-memory \
-msimd128 \
-flto \
-Isource \
-Wall \
-Wextra \
-Wpedantic \
-Wl,--allow-undefined \
-Wl,--import-undefined \
-Wl,--export-all \
-Wl,--lto-O3 \
-Wl,--initial-memory=$[16 * 1024 * 1024] \
-Wl,--max-memory=$[16 * 1024 * 1024] \
-Wl,-z,stack-size=$[8 * 1024 * 1024] \
-o build/zip.wasm \
source/zip.cxx
# -Wl,--no-entry \
# -Wl,--entry=_start \

mkdir -p build/assets
cp assets/Marigold.svg build/assets
inkscape assets/Marigold.svg \
--export-filename=build/assets/Marigold-180x180.png \
--export-width=180 \
--export-height=180
inkscape assets/Marigold.svg \
--export-filename=build/assets/Marigold-192x192.png \
--export-width=192 \
--export-height=192
inkscape assets/Marigold.svg \
--export-filename=build/assets/Marigold-384x384.png \
--export-width=384 \
--export-height=384
inkscape assets/Marigold.svg \
--export-filename=build/assets/Marigold-512x512.png \
--export-width=512 \
--export-height=512
inkscape assets/Marigold.svg \
--export-filename=build/assets/Marigold-1024x1024.png \
--export-width=1024 \
--export-height=1024

mkdir -p build/external/fonts
cp external/fonts/source-sans-3.052R/WOFF2/VF/SourceSans3VF-Upright.otf.woff2 build/external/fonts
cp external/fonts/source-sans-3.052R/WOFF2/VF/SourceSans3VF-Italic.otf.woff2 build/external/fonts
