#!/bin/bash
# Set optimization flags

#  build --target web -d `pwd`/target/pkg --out-name wonnx --scope webonnx ./wonnx-wasm
# Run wasm pack tool to build JS wrapper files and copy wasm to pkg directory.
mkdir -p pkg
RUSTFLAGS=--cfg=web_sys_unstable_apis wasm-pack build \
    -d pkg \
    --target web \
    --no-typescript \
    # --dev \