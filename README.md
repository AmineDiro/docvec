# DocVec - Wasm meets Semantic search

![Alt Text](./data/output.gif)

I wanted an excuse see what all the hype about `WebGPU` and `WebAssembly` was all about for a long time. Then I attended a Rust Wasm meetup and was eager to find a project to learn about these technologies.

`docVec` is a client-side fully working semantic search engine, ie. having the model run ENTIRELY on the client machine. This is NOT a production-ready project.

My goals for the project were to:

- Use Rust for NN inference
- Use the GPU for model inference and see how mature it is to use wgpu: Luckily, I found the amazing project wonnx. I had to hack around some issues of running transformers and also implement some missing ONNX operators (cf. PR) for this to work. Also, I am still working on re-implementing the project's MatMul broadcasting and trying if possible to improve the compute shader performance.
- Implement the whole logic in a webassembly module in Rust. The goal here is to understand some internals of wasm and the limitations that come from that
- Keep the JS to a minimum.
- Don't overcomplicate the search engine. For now a simple index of flat vector suffice.

## Maintainer

1. Download `gte-small` model from huggingface
   ```bash
   cd model/
   git clone https://huggingface.co/Supabase/gte-small
   ```
2. Install onnx simplifier : [`onnxsim`](https://github.com/daquexian/onnx-simplifier)
3. Simplify model and fix input batch size and sequence length
   ```bash
   python -m onnxsim gte-small/onnx/model.onnx  gte-small/onnx/sim_model.onnx \
    --overwrite-input-shape "input_ids:1,512" "attention_mask:1,512" "token_type_ids:1,512"
   ```
4. Install `wasm-pack`

   ```bash
   cargo install wasm-pack
   ```

5. Clone modified version of `wonnx` (temporary)

   ```bash
   cd ..
   git clone https://github.com/AmineDiro/wonnx.git
   cd wonnx
   git checkout broadcast-matmul
   ```

6. Build web assembly module & serve the page
   ```bash
   cd ..  # go to project root
   ./build.sh && python3 -m http.server 8000
   ```

Now you can access the semantic search module on `http://localhost:8000` 🌟

## TODO:

- Backend (wasm):

  - [x] Project scaffolding using `wasm-bindgen`
  - [ ] Generate string embedding using `wonnx` and `gte-small` model:
    - [x] Add `Erf` operator to wonnx
    - [x] Modify `MatMul` broadcasting checks ( this is temporary)
    - [ ] Reimplement _correct_ `MatMul` with broadcasting
    - [ ] Investigate float NaN issues on Vulkan backend for wgpu
  - [x] Tokenize input in wasm `tokenizers`
  - [x] Build index :
    - [x] Split page text
    - [x] Embed text using `sentence-transformers`
    - [x] Load index in wasm module
  - [x] Implement L2 distance and return k nearest neighbors (avec `Vec<String>`)

- Frontend:
  - [x] Download example wiki page as simple html
  - [x] Loop over page elements and search for matching html element
  - [ ] Highlight just the text and a littlebit the surrounding
