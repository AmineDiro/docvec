# DocVec - Wasm meets Semantic search

![Alt Text](./data/docvec.gif)

I wanted an excuse to learn `webgpu` and `wasm` for a longtime. Then I attended a Rust wasm meetup and was "forced" to find a project to learn about these technologies.
The idea is to have a fully working semantic search engine client side, meaning having the model run ENTIRELY on the client machine.
This is NOT a production ready project. The
My goal for the project were:

- Use the Hardware for model inference and if possible use `wgpu`: Luckily I found the amazing project`wonnx`. I had to hack around some issues of runnign transformers and also implement some missing onnx operators (cf. PR) for this to work. Also I am still working on reimplementing the project's `MatMul` broadcasting and try if possible improving the compute shader performance.
- Use Rust of course. Again amazing project `wasm-bindgen`
- Keep the JS to a minimum...
- Simple index : flat vector for now

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
