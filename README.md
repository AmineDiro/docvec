# DocVec - Wasm meets Semantic search

![Alt Text](./data/docvec.gif)

I wanted an excuse to learn `webgpu` and `wasm` for a longtime. Then I attended a Rust wasm meetup and was "forced" to find a project to learn about these technologies.
The idea is to have a fully working semantic search engine client side, meaning having the model run ENTIRELY on the client machine.
This is NOT a production ready project. The
My goal for the project were:

- Use the Hardware for model inference and if possible use `wgpu`: Luckily I found the amazing project`wonnx`. I had to hack around some issues of runnign transformers and also implement some missing onnx operators (cf. PR) for this to work. Also I am still working on reimplementing the project's `MatMul` broadcasting and try if possible improving the compute shader performance.
- Use Rust of course. Again amazing project `wasm-bindgen`
- Keep the JS to a minimum...
- Simple index:

# TODO:

- Backend:

  - [x] Project scaffolding using `wasm-bindgen`
  - [ ] Generate string embedding using `wonnx` and `gte-small` model:
    - [x] Add `Erf` operator to wonnx
    - [x] Modify `MatMul` broadcasting checks ( this is temporary)
    - [ ] Reimplement _correct_ `MatMul` with broadcasting
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
