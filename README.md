# DocVec - Wasm meets Semantic search

![Alt Text](./data/docvec.gif)

I wanted to have an excuse to learn `webgpu` and `wasm` for a longtime. Then I attended a Rust wasm meetup and was "forced" to find a project to learn about these technologies.
The idea is to have a fully working semantic search engine client side, meaning having the model run ENTIRELY on the client machine.
This is NOT a production ready project. The
My goal for the project were:

- Use the Hardware for model inference and if possible use `wgpu`: Luckily I found the amazing project`wonnx`. I had to hack around some issues of runnign transformers and also implement some missing onnx operators (cf. PR) for this to work. Also I am still working on reimplementing the project's `MatMul` broadcasting and try if possible improving the compute shader performance.
- Use Rust of course. Again amazing project `wasm-bindgen`
- Keep the JS to a minimum...
- Simple index:
