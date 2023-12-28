extern crate alloc;
use alloc::string::String;
use js_sys::Array;
use wasm_bindgen::prelude::*;

mod embedder;
mod index;
mod utils;
use embedder::Embedder;
use index::Index;
use web_sys::console;

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct VecSearch {
    embedder: Embedder,
    index: Index,
}

#[wasm_bindgen]
impl VecSearch {
    #[wasm_bindgen(constructor)]
    pub async fn new() -> Result<VecSearch, String> {
        let embedder = Embedder::load().await?;
        console::log_1(&format!("Loaded embedder").into());
        let index = Index::load();
        console::log_1(&format!("Loaded index into memory").into());
        Ok(VecSearch { embedder, index })
    }

    // TODO : return a REsult<Array,..>
    pub async fn search(&self, query: String, k: usize) -> Array {
        let query_emb = self.embedder.embed_query(query).await.unwrap();
        let neighbors = self.index.vec_search(&query_emb, k);
        let array = Array::new_with_length(neighbors.len() as u32);
        for value in neighbors {
            array.push(&(value).into());
        }
        array
    }
}
