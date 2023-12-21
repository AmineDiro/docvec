use std::{collections::HashMap, sync::Arc};

use tokenizers::tokenizer::Tokenizer;

use wonnx::{
    utils::{InputTensor, OutputTensor},
    Session,
};
extern crate alloc;
use alloc::string::String;
use wasm_bindgen::prelude::*;
use web_sys::console;

static MODEL_DATA: &'static [u8] = include_bytes!(
    "/Users/aminedirhoussi/Documents/coding/doc-wasm/model/all-MiniLM-L6-v2/optimum/sim_model.onnx"
);

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct Embedder {
    session: Arc<wonnx::Session>,
}

#[wasm_bindgen]
impl Embedder {
    #[wasm_bindgen(constructor)]
    pub async fn new() -> Result<Embedder, String> {
        console::log_1(&"Loading model from bytes".into());
        Ok(Self {
            session: Arc::new(
                wonnx::Session::from_bytes(MODEL_DATA)
                    .await
                    .map_err(|_| "Can't load model bytes")?,
            ),
        })
    }

    pub async fn random_inference(&self) -> JsValue {
        let mut input: HashMap<String, InputTensor> = HashMap::new();
        let tokens = vec![1f32; 512];
        let attention_mask = vec![1f32; 512];
        let token_type_ids = vec![0f32; 512];
        // For now ['input_ids', 'token_type_ids', 'attention_mask']
        input.insert("input_ids".to_string(), tokens[..].into());
        input.insert("attention_mask".to_string(), attention_mask[..].into());
        input.insert("token_type_ids".to_string(), token_type_ids[..].into());
        let output = self.session.clone().run(&input).await.unwrap();
        output.len().into()
    }
}

async fn run_test() -> HashMap<String, OutputTensor> {
    let session = Session::from_path(
        // "/Users/aminedirhoussi/Documents/coding/doc-wasm/model/all-MiniLM-L6-v2/sim_model.onnx",
        "/Users/aminedirhoussi/Documents/coding/doc-wasm/model/gte-small/onnx/sim_model.onnx",
    )
    .await
    .expect("can't create onnx inference session");

    let mut input: HashMap<String, InputTensor> = HashMap::new();
    let tokens = vec![1f32; 512];
    let attention_mask = vec![0f32; 512];
    let token_type_ids = vec![32f32; 512];
    // For now ['input_ids', 'token_type_ids', 'attention_mask']
    input.insert("input_ids".to_string(), tokens[..].into());
    input.insert("attention_mask".to_string(), attention_mask[..].into());
    input.insert("token_type_ids".to_string(), token_type_ids[..].into());
    let output = tokio_test::block_on(session.run(&input)).unwrap();
    dbg!(&output.keys());

    match output
        .get(&"last_hidden_state".to_string())
        .unwrap()
        .to_owned()
    {
        OutputTensor::F32(emb) => {
            dbg!(emb.len());
            dbg!(&emb[..10]);
        }
        _ => unreachable!("can't have other type"),
    }
    // dbg!("{}", embedding.);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use env_logger;
    use pollster;

    #[test]
    fn test_load_model() {
        env_logger::init();
        let output = pollster::block_on(run_test());

        assert!(output.len() >= 1)
    }

    #[test]
    fn test_tokenizer() {
        let tokenizer = Tokenizer::from_file(
            "/Users/aminedirhoussi/Documents/coding/doc-wasm/model/gte-small/tokenizer.json",
        )
        .unwrap();

        let encoding = tokenizer.encode("Hey there!", false).unwrap();

        dbg!(encoding.get_tokens().len());
        dbg!(encoding.get_ids().len());
        dbg!(encoding.get_attention_mask().len());
    }
}
