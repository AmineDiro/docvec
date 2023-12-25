use std::{collections::HashMap, sync::Arc};

use js_sys::Array;
use tokenizers::tokenizer::Tokenizer;
use wonnx::{
    utils::{InputTensor, OutputTensor},
    Session,
};
extern crate alloc;
use alloc::string::String;
use wasm_bindgen::prelude::*;
use web_sys::console;
mod utils;
use utils::average_pool;

static MODEL_DATA: &'static [u8] = include_bytes!("../model/gte-small/onnx/sim_model.onnx",);
static TOKENIZER_DATA: &'static [u8] = include_bytes!("../model/gte-small/tokenizer.json",);

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct Embedder {
    session: Arc<wonnx::Session>,
    tokenizer: Tokenizer,
}

#[wasm_bindgen]
impl Embedder {
    #[wasm_bindgen(constructor)]
    pub async fn new() -> Result<Embedder, String> {
        console::log_1(&"Loading model from bytes".into());
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_DATA).unwrap();
        Ok(Self {
            session: Arc::new(
                wonnx::Session::from_bytes(MODEL_DATA)
                    .await
                    .map_err(|_| "Can't load model bytes")?,
            ),
            tokenizer,
        })
    }

    pub async fn embed_query(&self, txt: String) -> Result<Array, String> {
        let mut input: HashMap<String, InputTensor> = HashMap::new();
        console::log_1(&format!("Embedding query {}", txt).into());
        let encoding = self.tokenizer.encode(txt, true).unwrap();
        let tokens: Vec<i32> = encoding
            .get_ids()
            .iter()
            .map(|&e| e as i32)
            .collect::<Vec<_>>();
        let token_type_ids = encoding
            .get_type_ids()
            .iter()
            .map(|&e| e as i32)
            .collect::<Vec<_>>();
        let attention_mask = encoding
            .get_attention_mask()
            .iter()
            .map(|&e| e as i32)
            .collect::<Vec<_>>();

        input.insert("input_ids".to_string(), tokens[..].into());
        input.insert("attention_mask".to_string(), attention_mask[..].into());
        input.insert("token_type_ids".to_string(), token_type_ids[..].into());
        let output = self.session.clone().run(&input).await.unwrap();

        match output.get(&"last_hidden_state".to_string()).unwrap() {
            OutputTensor::F32(last_hidden_layer) => {
                let emb = average_pool(last_hidden_layer, &attention_mask, 384, 512);
                console::log_1(
                    &format!(
                        "The generated embedding len: {}. first 10 eleemnts {:?}",
                        emb.len(),
                        &emb[..10]
                    )
                    .into(),
                );

                let array = Array::new_with_length(emb.len() as u32);
                for value in emb {
                    array.push(&(value).into());
                }
                Ok(array)
            }
            _ => Err("can't have other type".to_string()),
        }
    }
}

#[allow(dead_code)]
async fn run_test(query: String) -> HashMap<String, OutputTensor> {
    let session = Session::from_bytes(MODEL_DATA)
        .await
        .expect("can't create onnx inference session");

    let tokenizer = Tokenizer::from_bytes(TOKENIZER_DATA).unwrap();
    let mut input: HashMap<String, InputTensor> = HashMap::new();
    let encoding = tokenizer.encode(query, true).unwrap();
    let tokens: Vec<i32> = encoding
        .get_ids()
        .iter()
        .map(|&e| e as i32)
        .collect::<Vec<_>>();
    let token_type_ids = encoding
        .get_type_ids()
        .iter()
        .map(|&e| e as i32)
        .collect::<Vec<_>>();
    let attention_mask = encoding
        .get_attention_mask()
        .iter()
        .map(|&e| e as i32)
        .collect::<Vec<_>>();

    dbg!(&tokens[..10]);
    dbg!(&attention_mask[..10]);
    dbg!(&token_type_ids[..10]);

    input.insert("input_ids".to_string(), tokens[..].into());
    input.insert("attention_mask".to_string(), attention_mask[..].into());
    input.insert("token_type_ids".to_string(), token_type_ids[..].into());
    // For now ['input_ids', 'token_type_ids', 'attention_mask']
    input.insert("input_ids".to_string(), tokens[..].into());
    input.insert("attention_mask".to_string(), attention_mask[..].into());
    input.insert("token_type_ids".to_string(), token_type_ids[..].into());
    let output = pollster::block_on(session.run(&input)).unwrap();
    dbg!(&output.keys());

    match output
        .get(&"last_hidden_state".to_string())
        .unwrap()
        .to_owned()
    {
        OutputTensor::F32(last_hidden_layer) => {
            let emb = average_pool(&last_hidden_layer, &attention_mask, 384, 512);
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
    use pollster;

    #[test]
    fn test_random_embedding() {
        let query = String::from("active Python core developers elected");
        let output = pollster::block_on(run_test(query));
        assert!(output.len() >= 1)
    }

    #[test]
    fn test_tokenizer() {
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_DATA).unwrap();
        let encoding = tokenizer.encode("Hey there!", false).unwrap();
        assert_eq!(encoding.get_ids().len(), 512);
    }
}
