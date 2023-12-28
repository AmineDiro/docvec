use std::{collections::HashMap, sync::Arc};
use wonnx::utils::{InputTensor, OutputTensor};

// use alloc::string::String;
// use js_sys::Array;
use tokenizers::tokenizer::Tokenizer;

static MODEL_DATA: &'static [u8] = include_bytes!("../model/gte-small/onnx/sim_model.onnx",);
static TOKENIZER_DATA: &'static [u8] = include_bytes!("../model/gte-small/tokenizer.json",);

fn average_pool(last_hidden_layer: &[f32], mask: &[i32], embedding_dim: usize) -> Vec<f32> {
    // input 1,512,emb_d , len = 1x512
    // mask is 1,512
    // let mut avg: Vec<f32> = vec![0.0; 384];
    let mask_sum: i32 = mask.iter().sum();

    let avg = last_hidden_layer
        .chunks(embedding_dim)
        .enumerate()
        .filter(|(idx, _)| mask[*idx] == 1)
        .fold(vec![0.0; embedding_dim], |acc, (_, layer)| {
            dbg!(&layer.len());
            acc.into_iter()
                .zip(layer)
                .map(|(l, &r)| l + r)
                .collect::<Vec<_>>()
        });
    dbg!(&avg[..10]);
    avg.into_iter().map(|e| e / mask_sum as f32).collect()
}

pub struct Embedder {
    session: Arc<wonnx::Session>,
    tokenizer: Tokenizer,
}
impl Embedder {
    pub async fn load() -> Result<Embedder, String> {
        // console::log_1(&"Loading model from bytes".into());
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

    pub async fn embed_query(&self, txt: String) -> Result<Vec<f32>, String> {
        let mut input: HashMap<String, InputTensor> = HashMap::new();
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
                dbg!(&last_hidden_layer[..10]);
                let emb = average_pool(last_hidden_layer, &attention_mask, 384);
                Ok(emb)
            }
            _ => Err("can't have other type".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Index;
    use pollster;

    fn compare_vecs(left: &[f32], right: &[f32], tol: f32) {
        let diff: Vec<f32> = left.iter().zip(right.iter()).map(|(l, r)| l - r).collect();
        let diff = diff
            .iter()
            .filter(|e| f32::abs(**e) > tol)
            .collect::<Vec<_>>();
        assert_eq!(diff.len(), 0);
    }
    #[test]
    fn test_embedding_correctness() {
        let tol = 1e-2f32;
        let index = Index::load();
        let content = index.content[0].clone();
        let output = pollster::block_on(async move {
            let embdr = Embedder::load().await.unwrap();
            embdr.embed_query(content).await.unwrap()
        });

        dbg!(&output[..10]);
        compare_vecs(&output, &index.embeddings[..384], tol);
    }
    #[test]
    fn test_random_embedding() {
        let query = String::from("active Python core developers elected");
        let output = pollster::block_on(async {
            let embdr = Embedder::load().await.unwrap();
            embdr.embed_query(query).await.unwrap()
        });
        dbg!(&output[..10]);
        assert!(output.len() == 384)
    }

    #[test]
    fn test_tokenizer() {
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_DATA).unwrap();
        let encoding = tokenizer.encode("Hey there!", false).unwrap();

        assert_eq!(encoding.get_ids().len(), 512);
    }
}
