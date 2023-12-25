// def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
//     last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
//     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

pub(crate) fn average_pool(
    last_hidden_layer: &[f32],
    mask: &[i32],
    embedding_dim: usize,
    chunk_size: usize,
) -> Vec<f32> {
    // input 1,512,emb_d , len = 1x512
    // mask is 1,512
    // let mut avg: Vec<f32> = vec![0.0; 384];
    let mask_sum: i32 = mask.iter().sum();
    let avg = last_hidden_layer
        .chunks(chunk_size)
        .enumerate()
        .filter(|(idx, _)| mask[*idx] == 1)
        .fold(vec![0.0; embedding_dim], |acc, (_, x)| {
            acc.iter().zip(x).map(|(l, r)| l + r).collect::<Vec<_>>()
        });
    avg.into_iter().map(|e| e / mask_sum as f32).collect()
}
