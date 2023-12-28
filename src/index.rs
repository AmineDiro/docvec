static INDEX_CONTENT: &'static str = include_str!("../data/index_content.txt");
static INDEX_EMBEDDINGS: &'static [u8] = include_bytes!("../data/index_embedding.bin");
const DIM: usize = 384;

#[inline]
pub fn l2_distance(s1: &[f32], s2: &[f32]) -> f32 {
    f32::sqrt(
        s1.iter()
            .zip(s2.iter())
            .map(|(i, j)| f32::powi(i - j, 2))
            .sum(),
    )
}

pub struct Index {
    pub content: Vec<String>,
    pub embeddings: Vec<f32>,
}

impl Index {
    pub fn load() -> Self {
        let content: Vec<String> = INDEX_CONTENT
            .lines()
            .map(|l| l.to_string())
            .collect::<Vec<_>>();
        let embeddings = INDEX_EMBEDDINGS
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|e| f32::from_le_bytes([e[0], e[1], e[2], e[3]]))
            .collect::<Vec<_>>();
        Self {
            content,
            embeddings,
        }
    }

    pub fn vec_search(&self, query_emb: &[f32], k: usize) -> Vec<String> {
        //TODO : this is baad,
        let distances: Vec<f32> = self
            .embeddings
            .chunks(DIM)
            .map(|idx_emd| l2_distance(idx_emd, query_emb))
            .collect();
        let mut k_indices = (0..distances.len()).collect::<Vec<_>>();
        k_indices.sort_by(|&a, &b| distances[a].partial_cmp(&distances[b]).unwrap());
        // Return the k nearest
        k_indices[..k]
            .iter()
            .map(|&idx| self.content[idx].clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_index() {
        let index = Index::load();
        assert_eq!(index.content[0],
"Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.Python is dynamically typed and"
        );
        assert_eq!(
            &index.embeddings[..10],
            &[
                -0.3572080135345459,
                -0.17068633437156677,
                0.10957382619380951,
                -0.2546745538711548,
                -0.28215888142585754,
                -0.08235689997673035,
                0.3212471604347229,
                0.37563979625701904,
                0.020504314452409744,
                0.09850041568279266
            ]
        )
    }
    #[test]
    fn test_vec_search() {
        let index = Index::load();
        let res = index.vec_search(&index.embeddings[..384], 10);
        dbg!(&res);
        assert_eq!(res[0], index.content[0]);
    }
}
