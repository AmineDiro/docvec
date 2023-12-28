import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def read_input():
    with open("./data/index_content.txt", "r") as f:
        return f.read().splitlines()[:10]


input_texts = read_input()
tokenizer = AutoTokenizer.from_pretrained("./model/gte-small")
model = AutoModel.from_pretrained("./model/gte-small")

# Tokenize the input texts
batch_dict = tokenizer(
    input_texts,
    max_length=512,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
data = embeddings.flatten().tolist()
import struct

buffer = struct.pack(f"<{len(data)}f", *data)
with open("../data/index_embedding.bin", "wb") as f:
    f.write(buffer)

# (Optionally) normalize embeddings
# embeddings = F.normalize(embeddings, p=2, dim=1)
# scores = (embeddings[:1] @ embeddings[1:].T) * 100
# print(scores.tolist())
