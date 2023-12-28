import onnxruntime as ort
import torch
from transformers import AutoTokenizer

workdir = "/Users/aminedirhoussi/Documents/coding/doc-wasm/model/gte-small"
model_path = workdir + "/onnx/sim_model.onnx"

tokenizer = AutoTokenizer.from_pretrained(workdir)


def average_pool(last_hidden_states, mask):
    last_hidden = last_hidden_states.masked_fill(~mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / mask.sum(dim=1)[..., None]


# loading content
with open("./data/index_content.txt", "r") as f:
    first = f.read().splitlines()[0]

sentences = [first]

encoded_input = tokenizer(
    sentences,
    padding="max_length",
    truncation=True,
    return_tensors="np",
    max_length=512,
)

# Load Model
ort_sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
print(encoded_input.keys())
outputs = ort_sess.run(None, dict(encoded_input))
last_hidden_states = torch.Tensor(outputs[0])  # (b, 512,384)
mask = torch.Tensor(encoded_input["attention_mask"])
embd = average_pool(last_hidden_states, mask).numpy()
print("Output shape: ", embd.shape)
