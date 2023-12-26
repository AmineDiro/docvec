import onnxruntime as ort
from transformers import AutoModel, AutoTokenizer

workdir = "/Users/aminedirhoussi/Documents/coding/doc-wasm/model/all-MiniLM-L6-v2"
# model_path = workdir+'/model-prepared.onnx'
model_path = workdir + "/sim_model.onnx"
sentences = ["This is an example sentence"]

tokenizer = AutoTokenizer.from_pretrained(workdir)

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
embd = outputs[1].flatten()
print("Output shape: ", embd.shape)
