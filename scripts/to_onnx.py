import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# load model and tokenizer
model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
sentences = ["This is an example sentence", "Each sentence is converted"]

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
print(encoded_input.keys())

dummy_model_input = tokenizer("This is a sample", return_tensors="pt")

# export
torch.onnx.export(
    model,
    tuple(dummy_model_input.values()),
    f="torch-model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size", 1: "sequence"},
    },
    do_constant_folding=True,
    opset_version=13,
)
