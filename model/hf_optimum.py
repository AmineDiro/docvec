from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
save_directory = "tmp/onnx/"

# Load a model from transformers and export it to ONNX
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
ort_model = ORTModelForFeatureExtraction.from_pretrained(model_checkpoint, export=True)

# Save the ONNX model and tokenizer
ort_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
