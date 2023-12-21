#!/bin/bash
python -m onnxsim model.onnx sim_model_1.onnx \
    --overwrite-input-shape "input_ids:512" "attention_mask:512" "token_type_ids:512"