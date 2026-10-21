"""Export trained T5 model to ONNX for fast CPU inference."""

import os

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_model")
ONNX_PATH = os.path.join(MODEL_PATH, "robot_model.onnx")


def main():
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model.eval()

    dummy_input = tokenizer(
        "scene: Robot at (0,0) facing NORTH | tree 10m ahead | user: what do you see?",
        return_tensors="pt",
        max_length=256,
        truncation=True,
    )

    with torch.no_grad():
        torch.onnx.export(
            model,
            (
                dummy_input["input_ids"],
                dummy_input["attention_mask"],
                torch.tensor([[model.config.decoder_start_token_id]]),
            ),
            ONNX_PATH,
            input_names=["input_ids", "attention_mask", "decoder_input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "decoder_input_ids": {0: "batch", 1: "seq"},
            },
            opset_version=17,
        )

    print(f"ONNX model saved to {ONNX_PATH}")


if __name__ == "__main__":
    main()
