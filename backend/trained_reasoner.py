"""T5-small ONNX model — robot body control only (no speech).

Outputs command arrays like [{"action":"walk","params":{"distance_m":2}}].
Requires ``training/trained_model/robot_model.onnx`` (see training/README.md).
"""

import json
import re
import os
import numpy as np
from models import LLMCommand

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training", "trained_model")


class TrainedReasoner:
    def __init__(self, ollama_url=None, llm_model=None, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.session = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        from transformers import T5Tokenizer
        import onnxruntime

        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Trained model directory not found: {self.model_dir}. "
                "Run training/train.py and training/export_onnx.py first, or unset USE_TRAINED."
            )
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_dir)
        onnx_path = os.path.join(self.model_dir, "robot_model.onnx")
        self.session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._loaded = True

    async def reason(self, perception, user_input: str) -> list[LLMCommand]:
        self.load()
        scene_str = scene_to_str(perception)
        text = f"scene: {scene_str} | user: {user_input}"

        tokens = self.tokenizer(text, return_tensors="np", max_length=256, truncation=True, padding="max_length")
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)
        decoder_input = np.array([[self.tokenizer.pad_token_id]], dtype=np.int64)

        generated = []
        for _ in range(128):
            outputs = self.session.run(["logits"], {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": decoder_input,
            })
            next_token = int(np.argmax(outputs[0][0, -1, :]))
            if next_token == self.tokenizer.eos_token_id:
                break
            generated.append(next_token)
            decoder_input = np.concatenate([decoder_input, np.array([[next_token]], dtype=np.int64)], axis=1)

        output_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return self._parse_commands(output_text)

    def _parse_commands(self, text: str) -> list[LLMCommand]:
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if json_match:
            try:
                raw = json.loads(json_match.group())
                return [LLMCommand(action=c["action"], params=c.get("params", {})) for c in raw]
            except (json.JSONDecodeError, KeyError):
                pass
        return []


def scene_to_str(perception) -> str:
    robot = perception.robot
    obj_parts = [f"{o.label} {o.distance:.0f}m" for o in perception.objects[:5]]
    return f"Robot at ({robot.position[0]:.0f},{robot.position[1]:.0f}) | {' | '.join(obj_parts) or 'nothing'}"


if __name__ == "__main__":
    import asyncio
    import time

    from models import PerceptionData, RobotState

    async def demo():
        reasoner = TrainedReasoner()
        reasoner.load()
        scene = PerceptionData(
            robot=RobotState(position=(0.0, 0.0, 0.0), rotation=0.0, joints={}),
            objects=[],
            timestamp=time.time(),
        )
        for command in ("walk forward 2 meters", "turn left", "stop"):
            start = time.perf_counter()
            commands = await reasoner.reason(scene, command)
            elapsed = (time.perf_counter() - start) * 1000
            print(f"{command!r}: {[(c.action, c.params) for c in commands]} ({elapsed:.0f} ms)")

    asyncio.run(demo())
