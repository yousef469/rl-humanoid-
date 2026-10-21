# Training

This directory contains three independent training tracks:

1. **Command model** — a small T5 model that turns a scene description plus a
   user command into backend body-control JSON. This is optional.
2. **Red-ball YOLO detector** — a single-class YOLO11n model trained on synthetic
   head-camera frames. The exported weights ship as
   `models/red_ball_yolo.pt`.
3. **G1 walking policy** — a MuJoCo/Stable-Baselines3 PPO policy for the Unitree
   G1. This is separate from the live backend demo gait.

The shipped detector lets the web demo find and approach the red ball. The T5
and G1 tracks are optional.

## 1. Command model (T5)

### Schema

The target format matches `backend/controller.py`:

```json
[{"action": "walk", "params": {"distance_m": 2.0, "speed": 0.5}}]
[{"action": "turn", "params": {"degrees": 90}}]
[{"action": "wave", "params": {"arm": "right"}}]
[{"action": "stop", "params": {}}]
```

Positive `degrees` turns left; negative turns right. An empty list `[]` means "no body action". The model does not generate speech;
speech is handled separately by the backend.

### Dataset

`generate_data.py` writes `robot_training_data.jsonl` with 50,000 synthetic
examples. The current regenerated file has:

- 50,000 rows, 49,994 unique inputs
- 7,434 empty outputs
- actions: `turn` 21,560, `walk` 9,669, `wave` 6,434, `stop` 4,903

The original dataset was checked for duplicate/conflicting inputs and obvious
leakage; the main issue was label quality, which the rewritten generator fixes.
The data is still synthetic and does not cover every phrasing.

### Train and export

```bash
cd training
python -m pip install -r requirements.txt
python generate_data.py
python train.py
python export_onnx.py
```

`train.py` fine-tunes `t5-small` and saves to `training/trained_model/`.
`export_onnx.py` writes `training/trained_model/robot_model.onnx` (opset 17).

### Run with the backend

From the repository root:

```bash
cd backend
USE_TRAINED=1 ../.venv/bin/python main.py
```

The backend also needs the optional packages in
`backend/requirements-train.txt` (`transformers`, `onnxruntime`). If the model
directory or ONNX file is missing, it raises a clear `FileNotFoundError`.

### Test inference

The backend owns the inference smoke test now:

```bash
cd backend
USE_TRAINED=1 ../.venv/bin/python trained_reasoner.py
```

It prints parsed commands and latency for a few sample inputs.

## 2. Red-ball YOLO detector

`train_red_ball_yolo.py` instantiates the backend PyBullet world, moves the red
ball and robot through varied poses, uses the segmentation buffer to label the
ball, and fine-tunes `models/yolo11n.pt`.

```bash
cd training
python train_red_ball_yolo.py
```

The generated dataset is written to `training/red_ball_dataset/` and the run to
`training/red_ball_runs/` (both ignored). The best weights are copied to
`models/red_ball_yolo.pt`. Current validation on held-out synthetic frames is
about 0.99 mAP50.

## 3. G1 walking policy (MuJoCo PPO)

This track is for the full Unitree G1 model and is **not** loaded by the web
backend. Use it for experiments, evaluation, and rendering.

```bash
cd training
python -m pip install -r requirements_rl.txt
python train_g1.py
python watch_best.py
```

Useful entry points:

| File | Purpose |
| --- | --- |
| `g1_env.py` | G1 MuJoCo environment used by PPO |
| `train_g1.py` | Train PPO and save checkpoints |
| `watch_best.py` | Evaluate the latest/best checkpoint |
| `viewer.py` | Poll and render the latest checkpoint |
| `run_with_vision.py` | G1 with YOLO visualization and oracle steering |
| `test_vision_walk.py` | Vision/walking smoke test |
| `unitree_g1/g1.xml` | G1 scene with the first-person camera |

New runs save `vec_normalize_best.pkl` next to `best_model.zip`. Old checkpoints
without matching normalization stats should be treated as incompatible.

The backend demo walking controller is kinematic and hand-authored; it is not
this policy. See the root `README.md` for that distinction.

## Notes

- Paths in the scripts are relative to their own file, so they can be run from
  anywhere.
- `models/yolo11n.pt` is used by the G1 vision helpers; `models/yolov8n.pt` is
  used by the web backend.
- Training logs/checkpoints are generated artifacts and are ignored by git.
