# VRE Robot

A browser-controlled humanoid robot demo with a FastAPI + PyBullet backend, a
Three.js frontend, a head-camera YOLO red-ball tracker, optional language
models, and a separate MuJoCo training pipeline for a Unitree G1 policy.

## What runs

- **Backend** (`backend/`) owns the simulation, command queue, perception, and
  websocket state stream. It serves the frontend.
- **Red ball** is spawned 3 m in front of the robot. A YOLO11n model runs on the
  head camera and the robot can autonomously turn and walk to it.
- **Frontend** (`frontend/`) renders the robot, scene objects, camera overlay,
  and chat in a Three.js scene.
- **Training** (`training/`) contains command-model, red-ball YOLO, and MuJoCo
  G1 PPO pipelines. It is not loaded by the backend.

## Quick start

```bash
cd backend
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
python main.py
```

Open <http://localhost:8000>. The frontend is served by FastAPI.

The repo also includes a local `rl_env/` virtualenv. If you use it, invoke pip
through Python because its shebang points at an old path:

```bash
rl_env/bin/python -m pip install -r backend/requirements.txt
```

### Language model (optional)

The default command path expects Ollama:

```bash
ollama serve
ollama pull qwen2.5:0.5b
```

If Ollama is unavailable, the robot still renders and the websocket still
streams state, but natural-language commands will not be executed.

### Red-ball vision and seeking

The repository ships a single-class YOLO11n detector at
`models/red_ball_yolo.pt`, trained on synthetic frames rendered by the same
PyBullet head camera. It runs asynchronously on the head camera, with a red
colour check to reject false positives.

To use it:

- Click **Find Red Ball** in the browser, or type **“go to the red ball”**.
- The backend intercepts that request and starts closed-loop seeking.
- The robot turns toward the detected ball, walks to it, and stops inside
  `0.65 m`.
- A direct line-of-sight raycast supplies a precise ball centre and acts as a
  fallback if the ball is briefly missed by YOLO.

The detector is trained with:

```bash
cd training
python train_red_ball_yolo.py
```

Raycast perception still works without YOLO. If `ultralytics` is missing, the
red ball can still be tracked by raycast, but there is no camera detection or
overlay.

### Trained command model (optional)

After training and exporting the T5 model as described in
[`training/README.md`](training/README.md):

```bash
cd backend
USE_TRAINED=1 ../.venv/bin/python main.py
```

This requires `training/trained_model/robot_model.onnx` and the optional
packages in `backend/requirements-train.txt`.

## Architecture notes

- **Idle standing** is simulated with PyBullet contacts and joint motors.
- **Commanded walking and turning** use a kinematically guided base with an
  animated leg gait. This keeps the demo stable; it is not the trained MuJoCo
  policy.
- The backend language model only emits body commands:
  `walk {distance_m, speed}`, `turn {degrees}`, `wave {arm}`,
  `seek_red_ball`, `stop`.
- Red-ball seeking is a direct visual-servo behaviour. It intentionally skips
  the general obstacle-avoidance branch while chasing the target.
- `USE_TRAINED` switches the language/command reasoner, not the walking
  controller.
- `training/` is a separate MuJoCo environment. Its G1 policy is not wired into
  the backend.

## Project layout

```text
backend/      FastAPI app, PyBullet world, controller, perception, reasoners
frontend/     Three.js scene and websocket client
models/       YOLO weights, including red_ball_yolo.pt
training/     T5 command model, red-ball YOLO, and MuJoCo G1 PPO
rl_env/       Local virtualenv (not part of the app)
```

## Verification

```bash
python3 -m compileall -q backend training
cd backend && ../rl_env/bin/python test_walk.py 1.0 30 1
for f in frontend/js/*.js; do node --check "$f"; done
```

`test_walk.py` exercises the controller headlessly. The frontend is static and
does not require a build step.

Red-ball integration test (YOLO + controller, about 10 s of simulated time):

```bash
cd backend
../rl_env/bin/python - <<'PY'
from physics_world import PhysicsWorld
from perception import Perception
from controller import Controller
from models import RobotState

pw = PhysicsWorld(); perc = Perception(pw); ctrl = Controller(pw)
ctrl.start_red_ball_seek()
for i in range(900):
    ctrl.tick(1 / 60)
    rs = RobotState(ctrl.current_position, ctrl.current_rotation, ctrl.current_joints)
    ctrl.set_scene_objects(perc.get_scene(rs).objects)
    ctrl.set_red_ball_target(perc.get_red_ball_target())
    if not ctrl.seeking_red_ball and i > 60:
        print('found ball at', ctrl.current_position)
        break
perc.close(); pw.close()
PY
```

## Known limitations

- Walking is a stable demo gait, not a dynamically balanced policy.
- The T5 artifact is not included; `USE_TRAINED=1` fails clearly until it is
  trained/exported.
- The red-ball YOLO model is a small synthetic single-class detector. It is not
  a general COCO detector and may need retraining for other ball appearances,
  lighting, or real cameras.
- Red-ball seeking walks straight at the target and intentionally skips the
  general obstacle-avoidance branch.
- Older PPO checkpoints may lack a matching `vec_normalize_best.pkl`; new
  training runs save one alongside the best model.
