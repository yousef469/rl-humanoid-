"""G1 walking demo with optional YOLO camera overlay.

Steering uses the simulator's ground-truth goal position (oracle), not YOLO.
The YOLO window is a visualization of the robot's first-person camera.
"""
import os
import time
import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
import cv2
from ultralytics import YOLO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
SCENE_PATH = os.path.join(BASE_DIR, "unitree_g1", "scene_vision.xml")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
YOLO_PATH = os.path.join(ROOT_DIR, "models", "yolo11n.pt")
MODEL_STEP = 15000000

IMG_W, IMG_H = 320, 240
CMD_VX = 0.8
YAW_GAIN = 1.5
GOAL_THRESH = 0.5

NUM_OBS = 99
NUM_ACT = 29
FRAME_SKIP = 4
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

DEFAULT_POSE = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0,
], dtype=np.float32)

# ── Dummy env for VecNormalize ──────────────────────────────────────────────
class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (NUM_OBS,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (NUM_ACT,), dtype=np.float32)
    def reset(self): return np.zeros(NUM_OBS, dtype=np.float32), {}
    def step(self, a): return np.zeros(NUM_OBS, dtype=np.float32), 0.0, False, False, {}
    def render(self): return None
    def close(self): pass

# ── Load model & policy ─────────────────────────────────────────────────────
print("Loading MuJoCo scene...")
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)

for i in range(model.nu):
    model.actuator_gainprm[i, 0] = 100.0
    model.actuator_biasprm[i, 1] = -100.0
    model.actuator_biasprm[i, 2] = -20.0

print("Loading walking policy...")
policy = PPO.load(os.path.join(CKPT_DIR, f"ppo_g1_{MODEL_STEP}_steps.zip"))
policy.policy.eval()

dummy_vec = DummyVecEnv([lambda: DummyEnv()])
norm_env = VecNormalize.load(os.path.join(CKPT_DIR, f"vec_normalize_{MODEL_STEP}.pkl"), dummy_vec)
norm_env.training = False
norm_env.norm_reward = False

# ── Helper indices ──────────────────────────────────────────────────────────
_joint_ids = []
_jnt_qpos_adr = []
_jnt_dof_adr = []
for name in JOINT_NAMES:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    _joint_ids.append(jid)
    _jnt_qpos_adr.append(model.jnt_qposadr[jid])
    _jnt_dof_adr.append(model.jnt_dofadr[jid])

_actuator_ids = {}
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if name:
        _actuator_ids[name] = i

_prev_action = np.zeros(NUM_ACT, dtype=np.float32)
_last_action = np.zeros(NUM_ACT, dtype=np.float32)

def build_obs(cmd):
    qpos, qvel = data.qpos, data.qvel
    rot = data.xmat[1].reshape(3, 3)
    base_vel = rot.T @ qvel[0:3]
    base_ang = qvel[3:6]
    grav = rot.T @ np.array([0, 0, -1])
    jpos = np.array([qpos[adr] for adr in _jnt_qpos_adr], dtype=np.float32)
    jvel = np.array([qvel[adr] for adr in _jnt_dof_adr], dtype=np.float32)
    obs = np.concatenate([
        base_vel.astype(np.float32), base_ang.astype(np.float32),
        grav.astype(np.float32), jpos, jvel, _prev_action, cmd,
    ])
    return np.clip(np.nan_to_num(obs, nan=0.0, posinf=10, neginf=-10), -100, 100)

def apply_action(action):
    global _prev_action, _last_action
    _last_action = _prev_action.copy()
    smoothed = _prev_action * 0.6 + np.clip(action, -1, 1) * 0.4
    _prev_action = smoothed.copy()
    for i, name in enumerate(JOINT_NAMES):
        aid = _actuator_ids[name]
        lo, hi = model.jnt_range[_joint_ids[i]]
        target = np.clip(DEFAULT_POSE[i] + smoothed[i] * (hi - lo) * 0.3, lo, hi)
        data.ctrl[aid] = target

def reset_robot():
    global _prev_action, _last_action
    mujoco.mj_resetData(model, data)
    qpos = data.qpos.copy()
    qpos[2] = 0.79
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, adr in enumerate(_jnt_qpos_adr):
        qpos[adr] = DEFAULT_POSE[i]
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    _prev_action[:] = 0
    _last_action[:] = 0
    for i, name in enumerate(JOINT_NAMES):
        data.ctrl[_actuator_ids[name]] = DEFAULT_POSE[i]
    mujoco.mj_forward(model, data)

def fallen():
    qpos = data.qpos
    w, x, y, z = qpos[3:7]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return qpos[2] < 0.45 or abs(roll) > 0.7

# ── Body IDs ────────────────────────────────────────────────────────────────
torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")

# ── YOLO ────────────────────────────────────────────────────────────────────
print("Loading YOLO...")
yolo = YOLO(YOLO_PATH)

# ── Renderer ────────────────────────────────────────────────────────────────
renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

# ── Viewer loop ─────────────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 45
    viewer.cam.distance = 4.0
    viewer.cam.elevation = -20

    reset_robot()
    cmd = np.array([CMD_VX, 0.0, 0.0], dtype=np.float32)
    obs = build_obs(cmd)
    step = 0
    last_yolo_time = 0
    reached_count = 0
    bearing = 0.0
    yolo_warned = False

    print("Running vision-guided walking. Close viewer to exit.")
    while viewer.is_running():
        torso_pos = data.xpos[torso_id].copy()
        torso_mat = data.xmat[torso_id].reshape(3, 3)
        forward = torso_mat[:, 0]
        right = torso_mat[:, 1]
        goal_pos = data.xpos[goal_id].copy()
        goal_dir = goal_pos - torso_pos
        goal_dir[2] = 0
        dist = np.linalg.norm(goal_dir)

        # ── Steering ──
        if dist > GOAL_THRESH:
            gd = goal_dir / dist
            bearing = np.arctan2(np.dot(gd, right), np.dot(gd, forward))
            cmd[0] = CMD_VX * np.clip(np.dot(gd, forward), 0.3, 1.0)
            cmd[2] = np.clip(-bearing * YAW_GAIN, -1.0, 1.0)
        else:
            cmd[0] = cmd[2] = 0.0

        # ── Policy step ──
        obs_norm = norm_env.normalize_obs(obs.reshape(1, -1)).flatten()
        action, _ = policy.predict(obs_norm.reshape(1, -1), deterministic=True)
        apply_action(action.flatten())
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(model, data)

        step += 1
        obs = build_obs(cmd)

        # ── YOLO camera (throttled to ~10 fps) ──
        if time.time() - last_yolo_time > 0.1:
            last_yolo_time = time.time()
            try:
                renderer.update_scene(data, camera="g1_eye")
                rgb = np.ascontiguousarray(renderer.render())
                results = yolo(rgb, verbose=False)
                annotated = cv2.resize(results[0].plot(), (640, 480))
                info = f"Goal: {np.degrees(bearing):.0f}deg  {dist:.1f}m" if dist > GOAL_THRESH else "GOAL!"
                cv2.putText(annotated, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("G1 Vision - YOLO", annotated)
                cv2.waitKey(1)
            except Exception as exc:
                if not yolo_warned:
                    print(f"YOLO overlay disabled: {exc}")
                    yolo_warned = True

        # ── Reset on fall or goal ──
        if fallen() or step > 2000:
            reset_robot()
            cmd = np.array([CMD_VX, 0.0, 0.0], dtype=np.float32)
            obs = build_obs(cmd)
            step = 0
        elif dist < GOAL_THRESH:
            reached_count += 1
            print(f"Goal reached! ({reached_count}) Repositioning goal...")
            # Randomize goal position
            goal_pos_new = np.random.uniform([1, -2, 0.05], [5, 2, 0.05])
            model.body_pos[goal_id] = goal_pos_new
            mujoco.mj_forward(model, data)
            cmd = np.array([CMD_VX, 0.0, 0.0], dtype=np.float32)
            obs = build_obs(cmd)
            step = 0

        viewer.sync()
