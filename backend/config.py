import os

JOINT_LIMITS = {
    "neck_pitch": (-0.5, 0.5),
    "neck_yaw": (-0.8, 0.8),
    "l_shoulder_pitch": (-1.5, 1.5),
    "l_shoulder_roll": (-0.5, 1.0),
    "l_shoulder_yaw": (-1.0, 1.0),
    "l_elbow": (-1.5, 0.0),
    "r_shoulder_pitch": (-1.5, 1.5),
    "r_shoulder_roll": (-1.0, 0.5),
    "r_shoulder_yaw": (-1.0, 1.0),
    "r_elbow": (-1.5, 0.0),
    "l_hip_yaw": (-0.5, 0.5),
    "l_hip_roll": (-0.3, 0.5),
    "l_hip_pitch": (-0.5, 1.2),
    "l_knee": (0.0, 1.5),
    "l_ankle": (-0.5, 0.5),
    "r_hip_yaw": (-0.5, 0.5),
    "r_hip_roll": (-0.5, 0.3),
    "r_hip_pitch": (-0.5, 1.2),
    "r_knee": (0.0, 1.5),
    "r_ankle": (-0.5, 0.5),
    "torso_twist": (-0.5, 0.5),
    "torso_bend": (-0.4, 0.6),
}

BONE_LENGTHS = {
    "torso": 0.32,
    "upper_arm": 0.32,
    "forearm": 0.30,
    "thigh": 0.38,
    "shin": 0.36,
    "foot_height": 0.07,
    "neck": 0.12,
    "head": 0.18,
}

STEP_DURATION = 0.5

USE_TRAINED = os.getenv("USE_TRAINED", "").lower() in ("1", "true", "yes")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:0.5b")

WS_PORT = int(os.getenv("WS_PORT", "8000"))
TICK_RATE = 60
PHYSICS_TIMESTEP = 1.0 / 240.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_DIR = os.path.join(BASE_DIR, "models")
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "red_ball_yolo.pt")
YOLO_FALLBACK_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")
RED_BALL_STOP_DIST = 0.65
