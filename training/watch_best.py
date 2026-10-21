"""View a saved G1 policy with the observation-normalization stats it was trained with.

Examples:
    python watch_best.py
    python watch_best.py --checkpoint ppo_g1_15000000_steps
"""
import argparse
import os
import re

import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from g1_env import G1Env

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def find_vec_path(checkpoint):
    if checkpoint == "best_model":
        candidates = ["vec_normalize_best.pkl", "vec_normalize.pkl"]
    else:
        match = re.search(r"(\d+)_steps", checkpoint)
        candidates = ([f"vec_normalize_{match.group(1)}.pkl"] if match else []) + [
            "vec_normalize.pkl",
            "vec_normalize_best.pkl",
        ]
    for name in candidates:
        path = os.path.join(CHECKPOINT_DIR, name)
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_model",
                        help="Checkpoint name without .zip (e.g. best_model, ppo_g1_15000000_steps)")
    args = parser.parse_args()

    model_path = os.path.join(CHECKPOINT_DIR, f"{args.checkpoint}.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model = PPO.load(model_path)
    env = DummyVecEnv([lambda: G1Env(domain_randomize=False)])

    vec_path = find_vec_path(args.checkpoint)
    if vec_path:
        print(f"Using normalization stats: {os.path.basename(vec_path)}")
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: no matching VecNormalize stats found; observations will not be normalized.")

    obs = env.reset()
    viewer = mujoco.viewer.launch_passive(env.envs[0].model, env.envs[0].data)
    viewer.cam.azimuth = 45
    viewer.cam.distance = 3.5
    viewer.cam.elevation = -20

    step = 0
    while viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _ = env.step(action)
        viewer.sync()
        step += 1
        if terminated[0] or step > 2000:
            obs = env.reset()
            step = 0

    env.close()


if __name__ == "__main__":
    main()
