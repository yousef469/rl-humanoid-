"""Live viewer that continuously loads the newest G1 checkpoint.

The matching VecNormalize statistics are loaded from the same checkpoint
directory. Close the viewer window to stop.
"""
import glob
import os
import time

import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from g1_env import G1Env

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
POLL_INTERVAL = 3.0


def latest_checkpoint():
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "ppo_g1_*_steps.zip"))
    if not files:
        return None
    return max(files, key=lambda path: int(path.rsplit("_", 2)[1]))


def vec_stats_for(checkpoint_path):
    step = checkpoint_path.rsplit("_", 2)[1]
    for name in (f"vec_normalize_{step}.pkl", "vec_normalize.pkl", "vec_normalize_best.pkl"):
        path = os.path.join(CHECKPOINT_DIR, name)
        if os.path.exists(path):
            return path
    return None


def main():
    base_env = DummyVecEnv([lambda: G1Env(domain_randomize=False)])
    env = base_env
    obs = env.reset()
    model = None
    last_path = None
    last_poll = 0.0

    with mujoco.viewer.launch_passive(base_env.envs[0].model, base_env.envs[0].data) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -20

        while viewer.is_running():
            now = time.time()
            if now - last_poll > POLL_INTERVAL:
                last_poll = now
                path = latest_checkpoint()
                if path and path != last_path:
                    last_path = path
                    model = PPO.load(path)
                    vec_path = vec_stats_for(path)
                    if vec_path:
                        env = VecNormalize.load(vec_path, base_env)
                        env.training = False
                        env.norm_reward = False
                    else:
                        env = base_env
                    print(f"Loaded {os.path.basename(path)} ({os.path.basename(vec_path) if vec_path else 'no norm stats'})")
                    obs = env.reset()

            if model is None:
                time.sleep(0.05)
                continue

            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            viewer.sync()
            time.sleep(0.01)
            if done[0]:
                obs = env.reset()

    base_env.close()


if __name__ == "__main__":
    main()
