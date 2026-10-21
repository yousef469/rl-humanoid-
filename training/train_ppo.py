"""PPO training script for HumanoidEnv using Stable-Baselines3."""
import os
import sys
import argparse
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from humanoid_env import HumanoidEnv

TENSORBOARD_LOG = os.path.join(os.path.dirname(__file__), "logs")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
POLICY_DIR = os.path.join(os.path.dirname(__file__), "policy")

def make_env(rank, render_mode=None, seed=0, log_dir=TENSORBOARD_LOG):
    def _init():
        env = HumanoidEnv(render_mode=render_mode)
        env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}"))
        env.reset(seed=seed + rank)
        return env
    return _init


class LiveViewCallback(BaseCallback):
    def __init__(self, render_freq=1000, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.viewer_env = None
        self.viewer = None

    def _init_callback(self):
        import mujoco.viewer
        self.viewer_env = HumanoidEnv(domain_randomize=False)
        obs, _ = self.viewer_env.reset()
        self._last_obs = obs
        self.viewer = mujoco.viewer.launch_passive(
            self.viewer_env.model, self.viewer_env.data
        )
        self.viewer.__enter__()
        self.viewer.cam.azimuth = 45
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -30

    def _on_step(self):
        if self.n_calls % self.render_freq != 0:
            return True
        if self.viewer is None or not self.viewer.is_running():
            return True
        with torch.no_grad():
            action, _ = self.model.predict(self._last_obs, deterministic=True)
        self._last_obs, _, terminated, truncated, _ = self.viewer_env.step(action)
        self.viewer.sync()
        if terminated or truncated:
            self._last_obs, _ = self.viewer_env.reset()
        return True

    def _on_training_end(self):
        if self.viewer is not None:
            self.viewer.__exit__(None, None, None)
            self.viewer = None
        if self.viewer_env is not None:
            self.viewer_env.close()


def train(args):
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(POLICY_DIR, exist_ok=True)

    env = DummyVecEnv([make_env(i, seed=args.seed) for i in range(args.num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=100.0)

    eval_env = DummyVecEnv([make_env(0, seed=args.seed + 999)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log=TENSORBOARD_LOG,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
            activation_fn=torch.nn.ReLU if args.activation == "relu" else torch.nn.Tanh,
        ),
        verbose=1,
        device="auto",
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=CHECKPOINT_DIR,
            log_path=TENSORBOARD_LOG,
            eval_freq=args.eval_freq,
            deterministic=True,
            render=False,
        ),
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=CHECKPOINT_DIR,
            name_prefix="ppo_humanoid",
        ),
    ]

    if args.live_view:
        callbacks.append(LiveViewCallback(render_freq=args.live_freq))

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(os.path.join(POLICY_DIR, "ppo_humanoid_final.zip"))
    env.save(os.path.join(POLICY_DIR, "vec_normalize.pkl"))
    print(f"Model saved to {POLICY_DIR}")
    env.close()
    eval_env.close()


def test(args):
    if not args.model_path:
        args.model_path = os.path.join(POLICY_DIR, "ppo_humanoid_final.zip")
    model = PPO.load(args.model_path)
    env = DummyVecEnv([make_env(0)])
    vec_path = os.path.join(os.path.dirname(args.model_path), "vec_normalize.pkl")
    if os.path.exists(vec_path):
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False

    obs = env.reset()
    total_reward = 0
    for i in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        if done:
            print(f"Step {i}, total reward: {total_reward:.2f}")
            obs = env.reset()
            total_reward = 0
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--live-view", action="store_true")
    parser.add_argument("--live-freq", type=int, default=250)

    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--checkpoint-freq", type=int, default=50000)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        test(args)
