"""Gymnasium environment for Unitree G1 walking with MuJoCo + PPO.
Action space: target joint positions (position control, kp=500 native)."""
import math
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

XML_PATH = os.path.join(os.path.dirname(__file__), "unitree_g1", "g1.xml")
FRAME_SKIP = 4
DT = 0.005 * FRAME_SKIP
MAX_STEPS = 2000
FALL_HEIGHT = 0.45
FALL_ORIENTATION = 0.7

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

NUM_OBS = 99
NUM_ACT = 29

DEFAULT_POSE = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0,
], dtype=np.float32)


class G1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, domain_randomize=True):
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None
        self.domain_randomize = domain_randomize
        self._step_count = 0
        self._prev_action = np.zeros(NUM_ACT, dtype=np.float32)
        self._last_action = np.zeros(NUM_ACT, dtype=np.float32)
        self._cmd = np.zeros(3, dtype=np.float32)
        self._cmd_change_steps = 0
        self.observation_space = spaces.Box(-np.inf, np.inf, (NUM_OBS,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (NUM_ACT,), dtype=np.float32)
        self._sample_command()

        self._setup_indices()
        for i in range(self.model.nu):
            self.model.actuator_gainprm[i, 0] = 100.0
            self.model.actuator_biasprm[i, 1] = -100.0
            self.model.actuator_biasprm[i, 2] = -20.0
        self._nominal_gainprm = self.model.actuator_gainprm.copy()
        self._nominal_biasprm = self.model.actuator_biasprm.copy()
        self._nominal_damping = self.model.dof_damping.copy()
        self._nominal_armature = self.model.dof_armature.copy()
        self._nominal_mass = self.model.body_mass.copy()
        self._nominal_friction = self.model.geom_friction.copy()
        self.reset()

    def _setup_indices(self):
        self._joint_ids = []
        self._jnt_qpos_adr = []
        self._jnt_dof_adr = []
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_ids.append(jid)
            self._jnt_qpos_adr.append(self.model.jnt_qposadr[jid])
            self._jnt_dof_adr.append(self.model.jnt_dofadr[jid])

        self._actuator_ids = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._actuator_ids[name] = i

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        rot = self.data.xmat[1].reshape(3, 3)
        base_vel = rot.T @ qvel[0:3]
        base_ang = qvel[3:6]
        grav = rot.T @ np.array([0, 0, -1])

        jpos = np.array([qpos[adr] for adr in self._jnt_qpos_adr], dtype=np.float32)
        jvel = np.array([qvel[adr] for adr in self._jnt_dof_adr], dtype=np.float32)

        obs = np.concatenate([
            base_vel.astype(np.float32),
            base_ang.astype(np.float32),
            grav.astype(np.float32),
            jpos,
            jvel,
            self._prev_action,
            self._cmd,
        ])
        obs = np.clip(obs, -100, 100)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10, neginf=-10)

        if self.domain_randomize:
            noise = np.random.normal(0, 0.02, obs.shape).astype(np.float32)
            obs += noise

        return obs

    def _sample_command(self):
        self._cmd = np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.5, 0.5),
        ], dtype=np.float32)
        self._cmd_change_steps = np.random.randint(100, 300)

    def _apply_action(self, action):
        self._last_action = self._prev_action.copy()
        action = np.clip(action, -1.0, 1.0)
        smoothed = self._prev_action * 0.6 + action * 0.4
        self._prev_action = smoothed.copy()
        for i, name in enumerate(JOINT_NAMES):
            aid = self._actuator_ids[name]
            jid = self._joint_ids[i]
            lo, hi = self.model.jnt_range[jid]
            target = DEFAULT_POSE[i] + smoothed[i] * (hi - lo) * 0.3
            target = np.clip(target, lo, hi)
            self.data.ctrl[aid] = target

    def _randomize_dynamics(self):
        # Body 0 is the world; never randomize it.
        for i in range(1, self.model.nbody):
            self.model.body_mass[i] = self._nominal_mass[i] * np.random.uniform(0.9, 1.1)

        for i in range(self.model.nu):
            scale = np.random.uniform(0.8, 1.2)
            self.model.actuator_gainprm[i, 0] = self._nominal_gainprm[i, 0] * scale
            self.model.actuator_biasprm[i, 1] = self._nominal_biasprm[i, 1] * scale
            self.model.actuator_biasprm[i, 2] = self._nominal_biasprm[i, 2] * scale

        for i in range(self.model.nv):
            self.model.dof_damping[i] = self._nominal_damping[i] * np.random.uniform(0.5, 2.0)
            self.model.dof_armature[i] = self._nominal_armature[i] * np.random.uniform(0.8, 1.2)

        self.model.geom_friction[:] = self._nominal_friction * np.random.uniform(0.8, 1.2, (self.model.ngeom, 1))

    def _compute_reward(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        rot = self.data.xmat[1].reshape(3, 3)
        base_vel_body = rot.T @ qvel[0:3]
        base_ang_body = qvel[3:6]

        vx, vy, _ = base_vel_body
        _, _, yaw_rate = base_ang_body

        vx_err = self._cmd[0] - vx
        vy_err = self._cmd[1] - vy
        yaw_err = self._cmd[2] - yaw_rate

        tracking = (
            math.exp(-vx_err * vx_err / 0.25) * 1.2
            + math.exp(-vy_err * vy_err / 0.25) * 0.15
            + math.exp(-yaw_err * yaw_err / 0.25) * 0.25
        )

        forward_bonus = vx * 0.5 if self._cmd[0] > 0.1 else 0.0

        torques = np.clip(self.data.qfrc_actuator, -30, 30)
        torque_penalty = -np.sum(np.abs(torques)) * 2e-5

        action_rate = -np.sum((self._prev_action - self._last_action) ** 2) * 0.01

        euler = self._quat_to_euler(qpos[3:7])
        roll, pitch = euler[0], euler[1]
        orient_penalty = -abs(roll) * 1.0 - abs(pitch) * 1.0

        return max(tracking + forward_bonus + torque_penalty + action_rate + orient_penalty, -10)

    def _quat_to_euler(self, quat):
        w, x, y, z = quat
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
        return np.array([roll, pitch, 0.0])

    def _is_fallen(self):
        qpos = self.data.qpos
        root_z = qpos[2]
        euler = self._quat_to_euler(qpos[3:7])
        return root_z < FALL_HEIGHT or abs(euler[0]) > FALL_ORIENTATION or abs(euler[1]) > FALL_ORIENTATION

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if self.domain_randomize:
            self._randomize_dynamics()

        qpos = self.data.qpos.copy()
        qpos[2] = 0.79
        qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for i, adr in enumerate(self._jnt_qpos_adr):
            qpos[adr] = DEFAULT_POSE[i]

        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_action = np.zeros(NUM_ACT, dtype=np.float32)
        self._last_action = np.zeros(NUM_ACT, dtype=np.float32)
        self._sample_command()

        # Sync initial ctrl to default pose
        for i, name in enumerate(JOINT_NAMES):
            aid = self._actuator_ids[name]
            self.data.ctrl[aid] = DEFAULT_POSE[i]

        return self._get_obs(), {}

    def step(self, action):
        delay = np.random.randint(0, 3) if self.domain_randomize else 0
        self._apply_action(action)
        for _ in range(FRAME_SKIP + delay):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        self._cmd_change_steps -= 1
        if self._cmd_change_steps <= 0:
            self._sample_command()

        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            obs = np.zeros(NUM_OBS, dtype=np.float32)
            reward = -5.0
            terminated = True
            truncated = False
            info = {"nan_detected": True}
            return obs, reward, terminated, truncated, info

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_fallen()
        truncated = self._step_count >= MAX_STEPS
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        self._setup_viewer()
        width, height = 640, 480
        self._viewer_cam.lookat[:] = self.data.qpos[:3]
        mujoco.mjv_updateScene(
            self.model, self.data, self._viewer_opt,
            None, self._viewer_cam, mujoco.mjCAT_ALL,
            self._viewer_scene,
        )
        mujoco.mjr_render(
            mujoco.MjrRect(0, 0, width, height),
            self._viewer_scene,
            self._viewer_context,
        )
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, width, height), self._viewer_context)
        return np.flipud(rgb)

    def _setup_viewer(self):
        if not hasattr(self, "_viewer_cam") or self._viewer_cam is None:
            self._viewer_cam = mujoco.MjvCamera()
            self._viewer_cam.azimuth = 45
            self._viewer_cam.distance = 3.0
            self._viewer_cam.elevation = -30
            self._viewer_opt = mujoco.MjvOption()
            self._viewer_scene = mujoco.MjvScene(self.model, maxgeom=1000)
            self._viewer_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def close(self):
        self.viewer = None


if __name__ == "__main__":
    env = G1Env()
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if i % 20 == 0:
            print(f"Step {i}: reward={reward:.3f} fallen={terminated}")
        if terminated:
            print(f"Fell at step {i}")
            break
    print("Done")
    env.close()
