import math
import numpy as np

from models import LLMCommand
from config import JOINT_LIMITS, TICK_RATE, RED_BALL_STOP_DIST
from state_estimator import StateEstimator
from centroidal_mpc import CentroidalMPC


class Controller:
    def __init__(self, physics_world):
        self.physics_world = physics_world
        self.state_estimator = StateEstimator(physics_world)
        self.mpc = CentroidalMPC(dt=0.025, horizon_steps=24)

        self._speech = ""
        self._step_counter = 0
        self._sim_time = 0.0
        self._mpc_counter = 0
        self._mpc_interval = 3

        self.vx_des = 0.0
        self.vy_des = 0.0

        self._mpc_solution = None
        self._state_est = None

        self._prev_pitch = 0.0
        self._prev_roll = 0.0

        self._phase = 0.0
        self._step_duration = 0.35
        self._walk_enabled = False
        self._recovery_mode = False
        self.yaw_rate_des = 0.0
        self._scene_objects = []
        self._obstacle_yaw = 0.0
        self._obstacle_until = 0

        self._walk_start = None
        self._walk_stop_dist = None
        self._turn_target_yaw = None
        self._turn_sign = 0
        self._wave_until = 0
        self._wave_arm = "right"
        self._ramp_speed = 0.0

        self._seeking_red_ball = False
        self._red_ball_target = None
        self._red_ball_last_seen = -10**9
        self._red_ball_search_dir = 1
        self._neck_yaw = 0.0
        self._neck_pitch = 0.0

        pos0, yaw0 = physics_world.get_robot_position()
        self._stand_z = pos0[2]
        self._heading = yaw0
        self._walk_pos = [pos0[0], pos0[1]]

    @property
    def current_joints(self):
        return self.physics_world.get_joint_angles()

    @property
    def current_position(self):
        pos, _ = self.physics_world.get_robot_position()
        return pos

    @property
    def current_rotation(self):
        _, yaw = self.physics_world.get_robot_position()
        return yaw

    @property
    def gait_phase(self):
        return self._phase

    @property
    def foot_contacts(self):
        return self.state_estimator.get_contact_state()

    @property
    def walking(self):
        return self._walk_enabled

    @property
    def seeking_red_ball(self):
        return self._seeking_red_ball

    @property
    def state_est(self):
        return self._state_est

    @property
    def obstacle_warning(self):
        return self._obstacle_yaw != 0.0 or self._step_counter < self._obstacle_until

    def set_speech(self, text):
        self._speech = text

    def get_speech(self):
        s = self._speech
        self._speech = ""
        return s

    def enqueue(self, commands: list[LLMCommand]):
        for cmd in commands:
            if cmd.action in ("seek_red_ball", "go_to_red_ball", "find_red_ball"):
                self.start_red_ball_seek()
                continue
            if cmd.action in ("walk", "turn", "stop"):
                self._seeking_red_ball = False
            if cmd.action == "stop":
                self._stop_motion()
                return
            elif cmd.action == "walk":
                self._heading = self.current_rotation
                self._walk_pos = list(self.current_position[:2])
                speed = cmd.params.get("speed", 0.5)
                self.vx_des = min(max(speed * 0.4, 0.0), 0.6)
                self._walk_enabled = True
                distance = cmd.params.get("distance_m")
                if distance is not None:
                    self._walk_start = self.current_position[:2]
                    self._walk_stop_dist = max(0.1, float(distance))
                else:
                    self._walk_start = None
                    self._walk_stop_dist = None
            elif cmd.action == "turn":
                if "degrees" in cmd.params:
                    self._start_turn(float(cmd.params["degrees"]))
                else:
                    yaw = float(cmd.params.get("yaw_rate", 0.0))
                    self._turn_target_yaw = None
                    self.yaw_rate_des = max(-0.5, min(0.5, yaw))
            elif cmd.action == "wave":
                self._wave_until = self._step_counter + int(2.0 * TICK_RATE)
                self._wave_arm = cmd.params.get("arm", "right")
            elif cmd.action == "speak" or cmd.action == "speech":
                self.set_speech(cmd.params.get("text", ""))

    def _stop_motion(self):
        self.vx_des = 0.0
        self.vy_des = 0.0
        self.yaw_rate_des = 0.0
        self._walk_enabled = False
        self._walk_start = None
        self._walk_stop_dist = None
        self._turn_target_yaw = None
        self._seeking_red_ball = False
        self._red_ball_target = None
        self._neck_yaw = 0.0
        self._neck_pitch = 0.0

    def start_red_ball_seek(self):
        self._stop_motion()
        self._seeking_red_ball = True
        self._heading = self.current_rotation
        self._walk_pos = list(self.current_position[:2])
        self._red_ball_last_seen = -10**9
        self._red_ball_search_dir = 1
        self.set_speech("Looking for the red ball.")

    def set_red_ball_target(self, target):
        if target:
            self._red_ball_target = target
            self._red_ball_last_seen = self._step_counter

    def _update_red_ball_seek(self, dt):
        if not self._seeking_red_ball:
            return

        age = self._step_counter - self._red_ball_last_seen
        if self._red_ball_target is None or age > int(3.0 * TICK_RATE):
            self._red_ball_target = None
            self._walk_enabled = False
            self.vx_des = 0.0
            self.yaw_rate_des = 0.35 * self._red_ball_search_dir
            self._neck_yaw = 0.0
            self._neck_pitch = 0.0
            return

        target = self._red_ball_target
        tx, ty = target["position"][0], target["position"][1]
        px, py, _ = self.current_position
        dx, dy = tx - px, ty - py
        distance = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - self._heading
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        self._neck_yaw = max(-0.6, min(0.6, bearing * 0.8))
        self._neck_pitch = max(-0.2, min(0.5, math.atan2(
            max(0.0, self._stand_z + 0.55 - target["position"][2]),
            max(distance, 0.2),
        )))

        if distance <= RED_BALL_STOP_DIST:
            self._seeking_red_ball = False
            self._red_ball_target = None
            self._walk_enabled = False
            self.vx_des = 0.0
            self.yaw_rate_des = 0.0
            self.set_speech("I found the red ball.")
            return

        # When the ball gets very close it drops below the head camera. Keep
        # walking toward its last known position instead of losing the lock.
        self.yaw_rate_des = max(-0.5, min(0.5, bearing * 1.1))
        if abs(bearing) <= 0.4:
            self._walk_enabled = True
            self.vx_des = max(0.08, min(0.4, (distance - RED_BALL_STOP_DIST) * 0.5))
        else:
            self._walk_enabled = False
            self.vx_des = 0.0

    def _start_turn(self, degrees):
        while degrees > 180.0:
            degrees -= 360.0
        while degrees < -180.0:
            degrees += 360.0
        if abs(degrees) < 1.0:
            self.yaw_rate_des = 0.0
            self._turn_target_yaw = None
            return
        rate = 0.35 if degrees > 0 else -0.35
        travel = math.radians(degrees)
        self.yaw_rate_des = rate
        self._turn_sign = 1 if degrees > 0 else -1
        self._heading = self.current_rotation
        self._turn_target_yaw = self._heading + travel

    def _apply_wave(self, targets):
        if self._step_counter >= self._wave_until:
            return
        side = "r" if self._wave_arm == "right" else "l"
        t = math.sin(self._step_counter * 0.35)
        targets[f"{side}_shoulder_pitch"] = -1.3 + 0.25 * t
        targets[f"{side}_elbow"] = -0.7 + 0.35 * t
        targets[f"{side}_shoulder_roll"] = -0.25 if side == "r" else 0.25

    def _update_turn(self, yaw):
        if self._turn_target_yaw is None:
            return
        diff = math.atan2(
            math.sin(self._turn_target_yaw - yaw),
            math.cos(self._turn_target_yaw - yaw),
        )
        if abs(diff) < 0.02 or diff * self._turn_sign < 0:
            self.yaw_rate_des = 0.0
            self._turn_target_yaw = None
            self._turn_sign = 0

    def _update_walk_distance(self, pos):
        if self._walk_stop_dist is None or self._walk_start is None:
            return
        dx = pos[0] - self._walk_start[0]
        dy = pos[1] - self._walk_start[1]
        if math.hypot(dx, dy) >= self._walk_stop_dist:
            self._walk_enabled = False
            self.vx_des = 0.0
            self._walk_stop_dist = None
            self._walk_start = None

    def _move_kinematic(self, dt):
        if not self._walk_enabled and abs(self.yaw_rate_des) < 0.01:
            return
        self._heading += self.yaw_rate_des * dt
        step = self.vx_des * dt
        self._walk_pos[0] += math.cos(self._heading) * step
        self._walk_pos[1] += math.sin(self._heading) * step
        self.physics_world.reset_robot_position(
            self._walk_pos[0], self._walk_pos[1], self._stand_z, self._heading
        )
        self._update_walk_distance((self._walk_pos[0], self._walk_pos[1], self._stand_z))

    def tick(self, dt) -> dict:
        if not self._seeking_red_ball:
            decay = max(0.0, 1.0 - dt * 4.0)
            self._neck_yaw *= decay
            self._neck_pitch *= decay
        self._update_red_ball_seek(dt)
        self._update_turn(self._heading)
        self._move_kinematic(dt)
        state = self.state_estimator.estimate(dt)
        self._state_est = state
        self._sim_time += dt

        self._mpc_counter += 1
        if self._mpc_counter >= self._mpc_interval:
            self._mpc_counter = 0
            self._solve_mpc()

        pitch = state["base_orn"][0]
        roll = state["base_orn"][1]
        pitch_vel = (pitch - self._prev_pitch) / max(dt, 1e-6)
        roll_vel = (roll - self._prev_roll) / max(dt, 1e-6)
        self._prev_pitch = pitch
        self._prev_roll = roll

        mpc_speed = 0.0
        if self._mpc_solution is not None:
            X_sol, _ = self._mpc_solution
            mpc_speed = abs(X_sol[1, 0]) + abs(X_sol[3, 0])

        turning = abs(self.yaw_rate_des) > 0.01
        speed = max(self.vx_des, mpc_speed * 0.3)
        if not self._walk_enabled and not turning:
            speed = 0.0
        elif turning:
            speed = max(speed, 0.05)

        self._ramp_speed += (speed - self._ramp_speed) * min(1.0, dt * 3.0)

        eff_speed = self._ramp_speed
        amp = max(0.4, min(1.0, eff_speed / 0.12))

        kp_p = 0.8 + 0.6 * amp
        kd_p = 0.15 + 0.10 * amp
        kp_r = 0.3 + 0.2 * amp
        kd_r = 0.05 + 0.05 * amp

        pitch_corr = -(kp_p * pitch + kd_p * pitch_vel)
        roll_corr = -(kp_r * roll + kd_r * roll_vel)
        corr_limit = 0.2 + 0.2 * amp
        pitch_corr = max(-corr_limit, min(corr_limit, pitch_corr))
        roll_corr = max(-corr_limit * 0.5, min(corr_limit * 0.5, roll_corr))

        recovery_thresh = 0.25 + 0.15 * amp
        if abs(pitch) > recovery_thresh or abs(roll) > recovery_thresh * 0.8:
            self._recovery_mode = True
        else:
            self._recovery_mode = False

        if self._recovery_mode or eff_speed < 0.01:
            r = 1.5 if self._recovery_mode else 1.0
            self.physics_world.reset_robot_position(
                self._walk_pos[0], self._walk_pos[1], self._stand_z, self._heading
            )
            yaw_err = math.atan2(math.sin(state["base_orn"][2]),
                                 math.cos(state["base_orn"][2]))
            yaw_corr = max(-0.3, min(0.3, -0.5 * yaw_err))
            joint_targets = {
                'l_hip_yaw': yaw_corr, 'l_hip_roll': 0.0,
                'l_hip_pitch': 0.0,
                'l_knee': 0.06, 'l_ankle': -0.06,
                'r_hip_yaw': yaw_corr, 'r_hip_roll': 0.0,
                'r_hip_pitch': 0.0,
                'r_knee': 0.06, 'r_ankle': -0.06,
                'l_shoulder_pitch': 0.15 * r, 'r_shoulder_pitch': 0.15 * r,
                'l_elbow': -0.3, 'r_elbow': -0.3,
                'neck_pitch': self._neck_pitch, 'neck_yaw': self._neck_yaw,
                'torso_twist': 0.0, 'torso_bend': 0.0,
                'l_shoulder_roll': 0.0, 'r_shoulder_roll': 0.0,
                'l_shoulder_yaw': 0.0, 'r_shoulder_yaw': 0.0,
            }
            self._apply_wave(joint_targets)
            for name in joint_targets:
                lim = JOINT_LIMITS.get(name)
                if lim:
                    joint_targets[name] = max(lim[0], min(lim[1], joint_targets[name]))
            self.physics_world.set_joint_targets_walk(joint_targets)
            self.physics_world.step(dt)
            self._step_counter += 1
            self._update_walk_distance(self.current_position)
            return self.physics_world.get_joint_angles()

        step_dur = max(0.20, 0.40 - eff_speed * 0.3)
        self._phase += dt / step_dur

        phase = self._phase
        sin_2pi = math.sin(phase * 2.0 * math.pi)
        cos_2pi = math.cos(phase * 2.0 * math.pi)

        right_swing = (phase % 1.0) < 0.5
        swing_t = ((phase % 1.0) / 0.5) if right_swing else ((phase % 1.0) - 0.5) / 0.5
        swing_env = (math.sin(swing_t * math.pi) ** 2) if 0 < swing_t < 1 else 0

        P_AMP = 0.18
        K_AMP = 0.28
        S_AMP = 0.08

        hip_pitch_l = -P_AMP * sin_2pi
        hip_pitch_r = P_AMP * sin_2pi

        l_knee_t = 0.06 + K_AMP * swing_env if not right_swing else 0.06
        r_knee_t = 0.06 + K_AMP * swing_env if right_swing else 0.06

        joint_targets = {
            'l_hip_yaw': 0.0, 'l_hip_roll': 0.0,
            'l_hip_pitch': hip_pitch_l,
            'l_knee': l_knee_t,
            'l_ankle': -(hip_pitch_l + l_knee_t),
            'r_hip_yaw': 0.0, 'r_hip_roll': 0.0,
            'r_hip_pitch': hip_pitch_r,
            'r_knee': r_knee_t,
            'r_ankle': -(hip_pitch_r + r_knee_t),
            'l_shoulder_pitch': -S_AMP * sin_2pi,
            'r_shoulder_pitch': S_AMP * sin_2pi,
            'l_shoulder_roll': 0.0, 'r_shoulder_roll': 0.0,
            'l_shoulder_yaw': 0.0, 'r_shoulder_yaw': 0.0,
            'l_elbow': -0.3, 'r_elbow': -0.3,
            'neck_pitch': self._neck_pitch, 'neck_yaw': self._neck_yaw,
            'torso_twist': 0.0, 'torso_bend': -0.01,
        }

        self._apply_wave(joint_targets)
        for name in joint_targets:
            lim = JOINT_LIMITS.get(name)
            if lim:
                joint_targets[name] = max(lim[0], min(lim[1], joint_targets[name]))

        self.physics_world.set_joint_targets_walk(joint_targets)
        self.physics_world.step(dt)

        self._step_counter += 1
        self._update_walk_distance(self.current_position)
        return self.physics_world.get_joint_angles()

    def _solve_mpc(self):
        state = self._state_est
        if state is None:
            return

        if not self._walk_enabled or self._recovery_mode:
            return

        lf = state.get("l_foot_pos", np.zeros(3))
        rf = state.get("r_foot_pos", np.zeros(3))

        try:
            X, U = self.mpc.solve(
                com_pos=state["com_pos"],
                com_vel=state["com_vel"],
                vx_des=self.vx_des,
                vy_des=self.vy_des,
                left_foot_pos=lf,
                right_foot_pos=rf,
                step_number=self._step_counter,
            )
            self._mpc_solution = (X, U)
        except Exception:
            pass

    def set_scene_objects(self, objects):
        self._scene_objects = objects
        if not self._walk_enabled or self._seeking_red_ball:
            return
        if self._step_counter < self._obstacle_until:
            return
        for obj in objects:
            if obj.label == "red_ball":
                continue
            if abs(obj.angle) > 35:
                continue
            if obj.distance > 3.0 or obj.distance < 0.5:
                continue
            if obj.angle > 0:
                self._obstacle_yaw = -0.3
            else:
                self._obstacle_yaw = 0.3
            self._obstacle_until = self._step_counter + int(2.0 * TICK_RATE)
            self.vx_des = max(self.vx_des * 0.6, 0.02)
            return
        self._obstacle_yaw = 0.0
