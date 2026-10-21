import pybullet as p
import numpy as np


class StateEstimator:
    def __init__(self, physics_world):
        self.pw = physics_world
        self._link_indices = {}
        self._com_frame_idx = -1
        self.prev_base_pos = np.zeros(3)
        self.prev_base_orn_euler = np.zeros(3)
        self.base_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.prev_joint_pos = {}
        self.joint_vel = {}
        self._initialized = False

    def _cache_link_index(self, name):
        if name not in self._link_indices:
            pid = self.pw.robot_id
            for i in range(p.getNumJoints(pid, physicsClientId=self.pw.client)):
                info = p.getJointInfo(pid, i, physicsClientId=self.pw.client)
                link_name = info[12].decode("utf-8") if info[12] else info[1].decode("utf-8")
                if link_name == name:
                    self._link_indices[name] = i
                    return i
            self._link_indices[name] = -1
            return -1
        return self._link_indices[name]

    def estimate(self, dt):
        pid = self.pw.robot_id
        cid = self.pw.client

        base_pos, base_orn = p.getBasePositionAndOrientation(pid, physicsClientId=cid)
        base_orn_euler = p.getEulerFromQuaternion(base_orn)
        base_pos = np.array(base_pos)
        base_orn_euler = np.array(base_orn_euler)

        if self._initialized:
            self.base_vel = (base_pos - self.prev_base_pos) / max(dt, 1e-6)
            self.base_ang_vel = (base_orn_euler - self.prev_base_orn_euler) / max(dt, 1e-6)
            for jn, jpos in self._get_all_joint_positions().items():
                prev = self.prev_joint_pos.get(jn, jpos)
                self.joint_vel[jn] = (jpos - prev) / max(dt, 1e-6)

        self.prev_base_pos = base_pos.copy()
        self.prev_base_orn_euler = base_orn_euler.copy()
        self.prev_joint_pos = self._get_all_joint_positions()
        self._initialized = True

        l_foot_idx = self._cache_link_index("l_foot")
        r_foot_idx = self._cache_link_index("r_foot")

        l_foot_pos = self._get_link_pos(l_foot_idx)
        r_foot_pos = self._get_link_pos(r_foot_idx)

        left_contact = self._check_contact(l_foot_idx)
        right_contact = self._check_contact(r_foot_idx)

        base_lin_vel = self.base_vel.copy()
        return {
            "base_pos": base_pos,
            "base_orn": base_orn_euler,
            "base_vel_linear": base_lin_vel,
            "base_vel_angular": self.base_ang_vel.copy(),
            "com_pos": base_pos + np.array([0, 0, 0.4]),
            "com_vel": base_lin_vel,
            "l_foot_pos": l_foot_pos,
            "r_foot_pos": r_foot_pos,
            "left_contact": left_contact,
            "right_contact": right_contact,
            "joint_positions": self._get_all_joint_positions(),
            "joint_velocities": dict(self.joint_vel),
        }

    def _get_all_joint_positions(self):
        return self.pw.get_joint_angles()

    def _get_link_pos(self, link_idx):
        if link_idx < 0:
            return np.array([0, 0, 0])
        pid = self.pw.robot_id
        cid = self.pw.client
        state = p.getLinkState(pid, link_idx, physicsClientId=cid)
        return np.array(state[0])

    def _check_contact(self, link_idx):
        if link_idx < 0:
            return False
        pid = self.pw.robot_id
        cid = self.pw.client
        pts = p.getContactPoints(bodyA=pid, linkIndexA=link_idx, physicsClientId=cid)
        for pt in pts:
            # Contact tuple: (bodyA, bodyB, linkIndexA, linkIndexB, ...)
            if pt[1] == self.pw.ground_id or pt[2] == self.pw.ground_id:
                return True
        return False

    def get_contact_state(self):
        pid = self.pw.robot_id
        cid = self.pw.client
        l_foot_idx = self._cache_link_index("l_foot")
        r_foot_idx = self._cache_link_index("r_foot")
        left = self._check_contact(l_foot_idx)
        right = self._check_contact(r_foot_idx)
        return left, right
