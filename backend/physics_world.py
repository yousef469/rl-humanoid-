"""PyBullet physics simulation — ground, objects, robot with mass & inertia."""

import math
import random

import pybullet as p
import pybullet_data

from config import PHYSICS_TIMESTEP
from robot_urdf import generate_urdf

GRAVITY = -9.81
GROUND_FRICTION = 0.8
OBJECT_LABELS = ["tree", "rock", "house", "sign", "person", "car", "bush", "lamp"]


class PhysicsWorld:
    def __init__(self):
        self.client = p.connect(p.DIRECT)
        p.setTimeStep(PHYSICS_TIMESTEP, physicsClientId=self.client)
        p.setGravity(0, 0, GRAVITY, physicsClientId=self.client)
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            numSolverIterations=150,
            erp=0.2,
            contactERP=0.2,
            frictionERP=0.2,
            physicsClientId=self.client,
        )

        self.robot_id = None
        self.joint_indices = {}
        self.objects = {}
        self.head_link_index = -1
        self._setup_ground()
        self._load_robot()
        self._spawn_objects()
        self._spawn_red_ball()
        # Settle with active standing pose — real robots don't fall before walking
        self._initial_stand()

    def _setup_ground(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        plane = p.loadURDF("plane.urdf", useMaximalCoordinates=False, physicsClientId=self.client)
        p.changeDynamics(plane, -1, lateralFriction=GROUND_FRICTION, physicsClientId=self.client)
        self.ground_id = plane

    def _load_robot(self):
        urdf_path = generate_urdf()
        start_pos = [0, 0, 1.0]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(
            urdf_path, start_pos, start_orn,
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
            physicsClientId=self.client,
        )

        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            name = info[1].decode("utf-8")
            self.joint_indices[name] = i
            if info[12]:
                link_name = info[12].decode("utf-8")
                if link_name == "head":
                    self.head_link_index = i
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL,
                targetPosition=0.0,
                force=10.0,
                positionGain=0.6,
                velocityGain=0.8,
                maxVelocity=3.0,
                physicsClientId=self.client,
            )

    def _initial_stand(self, settle_steps=120):
        stand_pose = {
            'l_knee': 0.05, 'r_knee': 0.05,
            'l_hip_pitch': 0.0, 'r_hip_pitch': 0.0,
            'l_hip_roll': 0.0, 'r_hip_roll': 0.0,
            'l_hip_yaw': 0.0, 'r_hip_yaw': 0.0,
            'l_ankle': -0.05, 'r_ankle': -0.05,
            'l_shoulder_pitch': 0.0, 'r_shoulder_pitch': 0.0,
            'l_shoulder_roll': 0.0, 'r_shoulder_roll': 0.0,
            'l_shoulder_yaw': 0.0, 'r_shoulder_yaw': 0.0,
            'l_elbow': 0.0, 'r_elbow': 0.0,
            'neck_pitch': 0.0, 'neck_yaw': 0.0,
            'torso_twist': 0.0, 'torso_bend': 0.0,
        }
        for _ in range(settle_steps):
            self.set_joint_targets_walk(stand_pose)
            self.step()

        # Start every episode from a perfectly symmetric pose. Without this the
        # settle phase can leave one leg slightly longer/higher, which makes the
        # robot pivot on a single foot even while "standing".
        pos, _ = self.get_robot_position()
        self.reset_robot_position(0.0, 0.0, pos[2], 0.0)
        for name, idx in self.joint_indices.items():
            p.resetJointState(
                self.robot_id, idx,
                targetValue=stand_pose.get(name, 0.0),
                targetVelocity=0.0,
                physicsClientId=self.client,
            )

        # The reset above straightens the legs; lift the base so the soles rest
        # exactly on the ground instead of starting several centimetres inside it.
        foot_links = [self.get_link_index("l_foot"), self.get_link_index("r_foot")]
        min_foot_z = min(
            p.getAABB(self.robot_id, idx, physicsClientId=self.client)[0][2]
            for idx in foot_links
        )
        pos, _ = self.get_robot_position()
        self.reset_robot_position(0.0, 0.0, pos[2] - min_foot_z, 0.0)
        for _ in range(30):
            self.set_joint_targets_walk(stand_pose)
            self.step()
        self.reset_robot_position(0.0, 0.0, self.get_robot_position()[0][2], 0.0)

    def _spawn_objects(self, count=8):
        for i in range(count):
            label = random.choice(OBJECT_LABELS)
            for _ in range(12):
                x = random.uniform(-15, 15)
                y = random.uniform(-15, 15)
                if abs(x) < 2 and abs(y) < 2:
                    continue
                # Keep a clear approach corridor to the red ball.
                if abs(y) < 1.2 and 0.5 < x < 4.0:
                    continue
                if math.hypot(x - 3.0, y) < 1.5:
                    continue
                break
            else:
                continue
            self._create_object(f"obj_{i:03d}", label, [x, y, 0])

    def _spawn_red_ball(self):
        """Place the YOLO/navigation target directly in front of the robot."""
        self._create_object("red_ball", "red_ball", [3.0, 0.0, 0.18])

    def _create_object(self, obj_id, label, pos):
        pos = [pos[0], pos[1], 0]
        if label == "tree":
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.15, height=1.5, physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.15, length=1.5,
                                       rgbaColor=[0.4, 0.25, 0.1, 1], physicsClientId=self.client)
            body = p.createMultiBody(5.0, col, vis, basePosition=pos, physicsClientId=self.client)
            # Add leaves (no collision)
            leaf_pos = [pos[0], pos[1], 1.0]
            leaf_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.5,
                                            rgbaColor=[0.1, 0.6, 0.1, 1], physicsClientId=self.client)
            leaf_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01, physicsClientId=self.client)
            p.createMultiBody(0.01, leaf_col, leaf_vis, basePosition=leaf_pos, physicsClientId=self.client)
        elif label == "rock":
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.2, physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2,
                                       rgbaColor=[0.4, 0.4, 0.4, 1], physicsClientId=self.client)
            body = p.createMultiBody(2.0, col, vis, basePosition=pos, physicsClientId=self.client)
        elif label == "house":
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.3], physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.3],
                                       rgbaColor=[0.6, 0.3, 0.1, 1], physicsClientId=self.client)
            body = p.createMultiBody(50.0, col, vis, basePosition=[pos[0], pos[1], 0.3], physicsClientId=self.client)
        elif label == "person":
            col = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.15, height=0.5, physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_CAPSULE, radius=0.15, length=0.5,
                                       rgbaColor=[0.8, 0.5, 0.2, 1], physicsClientId=self.client)
            body = p.createMultiBody(10.0, col, vis, basePosition=[pos[0], pos[1], 0.3], physicsClientId=self.client)
        elif label == "red_ball":
            radius = 0.18
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                       rgbaColor=[0.95, 0.03, 0.03, 1], physicsClientId=self.client)
            body = p.createMultiBody(0.5, col, vis, basePosition=[pos[0], pos[1], radius],
                                     physicsClientId=self.client)
            p.changeDynamics(body, -1, lateralFriction=0.8, restitution=0.25,
                             rollingFriction=0.002, physicsClientId=self.client)
        elif label in ("car", "bush", "sign", "lamp"):
            size = {"car": 0.4, "bush": 0.25, "sign": 0.08, "lamp": 0.05}[label]
            color = {"car": [0.2, 0.4, 0.8, 1], "bush": [0.2, 0.5, 0.1, 1],
                     "sign": [0.9, 0.8, 0.1, 1], "lamp": [0.9, 0.9, 0.5, 1]}[label]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, size], physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, size],
                                       rgbaColor=color, physicsClientId=self.client)
            body = p.createMultiBody(3.0, col, vis, basePosition=[pos[0], pos[1], size], physicsClientId=self.client)
        else:
            return

        if label != "red_ball":
            p.changeDynamics(body, -1, lateralFriction=0.5, restitution=0.1, physicsClientId=self.client)
        self.objects[obj_id] = {
            "id": obj_id, "label": label, "body": body, "position": pos,
            "target": label == "red_ball",
        }

    def step(self, dt=None, steps=None):
        """Advance the simulation by `dt` seconds of simulated time.

        PyBullet uses a fixed 1/240 s internal timestep. Callers (the controller)
        run at 60 Hz, so a single call normally advances four physics steps.
        """
        if steps is None:
            steps = max(1, int(round(dt / PHYSICS_TIMESTEP))) if dt else 1
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client)

    LEG_STRENGTH = 120.0
    ARM_STRENGTH = 40.0
    LEG_JOINTS = {"l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ankle",
                  "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ankle"}

    def set_joint_target(self, joint_name, target_angle, max_force=10.0, kp=0.6):
        idx = self.joint_indices.get(joint_name)
        if idx is not None:
            p.setJointMotorControl2(
                self.robot_id, idx, p.POSITION_CONTROL,
                targetPosition=target_angle,
                force=max_force,
                maxVelocity=3.0,
                positionGain=kp,
                velocityGain=0.8,
                physicsClientId=self.client,
            )

    def set_joint_targets(self, joint_angles: dict, max_force=10.0):
        for name, angle in joint_angles.items():
            self.set_joint_target(name, angle, max_force)

    def set_joint_targets_walk(self, joint_angles: dict):
        all_joints = set(self.joint_indices.keys())
        for name, angle in joint_angles.items():
            force = self.LEG_STRENGTH if name in self.LEG_JOINTS else self.ARM_STRENGTH
            self.set_joint_target(name, angle, max_force=force, kp=1.0)
            all_joints.discard(name)
        for name in all_joints:
            if name in self.LEG_JOINTS:
                self.set_joint_target(name, 0.0, max_force=self.LEG_STRENGTH, kp=1.0)
            elif name not in ("torso_joint",):
                self.set_joint_target(name, 0.0, max_force=self.ARM_STRENGTH, kp=1.0)

    def get_joint_angles(self) -> dict:
        angles = {}
        for name, idx in self.joint_indices.items():
            state = p.getJointState(self.robot_id, idx, physicsClientId=self.client)
            angles[name] = state[0]
        return angles

    def get_robot_position(self) -> tuple:
        pos, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        euler = p.getEulerFromQuaternion(orn)
        return (pos[0], pos[1], pos[2]), euler[2]

    def get_robot_orientation_euler(self) -> tuple[float, float, float]:
        _, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        return p.getEulerFromQuaternion(orn)

    def get_link_index(self, link_name: str) -> int:
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            lname = info[12].decode("utf-8") if info[12] else ""
            if lname == link_name:
                return i
        return -1

    def get_link_world_pose(self, link_index: int):
        if link_index < 0:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        state = p.getLinkState(self.robot_id, link_index, physicsClientId=self.client)
        return state[0], state[1]

    def reset_robot_position(self, x, y, z=1.0, yaw=0.0):
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot_id, [x, y, z], orn,
                                          physicsClientId=self.client)
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self.client)

    def raycast_vision(self, num_rays=16, max_distance=50.0):
        """Robot 'sees' by casting rays from eye height toward ground-level targets.
        For each horizontal angle, multiple rays are cast at different range gates
        so nearby objects aren't missed."""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        eye_z = pos[2] + 0.65  # approximate eye height (~1.65m)
        head_pos = [pos[0], pos[1], eye_z]

        range_gates = [3, 7, 15, 30, max_distance]

        objects_seen = {}
        for i in range(num_rays):
            h_angle = yaw + math.radians(-60 + 120 * i / (num_rays - 1))
            cos_a, sin_a = math.cos(h_angle), math.sin(h_angle)

            for gate_dist in range_gates:
                to_pos = [
                    head_pos[0] + gate_dist * cos_a,
                    head_pos[1] + gate_dist * sin_a,
                    0.0,
                ]

                results = p.rayTest(head_pos, to_pos, physicsClientId=self.client)
                for hit in results:
                    body_id = int(hit[0])
                    if body_id in (self.ground_id, self.robot_id, -1):
                        continue
                    hit_pos = hit[3]
                    dist = math.sqrt((hit_pos[0] - head_pos[0])**2 + (hit_pos[1] - head_pos[1])**2)

                    for oid, obj in self.objects.items():
                        if obj["body"] == body_id and oid not in objects_seen:
                            angle_deg = math.degrees(math.atan2(
                                hit_pos[1] - head_pos[1],
                                hit_pos[0] - head_pos[0],
                            ) - yaw)
                            objects_seen[oid] = {
                                "id": oid,
                                "label": obj["label"],
                                "distance": round(dist, 1),
                                "angle": round(angle_deg, 1),
                                "position": (round(hit_pos[0], 2), round(hit_pos[1], 2), 0.0),
                                "confidence": round(random.uniform(0.85, 0.95), 2),
                            }
                            break

        # Small objects like the red ball can slip between the coarse ground
        # ray gates. Cast one direct line-of-sight ray at the target as well.
        target = self.objects.get("red_ball")
        if target is not None and "red_ball" not in objects_seen:
            body_pos = p.getBasePositionAndOrientation(
                target["body"], physicsClientId=self.client
            )[0]
            results = p.rayTest(head_pos, list(body_pos), physicsClientId=self.client)
            if results and int(results[0][0]) == target["body"]:
                hit_pos = results[0][3]
                dist = math.sqrt((hit_pos[0] - head_pos[0])**2 + (hit_pos[1] - head_pos[1])**2)
                angle_deg = math.degrees(math.atan2(
                    hit_pos[1] - head_pos[1], hit_pos[0] - head_pos[0]
                ) - yaw)
                objects_seen["red_ball"] = {
                    "id": "red_ball",
                    "label": "red_ball",
                    "distance": round(dist, 1),
                    "angle": round(angle_deg, 1),
                    "position": (round(hit_pos[0], 2), round(hit_pos[1], 2), 0.0),
                    "center": (round(body_pos[0], 3), round(body_pos[1], 3), 0.0),
                    "confidence": 0.99,
                }

        return list(objects_seen.values())

    def get_head_camera_image(self, width=320, height=240):
        """First-person view from the head link. Returns (width, height, rgba, depth, seg, view, proj)."""
        head_pos, head_orn = self.get_link_world_pose(self.head_link_index)
        rot = p.getMatrixFromQuaternion(head_orn, physicsClientId=self.client)
        # Link frame +X is forward, +Z is up.
        forward = [rot[0], rot[3], rot[6]]
        up = [rot[2], rot[5], rot[8]]

        cam_pos = [
            head_pos[0] + forward[0] * 0.09,
            head_pos[1] + forward[1] * 0.09,
            head_pos[2] + forward[2] * 0.09 + 0.08,
        ]
        cam_target = [
            cam_pos[0] + forward[0] * 10.0,
            cam_pos[1] + forward[1] * 10.0,
            cam_pos[2] + forward[2] * 10.0,
        ]

        view = p.computeViewMatrix(cam_pos, cam_target, up, physicsClientId=self.client)
        proj = p.computeProjectionMatrixFOV(70, width/height, 0.1, 100, physicsClientId=self.client)
        img = p.getCameraImage(width, height, view, proj, physicsClientId=self.client)
        return img[0], img[1], img[2], img[3], img[4], view, proj

    def get_visible_objects(self):
        seen = []
        for oid, obj in self.objects.items():
            pos = p.getBasePositionAndOrientation(obj["body"], physicsClientId=self.client)[0]
            robot_pos = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)[0]
            dist = math.sqrt((pos[0] - robot_pos[0])**2 + (pos[1] - robot_pos[1])**2)
            seen.append({
                "id": oid,
                "label": obj["label"],
                "distance": round(dist, 1),
                "position": pos,
            })
        seen.sort(key=lambda o: o["distance"])
        return seen

    def close(self):
        p.disconnect(physicsClientId=self.client)


if __name__ == "__main__":
    world = PhysicsWorld()
    for i in range(1000):
        world.step()
        if i % 100 == 0:
            pos, rot = world.get_robot_position()
            joints = world.get_joint_angles()
            print(f"Step {i}: pos=({pos[0]:.2f}, {pos[1]:.2f}), joints={len(joints)}")
    world.close()
