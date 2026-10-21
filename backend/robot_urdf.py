"""Generates a 24-DOF humanoid robot URDF for real-world transfer."""

import os

URDF_PATH = os.path.join(os.path.dirname(__file__), "humanoid.urdf")


def generate_urdf(path=URDF_PATH):
    # Robot dimensions (meters) — ~1.7m humanoid
    P = {
        "pelvis_w": 0.34, "pelvis_d": 0.20, "pelvis_h": 0.16,
        "torso_w": 0.36, "torso_d": 0.20, "torso_h": 0.32,
        "chest_w": 0.32, "chest_d": 0.18, "chest_h": 0.25,
        "neck_len": 0.12, "head_r": 0.10, "head_h": 0.18,
        "upper_arm_r": 0.04, "upper_arm_len": 0.32,
        "forearm_r": 0.035, "forearm_len": 0.30,
        "hand_r": 0.04,
        "thigh_r": 0.07, "thigh_len": 0.38,
        "shin_r": 0.055, "shin_len": 0.36,
        "foot_len": 0.22, "foot_w": 0.12, "foot_h": 0.07,
    }


    def box(name, size, mass, rgba, origin="0 0 0"):
        sx, sy, sz = size
        ixx = mass * (sy ** 2 + sz ** 2) / 12
        iyy = mass * (sx ** 2 + sz ** 2) / 12
        izz = mass * (sx ** 2 + sy ** 2) / 12
        return f'''  <link name="{name}">
    <inertial><origin xyz="{origin}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx:.4f}" ixy="0" ixz="0" iyy="{iyy:.4f}" iyz="0" izz="{izz:.4f}"/>
    </inertial>
    <visual><origin xyz="{origin}" rpy="0 0 0"/>
      <geometry><box size="{sx} {sy} {sz}"/></geometry>
      <material name="{rgba}"/>
    </visual>
    <collision><origin xyz="{origin}" rpy="0 0 0"/>
      <geometry><box size="{sx} {sy} {sz}"/></geometry>
    </collision>
  </link>'''

    def cyl(name, radius, length, mass, rgba, origin="0 0 0"):
        r, l = radius, length
        ixx = mass * (3 * r ** 2 + l ** 2) / 12
        izz = mass * r ** 2 / 2
        return f'''  <link name="{name}">
    <inertial><origin xyz="{origin}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx:.4f}" ixy="0" ixz="0" iyy="{ixx:.4f}" iyz="0" izz="{izz:.4f}"/>
    </inertial>
    <visual><origin xyz="{origin}" rpy="0 0 0"/>
      <geometry><cylinder radius="{r}" length="{l}"/></geometry>
      <material name="{rgba}"/>
    </visual>
    <collision><origin xyz="{origin}" rpy="0 0 0"/>
      <geometry><cylinder radius="{r}" length="{l}"/></geometry>
    </collision>
  </link>'''

    def fixed_joint(name, parent, child, origin):
        return f'''  <joint name="{name}" type="fixed">
    <parent link="{parent}"/><child link="{child}"/>
    <origin xyz="{origin}" rpy="0 0 0"/>
  </joint>'''

    def revolute(name, parent, child, origin, axis, limits):
        return f'''  <joint name="{name}" type="revolute">
    <parent link="{parent}"/><child link="{child}"/>
    <origin xyz="{origin}" rpy="0 0 0"/>
    <axis xyz="{axis}"/>
    <limit lower="{limits[0]}" upper="{limits[1]}" effort="{limits[2]}" velocity="{limits[3]}"/>
  </joint>'''

    ph = P["pelvis_h"] / 2
    th = P["torso_h"] / 2
    ch = P["chest_h"] / 2

    urdf = f'''<?xml version="1.0"?>
<robot name="humanoid_24dof">

  <material name="metal"><color rgba="0.53 0.60 0.73 1.0"/></material>
  <material name="white"><color rgba="0.90 0.90 0.92 1.0"/></material>
  <material name="dark"><color rgba="0.25 0.28 0.35 1.0"/></material>
  <material name="skin"><color rgba="0.80 0.72 0.65 1.0"/></material>

{box("pelvis", (P["pelvis_d"], P["pelvis_w"], P["pelvis_h"]), 8, "white", f"0 0 0")}

{fixed_joint("torso_joint", "pelvis", "torso", f"0 0 {P['pelvis_h']/2}")}
{box("torso", (P["torso_d"], P["torso_w"], P["torso_h"]), 10, "white", f"0 0 {th}")}

{revolute("torso_twist", "torso", "twist_link", f"0 0 {P['torso_h']}", "0 0 1", (-0.5, 0.5, 60, 3))}
{box("twist_link", (0.12, 0.12, 0.04), 0.5, "dark", "0 0 0.02")}

{revolute("torso_bend", "twist_link", "chest", f"0 0 0.04", "0 1 0", (-0.4, 0.6, 80, 3))}
{box("chest", (P["chest_d"], P["chest_w"], P["chest_h"]), 6, "white", f"0 0 {ch}")}

{revolute("neck_pitch", "chest", "neck", f"0 0 {P['chest_h']}", "0 1 0", (-0.5, 0.5, 10, 3))}
{cyl("neck", 0.04, P["neck_len"], 0.5, "dark", f"0 0 {P['neck_len']/2}")}

{revolute("neck_yaw", "neck", "head", f"0 0 {P['neck_len']}", "0 0 1", (-0.8, 0.8, 10, 3))}
{box("head", (P["head_r"]*1.8, P["head_r"]*2, P["head_h"]), 3, "skin", f"0 0 {P['head_h']/2}")}

{revolute("l_shoulder_pitch", "twist_link", "l_upperarm", f"0 {-P['chest_w']/2-0.02} 0.04", "0 1 0", (-1.5, 1.5, 40, 3))}
{cyl("l_upperarm", P["upper_arm_r"], P["upper_arm_len"], 2.5, "white", f"0 0 {-P['upper_arm_len']/2}")}

{revolute("l_shoulder_roll", "l_upperarm", "l_upperarm_roll", f"0 0 {-P['upper_arm_len']}", "1 0 0", (-0.8, 0.8, 40, 3))}
{box("l_upperarm_roll", (0.06, 0.06, 0.04), 0.3, "dark", "0 0 0.02")}

{revolute("l_shoulder_yaw", "l_upperarm_roll", "l_upperarm_yaw", "0 0 0.04", "0 0 1", (-1.0, 1.0, 20, 3))}
{box("l_upperarm_yaw", (0.06, 0.06, 0.04), 0.3, "dark", "0 0 0.02")}

{revolute("l_elbow", "l_upperarm_yaw", "l_forearm", "0 0 0.04", "0 1 0", (-1.5, 0.0, 20, 3))}
{cyl("l_forearm", P["forearm_r"], P["forearm_len"], 1.8, "white", f"0 0 {-P['forearm_len']/2}")}

{revolute("r_shoulder_pitch", "twist_link", "r_upperarm", f"0 {P['chest_w']/2+0.02} 0.04", "0 1 0", (-1.5, 1.5, 40, 3))}
{cyl("r_upperarm", P["upper_arm_r"], P["upper_arm_len"], 2.5, "white", f"0 0 {-P['upper_arm_len']/2}")}

{revolute("r_shoulder_roll", "r_upperarm", "r_upperarm_roll", f"0 0 {-P['upper_arm_len']}", "1 0 0", (-0.8, 0.8, 40, 3))}
{box("r_upperarm_roll", (0.06, 0.06, 0.04), 0.3, "dark", "0 0 0.02")}

{revolute("r_shoulder_yaw", "r_upperarm_roll", "r_upperarm_yaw", "0 0 0.04", "0 0 1", (-1.0, 1.0, 20, 3))}
{box("r_upperarm_yaw", (0.06, 0.06, 0.04), 0.3, "dark", "0 0 0.02")}

{revolute("r_elbow", "r_upperarm_yaw", "r_forearm", "0 0 0.04", "0 1 0", (-1.5, 0.0, 20, 3))}
{cyl("r_forearm", P["forearm_r"], P["forearm_len"], 1.8, "white", f"0 0 {-P['forearm_len']/2}")}

{revolute("l_hip_pitch", "pelvis", "l_thigh", f"0 {-P['pelvis_w']/4} {-P['pelvis_h']/2}", "0 1 0", (-0.5, 1.0, 120, 3))}
{cyl("l_thigh", P["thigh_r"], P["thigh_len"], 5, "white", f"0 0 {-P['thigh_len']/2}")}

{revolute("l_hip_roll", "l_thigh", "l_thigh_roll", f"0 0 {-P['thigh_len']}", "1 0 0", (-0.3, 0.5, 100, 3))}
{box("l_thigh_roll", (0.08, 0.08, 0.04), 0.3, "dark", "0 0 0.02")}

{revolute("l_hip_yaw", "l_thigh_roll", "l_thigh_yaw", "0 0 0.04", "0 0 1", (-0.5, 0.5, 80, 3))}
{box("l_thigh_yaw", (0.08, 0.08, 0.04), 0.3, "dark", "0 0 0.02")}

{revolute("l_knee", "l_thigh_yaw", "l_shin", "0 0 0.04", "0 1 0", (0.0, 1.5, 120, 3))}
{cyl("l_shin", P["shin_r"], P["shin_len"], 3, "white", f"0 0 {-P['shin_len']/2}")}

{revolute("l_ankle", "l_shin", "l_foot", f"0 0 {-P['shin_len']}", "0 1 0", (-0.5, 0.5, 100, 3))}
{box("l_foot", (P["foot_len"], P["foot_w"], P["foot_h"]), 1, "dark", f"0.03 0 {-P['foot_h']/2}")}

{revolute("r_hip_pitch", "pelvis", "r_thigh", f"0 {P['pelvis_w']/4} {-P['pelvis_h']/2}", "0 1 0", (-0.5, 1.0, 120, 3))}
{cyl("r_thigh", P["thigh_r"], P["thigh_len"], 5, "white", f"0 0 {-P['thigh_len']/2}")}

{revolute("r_hip_roll", "r_thigh", "r_thigh_roll", f"0 0 {-P['thigh_len']}", "1 0 0", (-0.5, 0.3, 100, 3))}
{box("r_thigh_roll", (0.08, 0.08, 0.04), 0.3, "dark", "0 0 0.02")}

{revolute("r_hip_yaw", "r_thigh_roll", "r_thigh_yaw", "0 0 0.04", "0 0 1", (-0.5, 0.5, 80, 3))}
{box("r_thigh_yaw", (0.08, 0.08, 0.04), 0.3, "dark", "0 0 0.02")}

{revolute("r_knee", "r_thigh_yaw", "r_shin", "0 0 0.04", "0 1 0", (0.0, 1.5, 120, 3))}
{cyl("r_shin", P["shin_r"], P["shin_len"], 3, "white", f"0 0 {-P['shin_len']/2}")}

{revolute("r_ankle", "r_shin", "r_foot", f"0 0 {-P['shin_len']}", "0 1 0", (-0.5, 0.5, 100, 3))}
{box("r_foot", (P["foot_len"], P["foot_w"], P["foot_h"]), 1, "dark", f"0.03 0 {-P['foot_h']/2}")}

</robot>'''

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(urdf)
    print(f"URDF written to {path}")
    return path


if __name__ == "__main__":
    generate_urdf()
