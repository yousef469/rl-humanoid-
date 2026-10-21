"""Generate MJCF model of our humanoid for MuJoCo training."""
import os
import sys

MJCF_PATH = os.path.join(os.path.dirname(__file__), "humanoid_walker.xml")

# Dimensions (meters) — matching robot_urdf.py
P = {
    "pelvis_w": 0.34, "pelvis_d": 0.20, "pelvis_h": 0.16,
    "torso_w": 0.36, "torso_d": 0.20, "torso_h": 0.32,
    "chest_w": 0.32, "chest_d": 0.18, "chest_h": 0.25,
    "neck_len": 0.12, "head_r": 0.10, "head_h": 0.18,
    "upper_arm_len": 0.32, "upper_arm_r": 0.04,
    "forearm_len": 0.30, "forearm_r": 0.035,
    "thigh_len": 0.38, "thigh_r": 0.07,
    "shin_len": 0.36, "shin_r": 0.055,
    "foot_w": 0.22, "foot_d": 0.12, "foot_h": 0.07,
}

ph = P["pelvis_h"]
th = P["torso_h"]
ch = P["chest_h"]


def gen_xml():
    geom = '<geom type="capsule" size="{r} {l}" mass="{m}" rgba="{c}"/>'
    box = '<geom type="box" size="{sx} {sy} {sz}" mass="{m}" rgba="{c}"/>'

    return f'''<?xml version="1.0"?>
<mujoco model="humanoid_24dof">
  <compiler angle="radian" autolimits="true" coordinate="local"/>
  <option timestep="0.005" iterations="50" tolerance="1e-10"
          impratio="1" noslip_iterations="5"/>
  <size nconmax="200" njmax="2000" nuserdata="3"/>

  <default>
    <joint damping="0.1" armature="0.01" limited="true"/>
    <geom friction="1.0 0.05 0.01" solimp="0.9 0.95 0.001" solref="0.02 1" condim="6"/>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
    <position kp="100" kv="10"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3"
             width="300" height="300"/>
    <material name="ground" texture="grid" texrepeat="5 5"/>
    <material name="white" rgba="0.9 0.9 0.92 1"/>
    <material name="metal" rgba="0.53 0.60 0.73 1"/>
    <material name="dark" rgba="0.25 0.28 0.35 1"/>
    <material name="skin" rgba="0.80 0.72 0.65 1"/>
  </asset>

  <worldbody>
    <light name="top" pos="0 0 5" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="20 20 0.1" material="ground"/>

    <!-- Pelvis (base of robot) -->
    <body name="pelvis" pos="0 0 0.85">
      <freejoint/>
      {box.format(sx=P["pelvis_w"]/2, sy=P["pelvis_d"]/2, sz=ph/2, m=8, c="0.9 0.9 0.92 1")}

      <!-- Torso (fixed to pelvis) -->
      <body name="torso" pos="0 0 {ph/2}">
        {box.format(sx=P["torso_w"]/2, sy=P["torso_d"]/2, sz=th/2, m=10, c="0.9 0.9 0.92 1")}

        <!-- Torso twist joint -->
        <body name="twist_link" pos="0 0 {th}">
          <joint name="torso_twist" type="hinge" axis="0 0 1" range="-0.5 0.5" limited="true"/>
          {box.format(sx=0.06, sy=0.06, sz=0.02, m=0.5, c="0.25 0.28 0.35 1")}

          <!-- Torso bend joint -->
          <body name="chest" pos="0 0 0.04">
            <joint name="torso_bend" type="hinge" axis="0 1 0" range="-0.4 0.6" limited="true"/>
            {box.format(sx=P["chest_w"]/2, sy=P["chest_d"]/2, sz=ch/2, m=6, c="0.9 0.9 0.92 1")}

            <!-- Neck -->
            <body name="neck" pos="0 0 {ch}">
              <joint name="neck_pitch" type="hinge" axis="0 1 0" range="-0.5 0.5" limited="true"/>
              {geom.format(r=0.04, l=P["neck_len"]/2, m=0.5, c="0.25 0.28 0.35 1")}

              <body name="head" pos="0 0 {P['neck_len']}">
                <joint name="neck_yaw" type="hinge" axis="0 0 1" range="-0.8 0.8" limited="true"/>
                {box.format(sx=P["head_r"], sy=P["head_r"]*0.9, sz=P["head_h"]/2, m=3, c="0.80 0.72 0.65 1")}
              </body>
            </body>

            <!-- Left arm -->
            <body name="l_upperarm" pos="{-P['chest_w']/2-0.02} 0 0.04">
              <joint name="l_shoulder_pitch" type="hinge" axis="0 1 0" range="-1.5 1.5" limited="true"/>
              {geom.format(r=P["upper_arm_r"], l=P["upper_arm_len"]/2, m=2.5, c="0.9 0.9 0.92 1")}

              <body name="l_upperarm_roll" pos="0 0 {-P['upper_arm_len']}">
                <joint name="l_shoulder_roll" type="hinge" axis="1 0 0" range="-0.8 0.8" limited="true"/>
                {box.format(sx=0.03, sy=0.03, sz=0.02, m=0.3, c="0.25 0.28 0.35 1")}

                <body name="l_upperarm_yaw" pos="0 0 0.04">
                  <joint name="l_shoulder_yaw" type="hinge" axis="0 0 1" range="-1.0 1.0" limited="true"/>
                  {box.format(sx=0.03, sy=0.03, sz=0.02, m=0.3, c="0.25 0.28 0.35 1")}

                  <body name="l_forearm" pos="0 0 0.04">
                    <joint name="l_elbow" type="hinge" axis="0 1 0" range="-1.5 0.0" limited="true"/>
                    {geom.format(r=P["forearm_r"], l=P["forearm_len"]/2, m=1.8, c="0.9 0.9 0.92 1")}
                  </body>
                </body>
              </body>
            </body>

            <!-- Right arm (mirror of left) -->
            <body name="r_upperarm" pos="{P['chest_w']/2+0.02} 0 0.04">
              <joint name="r_shoulder_pitch" type="hinge" axis="0 1 0" range="-1.5 1.5" limited="true"/>
              {geom.format(r=P["upper_arm_r"], l=P["upper_arm_len"]/2, m=2.5, c="0.9 0.9 0.92 1")}

              <body name="r_upperarm_roll" pos="0 0 {-P['upper_arm_len']}">
                <joint name="r_shoulder_roll" type="hinge" axis="1 0 0" range="-0.8 0.8" limited="true"/>
                {box.format(sx=0.03, sy=0.03, sz=0.02, m=0.3, c="0.25 0.28 0.35 1")}

                <body name="r_upperarm_yaw" pos="0 0 0.04">
                  <joint name="r_shoulder_yaw" type="hinge" axis="0 0 1" range="-1.0 1.0" limited="true"/>
                  {box.format(sx=0.03, sy=0.03, sz=0.02, m=0.3, c="0.25 0.28 0.35 1")}

                  <body name="r_forearm" pos="0 0 0.04">
                    <joint name="r_elbow" type="hinge" axis="0 1 0" range="-1.5 0.0" limited="true"/>
                    {geom.format(r=P["forearm_r"], l=P["forearm_len"]/2, m=1.8, c="0.9 0.9 0.92 1")}
                  </body>
                </body>
              </body>
            </body>

          </body>
        </body>
      </body>

      <!-- Left leg -->
      <body name="l_thigh" pos="{-P['pelvis_w']/4} 0 {-ph/2}">
        <joint name="l_hip_pitch" type="hinge" axis="0 1 0" range="-0.5 1.0" limited="true"/>
        {geom.format(r=P["thigh_r"], l=P["thigh_len"]/2, m=5, c="0.9 0.9 0.92 1")}

        <body name="l_thigh_roll" pos="0 0 {-P['thigh_len']}">
          <joint name="l_hip_roll" type="hinge" axis="1 0 0" range="-0.3 0.5" limited="true"/>
          {box.format(sx=0.04, sy=0.04, sz=0.02, m=0.3, c="0.25 0.28 0.35 1")}

          <body name="l_thigh_yaw" pos="0 0 0.04">
            <joint name="l_hip_yaw" type="hinge" axis="0 0 1" range="-0.5 0.5" limited="true"/>
            {box.format(sx=0.04, sy=0.04, sz=0.02, m=0.3, c="0.25 0.28 0.35 1")}

            <body name="l_shin" pos="0 0 0.04">
              <joint name="l_knee" type="hinge" axis="0 1 0" range="0.0 1.5" limited="true"/>
              {geom.format(r=P["shin_r"], l=P["shin_len"]/2, m=3, c="0.9 0.9 0.92 1")}

              <body name="l_foot" pos="0.03 0 {-P['shin_len']}">
                <joint name="l_ankle" type="hinge" axis="0 1 0" range="-0.5 0.5" limited="true"/>
                {box.format(sx=P["foot_w"]/2, sy=P["foot_d"]/2, sz=P["foot_h"]/2, m=1, c="0.25 0.28 0.35 1")}
              </body>
            </body>
          </body>
        </body>
      </body>

      <!-- Right leg (mirror of left) -->
      <body name="r_thigh" pos="{P['pelvis_w']/4} 0 {-ph/2}">
        <joint name="r_hip_pitch" type="hinge" axis="0 1 0" range="-0.5 1.0" limited="true"/>
        {geom.format(r=P["thigh_r"], l=P["thigh_len"]/2, m=5, c="0.9 0.9 0.92 1")}

        <body name="r_thigh_roll" pos="0 0 {-P['thigh_len']}">
          <joint name="r_hip_roll" type="hinge" axis="1 0 0" range="-0.5 0.3" limited="true"/>
          {box.format(sx=0.04, sy=0.04, sz=0.02, m=0.3, c="0.25 0.28 0.35 1")}

          <body name="r_thigh_yaw" pos="0 0 0.04">
            <joint name="r_hip_yaw" type="hinge" axis="0 0 1" range="-0.5 0.5" limited="true"/>
            {box.format(sx=0.04, sy=0.04, sz=0.02, m=0.3, c="0.25 0.28 0.35 1")}

            <body name="r_shin" pos="0 0 0.04">
              <joint name="r_knee" type="hinge" axis="0 1 0" range="0.0 1.5" limited="true"/>
              {geom.format(r=P["shin_r"], l=P["shin_len"]/2, m=3, c="0.9 0.9 0.92 1")}

              <body name="r_foot" pos="0.03 0 {-P['shin_len']}">
                <joint name="r_ankle" type="hinge" axis="0 1 0" range="-0.5 0.5" limited="true"/>
                {box.format(sx=P["foot_w"]/2, sy=P["foot_d"]/2, sz=P["foot_h"]/2, m=1, c="0.25 0.28 0.35 1")}
              </body>
            </body>
          </body>
        </body>
      </body>

    </body>
  </worldbody>

  <actuator>
    <position joint="l_hip_yaw" name="l_hip_yaw" kp="120" kv="10"/>
    <position joint="l_hip_roll" name="l_hip_roll" kp="120" kv="10"/>
    <position joint="l_hip_pitch" name="l_hip_pitch" kp="120" kv="10"/>
    <position joint="l_knee" name="l_knee" kp="120" kv="10"/>
    <position joint="l_ankle" name="l_ankle" kp="100" kv="10"/>
    <position joint="r_hip_yaw" name="r_hip_yaw" kp="120" kv="10"/>
    <position joint="r_hip_roll" name="r_hip_roll" kp="120" kv="10"/>
    <position joint="r_hip_pitch" name="r_hip_pitch" kp="120" kv="10"/>
    <position joint="r_knee" name="r_knee" kp="120" kv="10"/>
    <position joint="r_ankle" name="r_ankle" kp="100" kv="10"/>
    <position joint="l_shoulder_pitch" name="l_shoulder_pitch" kp="40" kv="5"/>
    <position joint="l_shoulder_roll" name="l_shoulder_roll" kp="40" kv="5"/>
    <position joint="l_shoulder_yaw" name="l_shoulder_yaw" kp="20" kv="5"/>
    <position joint="l_elbow" name="l_elbow" kp="20" kv="5"/>
    <position joint="r_shoulder_pitch" name="r_shoulder_pitch" kp="40" kv="5"/>
    <position joint="r_shoulder_roll" name="r_shoulder_roll" kp="40" kv="5"/>
    <position joint="r_shoulder_yaw" name="r_shoulder_yaw" kp="20" kv="5"/>
    <position joint="r_elbow" name="r_elbow" kp="20" kv="5"/>
    <position joint="neck_pitch" name="neck_pitch" kp="10" kv="3"/>
    <position joint="neck_yaw" name="neck_yaw" kp="10" kv="3"/>
    <position joint="torso_twist" name="torso_twist" kp="60" kv="8"/>
    <position joint="torso_bend" name="torso_bend" kp="80" kv="8"/>
  </actuator>

</mujoco>'''
    return f


def generate(path=MJCF_PATH):
    f = gen_xml()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f)
    print(f"MJCF written to {path}")
    return path


if __name__ == "__main__":
    generate()

