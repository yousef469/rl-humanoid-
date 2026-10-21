import mujoco
import mujoco.viewer
import os

xml_path = os.path.join(os.path.dirname(__file__), "unitree_h1/scene.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 45
    viewer.cam.distance = 3.5
    viewer.cam.elevation = -20

    while viewer.is_running():
        viewer.sync()
