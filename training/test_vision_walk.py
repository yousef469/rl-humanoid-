"""Minimal test: frozen policy walking toward a goal."""
import os, time, numpy as np, mujoco, mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(BASE_DIR, "unitree_g1", "scene_vision.xml")
CKPT = os.path.join(BASE_DIR, "checkpoints")
N = 15000000

NUM_OBS, NUM_ACT = 99, 29
JOINT_NAMES = [
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
    "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
    "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
]
DEFAULT_POSE = np.array([
    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,
    0.2,0.2,0, 1.28,0,0,0,
    0.2,-0.2,0, 1.28,0,0,0,
], dtype=np.float32)

print("Loading scene...")
m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
for i in range(m.nu):
    m.actuator_gainprm[i,0]=100; m.actuator_biasprm[i,1]=-100; m.actuator_biasprm[i,2]=-20

print("Loading policy...")
policy = PPO.load(f"{CKPT}/ppo_g1_{N}_steps.zip")
class DE(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (NUM_OBS,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1,1,(NUM_ACT,), dtype=np.float32)
    def reset(self): return np.zeros(NUM_OBS,dtype=np.float32),{}
    def step(self,a): return np.zeros(NUM_OBS,dtype=np.float32),0.0,False,False,{}
    def render(self): return None
    def close(self): pass

norm = VecNormalize.load(f"{CKPT}/vec_normalize_{N}.pkl", DummyVecEnv([lambda: DE()]))
norm.training = False; norm.norm_reward = False

# Build indices
ji,jq,jd=[],[],[]
for n in JOINT_NAMES:
    jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)
    ji.append(jid)
    jq.append(m.jnt_qposadr[jid]); jd.append(m.jnt_dofadr[jid])
aid={}
for i in range(m.nu):
    n_=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i)
    if n_: aid[n_]=i
pa=np.zeros(NUM_ACT,dtype=np.float32)

def reset():
    global pa
    mujoco.mj_resetData(m,d)
    q=d.qpos.copy(); q[2]=0.79; q[3:7]=[1,0,0,0]
    for i,a in enumerate(jq): q[a]=DEFAULT_POSE[i]
    d.qpos[:]=q; mujoco.mj_forward(m,d); pa[:]=0
    for i,n in enumerate(JOINT_NAMES): d.ctrl[aid[n]]=DEFAULT_POSE[i]
    mujoco.mj_forward(m,d)

def build_obs(cmd):
    rot=d.xmat[1].reshape(3,3)
    bv=rot.T@d.qvel[0:3]; ba=d.qvel[3:6]; gv=rot.T@np.array([0,0,-1])
    jp=np.array([d.qpos[a] for a in jq],dtype=np.float32)
    jv=np.array([d.qvel[a] for a in jd],dtype=np.float32)
    o=np.concatenate([bv.astype(np.float32),ba.astype(np.float32),gv.astype(np.float32),jp,jv,pa,cmd])
    return np.clip(np.nan_to_num(o,nan=0,posinf=10,neginf=-10),-100,100)

def apply(a):
    global pa
    a=np.clip(a,-1,1); s=pa*0.6+a*0.4; pa=s.copy()
    for i,n in enumerate(JOINT_NAMES):
        lo,hi=m.jnt_range[ji[i]]
        d.ctrl[aid[n]]=np.clip(DEFAULT_POSE[i]+s[i]*(hi-lo)*0.3,lo,hi)

torso_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"torso_link")
goal_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"goal")
print(f"torso={torso_id} goal={goal_id}")

reset()
cmd=np.array([0.8,0,0],dtype=np.float32)

with mujoco.viewer.launch_passive(m,d) as v:
    v.cam.distance=4.0; v.cam.azimuth=45; v.cam.elevation=-20
    print("Viewer open. Robot walking toward goal at (4,0,0)...")
    step=0
    bearing=0.0
    while v.is_running():
        # Steering (ground truth)
        tp=d.xpos[torso_id].copy(); tm=d.xmat[torso_id].reshape(3,3)
        fwd=tm[:,0]; right=tm[:,1]
        gp=d.xpos[goal_id].copy(); gd=gp-tp; gd[2]=0
        dist=np.linalg.norm(gd)
        if dist>0.5:
            gd/=dist
            bearing=np.arctan2(np.dot(gd,right),np.dot(gd,fwd))
            cmd[0]=0.8*np.clip(np.dot(gd,fwd),0.3,1.0)
            cmd[2]=np.clip(-bearing*1.5,-1.0,1.0)
        else:
            cmd[:]=0

        # Policy
        ob=build_obs(cmd)
        ob_n=norm.normalize_obs(ob.reshape(1,-1)).flatten()
        a,_=policy.predict(ob_n.reshape(1,-1),deterministic=True)
        apply(a.flatten())
        mujoco.mj_step(m,d); step+=1

        if step%100==0:
            print(f"  step={step} dist={dist:.2f} bearing={np.degrees(bearing):.0f}deg cmd={cmd}")
        if step>2000 or d.qpos[2]<0.45:
            print("RESET"); reset(); step=0

        v.sync()
