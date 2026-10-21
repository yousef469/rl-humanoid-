const JOINT_MAP = {
  neck_pitch:       { bone: "neck",        axis: "x" },
  neck_yaw:         { bone: "head",        axis: "y" },
  l_shoulder_pitch: { bone: "l_upper_arm", axis: "x" },
  l_shoulder_roll:  { bone: "l_shoulder",  axis: "z" },
  l_shoulder_yaw:   { bone: "l_upper_arm", axis: "y" },
  l_elbow:          { bone: "l_forearm",   axis: "x" },
  r_shoulder_pitch: { bone: "r_upper_arm", axis: "x" },
  r_shoulder_roll:  { bone: "r_shoulder",  axis: "z" },
  r_shoulder_yaw:   { bone: "r_upper_arm", axis: "y" },
  r_elbow:          { bone: "r_forearm",   axis: "x" },
  l_hip_yaw:        { bone: "l_thigh",     axis: "y" },
  l_hip_roll:       { bone: "l_thigh",     axis: "z" },
  l_hip_pitch:      { bone: "l_thigh",     axis: "x" },
  l_knee:           { bone: "l_shin",      axis: "x" },
  l_ankle:          { bone: "l_foot",      axis: "x" },
  r_hip_yaw:        { bone: "r_thigh",     axis: "y" },
  r_hip_roll:       { bone: "r_thigh",     axis: "z" },
  r_hip_pitch:      { bone: "r_thigh",     axis: "x" },
  r_knee:           { bone: "r_shin",      axis: "x" },
  r_ankle:          { bone: "r_foot",      axis: "x" },
  torso_twist:      { bone: "torso",       axis: "y" },
  torso_bend:       { bone: "chest",       axis: "x" },
};

class Humanoid {
  constructor(scene) {
    this.scene = scene;
    this.bones = {};
    this.meshes = [];
    this.hipOffset = 0.855;
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this._build();
  }

  _build() {
    const L = { torso:0.32, upper_arm:0.32, forearm:0.30, thigh:0.38, shin:0.36, foot:0.07, neck:0.12, head:0.18 };
    const hipY = this.hipOffset;

    const matBody = new THREE.MeshStandardMaterial({ color:0xccccd0, metalness:0.5, roughness:0.3 });
    const matDark = new THREE.MeshStandardMaterial({ color:0x2a2d35, metalness:0.7, roughness:0.2 });
    const matJoint = new THREE.MeshStandardMaterial({ color:0x3a3d45, metalness:0.6, roughness:0.3 });
    const matAccent = new THREE.MeshStandardMaterial({ color:0x00ccff, emissive:0x00ccff, emissiveIntensity:0.4 });
    const matFoot = new THREE.MeshStandardMaterial({ color:0x222222, metalness:0.4, roughness:0.6 });
    const matVisor = new THREE.MeshStandardMaterial({ color:0x00eeff, emissive:0x00eeff, emissiveIntensity:0.6 });

    const pelvis = new THREE.Group();
    pelvis.position.set(0, hipY, 0);
    this.root.add(pelvis);
    this.bones["root"] = pelvis;
    this._addMesh(new THREE.BoxGeometry(0.34,0.16,0.20), matBody, pelvis, [0,0,0]);

    const torso = new THREE.Group();
    torso.position.set(0, 0.08, 0);
    pelvis.add(torso);
    this.bones["torso"] = torso;
    this._addMesh(new THREE.BoxGeometry(0.36,0.30,0.18), matBody, torso, [0,0.15,0]);

    const chest = new THREE.Group();
    chest.position.set(0, 0.32, 0);
    torso.add(chest);
    this.bones["chest"] = chest;
    this._addMesh(new THREE.BoxGeometry(0.32,0.24,0.16), matBody, chest, [0,0.12,0]);
    this._addMesh(new THREE.SphereGeometry(0.04,10,10), matAccent, chest, [0,0.04,-0.09]);

    const neck = new THREE.Group();
    neck.position.set(0, 0.24, 0);
    chest.add(neck);
    this.bones["neck"] = neck;
    this._addMesh(new THREE.CylinderGeometry(0.04,0.05,0.12,8), matDark, neck, [0,0.06,0]);

    const head = new THREE.Group();
    head.position.set(0, 0.12, 0);
    neck.add(head);
    this.bones["head"] = head;
    this._addMesh(new THREE.BoxGeometry(0.22,0.20,0.20), matBody, head, [0,0,0]);
    this._addMesh(new THREE.BoxGeometry(0.16,0.035,0.015), matVisor, head, [0,0.02,-0.11]);
    this._addMesh(new THREE.CylinderGeometry(0.006,0.006,0.08,6), matDark, head, [0,0.14,0]);
    this._addMesh(new THREE.SphereGeometry(0.015,8,8), matAccent, head, [0,0.18,0]);

    // Arms
    this._buildArm("l", -0.19, torso, L, matBody, matJoint, matDark);
    this._buildArm("r", 0.19, torso, L, matBody, matJoint, matDark);

    // Legs
    this._buildLeg("l", -0.09, pelvis, L, matBody, matJoint, matFoot, matDark);
    this._buildLeg("r", 0.09, pelvis, L, matBody, matJoint, matFoot, matDark);

    this.setJointAngles({ l_knee:0.05, r_knee:0.05 });
  }

  _buildArm(side, x, parent, L, matBody, matJoint, matDark) {
    const shoulder = new THREE.Group();
    shoulder.position.set(x, 0.06, 0);
    parent.add(shoulder);
    this.bones[`${side}_shoulder`] = shoulder;
    this._addMesh(new THREE.SphereGeometry(0.055,8,8), matJoint, shoulder, [0,0,0]);

    const ua = new THREE.Group();
    shoulder.add(ua);
    this.bones[`${side}_upper_arm`] = ua;
    this._addMesh(new THREE.CylinderGeometry(0.04,0.04,L.upper_arm,10), matBody, ua, [0,-L.upper_arm/2,0]);

    const fa = new THREE.Group();
    fa.position.set(0, -L.upper_arm, 0);
    ua.add(fa);
    this.bones[`${side}_forearm`] = fa;
    this._addMesh(new THREE.CylinderGeometry(0.035,0.035,L.forearm,10), matBody, fa, [0,-L.forearm/2,0]);
  }

  _buildLeg(side, x, parent, L, matBody, matJoint, matFoot, matDark) {
    const thigh = new THREE.Group();
    thigh.position.set(x, -0.08, 0);
    parent.add(thigh);
    this.bones[`${side}_thigh`] = thigh;
    this._addMesh(new THREE.CylinderGeometry(0.07,0.06,L.thigh,8), matBody, thigh, [0,-L.thigh/2,0]);
    this._addMesh(new THREE.SphereGeometry(0.07,8,8), matJoint, thigh, [0,-L.thigh,0]);

    const shin = new THREE.Group();
    shin.position.set(0, -L.thigh, 0);
    thigh.add(shin);
    this.bones[`${side}_shin`] = shin;
    this._addMesh(new THREE.CylinderGeometry(0.055,0.045,L.shin,8), matBody, shin, [0,-L.shin/2,0]);
    this._addMesh(new THREE.SphereGeometry(0.055,8,8), matJoint, shin, [0,-L.shin,0]);

    const foot = new THREE.Group();
    foot.position.set(0, -L.shin, 0);
    shin.add(foot);
    this.bones[`${side}_foot`] = foot;
    this._addMesh(new THREE.BoxGeometry(0.12,0.07,0.22), matFoot, foot, [0,-0.035,-0.03]);
  }

  _addMesh(geo, mat, parent, pos) {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(pos[0], pos[1], pos[2]);
    m.castShadow = true;
    m.receiveShadow = true;
    parent.add(m);
    this.meshes.push(m);
    return m;
  }

  setJointAngles(joints) {
    for (const [name, angle] of Object.entries(joints)) {
      const m = JOINT_MAP[name];
      if (!m) continue;
      const bone = this.bones[m.bone];
      if (!bone) continue;
      const sign = m.axis === "y" ? 1 : -1;
      bone.rotation[m.axis] = sign * angle;
    }
  }

  setRootTransform(pos, yaw) {
    this.root.position.set(-pos[1], pos[2] - this.hipOffset, -pos[0]);
    this.root.rotation.y = yaw;
  }

  getBone(name) { return this.bones[name]; }
}
