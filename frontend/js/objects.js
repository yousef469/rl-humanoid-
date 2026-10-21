class ObjectManager {
  constructor(scene) {
    this.scene = scene;
    this.markers = {};
  }

  update(objects) {
    const ids = new Set(objects.map(o => o.id));
    for (const id of Object.keys(this.markers)) {
      if (!ids.has(id)) {
        this.scene.remove(this.markers[id]);
        delete this.markers[id];
      }
    }
    for (const obj of objects) {
      const p = this._worldPos(obj.position);
      if (this.markers[obj.id]) {
        this.markers[obj.id].position.copy(p);
      } else {
        this._create(obj, p);
      }
    }
  }

  _worldPos(p) {
    // PyBullet (x,y,z) = (forward, left, up) → Three.js (x,y,z) = (right, up, forward)
    return new THREE.Vector3(-p[1], p[2], -p[0]);
  }

  _create(obj, pos) {
    const g = new THREE.Group();
    g.position.copy(pos);
    const color = this._color(obj.label);

    switch (obj.label) {
      case "tree":
        this._addCyl(g, 0.12, 0.6, 0x8B4513, [0,0.3,0]);
        this._addCone(g, 0.45, 0.5, 0x228B22, [0,0.8,0]);
        break;
      case "rock":
        this._addSphere(g, 0.25, color, [0,0.15,0]);
        break;
      case "house": {
        this._addBox(g, 0.5, 0.35, 0.5, 0x8B4513, [0,0.17,0]);
        this._addCone(g, 0.45, 0.25, 0xCC3333, [0,0.45,0]);
        break;
      }
      case "person": {
        this._addCyl(g, 0.08, 0.22, color, [0,0.18,0]);
        this._addSphere(g, 0.06, color, [0,0.35,0]);
        break;
      }
      case "sign":
        this._addCyl(g, 0.02, 0.4, 0x888888, [0,0.2,0]);
        this._addBox(g, 0.18, 0.12, 0.02, color, [0,0.45,0]);
        break;
      case "car":
        this._addBox(g, 0.4, 0.18, 0.28, color, [0,0.1,0]);
        break;
      case "bush":
        this._addSphere(g, 0.22, color, [0,0.15,0]);
        break;
      case "red_ball":
        this._addSphere(g, 0.18, 0xff1a1a, [0,0.18,0]);
        break;
      case "lamp":
        this._addCyl(g, 0.015, 0.45, 0x666666, [0,0.22,0]);
        this._addSphere(g, 0.06, color, [0,0.48,0]);
        break;
      default:
        this._addBox(g, 0.2, 0.2, 0.2, color, [0,0.1,0]);
    }
    // label tag
    this.scene.add(g);
    this.markers[obj.id] = g;
  }

  _addBox(g, w, h, d, c, p) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w,h,d), new THREE.MeshStandardMaterial({color:c}));
    m.position.set(p[0],p[1],p[2]);
    m.castShadow = true; m.receiveShadow = true;
    g.add(m);
  }
  _addSphere(g, r, c, p) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(r,8,8), new THREE.MeshStandardMaterial({color:c}));
    m.position.set(p[0],p[1],p[2]);
    m.castShadow = true; m.receiveShadow = true;
    g.add(m);
  }
  _addCyl(g, r, h, c, p) {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(r,r,h,8), new THREE.MeshStandardMaterial({color:c}));
    m.position.set(p[0],p[1],p[2]);
    m.castShadow = true; m.receiveShadow = true;
    g.add(m);
  }
  _addCone(g, r, h, c, p) {
    const m = new THREE.Mesh(new THREE.ConeGeometry(r,h,8), new THREE.MeshStandardMaterial({color:c}));
    m.position.set(p[0],p[1],p[2]);
    m.castShadow = true; m.receiveShadow = true;
    g.add(m);
  }

  _color(label) {
    const map = { tree:0x228B22, rock:0x888888, house:0xCC4444, sign:0xFFDD44, person:0xFF8844, car:0x4488FF, bush:0x44AA44, lamp:0xFFFF88, red_ball:0xFF1A1A };
    return map[label] || 0xAAAAAA;
  }
}
