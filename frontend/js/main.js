const DIR_NAMES = ["N","NE","E","SE","S","SW","W","NW"];

class RobotApp {
  constructor() {
    const vp = document.getElementById('viewport');
    this._scene3d = new Scene3D(vp);
    this.humanoid = new Humanoid(this._scene3d.scene);
    this.objects = new ObjectManager(this._scene3d.scene);
    this.ui = new UI();

    this.ws = null;
    this.connected = false;
    this.lastSpeech = "";

    this.ui.onSend = text => this._sendCommand(text);
    this.ui.onSeekBall = () => this._seekRedBall();
    this.cameraCanvas = document.getElementById('camera-canvas');
    this.cameraCtx = this.cameraCanvas.getContext('2d');
    document.getElementById('camera-overlay').style.display = 'none';

    this._animate = this._animate.bind(this);
    this._connect();
    this._animate();
  }

  _connect() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || 'localhost:8000';
    const url = `${proto}//${host}/ws`;
    try {
      this.ws = new WebSocket(url);
      this.ws.onopen = () => { this.connected = true; this.ui.setConnection('Connected'); };
      this.ws.onmessage = e => this._onMessage(e);
      this.ws.onclose = () => {
        this.connected = false;
        this.ui.setConnection('Disconnected');
        setTimeout(() => this._connect(), 2000);
      };
      this.ws.onerror = () => { this.connected = false; this.ui.setConnection('Error'); };
    } catch (_) {
      this.ui.setConnection('Connection failed');
      setTimeout(() => this._connect(), 2000);
    }
  }

  _sendCommand(text) {
    if (!this.connected || !this.ws) { this.ui.showSpeech("Not connected."); return; }
    this.ui.setLlmState('thinking...');
    this.ws.send(JSON.stringify({ type: "command", text }));
  }

  _seekRedBall() {
    if (!this.connected || !this.ws) { this.ui.showSpeech("Not connected."); return; }
    this.ui.setLlmState('searching...');
    this.ws.send(JSON.stringify({ type: "seek_red_ball" }));
  }

  _onMessage(event) {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch (_) {
      return;
    }
    if (data.type === 'state') this._handleState(data);
    else if (data.type === 'camera') this._handleCamera(data);
  }

  _handleState(d) {
    // Animation
    if (d.joints) {
      this.humanoid.setJointAngles(d.joints);
    }
    if (d.robot_pos) {
      this.humanoid.setRootTransform(d.robot_pos, d.robot_rot || 0);
      // camera follows robot
      const p = this.humanoid.root.position;
      this._scene3d.followTarget.copy(p);
    }
    if (d.objects) {
      this.objects.update(d.objects);
    }
    // Speech
    if (d.speech && d.speech !== this.lastSpeech) {
      this.lastSpeech = d.speech;
      this.ui.showSpeech(d.speech);
      this.ui.setLlmState('idle');
    }
    // UI
    this.ui.updateStatus(d);
  }

  _handleCamera(d) {
    const ov = document.getElementById('camera-overlay');
    ov.style.display = 'block';
    const img = new Image();
    img.onload = () => {
      const ctx = this.cameraCtx;
      const w = this.cameraCanvas.width, h = this.cameraCanvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0, w, h);
      ctx.strokeStyle = '#00ff44';
      ctx.lineWidth = 2;
      ctx.font = '11px monospace';
      ctx.fillStyle = '#00ff44';
      for (const det of d.detections || []) {
        const [x1,y1,x2,y2] = det.bbox;
        const sx = x1*w/320, sy = y1*h/240, sw = (x2-x1)*w/320, sh = (y2-y1)*h/240;
        ctx.strokeRect(sx, sy, sw, sh);
        ctx.fillText(`${det.label} ${(det.confidence*100).toFixed(0)}%`, sx, sy-4);
      }
    };
    img.src = 'data:image/jpeg;base64,' + d.image;
  }

  _animate() {
    requestAnimationFrame(this._animate);
    this._scene3d.render();
  }
}

class Scene3D {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0d0d1a);
    this.scene.fog = new THREE.Fog(0x0d0d1a, 40, 120);

    const w = container.clientWidth, h = container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(45, w/h, 0.1, 200);
    this.camera.position.set(6, 5, 8);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.followTarget = new THREE.Vector3(0, 0.9, 0);
    this._setup();

    window.addEventListener('resize', () => this._resize());
  }

  _setup() {
    const amb = new THREE.HemisphereLight(0x4488ff, 0x002244, 0.5);
    this.scene.add(amb);
    const sun = new THREE.DirectionalLight(0xffeedd, 1.5);
    sun.position.set(8, 18, 10);
    sun.castShadow = true;
    sun.shadow.mapSize.width = 2048;
    sun.shadow.mapSize.height = 2048;
    sun.shadow.camera.near = 0.5;
    sun.shadow.camera.far = 100;
    sun.shadow.camera.left = -40;
    sun.shadow.camera.right = 40;
    sun.shadow.camera.top = 40;
    sun.shadow.camera.bottom = -40;
    this.scene.add(sun);
    const fill = new THREE.DirectionalLight(0x4488ff, 0.25);
    fill.position.set(-5, 8, -10);
    this.scene.add(fill);

    // Ground grid
    const grid = new THREE.GridHelper(120, 60, 0x333366, 0x222244);
    grid.position.y = -0.01;
    this.scene.add(grid);
    const gGeo = new THREE.PlaneGeometry(120, 120);
    const gMat = new THREE.MeshStandardMaterial({ color:0x0d0d1a, roughness:0.95, metalness:0.0, transparent:true, opacity:0.85 });
    const gnd = new THREE.Mesh(gGeo, gMat);
    gnd.rotation.x = -Math.PI/2;
    gnd.position.y = -0.01;
    gnd.receiveShadow = true;
    this.scene.add(gnd);
  }

  _resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    this.camera.aspect = w/h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  render() {
    // Smooth camera follow
    this.controls.target.lerp(this.followTarget, 0.03);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

document.addEventListener('DOMContentLoaded', () => { new RobotApp(); });
