(function () {
  class OrbitControls {
    constructor(camera, domElement) {
      this.camera = camera;
      this.domElement = domElement;
      this.target = new THREE.Vector3(0, 0.9, 0);
      this.spherical = new THREE.Spherical(6, 1.0, 0.5);
      this.mouse = new THREE.Vector2();
      this.prev = new THREE.Vector2();
      this.dragging = false;

      domElement.addEventListener('mousedown', e => {
        this.dragging = true;
        this.prev.set(e.clientX, e.clientY);
      });
      domElement.addEventListener('mousemove', e => {
        if (!this.dragging) return;
        const dx = e.clientX - this.prev.x;
        const dy = e.clientY - this.prev.y;
        this.spherical.theta -= dx * 0.005;
        this.spherical.phi -= dy * 0.005;
        this.spherical.phi = Math.max(0.15, Math.min(Math.PI - 0.15, this.spherical.phi));
        this.prev.set(e.clientX, e.clientY);
        this.update();
      });
      domElement.addEventListener('mouseup', () => { this.dragging = false; });
      domElement.addEventListener('wheel', e => {
        this.spherical.radius += e.deltaY * 0.01;
        this.spherical.radius = Math.max(1.5, Math.min(25, this.spherical.radius));
        this.update();
      });
      this.update();
    }
    update() {
      const pos = new THREE.Vector3().setFromSpherical(this.spherical).add(this.target);
      this.camera.position.copy(pos);
      this.camera.lookAt(this.target);
    }
    setTarget(p) {
      this.target.copy(p);
      this.camera.lookAt(this.target);
    }
  }
  window.OrbitControls = OrbitControls;
})();
