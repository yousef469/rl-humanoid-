class UI {
  constructor() {
    this.el = id => document.getElementById(id);
    this.speechBubble = this.el('speech-bubble');
    this.speechText = this.el('speech-text');
    this.cmdInput = this.el('command-input');
    this.sendBtn = this.el('send-btn');
    this.findBallBtn = this.el('find-ball-btn');
    this.statPos = this.el('stat-pos');
    this.statFacing = this.el('stat-facing');
    this.statSpeed = this.el('stat-speed');
    this.statObjects = this.el('stat-objects');
    this.statState = this.el('stat-state');
    this.statBall = this.el('stat-ball');
    this.contactL = this.el('contact-left');
    this.contactR = this.el('contact-right');
    this.obstaclePanel = this.el('obstacle-panel');
    this.connectionBadge = this.el('connection-badge');

    this.speechTimer = null;
    this.onSend = null;

    this.sendBtn.addEventListener('click', () => this._send());
    this.cmdInput.addEventListener('keydown', e => { if (e.key === 'Enter') this._send(); });
    this.findBallBtn.addEventListener('click', () => { if (this.onSeekBall) this.onSeekBall(); });
  }

  _send() {
    const t = this.cmdInput.value.trim();
    if (!t || !this.onSend) return;
    this.onSend(t);
    this.cmdInput.value = '';
  }

  showSpeech(text, duration = 5000) {
    if (this.speechTimer) clearTimeout(this.speechTimer);
    this.speechText.textContent = text;
    this.speechBubble.style.display = 'block';
    this.speechTimer = setTimeout(() => { this.speechBubble.style.display = 'none'; }, duration);
  }

  updateStatus(data) {
    const pos = data.robot_pos || [0,0,0];
    const yaw = data.robot_rot || 0;
    const vel = data.robot_vel || [0,0,0];
    const dirs = ["N","NE","E","SE","S","SW","W","NW"];
    const dir = dirs[Math.round(yaw / (Math.PI/4)) % 8];
    const speed = Math.sqrt(vel[0]**2 + vel[1]**2);
    this.statPos.textContent = `${pos[0].toFixed(1)}, ${pos[1].toFixed(1)}`;
    this.statFacing.textContent = dir;
    this.statSpeed.textContent = `${(speed * 3.6).toFixed(1)} km/h`;
    this.statObjects.textContent = (data.objects || []).length;
    this.statState.textContent = data.seeking_red_ball ? 'seeking ball' : (data.walking ? 'walking' : 'idle');
    this.statBall.textContent = data.red_ball_seen ? 'visible' : (data.seeking_red_ball ? 'searching' : '—');

    this.contactL.className = 'contact-dot ' + (data.left_contact ? 'on' : 'off');
    this.contactR.className = 'contact-dot ' + (data.right_contact ? 'on' : 'off');

    this.obstaclePanel.style.display = data.obstacle ? 'block' : 'none';
  }

  setConnection(state) {
    this.connectionBadge.textContent = state;
    this.connectionBadge.className = state === 'Connected' ? 'connected' : '';
  }

  setLlmState(state) {
    this.statState.textContent = state;
  }
}
