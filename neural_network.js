
/**
 * ╔══════════════════════════════════════════════════════════╗
 * ║   NEURAL NETWORK FROM SCRATCH                           ║
 * ║   Pure JavaScript — No Libraries — No Frameworks        ║
 * ║   Author : Ammar | github.com/ammar                     ║
 * ╚══════════════════════════════════════════════════════════╝
 *
 * Implements a fully connected neural network from scratch:
 *  - Matrix operations without any library
 *  - Forward pass + Backpropagation + Gradient Descent
 *  - Live training animation in browser canvas
 *  - Real-time loss curve rendering
 *  - Decision boundary visualization
 */

// ═══════════════════════════════════════
//  MATRIX MATH (from scratch)
// ═══════════════════════════════════════

const Matrix = {
  create: (rows, cols, fill = 0) =>
    Array.from({ length: rows }, () => Array(cols).fill(fill)),

  random: (rows, cols, scale = 1) =>
    Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
    ),

  zeros: (rows, cols) => Matrix.create(rows, cols, 0),

  dot: (A, B) => {
    const rows = A.length, cols = B[0].length, inner = B.length;
    const C = Matrix.zeros(rows, cols);
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++)
        for (let k = 0; k < inner; k++)
          C[i][j] += A[i][k] * B[k][j];
    return C;
  },

  add: (A, B) => A.map((row, i) => row.map((v, j) => v + B[i][j])),
  sub: (A, B) => A.map((row, i) => row.map((v, j) => v - B[i][j])),
  mul: (A, B) => A.map((row, i) => row.map((v, j) => v * B[i][j])),
  scale: (A, s) => A.map(row => row.map(v => v * s)),

  transpose: A => A[0].map((_, j) => A.map(row => row[j])),

  map: (A, fn) => A.map(row => row.map(fn)),

  addBias: (A, b) => A.map(row => row.map((v, j) => v + b[0][j])),

  sumRows: A => [A[0].map((_, j) => A.reduce((s, row) => s + row[j], 0) / A.length)],
};

// ═══════════════════════════════════════
//  ACTIVATION FUNCTIONS
// ═══════════════════════════════════════

const sigmoid  = x => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
const sigmoidD = x => { const s = sigmoid(x); return s * (1 - s); };
const relu     = x => Math.max(0, x);
const reluD    = x => x > 0 ? 1 : 0;
const tanh     = x => Math.tanh(x);
const tanhD    = x => 1 - Math.tanh(x) ** 2;

// ═══════════════════════════════════════
//  NEURAL NETWORK CLASS
// ═══════════════════════════════════════

class NeuralNetwork {
  constructor(layerSizes, lr = 0.5) {
    this.layers = layerSizes;
    this.lr     = lr;
    this.depth  = layerSizes.length - 1;
    this.lossHistory = [];

    this.weights = [];
    this.biases  = [];

    for (let i = 0; i < this.depth; i++) {
      const scale = Math.sqrt(2 / layerSizes[i]);
      this.weights.push(Matrix.random(layerSizes[i], layerSizes[i + 1], scale));
      this.biases.push(Matrix.zeros(1, layerSizes[i + 1]));
    }
  }

  forward(X) {
    this.activations = [X];
    this.zValues     = [];
    let current = X;

    for (let i = 0; i < this.depth; i++) {
      const z = Matrix.addBias(Matrix.dot(current, this.weights[i]), this.biases[i]);
      this.zValues.push(z);
      const a = Matrix.map(z, sigmoid);
      this.activations.push(a);
      current = a;
    }
    return current;
  }

  backward(X, yTrue) {
    const m      = X.length;
    const yPred  = this.activations[this.depth];
    const gradW  = Array(this.depth).fill(null);
    const gradB  = Array(this.depth).fill(null);

    let delta = Matrix.mul(
      Matrix.sub(yPred, yTrue),
      Matrix.map(this.zValues[this.depth - 1], sigmoidD)
    );

    for (let i = this.depth - 1; i >= 0; i--) {
      const aPrev = this.activations[i];
      gradW[i] = Matrix.scale(Matrix.dot(Matrix.transpose(aPrev), delta), 1 / m);
      gradB[i] = Matrix.sumRows(delta);

      if (i > 0) {
        delta = Matrix.mul(
          Matrix.dot(delta, Matrix.transpose(this.weights[i])),
          Matrix.map(this.zValues[i - 1], sigmoidD)
        );
      }
    }

    for (let i = 0; i < this.depth; i++) {
      this.weights[i] = Matrix.sub(this.weights[i], Matrix.scale(gradW[i], this.lr));
      this.biases[i]  = Matrix.sub(this.biases[i],  Matrix.scale(gradB[i], this.lr));
    }
  }

  binaryCrossEntropy(yTrue, yPred) {
    const eps = 1e-12;
    let loss = 0;
    for (let i = 0; i < yTrue.length; i++)
      loss += -(yTrue[i][0] * Math.log(yPred[i][0] + eps) +
               (1 - yTrue[i][0]) * Math.log(1 - yPred[i][0] + eps));
    return loss / yTrue.length;
  }

  predict(X, threshold = 0.5) {
    return this.forward(X).map(row => [row[0] >= threshold ? 1 : 0]);
  }

  accuracy(X, y) {
    const preds = this.predict(X);
    return preds.filter((p, i) => p[0] === y[i][0]).length / y.length;
  }

  getDecisionBoundary(resolution = 40) {
    const grid = [];
    for (let y = 0; y <= resolution; y++) {
      for (let x = 0; x <= resolution; x++) {
        const px = x / resolution;
        const py = y / resolution;
        const val = this.forward([[px, py]])[0][0];
        grid.push({ x: px, y: py, val });
      }
    }
    return grid;
  }
}

// ═══════════════════════════════════════
//  UI RENDERER
// ═══════════════════════════════════════

class NNRenderer {
  constructor(container) {
    this.container = container;
    this.nn        = null;
    this.animId    = null;
    this.epoch     = 0;
    this.maxEpochs = 4000;
    this.X = [[0,0],[0,1],[1,0],[1,1]];
    this.y = [[0],[1],[1],[0]];
    this._buildUI();
  }

  _buildUI() {
    this.container.innerHTML = `
      <section class="nn-section">
        <div class="section-label">PROJECT 01</div>
        <h2 class="section-title">Neural Network <em>from Scratch</em></h2>
        <p class="section-desc">
          A complete neural network built using only JavaScript math —
          no TensorFlow, no libraries. Trained live in your browser on the XOR problem.
        </p>

        <div class="nn-grid">
          <div class="nn-panel">
            <div class="panel-header">
              <span class="panel-tag">LIVE TRAINING</span>
              <span id="nn-epoch-label">Epoch 0 / ${this.maxEpochs}</span>
            </div>
            <canvas id="nn-loss-canvas" width="420" height="200"></canvas>
            <div class="nn-stats-row">
              <div class="stat-chip"><span class="stat-label">LOSS</span><span id="nn-loss-val">—</span></div>
              <div class="stat-chip"><span class="stat-label">ACCURACY</span><span id="nn-acc-val">—</span></div>
              <div class="stat-chip"><span class="stat-label">ARCHITECTURE</span><span>2→8→8→1</span></div>
            </div>
          </div>

          <div class="nn-panel">
            <div class="panel-header">
              <span class="panel-tag">DECISION BOUNDARY</span>
              <span>XOR Problem</span>
            </div>
            <canvas id="nn-boundary-canvas" width="420" height="300"></canvas>
          </div>
        </div>

        <div class="nn-truth-table">
          <div class="panel-header"><span class="panel-tag">XOR TRUTH TABLE</span><span>Predictions update live</span></div>
          <div class="truth-grid" id="nn-truth-grid">
            ${this.X.map((x, i) => `
              <div class="truth-row">
                <span class="truth-input">${x[0]} XOR ${x[1]}</span>
                <span class="truth-target">Target: ${this.y[i][0]}</span>
                <span class="truth-pred" id="pred-${i}">—</span>
                <span class="truth-match" id="match-${i}">⏳</span>
              </div>`).join('')}
          </div>
        </div>

        <div class="btn-row">
          <button id="nn-start-btn" class="btn-primary">▶ Start Training</button>
          <button id="nn-reset-btn" class="btn-ghost">↺ Reset</button>
        </div>
      </section>
    `;

    document.getElementById('nn-start-btn').onclick = () => this.start();
    document.getElementById('nn-reset-btn').onclick = () => this.reset();
    this._initNN();
  }

  _initNN() {
    this.nn    = new NeuralNetwork([2, 8, 8, 1], 0.5);
    this.epoch = 0;
    this._drawBoundary();
  }

  start() {
    if (this.animId) return;
    document.getElementById('nn-start-btn').textContent = '⏸ Training...';
    document.getElementById('nn-start-btn').disabled = true;
    this._trainStep();
  }

  reset() {
    cancelAnimationFrame(this.animId);
    this.animId = null;
    this._initNN();
    document.getElementById('nn-start-btn').textContent = '▶ Start Training';
    document.getElementById('nn-start-btn').disabled = false;
    document.getElementById('nn-epoch-label').textContent = `Epoch 0 / ${this.maxEpochs}`;
    document.getElementById('nn-loss-val').textContent = '—';
    document.getElementById('nn-acc-val').textContent = '—';
    const lossCanvas = document.getElementById('nn-loss-canvas');
    lossCanvas.getContext('2d').clearRect(0,0,lossCanvas.width,lossCanvas.height);
    for (let i = 0; i < 4; i++) {
      document.getElementById(`pred-${i}`).textContent = '—';
      document.getElementById(`match-${i}`).textContent = '⏳';
    }
  }

  _trainStep() {
    const stepsPerFrame = 20;
    for (let s = 0; s < stepsPerFrame && this.epoch < this.maxEpochs; s++) {
      const yPred = this.nn.forward(this.X);
      const loss  = this.nn.binaryCrossEntropy(this.y, yPred);
      this.nn.lossHistory.push(loss);
      this.nn.backward(this.X, this.y);
      this.epoch++;
    }

    this._updateStats();
    this._drawLossCurve();
    this._drawBoundary();
    this._updateTruthTable();

    if (this.epoch < this.maxEpochs) {
      this.animId = requestAnimationFrame(() => this._trainStep());
    } else {
      this.animId = null;
      document.getElementById('nn-start-btn').textContent = '✓ Complete';
    }
  }

  _updateStats() {
    const loss = this.nn.lossHistory[this.nn.lossHistory.length - 1];
    const acc  = this.nn.accuracy(this.X, this.y);
    document.getElementById('nn-epoch-label').textContent = `Epoch ${this.epoch} / ${this.maxEpochs}`;
    document.getElementById('nn-loss-val').textContent    = loss.toFixed(5);
    document.getElementById('nn-acc-val').textContent     = (acc * 100).toFixed(1) + '%';
  }

  _updateTruthTable() {
    const preds = this.nn.forward(this.X);
    this.X.forEach((x, i) => {
      const raw   = preds[i][0];
      const pred  = raw >= 0.5 ? 1 : 0;
      const match = pred === this.y[i][0];
      document.getElementById(`pred-${i}`).textContent  = `Output: ${raw.toFixed(4)}`;
      document.getElementById(`match-${i}`).textContent = match ? '✅' : '❌';
    });
  }

  _drawLossCurve() {
    const canvas = document.getElementById('nn-loss-canvas');
    const ctx    = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const history = this.nn.lossHistory;
    if (history.length < 2) return;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, W, H);

    const maxLoss = Math.max(...history);
    const minLoss = Math.min(...history);
    const range   = maxLoss - minLoss || 1;
    const pad     = { top: 20, right: 20, bottom: 30, left: 50 };
    const w = W - pad.left - pad.right;
    const h = H - pad.top  - pad.bottom;

    // Grid lines
    ctx.strokeStyle = '#1a1a2e';
    ctx.lineWidth   = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * h;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + w, y); ctx.stroke();
      const val = maxLoss - (i / 4) * range;
      ctx.fillStyle = '#444'; ctx.font = '10px monospace';
      ctx.fillText(val.toFixed(3), 2, y + 4);
    }

    // Loss line with glow
    ctx.shadowColor = '#00ff88';
    ctx.shadowBlur  = 8;
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth   = 2;
    ctx.beginPath();
    history.forEach((loss, i) => {
      const x = pad.left + (i / (history.length - 1)) * w;
      const y = pad.top  + (1 - (loss - minLoss) / range) * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Axis label
    ctx.fillStyle = '#666'; ctx.font = '10px monospace';
    ctx.fillText('Loss', pad.left, pad.top - 6);
    ctx.fillText('Epochs →', W - 70, H - 4);
  }

  _drawBoundary() {
    const canvas = document.getElementById('nn-boundary-canvas');
    const ctx    = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = 40;
    const grid = this.nn.getDecisionBoundary(60);

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, W, H);

    const cw = (W - pad * 2) / 60;
    const ch = (H - pad * 2) / 60;

    grid.forEach(({ x, y, val }) => {
      const px = pad + x * (W - pad * 2);
      const py = pad + y * (H - pad * 2);
      const r  = Math.round(val * 30);
      const g  = Math.round(val * 200);
      const b  = Math.round((1 - val) * 200 + 50);
      ctx.fillStyle = `rgba(${r},${g},${b},0.7)`;
      ctx.fillRect(px - cw / 2, py - ch / 2, cw + 1, ch + 1);
    });

    // Data points
    this.X.forEach((pt, i) => {
      const px = pad + pt[0] * (W - pad * 2);
      const py = pad + pt[1] * (H - pad * 2);
      ctx.shadowColor = this.y[i][0] === 1 ? '#00ff88' : '#ff4466';
      ctx.shadowBlur  = 12;
      ctx.fillStyle   = this.y[i][0] === 1 ? '#00ff88' : '#ff4466';
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    // Labels
    ctx.fillStyle = '#555'; ctx.font = '11px monospace';
    ctx.fillText('(0,0)', pad - 5, H - 10);
    ctx.fillText('(1,1)', W - pad - 25, 15);
  }
}

// ═══════════════════════════════════════
//  BOOT
// ═══════════════════════════════════════

window.__NNRenderer = NNRenderer;
