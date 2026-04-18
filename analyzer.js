/**
 * ╔══════════════════════════════════════════════════════════╗
 * ║   ML DATA ANALYZER                                      ║
 * ║   Pure JavaScript — Statistics + Visualization          ║
 * ║   Author : Ammar | github.com/ammar                     ║
 * ╚══════════════════════════════════════════════════════════╝
 *
 * Analyzes any CSV dataset and renders:
 *  - Full statistical summary (mean, std, median, IQR, skewness)
 *  - Outlier detection (Z-score + IQR methods)
 *  - Correlation heatmap (canvas rendered)
 *  - Interactive histogram per column
 *  - Category distribution charts
 */

// ═══════════════════════════════════════
//  STATISTICS ENGINE
// ═══════════════════════════════════════

const Stats = {
  mean: arr => arr.reduce((s, v) => s + v, 0) / arr.length,

  variance: arr => {
    if (arr.length < 2) return 0;
    const m = Stats.mean(arr);
    return arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1);
  },

  std: arr => Math.sqrt(Stats.variance(arr)),

  median: arr => {
    const s = [...arr].sort((a, b) => a - b);
    const n = s.length;
    return n % 2 ? s[Math.floor(n / 2)] : (s[n / 2 - 1] + s[n / 2]) / 2;
  },

  percentile: (arr, p) => {
    const s = [...arr].sort((a, b) => a - b);
    const k = (s.length - 1) * p / 100;
    const f = Math.floor(k), c = Math.ceil(k);
    return f === c ? s[f] : s[f] * (c - k) + s[c] * (k - f);
  },

  skewness: arr => {
    if (arr.length < 3) return 0;
    const m = Stats.mean(arr), s = Stats.std(arr), n = arr.length;
    if (s === 0) return 0;
    return (n / ((n - 1) * (n - 2))) * arr.reduce((sum, v) => sum + ((v - m) / s) ** 3, 0);
  },

  correlation: (x, y) => {
    const mx = Stats.mean(x), my = Stats.mean(y);
    const num = x.reduce((s, xi, i) => s + (xi - mx) * (y[i] - my), 0);
    const den = Math.sqrt(
      x.reduce((s, xi) => s + (xi - mx) ** 2, 0) *
      y.reduce((s, yi) => s + (yi - my) ** 2, 0)
    );
    return den ? num / den : 0;
  },

  outliersZ: (arr, thresh = 3) => {
    const m = Stats.mean(arr), s = Stats.std(arr);
    if (!s) return [];
    return arr.reduce((out, v, i) => (Math.abs((v - m) / s) > thresh ? [...out, i] : out), []);
  },

  outliersIQR: arr => {
    const q1 = Stats.percentile(arr, 25), q3 = Stats.percentile(arr, 75);
    const iqr = q3 - q1, lo = q1 - 1.5 * iqr, hi = q3 + 1.5 * iqr;
    return arr.reduce((out, v, i) => (v < lo || v > hi ? [...out, i] : out), []);
  },
};

// ═══════════════════════════════════════
//  CSV PARSER
// ═══════════════════════════════════════

function parseCSV(text) {
  const lines   = text.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
  const rows    = lines.slice(1).map(line => {
    const vals = line.split(',').map(v => v.trim().replace(/"/g, ''));
    return Object.fromEntries(headers.map((h, i) => [h, vals[i] ?? '']));
  });

  const columns = {};
  headers.forEach(h => {
    columns[h] = rows.map(r => r[h]);
  });

  const numeric = {}, categorical = {};
  headers.forEach(h => {
    const nums = columns[h].map(v => parseFloat(v));
    if (nums.every(n => !isNaN(n))) numeric[h] = nums;
    else categorical[h] = columns[h];
  });

  return { headers, rows, columns, numeric, categorical, n: rows.length };
}

function generateDemoCSV() {
  const rng = (() => { let s = 42; return () => { s=(s*1664525+1013904223)&0xffffffff; return(s>>>0)/0xffffffff; }; })();
  const gauss = (m, sd) => { const u = rng(), v = rng(); return m + sd * Math.sqrt(-2*Math.log(u)) * Math.cos(2*Math.PI*v); };

  const headers = ['study_hours','sleep_hours','attendance','exam_score','grade','gender'];
  const rows = [headers.join(',')];

  for (let i = 0; i < 250; i++) {
    const study   = Math.max(0, Math.min(12, +gauss(5, 2).toFixed(1)));
    const sleep   = Math.max(4, Math.min(10, +gauss(7, 1).toFixed(1)));
    const attend  = Math.max(30, Math.min(100, +gauss(75, 15).toFixed(1)));
    const score   = Math.max(0, Math.min(100, +(40 + study * 4 + sleep * 1.5 + gauss(0, 8)).toFixed(1)));
    const grade   = score >= 85 ? 'A' : score >= 70 ? 'B' : score >= 55 ? 'C' : 'D';
    const gender  = rng() > 0.5 ? 'Male' : 'Female';
    rows.push([study, sleep, attend, score, grade, gender].join(','));
  }
  return rows.join('\n');
}

// ═══════════════════════════════════════
//  CANVAS CHARTS
// ═══════════════════════════════════════

function drawHistogram(canvas, values, label, color = '#00d4ff') {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const bins = 16, pad = { top: 20, right: 16, bottom: 30, left: 44 };

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0a0a0f'; ctx.fillRect(0, 0, W, H);

  const mn = Math.min(...values), mx = Math.max(...values);
  const range = mx - mn || 1;
  const counts = Array(bins).fill(0);
  values.forEach(v => counts[Math.min(Math.floor((v - mn) / range * bins), bins - 1)]++);
  const maxC = Math.max(...counts);

  const w = W - pad.left - pad.right;
  const h = H - pad.top  - pad.bottom;
  const bw = w / bins - 2;

  // Grid
  ctx.strokeStyle = '#1a1a2e'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * h;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + w, y); ctx.stroke();
    ctx.fillStyle = '#444'; ctx.font = '9px monospace';
    ctx.fillText(Math.round(maxC * (1 - i / 4)), 2, y + 4);
  }

  counts.forEach((c, i) => {
    const x   = pad.left + i * (w / bins) + 1;
    const bh  = (c / maxC) * h;
    const y   = pad.top + h - bh;
    const grad = ctx.createLinearGradient(0, y, 0, y + bh);
    grad.addColorStop(0, color);
    grad.addColorStop(1, color + '44');
    ctx.fillStyle = grad;
    ctx.fillRect(x, y, bw, bh);
  });

  ctx.fillStyle = '#777'; ctx.font = '10px monospace';
  ctx.fillText(mn.toFixed(1), pad.left, H - 4);
  ctx.fillText(mx.toFixed(1), W - 36, H - 4);
  ctx.fillText(label, pad.left + 4, pad.top - 4);
}

function drawCorrelationHeatmap(canvas, cols, matrix) {
  const ctx = canvas.getContext('2d');
  const n   = cols.length;
  const W = canvas.width, H = canvas.height;
  const pad = 70;
  const cell = Math.min((W - pad) / n, (H - pad) / n);

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0a0a0f'; ctx.fillRect(0, 0, W, H);

  matrix.forEach((row, i) => {
    row.forEach((val, j) => {
      const x = pad + j * cell, y = pad + i * cell;
      const abs = Math.abs(val);
      let r, g, b;
      if (val > 0)      { r = 0;   g = Math.round(abs * 220); b = Math.round(abs * 100); }
      else if (val < 0) { r = Math.round(abs * 220); g = 40; b = Math.round(abs * 80); }
      else              { r = 30; g = 30; b = 40; }
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(x, y, cell - 2, cell - 2);

      ctx.fillStyle = abs > 0.4 ? '#fff' : '#666';
      ctx.font = `${Math.max(9, cell * 0.28)}px monospace`;
      ctx.textAlign = 'center';
      ctx.fillText(val.toFixed(2), x + cell / 2 - 1, y + cell / 2 + 4);
    });
  });

  ctx.fillStyle = '#888'; ctx.font = '11px monospace'; ctx.textAlign = 'right';
  cols.forEach((c, i) => ctx.fillText(c.slice(0, 8), pad - 4, pad + i * cell + cell / 2 + 4));

  ctx.textAlign = 'center';
  cols.forEach((c, j) => {
    ctx.save();
    ctx.translate(pad + j * cell + cell / 2, pad - 6);
    ctx.rotate(-Math.PI / 4);
    ctx.fillText(c.slice(0, 8), 0, 0);
    ctx.restore();
  });
  ctx.textAlign = 'left';
}

// ═══════════════════════════════════════
//  ANALYZER UI
// ═══════════════════════════════════════

class AnalyzerRenderer {
  constructor(container) {
    this.container = container;
    this._buildUI();
  }

  _buildUI() {
    this.container.innerHTML = `
      <section class="analyzer-section">
        <div class="section-label">PROJECT 02</div>
        <h2 class="section-title">ML Data <em>Analyzer</em></h2>
        <p class="section-desc">
          Drop any CSV file — the analyzer computes statistics, detects outliers,
          finds correlations, and renders interactive charts. Zero dependencies.
        </p>

        <div class="upload-zone" id="upload-zone">
          <div class="upload-icon">📂</div>
          <div class="upload-text">Drop a CSV file here</div>
          <div class="upload-sub">or</div>
          <button class="btn-primary" id="demo-btn">▶ Run on Demo Dataset</button>
          <input type="file" id="csv-input" accept=".csv" style="display:none">
          <button class="btn-ghost" id="file-btn">Choose File</button>
        </div>

        <div id="analysis-output" style="display:none"></div>
      </section>
    `;

    document.getElementById('demo-btn').onclick = () => this._runAnalysis(generateDemoCSV(), 'student_data.csv');
    document.getElementById('file-btn').onclick = () => document.getElementById('csv-input').click();
    document.getElementById('csv-input').onchange = e => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = ev => this._runAnalysis(ev.target.result, file.name);
      reader.readAsText(file);
    };

    const zone = document.getElementById('upload-zone');
    zone.ondragover = e => { e.preventDefault(); zone.classList.add('drag-over'); };
    zone.ondragleave = () => zone.classList.remove('drag-over');
    zone.ondrop = e => {
      e.preventDefault(); zone.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file) { const r = new FileReader(); r.onload = ev => this._runAnalysis(ev.target.result, file.name); r.readAsText(file); }
    };
  }

  _runAnalysis(csvText, filename) {
    const data = parseCSV(csvText);
    const out  = document.getElementById('analysis-output');
    out.style.display = 'block';
    document.getElementById('upload-zone').style.display = 'none';

    const colors = ['#00d4ff','#00ff88','#ff6b6b','#ffd93d','#c77dff','#ff9a3c'];
    const numCols  = Object.keys(data.numeric);
    const catCols  = Object.keys(data.categorical);

    // Overview
    out.innerHTML = `
      <div class="analysis-header">
        <div class="file-badge">📄 ${filename}</div>
        <div class="overview-chips">
          <div class="ov-chip"><span>${data.n}</span><small>Rows</small></div>
          <div class="ov-chip"><span>${data.headers.length}</span><small>Columns</small></div>
          <div class="ov-chip"><span>${numCols.length}</span><small>Numeric</small></div>
          <div class="ov-chip"><span>${catCols.length}</span><small>Categorical</small></div>
        </div>
        <button class="btn-ghost" id="reset-analyzer">↺ New Dataset</button>
      </div>

      <div class="stats-grid" id="stats-grid"></div>

      ${numCols.length > 0 ? `
        <div class="chart-section">
          <div class="section-label">DISTRIBUTIONS</div>
          <div class="histograms-grid" id="histograms-grid"></div>
        </div>` : ''}

      ${numCols.length >= 2 ? `
        <div class="chart-section">
          <div class="section-label">CORRELATION HEATMAP</div>
          <div class="heatmap-wrap">
            <canvas id="corr-canvas"></canvas>
          </div>
        </div>` : ''}

      ${catCols.length > 0 ? `
        <div class="chart-section">
          <div class="section-label">CATEGORY DISTRIBUTIONS</div>
          <div class="cat-grid" id="cat-grid"></div>
        </div>` : ''}
    `;

    document.getElementById('reset-analyzer').onclick = () => {
      out.style.display = 'none';
      document.getElementById('upload-zone').style.display = 'flex';
    };

    // Stats cards
    const statsGrid = document.getElementById('stats-grid');
    numCols.forEach((col, ci) => {
      const v   = data.numeric[col];
      const q1  = Stats.percentile(v, 25), q3 = Stats.percentile(v, 75);
      const skew = Stats.skewness(v);
      const zo  = Stats.outliersZ(v).length;
      const io  = Stats.outliersIQR(v).length;
      const skewLabel = skew > 0.5 ? 'right-skewed' : skew < -0.5 ? 'left-skewed' : 'normal';
      const color = colors[ci % colors.length];

      statsGrid.innerHTML += `
        <div class="stat-card" style="--accent:${color}">
          <div class="stat-card-header">
            <span class="stat-col-name">${col}</span>
            <span class="stat-badge" style="color:${color}">${skewLabel}</span>
          </div>
          <div class="stat-rows">
            <div class="sr"><span>Mean</span><strong>${Stats.mean(v).toFixed(3)}</strong></div>
            <div class="sr"><span>Std</span><strong>${Stats.std(v).toFixed(3)}</strong></div>
            <div class="sr"><span>Min</span><strong>${Math.min(...v).toFixed(3)}</strong></div>
            <div class="sr"><span>Median</span><strong>${Stats.median(v).toFixed(3)}</strong></div>
            <div class="sr"><span>Max</span><strong>${Math.max(...v).toFixed(3)}</strong></div>
            <div class="sr"><span>IQR</span><strong>${(q3-q1).toFixed(3)}</strong></div>
            <div class="sr"><span>Skewness</span><strong>${skew.toFixed(3)}</strong></div>
            <div class="sr outlier-row"><span>Outliers (Z)</span><strong ${zo>0?'style="color:#ff6b6b"':''}>${zo}</strong></div>
          </div>
        </div>`;
    });

    // Histograms
    const histGrid = document.getElementById('histograms-grid');
    if (histGrid) {
      numCols.forEach((col, ci) => {
        const id = `hist-${ci}`;
        histGrid.innerHTML += `<div class="hist-wrap"><canvas id="${id}" width="280" height="160"></canvas></div>`;
        requestAnimationFrame(() => drawHistogram(
          document.getElementById(id), data.numeric[col], col, colors[ci % colors.length]
        ));
      });
    }

    // Correlation heatmap
    if (numCols.length >= 2) {
      const n = numCols.length;
      const matrix = numCols.map(c1 =>
        numCols.map(c2 => c1 === c2 ? 1 : Stats.correlation(data.numeric[c1], data.numeric[c2]))
      );
      const size = Math.min(500, 70 + n * 80);
      const corrCanvas = document.getElementById('corr-canvas');
      corrCanvas.width  = size;
      corrCanvas.height = size;
      requestAnimationFrame(() => drawCorrelationHeatmap(corrCanvas, numCols, matrix));
    }

    // Categories
    const catGrid = document.getElementById('cat-grid');
    if (catGrid) {
      catCols.forEach(col => {
        const counts = {};
        data.categorical[col].forEach(v => counts[v] = (counts[v] || 0) + 1);
        const sorted  = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8);
        const maxCount = Math.max(...sorted.map(s => s[1]));

        catGrid.innerHTML += `
          <div class="cat-card">
            <div class="cat-title">${col} <small>(${Object.keys(counts).length} unique)</small></div>
            ${sorted.map(([val, cnt], i) => `
              <div class="cat-row">
                <span class="cat-label">${val}</span>
                <div class="cat-bar-wrap">
                  <div class="cat-bar" style="width:${cnt/maxCount*100}%;background:${colors[i%colors.length]}"></div>
                </div>
                <span class="cat-count">${cnt}</span>
              </div>`).join('')}
          </div>`;
      });
    }
  }
}

// ═══════════════════════════════════════
//  BOOT — builds full page
// ═══════════════════════════════════════

(function boot() {
  // Inject styles
  const style = document.createElement('style');
  style.textContent = `
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: #06060d;
      color: #e0e0e0;
      font-family: 'Courier New', monospace;
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* Header */
    .site-header {
      padding: 60px 40px 40px;
      border-bottom: 1px solid #111;
      background: linear-gradient(180deg, #0d0d1a 0%, #06060d 100%);
      position: relative;
      overflow: hidden;
    }
    .site-header::before {
      content: '';
      position: absolute; inset: 0;
      background: radial-gradient(ellipse at 20% 50%, #00d4ff08 0%, transparent 60%),
                  radial-gradient(ellipse at 80% 50%, #00ff8808 0%, transparent 60%);
      pointer-events: none;
    }
    .header-tag {
      font-size: 11px; letter-spacing: 4px; color: #00d4ff;
      text-transform: uppercase; margin-bottom: 16px;
    }
    .header-name {
      font-size: clamp(36px, 6vw, 72px);
      font-weight: 900;
      line-height: 1;
      letter-spacing: -2px;
      color: #fff;
      font-family: 'Courier New', monospace;
    }
    .header-name em { color: #00ff88; font-style: normal; }
    .header-sub {
      margin-top: 16px;
      font-size: 14px;
      color: #555;
      letter-spacing: 2px;
      text-transform: uppercase;
    }
    .header-badges {
      display: flex; gap: 10px; flex-wrap: wrap; margin-top: 28px;
    }
    .badge {
      padding: 5px 14px;
      border: 1px solid #1e1e30;
      border-radius: 2px;
      font-size: 11px;
      letter-spacing: 1px;
      color: #888;
      background: #0d0d1a;
    }
    .badge.active { color: #00ff88; border-color: #00ff8844; background: #00ff8808; }

    /* Nav */
    .site-nav {
      display: flex; gap: 0;
      border-bottom: 1px solid #111;
      overflow-x: auto;
    }
    .nav-tab {
      padding: 16px 32px;
      font-size: 12px; letter-spacing: 2px; text-transform: uppercase;
      color: #444; background: none; border: none; cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all .2s; white-space: nowrap;
    }
    .nav-tab:hover { color: #aaa; }
    .nav-tab.active { color: #00d4ff; border-bottom-color: #00d4ff; }

    /* Main */
    .main-content { max-width: 1100px; margin: 0 auto; padding: 60px 24px 100px; }

    .tab-panel { display: none; }
    .tab-panel.active { display: block; }

    /* Section */
    .section-label {
      font-size: 10px; letter-spacing: 4px; color: #333;
      text-transform: uppercase; margin-bottom: 12px;
    }
    .section-title {
      font-size: clamp(28px, 4vw, 48px);
      font-weight: 900; color: #fff; line-height: 1.1;
      margin-bottom: 16px; letter-spacing: -1px;
    }
    .section-title em { color: #00d4ff; font-style: normal; }
    .section-desc {
      font-size: 14px; color: #555; line-height: 1.7;
      max-width: 620px; margin-bottom: 48px;
    }

    /* NN Grid */
    .nn-section { }
    .nn-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 2px; margin-bottom: 2px;
    }
    .nn-panel {
      background: #0d0d1a;
      border: 1px solid #1a1a2e;
      padding: 24px;
    }
    .panel-header {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 16px; font-size: 11px; color: #444; letter-spacing: 1px;
    }
    .panel-tag {
      font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
      color: #00d4ff; background: #00d4ff11; padding: 3px 10px;
    }
    #nn-loss-canvas, #nn-boundary-canvas { width: 100%; display: block; }

    .nn-stats-row {
      display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap;
    }
    .stat-chip {
      flex: 1; min-width: 100px;
      background: #111120; border: 1px solid #1a1a2e;
      padding: 10px 14px;
    }
    .stat-label { display: block; font-size: 9px; letter-spacing: 2px; color: #444; margin-bottom: 4px; }
    .stat-chip span:last-child { font-size: 14px; color: #00ff88; font-weight: bold; }

    .nn-truth-table {
      background: #0d0d1a; border: 1px solid #1a1a2e;
      padding: 24px; margin-top: 2px;
    }
    .truth-grid { display: flex; flex-direction: column; gap: 6px; margin-top: 12px; }
    .truth-row {
      display: grid; grid-template-columns: 1fr 1fr 1fr auto;
      gap: 12px; align-items: center;
      padding: 10px 14px; background: #111120; font-size: 12px;
    }
    .truth-input { color: #888; }
    .truth-target { color: #555; }
    .truth-pred { color: #00d4ff; font-family: monospace; }
    .truth-match { font-size: 16px; text-align: right; }
    .btn-row { display: flex; gap: 12px; margin-top: 24px; }
    .btn-primary {
      padding: 12px 32px; background: #00d4ff; color: #000;
      font-size: 12px; letter-spacing: 2px; text-transform: uppercase;
      border: none; cursor: pointer; font-weight: bold; font-family: inherit;
      transition: all .2s;
    }
    .btn-primary:hover:not(:disabled) { background: #00ff88; }
    .btn-primary:disabled { opacity: .5; cursor: default; }
    .btn-ghost {
      padding: 12px 24px; background: none;
      border: 1px solid #1a1a2e; color: #666;
      font-size: 12px; letter-spacing: 2px; font-family: inherit;
      cursor: pointer; transition: all .2s;
    }
    .btn-ghost:hover { color: #aaa; border-color: #333; }

    /* Analyzer */
    .upload-zone {
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      gap: 16px; padding: 80px 40px;
      border: 1px dashed #1a1a2e; background: #0d0d1a;
      transition: all .2s; cursor: pointer;
    }
    .upload-zone.drag-over { border-color: #00d4ff; background: #00d4ff08; }
    .upload-icon { font-size: 48px; }
    .upload-text { font-size: 16px; color: #666; }
    .upload-sub { font-size: 12px; color: #333; }

    .analysis-header {
      display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
      margin-bottom: 40px; padding-bottom: 24px; border-bottom: 1px solid #111;
    }
    .file-badge {
      font-size: 12px; color: #00d4ff;
      background: #00d4ff11; border: 1px solid #00d4ff33;
      padding: 6px 16px;
    }
    .overview-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-left: auto; }
    .ov-chip {
      text-align: center; padding: 10px 20px;
      background: #0d0d1a; border: 1px solid #1a1a2e;
    }
    .ov-chip span { display: block; font-size: 22px; font-weight: bold; color: #00ff88; }
    .ov-chip small { font-size: 10px; color: #444; letter-spacing: 1px; }

    .stats-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 2px; margin-bottom: 48px;
    }
    .stat-card {
      background: #0d0d1a; border: 1px solid #1a1a2e;
      border-top: 2px solid var(--accent, #00d4ff);
      padding: 20px;
    }
    .stat-card-header {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 16px;
    }
    .stat-col-name { font-size: 13px; font-weight: bold; color: #ddd; }
    .stat-badge { font-size: 9px; letter-spacing: 1px; text-transform: uppercase; }
    .stat-rows { display: flex; flex-direction: column; gap: 6px; }
    .sr { display: flex; justify-content: space-between; font-size: 12px; }
    .sr span { color: #444; }
    .sr strong { color: #bbb; }
    .outlier-row strong { }

    .chart-section { margin-bottom: 48px; }
    .histograms-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 2px;
    }
    .hist-wrap { background: #0d0d1a; border: 1px solid #1a1a2e; padding: 12px; }
    .hist-wrap canvas { width: 100%; }

    .heatmap-wrap {
      background: #0d0d1a; border: 1px solid #1a1a2e;
      padding: 20px; display: inline-block; max-width: 100%; overflow: auto;
    }
    #corr-canvas { max-width: 100%; }

    .cat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 2px; }
    .cat-card { background: #0d0d1a; border: 1px solid #1a1a2e; padding: 20px; }
    .cat-title { font-size: 13px; font-weight: bold; color: #ddd; margin-bottom: 16px; }
    .cat-title small { color: #444; font-size: 11px; }
    .cat-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
    .cat-label { width: 80px; font-size: 11px; color: #666; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .cat-bar-wrap { flex: 1; height: 6px; background: #111; }
    .cat-bar { height: 100%; border-radius: 1px; transition: width .6s; }
    .cat-count { font-size: 11px; color: #444; width: 35px; text-align: right; }

    /* Docs tab */
    .docs-content { max-width: 700px; }
    .docs-block { margin-bottom: 48px; }
    .docs-block h3 { font-size: 14px; letter-spacing: 3px; color: #00d4ff; margin-bottom: 20px; text-transform: uppercase; }
    .docs-step {
      display: flex; gap: 16px; align-items: flex-start;
      padding: 16px; background: #0d0d1a; border: 1px solid #1a1a2e;
      margin-bottom: 2px;
    }
    .step-num {
      width: 28px; height: 28px; min-width: 28px;
      background: #00d4ff22; color: #00d4ff; font-size: 12px; font-weight: bold;
      display: flex; align-items: center; justify-content: center;
    }
    .step-body { flex: 1; }
    .step-title { font-size: 13px; color: #ccc; margin-bottom: 6px; }
    .step-desc { font-size: 12px; color: #555; line-height: 1.6; }
    code {
      background: #111120; color: #00ff88;
      padding: 2px 8px; font-family: monospace; font-size: 12px;
    }
    pre {
      background: #0a0a0f; border: 1px solid #1a1a2e;
      padding: 20px; overflow-x: auto; margin: 16px 0;
    }
    pre code { background: none; padding: 0; color: #e0e0e0; line-height: 1.7; }
    .file-tree {
      background: #0a0a0f; border: 1px solid #1a1a2e;
      padding: 20px; font-size: 13px; line-height: 2; color: #666;
    }
    .file-tree .highlight { color: #00d4ff; }
    .file-tree .comment { color: #333; font-size: 11px; }
  `;
  document.head.appendChild(style);

  // Build page structure
  document.body.innerHTML = `
    <header class="site-header">
      <div class="header-tag">AI / ML Engineer</div>
      <h1 class="header-name">Rakibul Hasan by artificial intelligence/machine learning<br><em>AI Lab</em></h1>
      <p class="header-sub">Neural Networks &amp; Data Analysis — Built from Scratch</p>
      <div class="header-badges">
        <span class="badge active">JavaScript</span>
        <span class="badge">Machine Learning</span>
        <span class="badge">Neural Networks</span>
        <span class="badge">Statistics</span>
        <span class="badge">No Frameworks</span>
      </div>
    </header>

    <nav class="site-nav">
      <button class="nav-tab active" data-tab="nn">01 — Neural Network</button>
      <button class="nav-tab" data-tab="analyzer">02 — Data Analyzer</button>
      <button class="nav-tab" data-tab="docs">Docs &amp; Setup</button>
    </nav>

    <main class="main-content">
      <div class="tab-panel active" id="tab-nn"></div>
      <div class="tab-panel" id="tab-analyzer"></div>
      <div class="tab-panel" id="tab-docs">
        <div class="docs-content">
          <div class="section-label">DOCUMENTATION</div>
          <h2 class="section-title">How to <em>Upload</em></h2>

          <div class="docs-block">
            <h3>Step-by-Step — GitHub Pages</h3>
            <div class="docs-step"><div class="step-num">1</div><div class="step-body"><div class="step-title">Create a new repository on GitHub</div><div class="step-desc">Go to github.com → click "New repository" → name it <code>ai-ml-lab</code> → set to Public → click Create.</div></div></div>
            <div class="docs-step"><div class="step-num">2</div><div class="step-body"><div class="step-title">Upload the 3 files</div><div class="step-desc">Click "Add file" → "Upload files" → drag and drop all 3 files below.</div></div></div>
            <div class="docs-step"><div class="step-num">3</div><div class="step-body"><div class="step-title">Enable GitHub Pages</div><div class="step-desc">Go to Settings → Pages → Source: select "main" branch → click Save.</div></div></div>
            <div class="docs-step"><div class="step-num">4</div><div class="step-body"><div class="step-title">Your live link will be:</div><div class="step-desc"><code>https://YOUR-USERNAME.github.io/ai-ml-lab/</code><br><br>Share this link — it works in any browser, anywhere.</div></div></div>
          </div>

          <div class="docs-block">
            <h3>Files to Upload</h3>
            <div class="file-tree">
              <span class="highlight">index.html</span>   <span class="comment">← 8 lines only (loads JS)</span><br>
              <span class="highlight">neural_network.js</span>   <span class="comment">← Project 1: Neural net logic + UI</span><br>
              <span class="highlight">analyzer.js</span>   <span class="comment">← Project 2: Data analysis + UI</span>
            </div>
          </div>

          <div class="docs-block">
            <h3>GitHub Language Detection</h3>
            <pre><code>index.html        →   8 lines   (2%)
neural_network.js →  ~350 lines  (49%)
analyzer.js       →  ~400 lines  (49%)

GitHub will show:  🟡 JavaScript 98%  |  HTML 2%</code></pre>
          </div>
        </div>
      </div>
    </main>
  `;

  // Tab switching
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.onclick = () => {
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    };
  });

  // Mount projects
  new window.__NNRenderer(document.getElementById('tab-nn'));
  new AnalyzerRenderer(document.getElementById('tab-analyzer'));
})();
