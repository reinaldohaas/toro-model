/* ============================================================
   Toró Model — Visualization Engine
   ============================================================ */

// Global state
let DATA = null;

// Colors
const C = {
    blue: '#3b82f6',
    cyan: '#22d3ee',
    emerald: '#34d399',
    amber: '#fbbf24',
    rose: '#f43f5e',
    violet: '#a78bfa',
    white: '#f1f5f9',
    muted: '#64748b',
    grid: 'rgba(148, 163, 184, 0.08)',
    gridLine: 'rgba(148, 163, 184, 0.15)',
};

// ============================================================
// NAVIGATION
// ============================================================
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('section-' + btn.dataset.section).classList.add('active');
    });
});

// ============================================================
// LOAD DATA
// ============================================================
async function loadData() {
    try {
        const resp = await fetch('../output/results.json');
        DATA = await resp.json();
        populateValues();
        drawAllCharts();
    } catch (e) {
        console.warn('No results.json found, using demo data:', e);
        DATA = generateDemoData();
        populateValues();
        drawAllCharts();
    }
}

function generateDemoData() {
    // Fallback data based on the simulation output
    const nz = 50;
    const dz = 300;
    const z = Array.from({length: nz}, (_, i) => i * dz);
    
    // Simulated w profile (gaussian-ish)
    const w = z.map(zi => 23.3 * Math.exp(-0.5 * Math.pow((zi - 3000) / 2000, 2)) * (1 - Math.exp(-zi / 500)));
    
    // Temperature profile
    const T = z.map(zi => 300 - 6.5e-3 * zi);
    
    // LWC/IWC profiles
    const lwc = z.map(zi => {
        if (zi < 1000 || zi > 10000) return 0;
        return 0.5 * Math.exp(-0.5 * Math.pow((zi - 3000) / 2000, 2));
    });
    const iwc = z.map(zi => {
        if (zi < 4000 || zi > 12000) return 0;
        return 0.3 * Math.exp(-0.5 * Math.pow((zi - 6000) / 2000, 2));
    });
    
    // Reflectivity
    const Z_dbz = z.map((zi, i) => {
        const total = lwc[i] + iwc[i];
        if (total < 0.001) return -30;
        return 10 * Math.log10(total * 1e6) + 20;
    });
    
    // Time history
    const nt = 61;
    const t_hist = Array.from({length: nt}, (_, i) => i * 10);
    const w_hist = t_hist.map(t => {
        if (t < 10) return 2.9;
        return Math.min(23.3, 2.9 + 20.4 * (1 - Math.exp(-t / 150)));
    });
    
    // Fall trajectory
    const nf = 50;
    const t_fall = Array.from({length: nf}, (_, i) => i * 0.5);
    const z_fall = [], v_fall = [];
    let zf = 4000, vf = 0;
    for (let i = 0; i < nf; i++) {
        z_fall.push(zf);
        v_fall.push(vf);
        const dt = 0.5;
        const drag = 0.5 * 0.9 * 1.0 * 125664 * vf * vf;
        const dvdt = 9.81 - drag / 3e6;
        vf += dvdt * dt;
        if (vf > 22.8) vf = 22.8;
        zf -= vf * dt;
        if (zf < 0) { zf = 0; break; }
    }
    
    return {
        config: {
            microphysics: { mu: 20, D_mean: 20e-6, n_bins: 25 },
            dynamics: { V_max_tornado: 70, R_max_tornado: 150 },
        },
        phase1: {
            z: z, w: w, T: T,
            lwc: lwc, iwc: iwc,
            Z_dbz: Z_dbz,
            bwer: { detected: true, z_bottom: 3000, z_top: 7000 },
            w_max: 23.3,
            history: { time: t_hist, w_max: w_hist },
            tornado: { swirl_ratio: 1.2, pressure_deficit: 5400 },
        },
        phase2: {
            M_piston_kg: 3e6,
            M_piston_ton: 3000,
            rho_mix: 500,
            H_piston: 1500,
            A_cross: 125664,
            v_terminal: 22.8,
            v_impact: 22.8,
            P_impact: 5.7e6,
            E_impact: 7.81e8,
            z_start: 4000,
            t_fall: t_fall.slice(0, z_fall.length),
            z_fall: z_fall,
            v_fall: v_fall,
        },
        phase3: {
            sound: {
                SPL_1km: 189.1,
                components: {
                    infrasound: { freq_Hz: 5, amplitude: 0.6 },
                    boom: { freq_Hz: 15, amplitude: 1.0 },
                    rumble: { freq_Hz: 80, amplitude: 0.4 },
                    crackle: { freq_Hz: 1000, amplitude: 0.15 },
                },
            },
            seismic: {
                M_L: 3.89,
                E_seismic: 3.9e7,
                f_dominant: 3,
            },
            erosion: {
                M_eroded_kg: 10000,
                M_eroded_ton: 10,
                mud_percent: 0,
                composition: { rock_pct: 60, soil_mineral_pct: 25, trees_pct: 15, mud_pct: 0 },
                D_max_mobilized: 2386,
                channel: { width: 400, depth: 5, length: 800 },
                wash: {
                    clay: { ratio: 999, mobilized: true },
                    silt: { ratio: 500, mobilized: true },
                    sand: { ratio: 50, mobilized: true },
                    gravel: { ratio: 2, mobilized: true },
                },
            },
        },
    };
}

// ============================================================
// POPULATE UI VALUES
// ============================================================
function populateValues() {
    if (!DATA) return;
    const p2 = DATA.phase2;
    const p3 = DATA.phase3;
    
    // Stage cards
    setText('val-mpiston', fmtTon(p2.M_piston_ton || p2.M_piston_kg / 1000));
    setText('val-rhomix', fmtNum(p2.rho_mix) + ' kg/m³');
    setText('val-hpiston', fmtNum(p2.H_piston) + ' m');
    setText('val-vimpact', fmtNum(p2.v_impact, 1) + ' m/s');
    setText('val-pimpact', fmtNum(p2.P_impact / 1e6, 1) + ' MPa');
    setText('val-eimpact', fmtSci(p2.E_impact) + ' J');
    
    setText('val-ml', fmtNum(p3.seismic.M_L, 2));
    setText('val-eroded', fmtNum((p3.erosion.M_eroded_ton || p3.erosion.M_eroded_kg / 1000), 0) + ' ton');
    setText('val-mud', p3.erosion.mud_percent + '%');
    
    // Impact section
    setText('val-spl', fmtNum(p3.sound.SPL_1km, 1) + ' dB');
    setText('val-ml2', fmtNum(p3.seismic.M_L, 2));
    setText('val-eseis', fmtSci(p3.seismic.E_seismic) + ' J');
    setText('val-eroded2', fmtNum((p3.erosion.M_eroded_ton || p3.erosion.M_eroded_kg / 1000), 0) + ' ton');
    
    // Composition bars
    const comp = p3.erosion.composition;
    const compEl = document.getElementById('composition-bars');
    compEl.innerHTML = '';
    const items = [
        { label: 'Rocha', pct: comp.rock_pct, color: C.muted },
        { label: 'Solo', pct: comp.soil_mineral_pct, color: C.amber },
        { label: 'Madeira', pct: comp.trees_pct, color: C.emerald },
        { label: 'Barro', pct: comp.mud_pct, color: C.rose },
    ];
    items.forEach(it => {
        const bar = document.createElement('div');
        bar.className = 'comp-bar';
        bar.innerHTML = `
            <span class="bar-label">${it.label}</span>
            <span class="bar-track"><span class="bar-fill" style="width:${it.pct}%;background:${it.color}"></span></span>
            <span class="bar-value">${it.pct}%</span>`;
        compEl.appendChild(bar);
    });
    
    // Wash table
    const wash = p3.erosion.wash || {};
    setWashRow('wash-clay', 'Argila', wash.clay);
    setWashRow('wash-silt', 'Silte', wash.silt);
    setWashRow('wash-sand', 'Areia', wash.sand);
    setWashRow('wash-gravel', 'Cascalho', wash.gravel);
}

function setWashRow(id, name, data) {
    const el = document.getElementById(id);
    if (!el || !data) return;
    const ratio = data.ratio > 100 ? '>100' : fmtNum(data.ratio, 1);
    const mob = data.mobilized ? '✅ SIM' : '❌ NÃO';
    el.innerHTML = `<span>${name}</span><span>${ratio}</span><span>${mob}</span>`;
    el.className = 'wash-row ' + (data.mobilized ? 'mobilized' : '');
}

// ============================================================
// CHART DRAWING UTILITIES
// ============================================================
function getCtx(id) {
    const canvas = document.getElementById(id);
    if (!canvas) return null;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr || canvas.width;
    canvas.height = rect.height * dpr || canvas.height;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return { ctx, w: rect.width || canvas.width / dpr, h: rect.height || canvas.height / dpr };
}

function drawGrid(ctx, w, h, pad, xTicks, yTicks) {
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 0.5;
    for (let x of xTicks) {
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, h - pad.bottom);
        ctx.stroke();
    }
    for (let y of yTicks) {
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(w - pad.right, y);
        ctx.stroke();
    }
}

function drawLine(ctx, points, color, width = 2) {
    if (points.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.stroke();
}

function drawFilledLine(ctx, points, color, alpha = 0.15, baseY = 0) {
    if (points.length < 2) return;
    ctx.fillStyle = color.replace(')', `,${alpha})`).replace('rgb', 'rgba');
    if (color.startsWith('#')) {
        const r = parseInt(color.slice(1,3),16);
        const g = parseInt(color.slice(3,5),16);
        const b = parseInt(color.slice(5,7),16);
        ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
    }
    ctx.beginPath();
    ctx.moveTo(points[0][0], baseY);
    for (const p of points) ctx.lineTo(p[0], p[1]);
    ctx.lineTo(points[points.length - 1][0], baseY);
    ctx.closePath();
    ctx.fill();
}

function drawLabel(ctx, text, x, y, color = C.muted, size = 10, align = 'center') {
    ctx.fillStyle = color;
    ctx.font = `${size}px Inter, sans-serif`;
    ctx.textAlign = align;
    ctx.fillText(text, x, y);
}

function mapX(val, vmin, vmax, xmin, xmax) {
    return xmin + (val - vmin) / (vmax - vmin) * (xmax - xmin);
}

function mapY(val, vmin, vmax, ymin, ymax) {
    return ymax - (val - vmin) / (vmax - vmin) * (ymax - ymin);
}

// ============================================================
// DRAW ALL CHARTS
// ============================================================
function drawAllCharts() {
    if (!DATA) return;
    drawProfileChart();
    drawTimeseriesChart();
    drawRadarChart();
    drawTornadoChart();
    drawFallChart();
    drawDSDChart();
    drawColumnDiagram();
    drawSoundWaveform();
    drawSeismicWaveform();
    drawErosionChannel();
}

// ============================================================
// 1. VERTICAL PROFILE
// ============================================================
function drawProfileChart() {
    const {ctx, w, h} = getCtx('canvas-profile');
    const pad = {top: 30, right: 20, bottom: 40, left: 50};
    const pw = w - pad.left - pad.right;
    const ph = h - pad.top - pad.bottom;
    
    const z = DATA.phase1.z;
    const wArr = DATA.phase1.w;
    const zMax = Math.max(...z);
    const wMax = Math.max(...wArr) * 1.1;
    
    // Y axis: altitude
    const yTicks = [0, 3000, 6000, 9000, 12000, 15000];
    yTicks.forEach(zt => {
        const y = mapY(zt, 0, zMax, pad.top, h - pad.bottom);
        drawLabel(ctx, (zt / 1000) + 'km', pad.left - 5, y + 3, C.muted, 9, 'right');
    });
    
    // w(z)
    const pts = z.map((zi, i) => [
        mapX(wArr[i], 0, wMax, pad.left, w - pad.right),
        mapY(zi, 0, zMax, pad.top, h - pad.bottom)
    ]);
    drawFilledLine(ctx, pts, C.blue, 0.2, mapY(0, 0, zMax, pad.top, h - pad.bottom));
    drawLine(ctx, pts, C.blue, 2);
    
    // LWC
    const lwc = DATA.phase1.lwc;
    const lwcMax = Math.max(...lwc) * 1.1 || 1;
    const ptsLwc = z.map((zi, i) => [
        mapX(lwc[i], 0, lwcMax, pad.left, w - pad.right),
        mapY(zi, 0, zMax, pad.top, h - pad.bottom)
    ]);
    drawLine(ctx, ptsLwc, C.cyan, 1.5);
    
    // IWC
    const iwc = DATA.phase1.iwc;
    const iwcMax = Math.max(...iwc) * 1.1 || 1;
    const ptsIwc = z.map((zi, i) => [
        mapX(iwc[i], 0, iwcMax, pad.left, w - pad.right),
        mapY(zi, 0, zMax, pad.top, h - pad.bottom)
    ]);
    drawLine(ctx, ptsIwc, C.violet, 1.5);
    
    // Freezing level
    const zFreeze = 4000;
    const yFreeze = mapY(zFreeze, 0, zMax, pad.top, h - pad.bottom);
    ctx.setLineDash([5, 3]);
    ctx.strokeStyle = C.rose;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, yFreeze);
    ctx.lineTo(w - pad.right, yFreeze);
    ctx.stroke();
    ctx.setLineDash([]);
    drawLabel(ctx, '0°C', w - pad.right - 5, yFreeze - 5, C.rose, 9, 'right');
    
    // HM Zone
    const yHM1 = mapY(4250, 0, zMax, pad.top, h - pad.bottom);
    const yHM2 = mapY(4750, 0, zMax, pad.top, h - pad.bottom);
    ctx.fillStyle = 'rgba(34, 211, 238, 0.08)';
    ctx.fillRect(pad.left, yHM2, pw, yHM1 - yHM2);
    drawLabel(ctx, 'H-M Zone', pad.left + 5, (yHM1 + yHM2) / 2 + 3, C.cyan, 8, 'left');
    
    // Legend
    drawLabel(ctx, 'w (m/s)', w / 2, h - 8, C.blue, 10);
    
    // Legend items
    const ly = pad.top + 10;
    ctx.fillStyle = C.blue; ctx.fillRect(pad.left + 5, ly, 12, 3);
    drawLabel(ctx, 'w', pad.left + 22, ly + 4, C.blue, 9, 'left');
    ctx.fillStyle = C.cyan; ctx.fillRect(pad.left + 45, ly, 12, 3);
    drawLabel(ctx, 'LWC', pad.left + 62, ly + 4, C.cyan, 9, 'left');
    ctx.fillStyle = C.violet; ctx.fillRect(pad.left + 95, ly, 12, 3);
    drawLabel(ctx, 'IWC', pad.left + 112, ly + 4, C.violet, 9, 'left');
}

// ============================================================
// 2. TIME SERIES
// ============================================================
function drawTimeseriesChart() {
    const {ctx, w, h} = getCtx('canvas-timeseries');
    const pad = {top: 25, right: 20, bottom: 40, left: 50};
    
    const hist = DATA.phase1.history;
    if (!hist) return;
    const t = hist.time;
    const wm = hist.w_max;
    const tMax = Math.max(...t);
    const wmMax = Math.max(...wm) * 1.1;
    
    // Grid
    const xTicks = [0, 100, 200, 300, 400, 500, 600].filter(v => v <= tMax).map(v =>
        mapX(v, 0, tMax, pad.left, w - pad.right)
    );
    drawGrid(ctx, w, h, pad, xTicks, []);
    
    // X labels
    [0, 100, 200, 300, 400, 500, 600].filter(v => v <= tMax).forEach(v => {
        drawLabel(ctx, v + 's', mapX(v, 0, tMax, pad.left, w - pad.right), h - pad.bottom + 15, C.muted, 9);
    });
    
    // w_max(t)
    const pts = t.map((ti, i) => [
        mapX(ti, 0, tMax, pad.left, w - pad.right),
        mapY(wm[i], 0, wmMax, pad.top, h - pad.bottom)
    ]);
    drawFilledLine(ctx, pts, C.emerald, 0.15, h - pad.bottom);
    drawLine(ctx, pts, C.emerald, 2);
    
    // Y labels
    [0, 5, 10, 15, 20, 25].filter(v => v <= wmMax).forEach(v => {
        drawLabel(ctx, v + '', pad.left - 5, mapY(v, 0, wmMax, pad.top, h - pad.bottom) + 3, C.muted, 9, 'right');
    });
    
    drawLabel(ctx, 'w_max (m/s)', pad.left + 5, pad.top - 8, C.emerald, 10, 'left');
    drawLabel(ctx, 'tempo (s)', w / 2, h - 5, C.muted, 10);
}

// ============================================================
// 3. RADAR / BWER
// ============================================================
function drawRadarChart() {
    const {ctx, w, h} = getCtx('canvas-radar');
    const pad = {top: 25, right: 30, bottom: 40, left: 50};
    
    const z = DATA.phase1.z;
    const Z = DATA.phase1.Z_dbz;
    const zMax = Math.max(...z);
    const Zmin = -30, Zmax = 60;
    
    // Z(z) profile
    const pts = z.map((zi, i) => [
        mapX(Z[i], Zmin, Zmax, pad.left, w - pad.right),
        mapY(zi, 0, zMax, pad.top, h - pad.bottom)
    ]);
    drawLine(ctx, pts, C.amber, 2);
    
    // Color scale
    const colors = ['#1e3a5f', '#2563eb', '#22c55e', '#eab308', '#ef4444', '#7c3aed'];
    const vals = [-30, -10, 10, 30, 50, 60];
    vals.forEach((v, i) => {
        const x = mapX(v, Zmin, Zmax, pad.left, w - pad.right);
        drawLabel(ctx, v + '', x, h - pad.bottom + 15, C.muted, 8);
    });
    
    // BWER highlight
    const bwer = DATA.phase1.bwer;
    if (bwer && bwer.detected) {
        const yb = mapY(bwer.z_bottom, 0, zMax, pad.top, h - pad.bottom);
        const yt = mapY(bwer.z_top, 0, zMax, pad.top, h - pad.bottom);
        ctx.fillStyle = 'rgba(244, 63, 94, 0.1)';
        ctx.fillRect(pad.left, yt, w - pad.left - pad.right, yb - yt);
        ctx.strokeStyle = C.rose;
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1;
        ctx.strokeRect(pad.left, yt, w - pad.left - pad.right, yb - yt);
        ctx.setLineDash([]);
        drawLabel(ctx, 'BWER', w - pad.right - 25, (yt + yb) / 2, C.rose, 11);
    }
    
    // Y axis
    [0, 3000, 6000, 9000, 12000, 15000].forEach(zt => {
        const y = mapY(zt, 0, zMax, pad.top, h - pad.bottom);
        drawLabel(ctx, (zt / 1000) + 'km', pad.left - 5, y + 3, C.muted, 9, 'right');
    });
    
    drawLabel(ctx, 'Z (dBZ)', w / 2, h - 5, C.amber, 10);
}

// ============================================================
// 4. TORNADO WIND
// ============================================================
function drawTornadoChart() {
    const {ctx, w, h} = getCtx('canvas-tornado');
    const pad = {top: 25, right: 20, bottom: 40, left: 50};
    
    const Vmax = DATA.config?.dynamics?.V_max_tornado || 70;
    const Rmax = DATA.config?.dynamics?.R_max_tornado || 150;
    
    const rArr = [];
    for (let r = 0; r <= 1000; r += 5) rArr.push(r);
    
    const Vt = rArr.map(r => {
        if (r <= Rmax) return Vmax * r / Rmax;
        return Vmax * Rmax / r;
    });
    
    const rMax = 1000;
    const vMax = Vmax * 1.15;
    
    // Draw
    const pts = rArr.map((r, i) => [
        mapX(r, 0, rMax, pad.left, w - pad.right),
        mapY(Vt[i], 0, vMax, pad.top, h - pad.bottom)
    ]);
    drawFilledLine(ctx, pts, C.rose, 0.15, h - pad.bottom);
    drawLine(ctx, pts, C.rose, 2);
    
    // Rmax line
    const xRmax = mapX(Rmax, 0, rMax, pad.left, w - pad.right);
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = C.muted;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xRmax, pad.top);
    ctx.lineTo(xRmax, h - pad.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    drawLabel(ctx, 'R_max', xRmax, pad.top + 15, C.muted, 9);
    
    // Labels
    drawLabel(ctx, 'V_t (m/s)', pad.left + 5, pad.top - 8, C.rose, 10, 'left');
    drawLabel(ctx, 'r (m)', w / 2, h - 5, C.muted, 10);
    
    [0, 200, 400, 600, 800, 1000].forEach(v => {
        drawLabel(ctx, v + '', mapX(v, 0, rMax, pad.left, w - pad.right), h - pad.bottom + 15, C.muted, 9);
    });
    [0, 20, 40, 60, 80].filter(v => v <= vMax).forEach(v => {
        drawLabel(ctx, v + '', pad.left - 5, mapY(v, 0, vMax, pad.top, h - pad.bottom) + 3, C.muted, 9, 'right');
    });
    
    // Annotation
    drawLabel(ctx, `EF2-EF3 (${Vmax} m/s)`, w - pad.right - 10, pad.top + 10, C.rose, 9, 'right');
}

// ============================================================
// 5. PISTON FALL
// ============================================================
function drawFallChart() {
    const {ctx, w, h} = getCtx('canvas-fall');
    const pad = {top: 25, right: 50, bottom: 40, left: 50};
    
    const p2 = DATA.phase2;
    const tf = p2.t_fall || [];
    const zf = p2.z_fall || [];
    const vf = p2.v_fall || [];
    
    if (tf.length < 2) {
        drawLabel(ctx, 'Sem dados de queda', w / 2, h / 2, C.muted, 14);
        return;
    }
    
    const tMax = Math.max(...tf) * 1.1;
    const zMax = Math.max(...zf) * 1.1;
    const vMax = Math.max(...vf) * 1.1;
    
    // z(t)
    const ptsZ = tf.map((t, i) => [
        mapX(t, 0, tMax, pad.left, w - pad.right),
        mapY(zf[i], 0, zMax, pad.top, h - pad.bottom)
    ]);
    drawLine(ctx, ptsZ, C.blue, 2);
    
    // v(t) — secondary axis
    const ptsV = tf.map((t, i) => [
        mapX(t, 0, tMax, pad.left, w - pad.right),
        mapY(vf[i], 0, vMax, pad.top, h - pad.bottom)
    ]);
    drawLine(ctx, ptsV, C.rose, 2);
    
    // Labels
    drawLabel(ctx, 'z (m)', pad.left + 5, pad.top - 8, C.blue, 10, 'left');
    drawLabel(ctx, 'v (m/s)', w - pad.right - 5, pad.top - 8, C.rose, 10, 'right');
    drawLabel(ctx, 't (s)', w / 2, h - 5, C.muted, 10);
    
    // Impact annotation
    const vimp = p2.v_impact;
    drawLabel(ctx, `v_impact = ${fmtNum(vimp, 1)} m/s`, w / 2, h - pad.bottom - 20, C.amber, 11);
}

// ============================================================
// 6. DSD
// ============================================================
function drawDSDChart() {
    const {ctx, w, h} = getCtx('canvas-dsd');
    const pad = {top: 25, right: 20, bottom: 40, left: 50};
    
    const mu = DATA.config?.microphysics?.mu || 20;
    const Dmean = (DATA.config?.microphysics?.D_mean || 20e-6) * 1e6; // µm
    const lambda = (mu + 1) / Dmean;
    
    // Generate DSD
    const npts = 200;
    const Dmax = Dmean * 5;
    const D = Array.from({length: npts}, (_, i) => (i + 1) / npts * Dmax);
    const N = D.map(d => Math.pow(d, mu) * Math.exp(-lambda * d));
    const Nmax = Math.max(...N);
    const Nnorm = N.map(n => n / Nmax);
    
    const pts = D.map((d, i) => [
        mapX(d, 0, Dmax, pad.left, w - pad.right),
        mapY(Nnorm[i], 0, 1.05, pad.top, h - pad.bottom)
    ]);
    
    drawFilledLine(ctx, pts, C.violet, 0.2, h - pad.bottom);
    drawLine(ctx, pts, C.violet, 2);
    
    // D_mean marker
    const xDm = mapX(Dmean, 0, Dmax, pad.left, w - pad.right);
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = C.amber;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xDm, pad.top);
    ctx.lineTo(xDm, h - pad.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    drawLabel(ctx, `D̄=${Dmean.toFixed(0)}µm`, xDm + 5, pad.top + 15, C.amber, 9, 'left');
    
    // Labels
    drawLabel(ctx, `N(D) normalizado — μ=${mu}`, pad.left + 5, pad.top - 8, C.violet, 10, 'left');
    drawLabel(ctx, 'D (µm)', w / 2, h - 5, C.muted, 10);
    
    // Annotation: narrow
    drawLabel(ctx, 'DSD ESTREITA', w - pad.right - 10, pad.top + 30, C.cyan, 11, 'right');
    drawLabel(ctx, 'Difração → Iridescência', w - pad.right - 10, pad.top + 45, C.muted, 8, 'right');
}

// ============================================================
// COLUMN DIAGRAM (Conceptual)
// ============================================================
function drawColumnDiagram() {
    const {ctx, w, h} = getCtx('canvas-column');
    const pad = {top: 20, right: 30, bottom: 30, left: 60};
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bottom;
    
    // Altitude axis
    const zLevels = [
        {z: 0, label: 'Solo (200m)', color: '#6b7280'},
        {z: 1000, label: 'LCL', color: C.muted},
        {z: 4000, label: '0°C (Congelamento)', color: C.cyan},
        {z: 4250, label: '-3°C (H-M base)', color: C.emerald},
        {z: 4750, label: '-8°C (H-M topo)', color: C.emerald},
        {z: 8000, label: '-15°C (Phillips)', color: C.violet},
        {z: 10000, label: '-38°C (Nucleação)', color: C.blue},
        {z: 12000, label: 'Tropopausa', color: C.rose},
    ];
    const zTop = 15000;
    
    // Background gradient (atmosphere)
    const grad = ctx.createLinearGradient(0, pad.top, 0, h - pad.bottom);
    grad.addColorStop(0, 'rgba(10, 14, 40, 0.9)');
    grad.addColorStop(0.3, 'rgba(15, 30, 80, 0.8)');
    grad.addColorStop(0.7, 'rgba(30, 60, 100, 0.5)');
    grad.addColorStop(1, 'rgba(80, 120, 80, 0.3)');
    ctx.fillStyle = grad;
    ctx.fillRect(pad.left, pad.top, cw, ch);
    
    // Z levels
    zLevels.forEach(lv => {
        const y = mapY(lv.z, 0, zTop, pad.top, h - pad.bottom);
        ctx.strokeStyle = lv.color;
        ctx.lineWidth = 0.8;
        ctx.setLineDash([3, 5]);
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(w - pad.right, y);
        ctx.stroke();
        ctx.setLineDash([]);
        drawLabel(ctx, lv.label, pad.left - 5, y + 3, lv.color, 9, 'right');
    });
    
    // Updraft column
    const colX = w / 2 - 40;
    const colW = 80;
    const yLCL = mapY(1000, 0, zTop, pad.top, h - pad.bottom);
    const yTop = mapY(11000, 0, zTop, pad.top, h - pad.bottom);
    
    const updraftGrad = ctx.createLinearGradient(0, yTop, 0, yLCL);
    updraftGrad.addColorStop(0, 'rgba(59, 130, 246, 0.05)');
    updraftGrad.addColorStop(0.5, 'rgba(59, 130, 246, 0.15)');
    updraftGrad.addColorStop(1, 'rgba(59, 130, 246, 0.08)');
    ctx.fillStyle = updraftGrad;
    ctx.fillRect(colX, yTop, colW, yLCL - yTop);
    
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)';
    ctx.lineWidth = 1;
    ctx.strokeRect(colX, yTop, colW, yLCL - yTop);
    
    // Upward arrows
    const arrowX = colX + colW / 2;
    for (let zi = 1500; zi < 10000; zi += 800) {
        const y = mapY(zi, 0, zTop, pad.top, h - pad.bottom);
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(arrowX, y + 10);
        ctx.lineTo(arrowX, y - 10);
        ctx.lineTo(arrowX - 4, y - 5);
        ctx.moveTo(arrowX, y - 10);
        ctx.lineTo(arrowX + 4, y - 5);
        ctx.stroke();
    }
    
    // HM Zone highlight
    const yHM1 = mapY(4250, 0, zTop, pad.top, h - pad.bottom);
    const yHM2 = mapY(4750, 0, zTop, pad.top, h - pad.bottom);
    ctx.fillStyle = 'rgba(34, 211, 238, 0.15)';
    ctx.fillRect(colX - 15, yHM2, colW + 30, yHM1 - yHM2);
    drawLabel(ctx, '❄️ SIP H-M', colX + colW + 25, (yHM1 + yHM2) / 2, C.cyan, 10, 'left');
    
    // Phillips zone
    const yPh = mapY(8000, 0, zTop, pad.top, h - pad.bottom);
    drawLabel(ctx, '💎 Phillips Breakup', colX + colW + 25, yPh, C.violet, 10, 'left');
    
    // Cloud droplets zone
    const yCloud = mapY(2500, 0, zTop, pad.top, h - pad.bottom);
    drawLabel(ctx, '💧 Gotas Supercongeladas', colX + colW + 25, yCloud, C.blue, 10, 'left');
    
    // Tornado at base
    const yBase = mapY(0, 0, zTop, pad.top, h - pad.bottom);
    drawLabel(ctx, '🌪️ Vórtice Tornado', colX + colW + 25, yBase - 10, C.rose, 10, 'left');
    
    // Piston annotation
    const yPiston = mapY(5500, 0, zTop, pad.top, h - pad.bottom);
    ctx.fillStyle = 'rgba(251, 191, 36, 0.08)';
    const yPistTop = mapY(7000, 0, zTop, pad.top, h - pad.bottom);
    const yPistBot = mapY(4000, 0, zTop, pad.top, h - pad.bottom);
    ctx.fillRect(colX - 5, yPistTop, colW + 10, yPistBot - yPistTop);
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(colX - 5, yPistTop, colW + 10, yPistBot - yPistTop);
    drawLabel(ctx, '🧊 PISTÃO', colX - 60, yPiston, C.amber, 11, 'left');
    
    // Title
    drawLabel(ctx, 'Coluna Vertical — Estrutura do Toró', w / 2, pad.top + 12, C.white, 12);
}

// ============================================================
// SOUND WAVEFORM
// ============================================================
function drawSoundWaveform() {
    const {ctx, w, h} = getCtx('canvas-sound');
    const pad = {top: 15, right: 15, bottom: 25, left: 40};
    
    const sr = 1000; // Simplified sampling
    const dur = 5;
    const n = sr * dur;
    const t = Array.from({length: n}, (_, i) => i / sr);
    
    const comp = DATA.phase3.sound.components;
    const signal = t.map(ti => {
        const infra = (comp.infrasound?.amplitude || 0.6) * Math.sin(2 * Math.PI * (comp.infrasound?.freq_Hz || 3) * ti) * Math.exp(-ti / 3);
        const boom = (comp.boom?.amplitude || 1.0) * Math.sin(2 * Math.PI * (comp.boom?.freq_Hz || 15) * ti) * Math.exp(-ti / 2);
        const rumble = (comp.rumble?.amplitude || 0.4) * Math.sin(2 * Math.PI * (comp.rumble?.freq_Hz || 80) * ti) * Math.exp(-ti / 1.5);
        return (infra + boom + rumble) * Math.min(1, ti / 0.05);
    });
    
    const maxAmp = Math.max(...signal.map(Math.abs)) || 1;
    
    // Draw waveform
    ctx.strokeStyle = C.emerald;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < n; i += 2) {
        const x = mapX(t[i], 0, dur, pad.left, w - pad.right);
        const y = mapY(signal[i] / maxAmp, -1.1, 1.1, pad.top, h - pad.bottom);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    
    // Zero line
    const y0 = mapY(0, -1.1, 1.1, pad.top, h - pad.bottom);
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, y0);
    ctx.lineTo(w - pad.right, y0);
    ctx.stroke();
    
    drawLabel(ctx, '0s', pad.left, h - 5, C.muted, 9, 'left');
    drawLabel(ctx, dur + 's', w - pad.right, h - 5, C.muted, 9, 'right');
    drawLabel(ctx, 'Som "Tó" — vagão desgovernado', w / 2, pad.top + 5, C.emerald, 10);
}

// ============================================================
// SEISMIC WAVEFORM
// ============================================================
function drawSeismicWaveform() {
    const {ctx, w, h} = getCtx('canvas-seismic');
    const pad = {top: 15, right: 15, bottom: 25, left: 40};
    
    const f0 = DATA.phase3.seismic.f_dominant || 3;
    const dur = 5;
    const sr = 500;
    const n = sr * dur;
    const t = Array.from({length: n}, (_, i) => i / sr);
    
    // Ricker wavelet
    const signal = t.map(ti => {
        const u = Math.PI * f0 * (ti - 0.5);
        return (1 - 2 * u * u) * Math.exp(-u * u) * Math.exp(-0.3 * ti);
    });
    
    const maxAmp = Math.max(...signal.map(Math.abs)) || 1;
    
    ctx.strokeStyle = C.amber;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
        const x = mapX(t[i], 0, dur, pad.left, w - pad.right);
        const y = mapY(signal[i] / maxAmp, -1.2, 1.2, pad.top, h - pad.bottom);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    
    const y0 = mapY(0, -1.2, 1.2, pad.top, h - pad.bottom);
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, y0);
    ctx.lineTo(w - pad.right, y0);
    ctx.stroke();
    
    drawLabel(ctx, '0s', pad.left, h - 5, C.muted, 9, 'left');
    drawLabel(ctx, dur + 's', w - pad.right, h - 5, C.muted, 9, 'right');
    drawLabel(ctx, `Sismograma — Ricker f₀=${f0} Hz`, w / 2, pad.top + 5, C.amber, 10);
}

// ============================================================
// EROSION CHANNEL CROSS-SECTION
// ============================================================
function drawErosionChannel() {
    const {ctx, w, h} = getCtx('canvas-erosion');
    const pad = {top: 20, right: 20, bottom: 30, left: 20};
    
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bottom;
    
    // Canyon walls
    ctx.fillStyle = '#4a3728';
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, h - pad.bottom);
    ctx.lineTo(pad.left + cw * 0.15, h - pad.bottom);
    ctx.lineTo(pad.left + cw * 0.2, pad.top + ch * 0.3);
    ctx.closePath();
    ctx.fill();
    
    ctx.beginPath();
    ctx.moveTo(w - pad.right, pad.top);
    ctx.lineTo(w - pad.right, h - pad.bottom);
    ctx.lineTo(w - pad.right - cw * 0.15, h - pad.bottom);
    ctx.lineTo(w - pad.right - cw * 0.2, pad.top + ch * 0.3);
    ctx.closePath();
    ctx.fill();
    
    // Floor (bare rock)
    ctx.fillStyle = '#6b7280';
    ctx.fillRect(pad.left + cw * 0.15, h - pad.bottom - ch * 0.08, cw * 0.7, ch * 0.08);
    
    // Erosion scar (linear)
    ctx.fillStyle = '#1e293b';
    const scarW = cw * 0.2;
    const scarX = w / 2 - scarW / 2;
    ctx.fillRect(scarX, h - pad.bottom - ch * 0.15, scarW, ch * 0.15);
    
    // Arrow showing impact
    ctx.strokeStyle = C.rose;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(w / 2, pad.top + 10);
    ctx.lineTo(w / 2, h - pad.bottom - ch * 0.2);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Arrow head
    ctx.fillStyle = C.rose;
    ctx.beginPath();
    ctx.moveTo(w / 2, h - pad.bottom - ch * 0.15);
    ctx.lineTo(w / 2 - 6, h - pad.bottom - ch * 0.25);
    ctx.lineTo(w / 2 + 6, h - pad.bottom - ch * 0.25);
    ctx.closePath();
    ctx.fill();
    
    // Labels
    drawLabel(ctx, 'IMPACTO', w / 2, pad.top + 8, C.rose, 9);
    drawLabel(ctx, 'Rocha nua', w / 2, h - pad.bottom + 12, C.muted, 8);
    drawLabel(ctx, 'ZERO barro', w / 2, h - pad.bottom - ch * 0.05, C.amber, 9);
}

// ============================================================
// HELPERS
// ============================================================
function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function fmtNum(v, d = 0) {
    if (v === undefined || v === null) return '—';
    return Number(v).toFixed(d);
}

function fmtTon(v) {
    if (v === undefined || v === null) return '—';
    return Number(v).toFixed(0) + ' ton';
}

function fmtSci(v) {
    if (v === undefined || v === null || v === 0) return '0';
    const exp = Math.floor(Math.log10(Math.abs(v)));
    const man = v / Math.pow(10, exp);
    return man.toFixed(2) + '×10' + superscript(exp);
}

function superscript(n) {
    const sup = {'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','-':'⁻'};
    return String(n).split('').map(c => sup[c] || c).join('');
}

// ============================================================
// INIT
// ============================================================
window.addEventListener('DOMContentLoaded', loadData);
window.addEventListener('resize', () => { if (DATA) drawAllCharts(); });
