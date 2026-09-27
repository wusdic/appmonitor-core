// Dependency-free inline-SVG charts (works offline, dark theme).
// Series colours follow a fixed categorical order (never cycled by rank);
// status colours (tiers / severities) are reserved and always ship with a label.
const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const el = (n, a = {}) => { const e = document.createElementNS(NS, n);
    for (const k in a) e.setAttribute(k, a[k]); return e; };
  const INK = "#e6edf7", MUTED = "#8ba0be", GRID = "#243149", SURF = "#131c2e";
  // categorical slots (dark steps of the validated reference palette)
  const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
  // sequential blue ramp (near-zero recedes to the surface)
  const SEQ = ["#1a2438", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#86b6ef", "#b7d3f6"];
  const seq = (v) => (v == null || !isFinite(v)) ? null : SEQ[Math.max(0, Math.min(SEQ.length - 1, Math.round(v * (SEQ.length - 1))))];

  // ------------------------------------------------------------ tooltip
  let tipEl = null;
  function tip() {
    if (!tipEl) { tipEl = document.createElement("div"); tipEl.className = "viz-tip"; document.body.appendChild(tipEl); }
    return tipEl;
  }
  function showTip(evt, html) {
    const t = tip(); t.innerHTML = html; t.style.display = "block";
    const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
    t.style.left = Math.max(4, x) + "px"; t.style.top = Math.max(4, y) + "px";
  }
  function hideTip() { if (tipEl) tipEl.style.display = "none"; }
  function hover(node, htmlFn) {
    node.addEventListener("mousemove", (e) => showTip(e, htmlFn(e)));
    node.addEventListener("mouseleave", hideTip);
    return node;
  }
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const txt = (x, y, s, a = {}) => { const t = el("text", Object.assign({ x, y, fill: MUTED, "font-size": 11 }, a)); t.textContent = s; return t; };
  const empty = (w, h, msg) => { const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", height: h });
    svg.appendChild(txt(8, h / 2, msg || "数据不足 / no data")); return svg; };

  // ------------------------------------------------------------ sparkline (legacy)
  function sparkline(points, opts = {}) {
    const w = opts.w || 260, h = opts.h || 60, pad = 4;
    const vals = points.map(p => (typeof p === "number" ? p : p.value)).filter(v => typeof v === "number" && isFinite(v));
    if (vals.length < 2) return empty(w, h);
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", height: h });
    const min = opts.min != null ? opts.min : Math.min(...vals), max = opts.max != null ? opts.max : Math.max(...vals);
    const span = (max - min) || 1;
    const dx = (w - pad * 2) / (vals.length - 1);
    const y = v => h - pad - ((v - min) / span) * (h - pad * 2);
    let d = "";
    vals.forEach((v, i) => { d += (i ? "L" : "M") + (pad + i * dx).toFixed(1) + " " + y(v).toFixed(1) + " "; });
    const col = opts.color || SERIES[0];
    svg.appendChild(el("path", { d: d + `L ${pad + (vals.length - 1) * dx} ${h - pad} L ${pad} ${h - pad} Z`, fill: col + "22", stroke: "none" }));
    svg.appendChild(el("path", { d, fill: "none", stroke: col, "stroke-width": 2, "stroke-linejoin": "round" }));
    svg.appendChild(el("circle", { cx: pad + (vals.length - 1) * dx, cy: y(vals[vals.length - 1]), r: 3, fill: col }));
    return svg;
  }

  // ------------------------------------------------------------ bars (legacy)
  function bars(data, opts = {}) {
    const h = opts.h || (data.length * 26 + 8), w = opts.w || 320, labelW = opts.labelW || 120;
    if (!data.length) return empty(w, 40);
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", height: h });
    const max = Math.max(1e-9, ...data.map(d => d.value));
    data.forEach((d, i) => {
      const y = i * 26 + 4;
      svg.appendChild(txt(0, y + 13, d.label));
      const bw = Math.max(2, (w - labelW - 44) * d.value / max);
      hover(svg.appendChild(el("rect", { x: labelW, y: y + 3, width: bw, height: 12, rx: 3, fill: d.color || SERIES[0] })),
        () => `<b>${esc(d.label)}</b><br>${esc(d.value)}`);
      svg.appendChild(txt(w - 4, y + 13, String(Math.round(d.value * 100) / 100), { fill: INK, "text-anchor": "end" }));
    });
    return svg;
  }

  function donut(segments, opts = {}) {
    const size = opts.size || 150, r = size / 2 - 12, cx = size / 2, cy = size / 2;
    const svg = el("svg", { viewBox: `0 0 ${size} ${size}`, width: size, height: size });
    const total = segments.reduce((a, s) => a + s.value, 0) || 1;
    let ang = -Math.PI / 2;
    segments.forEach(s => {
      const a2 = ang + (s.value / total) * Math.PI * 2 - 0.02;
      const large = (a2 - ang) > Math.PI ? 1 : 0;
      const p = el("path", { d: `M ${cx} ${cy} L ${cx + r * Math.cos(ang)} ${cy + r * Math.sin(ang)} A ${r} ${r} 0 ${large} 1 ${cx + r * Math.cos(a2)} ${cy + r * Math.sin(a2)} Z`, fill: s.color });
      hover(svg.appendChild(p), () => `<b>${esc(s.label)}</b><br>${s.value}`);
      ang = a2 + 0.02;
    });
    svg.appendChild(el("circle", { cx, cy, r: r * 0.6, fill: SURF }));
    svg.appendChild(txt(cx, cy + 6, String(total), { fill: INK, "font-size": 20, "text-anchor": "middle", "font-weight": 700 }));
    return svg;
  }

  // ------------------------------------------------------------ heatmap
  // grid: rows x cols of [0,1] (null = unknown). overlay: same shape, drawn
  // as a dot whose size is the observed share. highlight: {r, c}.
  function heatmap(grid, opts = {}) {
    const rows = grid.length, cols = rows ? grid[0].length : 0;
    if (!rows || !cols) return empty(600, 60);
    const lw = opts.labelW || 40, top = 16, cw = opts.cell || 22, ch = opts.cellH || 20, gap = 2;
    const w = lw + cols * cw + 4, h = top + rows * ch + 4;
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", style: `max-width:${w * 1.6}px`, class: "heatmap" });
    const colLabels = opts.colLabels || [...Array(cols).keys()].map(String);
    colLabels.forEach((c, j) => { if (j % (opts.colEvery || 3) === 0) svg.appendChild(txt(lw + j * cw + cw / 2, 11, c, { "text-anchor": "middle", "font-size": 9 })); });
    grid.forEach((row, i) => {
      svg.appendChild(txt(lw - 6, top + i * ch + ch / 2 + 4, (opts.rowLabels || [])[i] || "", { "text-anchor": "end", "font-size": 10 }));
      row.forEach((v, j) => {
        const x = lw + j * cw, y = top + i * ch;
        const fill = seq(v) || "#0f1626";
        const cell = el("rect", { x: x + gap / 2, y: y + gap / 2, width: cw - gap, height: ch - gap, rx: 3, fill });
        svg.appendChild(cell);
        const ov = opts.overlay && opts.overlay[i] ? opts.overlay[i][j] : null;
        if (ov != null && isFinite(ov)) {
          svg.appendChild(el("circle", { cx: x + cw / 2, cy: y + ch / 2, r: 1.5 + 3.5 * ov, fill: "none", stroke: "#ffd166", "stroke-width": 1.5 }));
        }
        const hi = opts.highlight && opts.highlight.r === i && opts.highlight.c === j;
        if (hi) svg.appendChild(el("rect", { x: x + 0.5, y: y + 0.5, width: cw - 1, height: ch - 1, rx: 4, fill: "none", stroke: "#ffffff", "stroke-width": 2 }));
        const hit = el("rect", { x, y, width: cw, height: ch, fill: "transparent" });
        hover(svg.appendChild(hit), () => (opts.tipFn ? opts.tipFn(i, j, v, ov) :
          `<b>${esc((opts.rowLabels || [])[i] || i)} ${esc(colLabels[j])}</b><br>${v == null ? "—" : (v * 100).toFixed(0) + "%"}`));
      });
    });
    return svg;
  }

  function seqLegend(label) {
    const d = document.createElement("div"); d.className = "legend";
    const bar = SEQ.map(c => `<i style="display:inline-block;width:14px;height:10px;background:${c}"></i>`).join("");
    d.innerHTML = `<span>0</span><span style="display:inline-flex">${bar}</span><span>1</span><span>${esc(label || "")}</span>`;
    return d;
  }

  // ------------------------------------------------------------ bands
  // One feature: rows [{label, lo, mid, hi}] on a shared x scale plus the
  // current value as a vertical marker. log: log10 scale for heavy tails.
  function bands(rows, opts = {}) {
    const w = opts.w || 360, rh = 22, top = 8, lw = opts.labelW || 64;
    const cur = opts.current;
    const vals = [];
    rows.forEach(r => [r.lo, r.mid, r.hi].forEach(v => { if (v != null && isFinite(v)) vals.push(v); }));
    if (cur != null && isFinite(cur)) vals.push(cur);
    if (!vals.length) return empty(w, 40);
    const log = !!opts.log;
    const f = (v) => log ? Math.log10(Math.max(v, 0) + 1) : v;
    let lo = Math.min(...vals.map(f)), hi = Math.max(...vals.map(f));
    if (!log) lo = Math.min(lo, 0);
    if (hi - lo < 1e-9) hi = lo + 1;
    const pad = (hi - lo) * 0.06; lo -= log ? 0 : 0; hi += pad;
    const h = top + rows.length * rh + 18;
    const X = (v) => lw + (w - lw - 8) * (f(v) - lo) / (hi - lo);
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", style: `max-width:${w}px;display:block` });
    svg.appendChild(el("line", { x1: lw, x2: w - 8, y1: h - 14, y2: h - 14, stroke: GRID }));
    svg.appendChild(txt(lw, h - 2, fmtNum(log ? Math.pow(10, lo) - 1 : lo), { "font-size": 9 }));
    svg.appendChild(txt(w - 8, h - 2, fmtNum(log ? Math.pow(10, hi) - 1 : hi), { "font-size": 9, "text-anchor": "end" }));
    rows.forEach((r, i) => {
      const y = top + i * rh;
      svg.appendChild(txt(lw - 6, y + 13, r.label, { "text-anchor": "end" }));
      if (r.lo != null && r.hi != null) {
        const band = el("rect", { x: X(r.lo), y: y + 4, width: Math.max(3, X(r.hi) - X(r.lo)), height: 12, rx: 4, fill: (opts.color || SERIES[0]) + "66" });
        hover(svg.appendChild(band), () => `<b>${esc(r.label)}</b><br>${esc(opts.qLabel || "p10–p90")}: ${fmtNum(r.lo)} – ${fmtNum(r.hi)}<br>p50: ${fmtNum(r.mid)} ${esc(opts.unit || "")}`);
      }
      if (r.mid != null) svg.appendChild(el("line", { x1: X(r.mid), x2: X(r.mid), y1: y + 3, y2: y + 17, stroke: INK, "stroke-width": 2 }));
    });
    if (cur != null && isFinite(cur)) {
      const x = X(cur);
      const out = rows.some(r => r.hi != null && cur > r.hi) && rows.every(r => r.hi == null || cur > r.hi);
      const col = out ? "#ec835a" : "#ffd166";
      svg.appendChild(el("line", { x1: x, x2: x, y1: top - 2, y2: h - 14, stroke: col, "stroke-width": 2, "stroke-dasharray": "3 2" }));
      hover(svg.appendChild(el("circle", { cx: x, cy: top - 1, r: 4, fill: col })), () => `当前 current: <b>${fmtNum(cur)}</b> ${esc(opts.unit || "")}`);
    }
    return svg;
  }

  // ------------------------------------------------------------ gauge
  function gauge(value, ci, opts = {}) {
    const w = opts.w || 200, h = w * 0.62, cx = w / 2, cy = h - 12, r = w / 2 - 16;
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h });
    const pt = (v) => { const a = Math.PI * (1 - Math.max(0, Math.min(1, v))); return [cx + r * Math.cos(a), cy - r * Math.sin(a)]; };
    const arc = (a, b, col, sw) => { const [x1, y1] = pt(a), [x2, y2] = pt(b);
      return el("path", { d: `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`, fill: "none", stroke: col, "stroke-width": sw, "stroke-linecap": "round" }); };
    svg.appendChild(arc(0, 1, "#1c2740", 14));
    if (value != null && isFinite(value)) {
      const col = value >= 0.8 ? "#0ca30c" : value >= 0.5 ? "#fab219" : "#d03b3b";
      if (ci && ci.length === 2 && ci[0] != null) svg.appendChild(arc(ci[0], Math.max(ci[0] + 0.005, ci[1]), col + "55", 22));
      svg.appendChild(arc(0, Math.max(0.005, value), col, 14));
      const [nx, ny] = pt(value);
      svg.appendChild(el("circle", { cx: nx, cy: ny, r: 5, fill: INK }));
    }
    svg.appendChild(txt(cx, cy - 8, value == null ? "—" : value.toFixed(2), { fill: INK, "font-size": 24, "font-weight": 800, "text-anchor": "middle" }));
    if (ci && ci[0] != null) svg.appendChild(txt(cx, cy + 8, `95% CI ${ci[0].toFixed(2)}–${ci[1].toFixed(2)}`, { "text-anchor": "middle", "font-size": 10 }));
    svg.appendChild(txt(cx - r, cy + 10, "0", { "text-anchor": "middle", "font-size": 9 }));
    svg.appendChild(txt(cx + r, cy + 10, "1", { "text-anchor": "middle", "font-size": 9 }));
    return svg;
  }

  // ------------------------------------------------------------ time series
  // series: [{name, ts[], values[], color}] on ONE y scale; markers:
  // [{ts, label, color, shape: 'tri'|'line'|'dot', row}]; thresholds: [{y, label, color}].
  function timeSeries(series, opts = {}) {
    const w = opts.w || 640, h = opts.h || 220, L = 44, R = 12, T = 14, B = 34;
    const all = [];
    series.forEach(s => s.ts.forEach((t, i) => { if (s.values[i] != null) all.push([t, s.values[i]]); }));
    const mk = opts.markers || [];
    if (!all.length && !mk.length) return empty(w, 60);
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", height: h, class: "tschart" });
    const ts = all.map(a => a[0]).concat(mk.map(m => m.ts));
    let t0 = opts.t0 != null ? opts.t0 : Math.min(...ts), t1 = opts.t1 != null ? opts.t1 : Math.max(...ts);
    if (t1 - t0 < 1) t1 = t0 + 1;
    const ys = all.map(a => a[1]);
    let y0 = opts.yMin != null ? opts.yMin : Math.min(0, ...ys), y1 = opts.yMax != null ? opts.yMax : Math.max(1e-9, ...ys);
    if (opts.logY) { y0 = Math.max(opts.yMin != null ? opts.yMin : 1e-12, Math.min(...ys.filter(v => v > 0), 1)); }
    const fy = (v) => opts.logY ? Math.log10(Math.max(v, 1e-300)) : v;
    const Y = (v) => T + (h - T - B) * (1 - (fy(v) - fy(y0)) / ((fy(y1) - fy(y0)) || 1));
    const X = (t) => L + (w - L - R) * (t - t0) / (t1 - t0);
    // grid + axes
    const ticksY = opts.yTicks || [y0, (y0 + y1) / 2, y1];
    ticksY.forEach(v => { svg.appendChild(el("line", { x1: L, x2: w - R, y1: Y(v), y2: Y(v), stroke: GRID, "stroke-width": 1 }));
      svg.appendChild(txt(L - 6, Y(v) + 4, fmtNum(v), { "text-anchor": "end", "font-size": 10 })); });
    (opts.thresholds || []).forEach(th => {
      svg.appendChild(el("line", { x1: L, x2: w - R, y1: Y(th.y), y2: Y(th.y), stroke: th.color, "stroke-dasharray": "4 3", "stroke-width": 1, opacity: 0.7 }));
      svg.appendChild(txt(w - R - 2, Y(th.y) - 3, th.label, { "text-anchor": "end", "font-size": 9, fill: MUTED }));
    });
    const nT = 6;
    for (let i = 0; i <= nT; i++) { const t = t0 + (t1 - t0) * i / nT;
      svg.appendChild(txt(X(t), h - B + 16, opts.fmtT ? opts.fmtT(t) : String(Math.round(t)), { "text-anchor": i === 0 ? "start" : i === nT ? "end" : "middle", "font-size": 10 })); }
    // markers (under the lines)
    mk.forEach(m => {
      const x = X(m.ts);
      if (m.shape === "line") svg.appendChild(el("line", { x1: x, x2: x, y1: T, y2: h - B, stroke: m.color, "stroke-width": 1.5, "stroke-dasharray": "2 3", opacity: 0.8 }));
      const yy = h - B + 2 + (m.row || 0) * 0;
      const node = m.shape === "tri"
        ? el("path", { d: `M ${x} ${T + 2 + (m.row || 0) * 11} l 5 9 l -10 0 Z`, fill: m.color })
        : el("circle", { cx: x, cy: m.shape === "line" ? T + 4 + (m.row || 0) * 11 : yy, r: 4.5, fill: m.color, stroke: SURF, "stroke-width": 2 });
      hover(svg.appendChild(node), () => `<b>${esc(m.label)}</b><br>${esc(opts.fmtFull ? opts.fmtFull(m.ts) : m.ts)}`);
    });
    // lines
    series.forEach((s, si) => {
      let d = "", pen = false;
      s.ts.forEach((t, i) => { const v = s.values[i];
        if (v == null || !isFinite(fy(v))) { pen = false; return; }
        d += (pen ? "L" : "M") + X(t).toFixed(1) + " " + Y(v).toFixed(1) + " "; pen = true; });
      svg.appendChild(el("path", { d, fill: "none", stroke: s.color || SERIES[si % SERIES.length], "stroke-width": 2, "stroke-linejoin": "round" }));
    });
    // crosshair + tooltip
    const cross = el("line", { x1: 0, x2: 0, y1: T, y2: h - B, stroke: MUTED, "stroke-width": 1, visibility: "hidden" });
    svg.appendChild(cross);
    const hit = el("rect", { x: L, y: T, width: w - L - R, height: h - T - B, fill: "transparent" });
    svg.appendChild(hit);
    hit.addEventListener("mousemove", (e) => {
      const box = svg.getBoundingClientRect();
      const vx = (e.clientX - box.left) * (w / box.width);
      const t = t0 + (vx - L) / (w - L - R) * (t1 - t0);
      cross.setAttribute("x1", vx); cross.setAttribute("x2", vx); cross.setAttribute("visibility", "visible");
      let html = `<b>${esc(opts.fmtFull ? opts.fmtFull(t) : Math.round(t))}</b>`;
      series.forEach((s, si) => { if (!s.ts.length) return;
        let bi = 0, bd = Infinity; s.ts.forEach((tt, i) => { const dd = Math.abs(tt - t); if (dd < bd) { bd = dd; bi = i; } });
        html += `<br><i class="dot" style="background:${s.color || SERIES[si]}"></i>${esc(s.name)}: <b>${fmtNum(s.values[bi])}</b>`; });
      showTip(e, html);
    });
    hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
    return svg;
  }

  // ------------------------------------------------------------ graph
  // nodes: [{id, label, color, r, group}], edges: [{from, to, type}]
  function graph(nodes, edges, opts = {}) {
    const w = opts.w || 700, h = opts.h || 340;
    if (!nodes.length) return empty(w, 60, "无关联事件 / no linked incidents");
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", height: h });
    // group nodes by component so each campaign sits together, on a circle
    const pos = {}, n = nodes.length, cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 40;
    const order = nodes.slice().sort((a, b) => String(a.group || "~").localeCompare(String(b.group || "~")) || String(a.id).localeCompare(String(b.id)));
    order.forEach((nd, i) => { const a = 2 * Math.PI * i / n - Math.PI / 2;
      pos[nd.id] = n === 1 ? [cx, cy] : [cx + R * Math.cos(a), cy + R * Math.sin(a)]; });
    edges.forEach(e => { const a = pos[e.from], b = pos[e.to]; if (!a || !b) return;
      svg.appendChild(el("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: e.type === "campaign" ? "#9085e9" : "#199e70", "stroke-width": 2, "stroke-dasharray": e.type === "campaign" ? "" : "5 3", opacity: 0.85 })); });
    order.forEach(nd => { const [x, y] = pos[nd.id];
      const c = el("circle", { cx: x, cy: y, r: nd.r || 9, fill: nd.color || SERIES[0], stroke: SURF, "stroke-width": 2, style: nd.onclick ? "cursor:pointer" : "" });
      if (nd.onclick) c.addEventListener("click", nd.onclick);
      hover(svg.appendChild(c), () => nd.tip || esc(nd.label));
      svg.appendChild(txt(x, y + (nd.r || 9) + 12, nd.label, { "text-anchor": "middle", "font-size": 10, fill: INK })); });
    return svg;
  }

  function fmtNum(v) {
    if (v == null || !isFinite(v)) return "—";
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(1) + "G";
    if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (a >= 1e4) return (v / 1e3).toFixed(1) + "k";
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(2).replace(/\.?0+$/, "");
    if (a === 0) return "0";
    if (a >= 1e-3) return v.toFixed(3);
    return v.toExponential(1);
  }

  return { sparkline, bars, donut, heatmap, seqLegend, bands, gauge, timeSeries, graph, fmtNum, hover, esc, SERIES, SEQ, seq };
})();
