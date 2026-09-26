// Dependency-free inline-SVG charts (works offline, theme-aware colours).
const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const el = (n, a = {}) => { const e = document.createElementNS(NS, n);
    for (const k in a) e.setAttribute(k, a[k]); return e; };

  function sparkline(points, opts = {}) {
    const w = opts.w || 260, h = opts.h || 60, pad = 4;
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", height: h });
    const vals = points.map(p => (typeof p === "number" ? p : p.value)).filter(v => v != null);
    if (vals.length < 2) { svg.appendChild(el("text",{x:6,y:h/2,fill:"#8ba0be","font-size":11}))
      .textContent = "数据不足"; return svg; }
    const min = Math.min(...vals), max = Math.max(...vals), span = (max - min) || 1;
    const dx = (w - pad * 2) / (vals.length - 1);
    const y = v => h - pad - ((v - min) / span) * (h - pad * 2);
    let d = "";
    vals.forEach((v, i) => { d += (i ? "L" : "M") + (pad + i * dx).toFixed(1) + " " + y(v).toFixed(1) + " "; });
    const area = d + `L ${pad + (vals.length-1)*dx} ${h-pad} L ${pad} ${h-pad} Z`;
    svg.appendChild(el("path", { d: area, fill: (opts.color||"#4da3ff") + "22", stroke: "none" }));
    svg.appendChild(el("path", { d, fill: "none", stroke: opts.color || "#4da3ff", "stroke-width": 1.8 }));
    const last = vals[vals.length-1];
    svg.appendChild(el("circle", { cx: pad+(vals.length-1)*dx, cy: y(last), r: 2.6, fill: opts.color||"#4da3ff" }));
    return svg;
  }

  function bars(data, opts = {}) { // data: [{label,value,color?}]
    const h = opts.h || (data.length * 26 + 8), w = opts.w || 320, labelW = opts.labelW || 120;
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", height: h });
    const max = Math.max(1, ...data.map(d => d.value));
    data.forEach((d, i) => {
      const y = i * 26 + 4;
      const t = el("text", { x: 0, y: y + 13, fill: "#8ba0be", "font-size": 11 });
      t.textContent = d.label; svg.appendChild(t);
      svg.appendChild(el("rect", { x: labelW, y: y + 3, width: (w - labelW - 40) * d.value / max,
        height: 12, rx: 3, fill: d.color || "#4da3ff" }));
      const v = el("text", { x: w - 4, y: y + 13, fill: "#e6edf7", "font-size": 11, "text-anchor": "end" });
      v.textContent = (Math.round(d.value * 100) / 100); svg.appendChild(v);
    });
    return svg;
  }

  function donut(segments, opts = {}) { // segments: [{label,value,color}]
    const size = opts.size || 150, r = size/2 - 12, cx = size/2, cy = size/2;
    const svg = el("svg", { viewBox: `0 0 ${size} ${size}`, width: size, height: size });
    const total = segments.reduce((a, s) => a + s.value, 0) || 1;
    let ang = -Math.PI/2;
    segments.forEach(s => {
      const a2 = ang + (s.value/total) * Math.PI*2;
      const large = (a2-ang) > Math.PI ? 1 : 0;
      const x1 = cx + r*Math.cos(ang), y1 = cy + r*Math.sin(ang);
      const x2 = cx + r*Math.cos(a2), y2 = cy + r*Math.sin(a2);
      svg.appendChild(el("path", { d:`M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z`,
        fill: s.color, opacity: .9 }));
      ang = a2;
    });
    svg.appendChild(el("circle", { cx, cy, r: r*0.6, fill: "#131c2e" }));
    const c = el("text", { x: cx, y: cy+5, fill: "#e6edf7", "font-size": 20, "text-anchor":"middle", "font-weight":700 });
    c.textContent = total; svg.appendChild(c);
    return svg;
  }
  return { sparkline, bars, donut };
})();
