// 画像模式 — progressive profile core pages (API v3, docs/lib3/progressive.md §9.4).
// Two views of one pattern store: the SYSTEM view (actions -> who -> when ->
// content -> workflow) and the GROUP / IP view (group -> systems -> actions),
// plus the pattern lattice, typed violations, facet composition, the system's
// strategy, the attribute registry and the resource budget.
// A "user" is an IP or an IP class, never a person. Defensive like pages.js:
// a model not fitted yet renders as "—" / an empty state, never an error.
(() => {
  const enc = encodeURIComponent;
  const esc = Charts.esc;
  async function get(path) {
    const r = await fetch("/api/v3" + path);
    if (!r.ok) { const e = new Error(path + " -> " + r.status); e.status = r.status;
      try { e.body = await r.json(); } catch (_) { /* not JSON */ } throw e; }
    return r.json();
  }
  async function post(path, body) {
    const r = await fetch("/api/v3" + path, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body) });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) { const e = new Error((d && d.detail) || path + " -> " + r.status); e.status = r.status; throw e; }
    return d;
  }
  const qs = (o) => { const p = Object.entries(o || {}).filter(([, v]) => v !== undefined && v !== null && v !== "");
    return p.length ? "?" + p.map(([k, v]) => `${k}=${enc(v)}`).join("&") : ""; };
  const API3 = {
    status: () => get("/status"),
    systems: () => get("/systems"),
    view: (s, fresh) => get(`/systems/${enc(s)}/view` + qs({ fresh: fresh ? "true" : "" })),
    precision: (s) => get(`/systems/${enc(s)}/precision`),
    lattice: (s, o) => get(`/systems/${enc(s)}/lattice` + qs(o)),
    ipView: (s, ip) => get(`/systems/${enc(s)}/entities/${enc(ip)}/view`),
    ipFacets: (s, ip) => get(`/systems/${enc(s)}/entities/${enc(ip)}/facets`),
    facets: (s) => get(`/systems/${enc(s)}/facets`),
    strategy: (s) => get(`/systems/${enc(s)}/strategy`),
    attributes: (s) => get(`/systems/${enc(s)}/attributes`),
    groups: () => get("/groups"),
    groupView: (g) => get(`/groups/${enc(g)}/view`),
    groupFacets: (g) => get(`/groups/${enc(g)}/facets`),
    groupName: (g, name) => post(`/groups/${enc(g)}/name`, { name }),
    pattern: (pid) => get(`/patterns/${enc(pid)}`),
    violations: (f) => get("/violations" + qs(f)),
    budget: () => get("/budget"),
  };
  window.API3 = API3;

  // ------------------------------------------------------------ helpers
  const L = (zh, en) => t(zh, en);
  const txt = (o) => o ? (STATE.lang === "en" ? (o.text_en || o.text_zh) : (o.text_zh || o.text_en)) : "";
  const num = (v, d = 2) => v == null || !isFinite(v) ? "—" : Number(v).toFixed(d);
  const pctS = (v) => v == null || !isFinite(v) ? "—" : (v * 100).toFixed(v < 0.01 && v > 0 ? 2 : 0) + "%";
  const hhmm = (m) => { m = Math.round(m); return String(Math.floor(m / 60) % 24).padStart(2, "0") + ":" + String(m % 60).padStart(2, "0"); };
  const STATE_CLS = { candidate: "sev-info", confirmed: "sev-low", stable: "sev-low", evolving: "sev-medium",
    stale: "sev-medium", dormant: "sev-info", retired: "sev-critical" };
  const STATE_ZH = { candidate: "候选", confirmed: "已确认", stable: "稳定", evolving: "演化中", stale: "陈旧",
    dormant: "休眠", retired: "已退役" };
  const stateBadge = (s) => h("span", { class: "badge " + (STATE_CLS[s] || "sev-info") }, L(STATE_ZH[s] || s || "—", s || "—"));
  const WHO_ZH = { ip: "IP 清单", grp: "行为群组", prefix: "网段", reg: "区域", any: "任意 IP", none: "IP 不作为特征" };
  const chip = (label, val, title) => h("span", { class: "chip", title: title || null }, label, val != null && val !== "" ? h("b", {}, " " + val) : "");
  const sysHref = (s) => `#/pp/system/${enc(s)}`;
  const patHref = (pid) => `#/pp/pattern/${enc(pid)}`;
  const grpHref = (g) => `#/pp/group/${enc(g)}`;
  const ipHref = (s, ip) => `#/pp/ip/${enc(s)}/${enc(ip)}`;
  const latHref = (s, root) => `#/pp/lattice/${enc(s)}` + (root != null ? `?root=${root}` : "");
  function confMeter(v) { return h("div", { class: "rowline", title: L("置信度", "confidence") },
    h("div", { class: "meter pp-conf", style: "flex:1" }, h("span", { style: `width:${Math.round((v || 0) * 100)}%` })),
    h("span", { class: "small muted nowrap" }, v == null ? "—" : num(v, 2))); }
  function isIp(x) { return /^\d+\.\d+\.\d+\.\d+$/.test(String(x)); }
  function ipLink(sys, x) { return isIp(x) && sys ? link(ipHref(sys, x), x) : String(x); }

  function subnav(active, sys) {
    const items = [["overview", L("总览", "Overview"), "#/pp"],
      ["system", L("系统视图", "System view"), sys ? sysHref(sys) : null],
      ["groups", L("群组/用户视图", "Groups / users"), "#/pp/groups"],
      ["lattice", L("模式格", "Lattice"), sys ? latHref(sys) : null],
      ["violations", L("违例", "Violations"), "#/pp/violations" + (sys ? `?system=${enc(sys)}` : "")],
      ["facets", L("多维刻面", "Facets"), sys ? `#/pp/facets/${enc(sys)}` : null],
      ["strategy", L("系统特征与策略", "Strategy"), sys ? `#/pp/strategy/${enc(sys)}` : null],
      ["attributes", L("指标/属性", "Attributes"), sys ? `#/pp/attributes/${enc(sys)}` : null],
      ["budget", L("资源预算", "Budget"), "#/pp/budget"]];
    const bar = h("div", { class: "pp-subnav" });
    items.forEach(([k, label, href]) => { if (!href) return;
      bar.appendChild(h("a", { class: "pp-sub" + (k === active ? " active" : ""), href }, label)); });
    if (sys && STATE.ppSystems && STATE.ppSystems.length) {
      const sel = h("select", { class: "search small", "aria-label": L("业务系统", "system"), onchange: (e) => {
        const ns = e.target.value; STATE.system = ns;
        const map = { lattice: latHref(ns), facets: `#/pp/facets/${enc(ns)}`, strategy: `#/pp/strategy/${enc(ns)}`,
          attributes: `#/pp/attributes/${enc(ns)}` };
        go(map[active] || sysHref(ns)); } },
        ...STATE.ppSystems.map((x) => h("option", { value: x, selected: x === sys ? "selected" : null }, x)));
      bar.appendChild(h("span", { class: "right" }, sel));
    }
    return bar;
  }
  async function ensureSystems() {
    if (STATE.ppSystems) return STATE.ppSystems;
    try { const s = await API3.systems(); STATE.ppSystems = s.systems.map((x) => x.system); }
    catch (_) { STATE.ppSystems = []; }
    return STATE.ppSystems;
  }
  function pickSystem(params) {
    const s = params[0] || STATE.ppSystem || (STATE.ppSystems || [])[0];
    if (s) STATE.ppSystem = s;
    return s;
  }
  function page(active, sys, title, ...kids) {
    return h("section", { class: "pp" }, subnav(active, sys),
      title ? h("div", { class: "detail-head" }, h("h2", {}, title)) : "", ...kids);
  }

  // ------------------------------------------------------------ statement blocks
  function whoBlock(w, sys) {
    if (!w) return h("span", { class: "muted" }, "—");
    const lvl = w.level || "any";
    const items = (w.members && w.members.length ? w.members : w.items || []).slice(0, 12);
    const d = h("div", { class: "pp-who" },
      h("span", { class: "tag" }, L(WHO_ZH[lvl] || lvl, lvl)),
      w.name ? h("b", {}, " " + w.name) : "",
      w.group && !w.name ? link(grpHref(w.group), " " + w.group) : "",
      " ", ...items.map((x) => h("span", { class: "chip small mono" }, String(x).startsWith("grp:") ? link(grpHref(String(x).slice(4)), x) : ipLink(sys, x))),
      (w.members || w.items || []).length > 12 ? h("span", { class: "muted small" }, ` +${(w.members || w.items).length - 12}`) : "",
      " ", w.closed ? h("span", { class: "badge sev-low", title: "U" }, L("来源封闭", "closed") + (w.U != null ? ` · U=${num(w.U, 3)}` : ""))
        : h("span", { class: "badge sev-info" }, L("未封闭", "open")),
      w.distinct != null ? h("span", { class: "muted small" }, L(` 约 ${w.distinct} 个 IP`, ` ~${w.distinct} IPs`)) : "",
      w.share != null ? h("span", { class: "muted small" }, L(` · 占该模式 ${pctS(w.share)}`, ` · ${pctS(w.share)} of the pattern`)) : "");
    return d;
  }
  function whenBlock(w) {
    if (!w) return h("span", { class: "muted" }, L("（未拟合时间窗）", "(no windows fitted)"));
    const seg = (arr) => (arr || []).map((x) => `${hhmm(x[0])}–${hhmm(x[1])}`).join(", ") || "—";
    return h("div", {},
      h("span", { class: "chip" }, L("工作日 ", "workdays "), h("b", {}, seg(w.workday))),
      " ", h("span", { class: "chip" }, L("休息日 ", "non-workdays "), h("b", {}, seg(w.nonworkday))),
      w.coverage != null ? h("span", { class: "muted small" }, L(` 覆盖 ${pctS(w.coverage)}`, ` coverage ${pctS(w.coverage)}`)) : "");
  }
  function fmtBytes(v) { if (v == null || !isFinite(v)) return "—";
    if (v >= 1048576) return (v / 1048576).toFixed(1) + " MB"; if (v >= 1024) return (v / 1024).toFixed(1) + " KB";
    return Math.round(v) + " B"; }
  function contentBlock(c) {
    const keys = Object.keys(c || {});
    if (!keys.length) return h("span", { class: "muted" }, "—");
    const tb = h("table", { class: "pp-mini" }, h("tr", {}, h("th", {}, L("属性", "attribute")), h("th", {}, L("约束", "constraint"))));
    keys.forEach((a) => { const r = c[a] || {}; const parts = [];
      if (r.band90) parts.push(L(`90% 在 ${fmtBytes(r.band90[0])}–${fmtBytes(r.band90[1])}`, `90 % in ${fmtBytes(r.band90[0])}–${fmtBytes(r.band90[1])}`));
      if (r.range) parts.push(L(`全部在 ${fmtBytes(r.range[0])}–${fmtBytes(r.range[1])}`, `all in ${fmtBytes(r.range[0])}–${fmtBytes(r.range[1])}`));
      if (r.required) parts.push(L("必含键 ", "required keys ") + r.required.join(", "));
      if (r.optional && r.optional.length) parts.push(L("可选键 ", "optional ") + r.optional.join(", "));
      if (r.grammar) parts.push(L("语法 ", "grammar ") + r.grammar);
      if (r.len) parts.push(L(`长度 ${r.len[0]}–${r.len[1]}`, `length ${r.len[0]}–${r.len[1]}`));
      if (r.closed) parts.push(L("封闭集合 {", "closed set {") + r.closed.slice(0, 10).join(", ") + (r.closed.length > 10 ? ", …" : "") + "}");
      if (r.value != null) parts.push(L("恒定 ", "constant ") + r.value);
      if (!parts.length) parts.push(Object.entries(r).filter(([, v]) => typeof v !== "object").slice(0, 4).map(([k, v]) => `${k}=${typeof v === "number" ? num(v, 3) : v}`).join(" "));
      tb.appendChild(h("tr", {}, h("td", { class: "mono small" }, a), h("td", { class: "small" }, parts.join("；")))); });
    return h("div", { class: "tablewrap" }, tb);
  }
  function bindingsBlock(b, sys) {
    const keys = Object.keys(b || {});
    if (!keys.length) return h("span", { class: "muted" }, "—");
    return h("div", {}, ...keys.map((a) => { const r = b[a] || {}; const tbl = r.table || {};
      return h("div", { class: "pp-bind" }, h("span", { class: "mono small muted" }, `${r.x || "net.src"} → ${a}`),
        r.g3 != null ? h("span", { class: "muted small" }, ` g3=${num(r.g3, 3)}`) : "",
        h("div", { class: "chips" }, ...Object.entries(tbl).slice(0, 24).map(([ip, v]) =>
          h("span", { class: "chip small" }, ipLink(sys, ip), " → ", h("b", { class: "mono" }, Array.isArray(v) ? v.join("|") : String(v)),
            (r.LB || {})[ip] != null ? h("span", { class: "muted" }, ` LB ${num(r.LB[ip], 2)}`) : "")))); }));
  }
  function workflowBlock(wf) {
    if (!wf || !wf.length) return h("span", { class: "muted" }, "—");
    return h("div", {}, ...wf.slice(0, 8).map((e) => h("div", { class: "small" },
      h("span", { class: "mono" }, routeShort(e.from)), " → ", h("span", { class: "mono" }, routeShort(e.to)),
      e.band ? h("span", { class: "muted" }, L(`（间隔 ${num(e.band[0], 0)}–${num(e.band[1], 0)} 秒）`, ` (gap ${num(e.band[0], 0)}–${num(e.band[1], 0)} s)`)) : "",
      e.dep != null ? h("span", { class: "muted" }, ` dep ${num(e.dep, 2)}`) : "")));
  }
  function routeShort(r) { const p = String(r || "").split(" "); return p.length >= 3 ? p[0] + " " + p.slice(2).join(" ") : String(r || ""); }
  function statementCard(st, sys, opts = {}) {
    const cls = "card pp-stmt" + (st.negative ? " pp-neg" : "") + (st.part_of ? " pp-part" : "");
    const head = h("div", { class: "rowline wrap" }, stateBadge(st.state),
      st.negative ? h("span", { class: "badge sev-high" }, L("否定事实", "negative")) : "",
      st.part_of ? h("span", { class: "tag" }, L("某类人分解", "group part")) : "",
      st.is_exc ? h("span", { class: "tag" }, L("IP 例外", "IP exception")) : "",
      h("span", { class: "small muted" }, `v${st.version ?? "?"}.${st.cver ?? 0}`),
      h("span", { class: "small muted" }, L("支持 ", "support ") + num(st.support, 1)),
      h("span", { class: "right", style: "min-width:150px" }, confMeter(st.confidence)));
    const body = h("div", { class: "narrative" }, txt(st));
    const det = h("details", {}, h("summary", {}, L("约束明细", "constraint details")),
      h("div", { class: "kvgrid pp-blocks" },
        h("div", {}, h("div", { class: "muted small" }, L("谁（来源）", "who")), whoBlock(st.who, sys)),
        h("div", {}, h("div", { class: "muted small" }, L("何时（时间窗）", "when")), whenBlock(st.when)),
        h("div", {}, h("div", { class: "muted small" }, L("内容约束", "content")), contentBlock(st.content)),
        h("div", {}, h("div", { class: "muted small" }, L("绑定", "bindings")), bindingsBlock(st.bindings, sys)),
        h("div", {}, h("div", { class: "muted small" }, L("流程", "workflow")), workflowBlock(st.workflow))),
      h("div", { class: "small muted mt6" }, L("首次 ", "first "), st.first_seen_local || "—", " · ", L("最近 ", "last "), st.last_seen_local || "—",
        " · ", L("刻面 ", "facets "), (st.facets || []).join(", ")));
    const links = h("div", { class: "links small" },
      st.pattern_id && !st.negative ? link(patHref(st.pattern_id), L("模式详情 →", "pattern →")) : "",
      st.node != null && sys ? link(latHref(sys, st.node), L("在模式格中查看 →", "in the lattice →")) : "");
    return h("div", { class: cls }, head, body, opts.compact ? "" : det, links);
  }

  // ------------------------------------------------------------ overview
  async function overview() {
    const s = await API3.status();
    STATE.ppSystems = s.systems.map((x) => x.system);
    if (!STATE.ppSystem && STATE.ppSystems.length) STATE.ppSystem = STATE.ppSystems[0];
    const head = h("div", { class: "grid g4" },
      kpi(L("业务系统", "Systems"), s.n_systems, s.registry_mode || ""),
      kpi(L("已确认模式", "Confident patterns"), s.n_confident, L("节点", "nodes")),
      kpi(L("画像语句", "Statements"), s.n_statements, L("系统视图", "system views")),
      kpi(L("行为群组", "Groups"), s.n_groups, L("P11 学习", "learned (P11)")));
    const banner = !s.running || !s.present ? h("div", { class: "card pp-hint" }, h("b", {}, txt({ text_zh: s.hint_zh, text_en: s.hint_en }))) : "";
    const tb = h("table", {}, h("tr", {}, h("th", {}, L("系统", "System")), h("th", {}, L("模式树", "Tree")),
      h("th", { class: "num" }, L("节点", "Nodes")), h("th", { class: "num" }, L("已确认", "Confident")),
      h("th", { class: "num" }, L("动作", "Actions")), h("th", { class: "num" }, L("语句", "Statements")),
      h("th", {}, L("谁的粒度", "Who")), h("th", {}, L("档位", "Tier")), h("th", {})));
    s.systems.forEach((x) => tb.appendChild(h("tr", { class: "clickable", onclick: () => go(sysHref(x.system)) },
      h("td", {}, h("b", {}, x.system), x.address ? h("div", { class: "small muted" }, x.address) : ""),
      h("td", { class: "mono small" }, x.tree_key + (x.family ? " (fam)" : "")),
      h("td", { class: "num" }, String(x.n_nodes)), h("td", { class: "num" }, String(x.n_confident)),
      h("td", { class: "num" }, String(x.n_actions)), h("td", { class: "num" }, String(x.n_statements)),
      h("td", {}, x.who_mode ? h("span", { class: "tag" }, L(WHO_ZH[x.who_mode] || x.who_mode, x.who_mode)) : "—"),
      h("td", {}, x.tier || "—"),
      h("td", { class: "links" }, link(latHref(x.system), L("模式格", "lattice")), link(`#/pp/strategy/${enc(x.system)}`, L("策略", "strategy"))))));
    const intro = h("div", { class: "card small muted" }, L(
      "渐进画像：不遍历全部用户与服务器，而是在一棵有预算的模式树上逐步细化（系统 → 动作 → 群组/网段/IP → 时间窗与内容约束），观测时间越长语句越精确，并随行为变化而演化。用户即 IP 或 IP 类。",
      "Progressive profiling: no enumeration of every user and server; a budgeted pattern tree is refined step by step (system → action → group / prefix / IP → windows and content constraints). Statements sharpen with observation time and follow behaviour. A user is an IP or an IP class."));
    return page("overview", STATE.ppSystem, L("画像模式 · 渐进行为画像", "Progressive behaviour profiles"),
      banner, intro, h("div", { class: "mt" }, head), h("div", { class: "card tablewrap mt" }, h("h3", {}, L("系统", "Systems")), tb));
  }

  // ------------------------------------------------------------ system view
  async function systemPage(params, q) {
    await ensureSystems();
    const sys = pickSystem(params);
    if (!sys) return page("system", null, L("系统视图", "System view"), empty());
    const [v, p] = await Promise.all([API3.view(sys, q.fresh === "1"), API3.precision(sys).catch(() => null)]);
    const hdr = h("div", { class: "card" },
      h("div", { class: "narrative" }, txt(v.header) || sys),
      h("div", { class: "chips" },
        chip(L("谁的粒度", "who"), v.who_mode ? L(WHO_ZH[v.who_mode] || v.who_mode, v.who_mode) : "—"),
        chip(L("动作", "actions"), v.n_actions), chip(L("语句", "statements"), v.n_statements),
        chip(L("置信中位数", "median confidence"), num(v.confidence && v.confidence.median, 2)),
        chip(L("版本", "version"), v.version ?? "—"), chip(L("来源", "source"), v.source),
        chip(L("更新", "updated"), v.updated_local ? v.updated_local.replace("T", " ").slice(0, 16) : "—"),
        h("button", { class: "mini", onclick: () => go(`${sysHref(sys)}?fresh=1`) }, L("立即重新渲染", "render now"))),
      v.header && v.header.hint ? h("div", { class: "small", style: "color:var(--medium)" }, v.header.hint) : "");
    const acts = h("div", {});
    if (!v.actions.length) acts.appendChild(card(L("动作", "Actions"), empty(L("尚无已确认的模式（模式在积累证据后确认）", "no confirmed pattern yet (patterns confirm as evidence accumulates)"))));
    v.actions.forEach((a) => {
      acts.appendChild(h("div", { class: "card pp-action mt" },
        h("div", { class: "rowline wrap" }, h("b", { class: "mono" }, a.route_text || a.route),
          a.write ? h("span", { class: "badge sev-medium" }, L("写操作", "write")) : h("span", { class: "badge sev-info" }, L("读", "read")),
          h("span", { class: "small muted" }, L(`占系统 ${pctS(a.share)}`, `${pctS(a.share)} of the system`)),
          h("span", { class: "small muted" }, L(`${a.n_statements} 条语句`, `${a.n_statements} statements`)),
          a.act_node != null ? link(latHref(sys, a.act_node), L("子树 →", "subtree →")) : ""),
        ...a.statements.map((s) => statementCard(s, sys))));
    });
    return page("system", sys, L(`系统视图 · ${sys}`, `System view · ${sys}`), hdr,
      p ? precisionCard(p) : "", h("h3", { class: "pp-h3" }, L("动作 → 谁 → 何时 → 内容 → 流程", "Actions → who → when → content → workflow")), acts);
  }

  function precisionCard(p) {
    const days = p.days || [];
    const ts = days.map((d) => Date.parse(d.date + "T00:00:00Z") / 1000);
    const fmtD = (x) => new Date(x * 1000).toISOString().slice(5, 10);
    const opts = { w: 420, h: 170, fmtT: fmtD, fmtFull: fmtD };
    const c1 = Charts.timeSeries([{ name: L("已确认模式", "confirmed patterns"), ts, values: days.map((d) => d.confirmed), color: Charts.SERIES[0] }], opts);
    const c2 = Charts.timeSeries([{ name: L("平均特化深度", "mean specificity (depth)"), ts, values: days.map((d) => d.mean_depth), color: Charts.SERIES[1] }], opts);
    const c3 = Charts.timeSeries([{ name: L("违例/天", "violations / day"), ts, values: days.map((d) => d.violations), color: Charts.SERIES[2] },
      { name: L("≥中 违例/天", "≥ medium / day"), ts, values: days.map((d) => d.violations_medium), color: Charts.SERIES[3] }], opts);
    const now = p.now || {};
    const live = h("div", { class: "grid g3" },
      h("div", {}, h("div", { class: "small muted" }, L("已确认模式（累计）", "confirmed patterns (cumulative)")), c1),
      h("div", {}, h("div", { class: "small muted" }, L("特化深度（越深越具体）", "specificity (deeper = more specific)")), c2),
      h("div", {}, h("div", { class: "small muted" }, L("违例数/天", "violations per day")), c3,
        h("div", { class: "legend" }, h("span", {}, h("i", { class: "dot", style: `background:${Charts.SERIES[2]}` }), L("全部", "all")),
          h("span", {}, h("i", { class: "dot", style: `background:${Charts.SERIES[3]}` }), L("≥中", "≥ medium")))));
    const nowRow = h("div", { class: "chips mt6" },
      chip(L("当前置信中位数", "confidence median"), num(now.confidence && now.confidence.median, 2)),
      chip("p10–p90", now.confidence ? `${num(now.confidence.p10, 2)}–${num(now.confidence.p90, 2)}` : "—"),
      chip(L("已确认节点平均深度", "mean depth of confident nodes"), num(now.mean_depth_confident, 2)),
      chip(L("支持度中位数", "support median"), num(now.support_median, 1)),
      ...Object.entries(now.nodes || {}).map(([k, n]) => chip(L(STATE_ZH[k] || k, k), n)));
    const m = p.measured || {};
    let meas = h("div", { class: "small muted" }, L("没有针对该系统的评估运行", "no evaluation run covers this system"));
    if (m.available) {
      const series = [], legend = [];
      m.runs.forEach((r, i) => {
        const tt = r.days.map((d) => d.day);
        const col = Charts.SERIES[i % Charts.SERIES.length];
        series.push({ name: L(`种子 ${r.seed} 本系统召回`, `seed ${r.seed} system recall`), ts: tt, values: r.days.map((d) => d.system_recall), color: col });
        legend.push(h("span", {}, h("i", { class: "dot", style: `background:${col}` }), L(`种子 ${r.seed}（${r.file}）`, `seed ${r.seed} (${r.file})`)));
      });
      const prec = m.runs.map((r, i) => ({ name: L(`种子 ${r.seed} 全局精确率`, `seed ${r.seed} global precision`), ts: r.days.map((d) => d.day),
        values: r.days.map((d) => d.precision), color: Charts.SERIES[i % Charts.SERIES.length] }));
      const o2 = { h: 150, yMin: 0, yMax: 1, yTicks: [0, 0.5, 1], fmtT: (x) => L(`第${Math.round(x)}天`, `d${Math.round(x)}`), fmtFull: (x) => L(`第 ${Math.round(x)} 天`, `day ${Math.round(x)}`) };
      meas = h("div", { class: "grid g2" },
        h("div", {}, h("div", { class: "small muted" }, L("本系统模式召回（对生成器真值）", "this system's pattern recall (vs generator truth)")), Charts.timeSeries(series, o2)),
        h("div", {}, h("div", { class: "small muted" }, L("全局语句精确率（对生成器真值）", "global statement precision (vs generator truth)")), Charts.timeSeries(prec, o2)));
      meas = h("div", {}, meas, h("div", { class: "legend" }, ...legend));
    }
    return h("div", { class: "card mt" }, h("h3", {}, L("精度随时间", "Precision over time")),
      h("div", { class: "small muted" }, txt({ text_zh: p.note_zh, text_en: p.note_en })),
      live, nowRow, h("details", { class: "mt6", open: m.available ? "open" : null }, h("summary", {}, L("评估实测曲线", "measured in evaluation")), meas));
  }

  // ------------------------------------------------------------ groups
  async function groupsPage() {
    const g = await API3.groups();
    const tb = h("table", {}, h("tr", {}, h("th", {}, L("群组", "Group")), h("th", { class: "num" }, L("成员", "Members")),
      h("th", {}, L("网段覆盖", "Prefix covers")), h("th", {}, L("使用的系统", "Systems")), h("th", {}, L("特征动作", "Top actions")),
      h("th", { class: "num" }, L("内聚度", "Cohesion"))));
    g.groups.forEach((x) => tb.appendChild(h("tr", { class: "clickable", onclick: () => go(grpHref(x.id)) },
      h("td", {}, h("b", {}, x.name), h("div", { class: "small muted mono" }, x.id + (x.name_source ? ` · ${x.name_source}` : ""))),
      h("td", { class: "num" }, String(x.n)),
      h("td", { class: "small mono" }, (x.covers || []).slice(0, 4).join(", ") || "—"),
      h("td", { class: "small" }, Object.entries(x.systems || {}).map(([s, v]) => `${s} ${pctS(v)}`).join(", ")),
      h("td", { class: "small" }, (x.labels || []).map((l) => l.label).join("；") || "—"),
      h("td", { class: "num" }, num(x.cohesion, 2)))));
    return page("groups", STATE.ppSystem, L("群组 / 用户视图", "Groups / users"),
      h("div", { class: "chips" }, chip(L("群组", "groups"), g.n_groups), chip(L("已归组 IP", "grouped IPs"), g.n_grouped_ips),
        ...Object.entries(g.modes || {}).map(([s, m]) => chip(s, L(WHO_ZH[m] || m || "—", m || "—")))),
      h("div", { class: "card tablewrap mt" }, g.groups.length ? tb : empty(L("尚未形成行为群组（每日聚类）", "no group yet (daily clustering)"))));
  }
  async function groupPage(params) {
    const gid = params[0];
    const [v, f] = await Promise.all([API3.groupView(gid), API3.groupFacets(gid).catch(() => null)]);
    const nameIn = h("input", { class: "search small", placeholder: L("为该组命名，例如 综合部", "name this group, e.g. 综合部"), value: v.name_source === "auto" ? "" : v.name });
    const nameBox = h("div", { class: "rowline wrap mt6" }, nameIn, h("button", { class: "mini", onclick: async () => {
      try { const r = await API3.groupName(gid, nameIn.value); toast(txt({ text_zh: r.note_zh, text_en: r.note_en })); }
      catch (e) { toast(String(e.message || e), false); } } }, L("保存名称", "save name")));
    const hdr = h("div", { class: "card" }, h("div", { class: "narrative" }, txt(v.header) || v.name),
      h("div", { class: "small muted" }, L("成员", "members"), ` (${v.members.length})`),
      h("div", { class: "chips" }, ...v.members.slice(0, 40).map((m) => h("span", { class: "chip small mono" }, m))),
      v.covers && v.covers.length ? h("div", { class: "small muted mt6" }, L("网段覆盖：", "prefix covers: "), v.covers.join(", ")) : "",
      nameBox);
    const sysCards = v.systems.map((s) => h("div", { class: "card mt" },
      h("div", { class: "rowline" }, h("h3", {}, link(sysHref(s.system), s.system)),
        h("span", { class: "small muted" }, L(`占该组行为 ${pctS(s.share)}`, `${pctS(s.share)} of the group's activity`))),
      s.actions.length ? h("div", {}, ...s.actions.map((a) => h("div", { class: "pp-action" },
        h("div", { class: "rowline" }, h("b", { class: "mono" }, a.route_text), a.write ? h("span", { class: "badge sev-medium" }, L("写", "write")) : ""),
        ...a.statements.map((st) => statementCard(st, s.system, { compact: true })))))
        : empty(L("该系统中该组尚无已确认模式", "no confirmed pattern of the group here yet"))));
    const neg = v.negative.length ? h("div", { class: "card mt" }, h("h3", {}, L("从未发生（据此判定越界）", "Never happens (basis of who violations)")),
      ...v.negative.map((st) => statementCard(st, null, { compact: true }))) : "";
    return page("groups", STATE.ppSystem, L(`群组视图 · ${v.name}`, `Group · ${v.name}`), hdr, neg, ...sysCards,
      f ? facetCard(f, L("该组的多维刻面", "Facets of the group")) : "");
  }

  // ------------------------------------------------------------ IP view
  async function ipPage(params) {
    const [sys, ip] = params;
    STATE.ppSystem = sys;
    const v = await API3.ipView(sys, ip);
    const g = v.group;
    const hdr = h("div", { class: "card" }, h("div", { class: "rowline wrap" }, h("h3", { class: "mono" }, ip),
      chip("/24", v.prefix24), g ? chip(L("所属群组", "group"), "") : chip(L("所属群组", "group"), L("未归组", "ungrouped")),
      g ? link(grpHref(g.id), `${g.name} (${g.n})`) : ""),
      h("div", { class: "small muted" }, L("用户即 IP 或 IP 类：该 IP 继承其群组的模式，只有它与群组不同的地方才单独成为例外。",
        "A user is an IP or an IP class: the IP inherits its group's patterns; only where it differs does it earn its own exception.")));
    const sec = (title, arr, emptyMsg) => h("div", { class: "card mt" }, h("h3", {}, title),
      arr.length ? h("div", {}, ...arr.map((s) => statementCard(s, sys, { compact: true }))) : empty(emptyMsg));
    const binds = h("div", { class: "card mt" }, h("h3", {}, L("绑定值", "Bound values")),
      v.bindings.length ? h("table", {}, h("tr", {}, h("th", {}, L("动作", "Action")), h("th", {}, L("字段", "Pair")), h("th", {}, L("取值", "Value")), h("th", { class: "num" }, "LB"), h("th", { class: "num" }, "n")),
        ...v.bindings.map((b) => h("tr", {}, h("td", { class: "mono small" }, routeShort(b.route)), h("td", { class: "mono small" }, b.pair),
          h("td", { class: "mono" }, b.value != null ? String(b.value) : (b.set || []).join("|")), h("td", { class: "num" }, num(b.LB, 2)), h("td", { class: "num" }, num(b.n, 0)))))
        : empty(L("没有以该 IP 为来源的已确认绑定", "no confirmed binding from this IP")));
    return page("system", sys, L(`IP 视图 · ${ip} @ ${sys}`, `IP · ${ip} @ ${sys}`), hdr,
      sec(L("点名该 IP 的系统模式", "System patterns naming this IP"), v.statements, L("暂无", "none yet")),
      sec(L("继承的群组模式（本系统）", "Inherited group patterns (this system)"), v.inherited_here, L("暂无", "none yet")),
      sec(L("IP 例外", "IP exceptions"), v.exceptions, L("该 IP 没有偏离群组的例外模式", "no exception: the IP behaves like its group")),
      binds, violationsCard(v.violations, L("该 IP 的违例", "Violations of this IP")));
  }

  // ------------------------------------------------------------ lattice
  async function latticePage(params, q) {
    await ensureSystems();
    const sys = pickSystem(params);
    const kind = q.kind || 0;
    const lat = await API3.lattice(sys, { kind, root: q.root, depth: q.depth || 5 });
    const byId = {}; lat.nodes.forEach((n) => { byId[n.id] = n; });
    const detail = h("div", { class: "card pp-detail" }, empty(L("点击节点查看详情", "click a node for its detail")));
    const show = async (n) => { detail.innerHTML = ""; detail.appendChild(h("div", { class: "muted small" }, L("加载中…", "loading…")));
      try { const d = await API3.pattern(n.pattern_id); detail.innerHTML = ""; detail.appendChild(patternBody(d, sys, true)); }
      catch (e) { detail.innerHTML = ""; detail.appendChild(empty(String(e.message || e))); } };
    const maxMass = Math.max(1e-9, ...lat.nodes.map((n) => n.mass || 0));
    function li(n) {
      const kids = (n.children || []).map((c) => byId[c]).filter(Boolean);
      const row = h("div", { class: "pp-node", onclick: (e) => { e.stopPropagation(); show(n); } },
        stateBadge(n.state), " ", n.is_exc ? h("span", { class: "tag" }, "exc") : "",
        h("span", { class: "mono small" }, " " + n.label), n.route_text && n.label.indexOf("route") < 0 ? h("span", { class: "small muted" }, " · " + n.route_text) : "",
        h("span", { class: "pp-massbar", title: L("权重", "mass") }, h("i", { style: `width:${Math.max(2, 100 * (n.mass || 0) / maxMass)}%` })),
        h("span", { class: "small muted" }, ` n_c ${num(n.n_c, 1)} · ${n.distinct_ips ?? "—"} IP`),
        n.split ? h("span", { class: "small muted" }, L(` · 按 ${n.split.attr}@${n.split.level_name} 细分`, ` · split on ${n.split.attr}@${n.split.level_name}`)) : "",
        !n.expanded && n.children.length ? link(latHref(sys, n.id), L(` 展开 ${n.children.length} →`, ` expand ${n.children.length} →`)) : "");
      const el = h("li", {}, row);
      if (kids.length) el.appendChild(h("ul", {}, ...kids.map(li)));
      return el;
    }
    const root = byId[lat.root];
    const kinds = h("span", { class: "chips" }, ...(lat.kinds || []).map((k) => h("a", { class: "chip" + (String(k) === String(kind) ? " pp-on" : ""),
      href: `#/pp/lattice/${enc(sys)}?kind=${k}` }, k === 0 ? L("事务事件", "transactions") : L("窗口事件", "windows"))));
    const counts = h("div", { class: "chips" }, chip(L("节点", "nodes"), lat.n_nodes), chip(L("预算上限", "n_max"), (lat.budget || {}).n_max),
      chip(L("档位", "tier"), (lat.budget || {}).tier), chip(L("IP 例外", "exceptions"), lat.n_exceptions), chip(L("已退役", "retired"), lat.n_retired),
      ...Object.entries(lat.counts || {}).filter(([, v]) => v).map(([k, v]) => chip(L(STATE_ZH[k] || k, k), v)),
      lat.truncated ? h("span", { class: "badge sev-medium" }, L("已截断", "truncated")) : "",
      lat.root_depth ? link(latHref(sys), L("回到根", "back to root")) : "");
    const tree = h("ul", { class: "tree pp-tree" }, root ? li(root) : h("li", {}, empty()));
    return page("lattice", sys, L(`模式格 · ${sys}`, `Pattern lattice · ${sys}`),
      h("div", { class: "small muted" }, L("从系统根节点开始，模式树只在证据表明能更好地预测行为时才细分（按路由、群组、网段、时间、内容）；节点按证据确认、演化、陈旧与退役。",
        "From the system root the tree only specialises where evidence shows a better prediction of behaviour (route, group, prefix, time, content); nodes confirm, evolve, go stale and retire on evidence.")),
      h("div", { class: "rowline wrap mt6" }, kinds, counts),
      h("div", { class: "grid g2 mt" }, h("div", { class: "card scroll pp-treewrap" }, tree), detail));
  }

  // ------------------------------------------------------------ pattern detail
  function hist96(hd) {
    if (!hd) return empty();
    const rows = [hd.workday || [], hd.nonworkday || []];
    const mx = Math.max(1e-9, ...rows.flat());
    const grid = rows.map((r) => r.map((v) => v / mx));
    return Charts.heatmap(grid, { rowLabels: [L("工作日", "wd"), L("休息日", "nwd")], colLabels: [...Array(96).keys()].map((i) => i % 4 === 0 ? String(i / 4) : ""),
      colEvery: 8, cell: 7, cellH: 16, labelW: 44 });
  }
  function patternBody(d, sys, compact) {
    if (!d.alive) return h("div", {}, h("div", { class: "rowline" }, stateBadge(d.state), h("span", { class: "mono small" }, d.pattern_id)),
      h("pre", {}, JSON.stringify(d.record, null, 1).slice(0, 2000)), lineageTable(d.lineage), lifecycleTable(d.lifecycle));
    const b = d.brief;
    sys = sys || (d.systems || [])[0];
    const path = h("div", { class: "pp-path small" }, ...d.path.map((p, i) => h("span", {}, i ? " › " : "",
      link(latHref(sys, p.id), p.label))));
    const head = h("div", {}, h("div", { class: "rowline wrap" }, stateBadge(b.state), h("b", { class: "mono small" }, b.pattern_id),
      d.stale_id ? h("span", { class: "badge sev-medium" }, L("请求的版本已过期", "requested version superseded")) : "",
      compact ? link(patHref(b.pattern_id), L("全页 →", "full page →")) : ""), path);
    const facts = h("div", { class: "kvgrid" }, kvRow(L("上下文", "context"), b.context), kvRow(L("动作", "action"), b.route_text || "—"),
      kvRow(L("深度", "depth"), b.depth), kvRow(L("权重", "mass"), num(b.mass, 2)), kvRow("n_c", num(b.n_c, 1)),
      kvRow(L("不同 IP", "distinct IPs"), b.distinct_ips), kvRow(L("活跃天数", "days"), b.days),
      kvRow(L("版本", "version"), `v${b.version}.${b.cver}`), kvRow(L("首次", "first"), fmtShort(b.first_seen)), kvRow(L("最近", "last"), fmtShort(b.last_seen)),
      kvRow(L("细分", "split"), b.split ? `${b.split.attr}@${b.split.level_name} (${b.split.n_groups})` : "—"));
    const kids = d.children.length ? h("div", { class: "chips" }, ...d.children.map((c) => h("a", { class: "chip", href: latHref(sys, c.id) },
      stateBadge(c.state), " ", c.label))) : empty(L("叶节点", "leaf"));
    const exc = d.exceptions.length ? h("div", { class: "chips" }, ...d.exceptions.map((x) => h("span", { class: "chip mono" }, ipLink(sys, x.ip), " ", stateBadge(x.state)))) : empty(L("无", "none"));
    const who = h("div", {}, h("div", { class: "small" }, txt(d.who)), whoBlock(d.who.evidence, sys));
    const cons = h("details", {}, h("summary", {}, L("拟合的约束（原始）", "fitted constraints (raw)")),
      ...Object.entries(d.constraints).filter(([, v]) => v && (Array.isArray(v) ? v.length : true)).map(([k, v]) =>
        h("div", {}, h("div", { class: "small muted" }, k), h("pre", { class: "small" }, JSON.stringify(v, null, 1).slice(0, 3000)))));
    return h("div", {}, head,
      d.statement ? statementCard(d.statement, sys) : h("div", { class: "small muted mt6" }, L("该节点尚无画像语句（候选或未覆盖动作）", "no statement for this node (candidate, or not an action node)")),
      ...(d.parts || []).map((s) => statementCard(s, sys, { compact: true })),
      h("h4", {}, L("事实", "Facts")), facts,
      h("h4", {}, L("谁", "Who")), who,
      h("h4", {}, L("时间分布（15 分钟）", "Arrival profile (15 min)")), hist96(d.when_hist96),
      h("h4", {}, L("子节点", "Children")), kids,
      h("h4", {}, L("IP 例外", "IP exceptions")), exc, cons,
      h("h4", {}, L("演化与漂移历史", "Lifecycle and drift history")), lifecycleTable(d.lifecycle),
      h("h4", {}, L("谱系", "Lineage")), lineageTable(d.lineage),
      violationsCard(d.violations, L("该模式判定的违例", "Violations judged by this pattern")));
  }
  function lifecycleTable(rows) {
    if (!rows || !rows.length) return empty(L("无事件", "no event"));
    return h("div", { class: "tablewrap" }, h("table", {}, h("tr", {}, h("th", {}, L("时间", "Time")), h("th", {}, L("事件", "Event")), h("th", {}, L("说明", "Description"))),
      ...rows.map((r) => h("tr", {}, h("td", { class: "small nowrap" }, fmtShort(r.ts)), h("td", {}, kindBadge(r.kind)), h("td", { class: "small" }, r.description)))));
  }
  function lineageTable(rows) {
    if (!rows || !rows.length) return empty(L("无谱系记录", "no lineage"));
    return h("div", { class: "tablewrap scroll" }, h("table", {}, h("tr", {}, h("th", {}, L("时间", "Time")), h("th", {}, L("操作", "Op")), h("th", {}, L("节点", "Node")), h("th", {}, L("细节", "Detail"))),
      ...rows.slice().reverse().map((r) => h("tr", {}, h("td", { class: "small nowrap" }, fmtShort(r.ts)), h("td", {}, h("span", { class: "tag" }, r.op)),
        h("td", { class: "small mono" }, `${r.node} [${r.parents.join(",")}] → [${r.children.join(",")}]`),
        h("td", { class: "small mono" }, typeof r.detail === "object" && r.detail ? JSON.stringify(r.detail).slice(0, 160) : String(r.detail ?? ""))))));
  }
  async function patternPage(params) {
    const pid = params.join("/");
    const d = await API3.pattern(pid);
    const sys = (d.systems || [])[0] || d.tree_key;
    return page("lattice", sys, L("模式详情", "Pattern"), h("div", { class: "card" }, patternBody(d, sys, false)));
  }

  // ------------------------------------------------------------ violations
  function violationsCard(rows, title) {
    return h("div", { class: "card mt" }, h("h3", {}, title), violationsTable(rows || []));
  }
  function violationsTable(rows) {
    if (!rows.length) return empty(L("暂无违例", "no violation"));
    const tb = h("table", {}, h("tr", {}, h("th", {}, L("时间", "Time")), h("th", {}, L("来源", "Source")), h("th", {}, L("类型", "Type")),
      h("th", {}, L("严重度", "Severity")), h("th", {}, L("原因", "Reason")), h("th", {}, "p_day"), h("th", {})));
    rows.forEach((v) => tb.appendChild(h("tr", {},
      h("td", { class: "small nowrap" }, fmtShort(v.ts)),
      h("td", { class: "mono small" }, link(ipHref(v.system, v.entity), v.entity), h("div", { class: "muted" }, v.system)),
      h("td", {}, h("span", { class: "badge sev-info", title: L(v.explain_zh, v.explain_en) }, L(v.type_zh, v.type_en))),
      h("td", {}, sevBadge(v.severity)),
      h("td", { class: "small" }, L(v.reason_zh, v.reason_en || v.reason_zh),
        h("div", { class: "chips" }, ...(v.flags || []).map((f) => h("span", { class: "chip small", title: f.flag }, L(f.zh, f.en))))),
      h("td", { class: "num small" }, v.p_day == null ? "—" : Charts.fmtNum(v.p_day)),
      h("td", { class: "links small" }, v.pattern_id ? link(patHref(v.pattern_id), L("模式", "pattern")) : "",
        v.incident_id ? link(`#/incident/${enc(v.incident_id)}`, L("事件", "incident")) : ""))));
    return h("div", { class: "tablewrap" }, tb);
  }
  async function violationsPage(_params, q) {
    await ensureSystems();
    const f = { system: q.system, type: q.type, severity: q.severity, entity: q.entity, limit: 300 };
    const v = await API3.violations(f);
    const setQ = (k, val) => { const n = Object.assign({}, q, { [k]: val || undefined });
      go("#/pp/violations" + qs(n)); };
    const sel = (k, opts, cur) => h("select", { class: "search small", onchange: (e) => setQ(k, e.target.value) },
      ...opts.map(([val, label]) => h("option", { value: val, selected: String(val) === String(cur || "") ? "selected" : null }, label)));
    const filters = h("div", { class: "rowline wrap" },
      sel("system", [["", L("全部系统", "all systems")], ...(STATE.ppSystems || []).map((s) => [s, s])], q.system),
      sel("type", [["", L("全部类型", "all types")], ...Object.entries(v.types).map(([k, x]) => [k, L(x.zh, x.en)])], q.type),
      sel("severity", [["", L("全部严重度", "any severity")], ["low", "≥ low"], ["medium", "≥ medium"], ["high", "≥ high"]], q.severity));
    const counts = h("div", { class: "grid g3 mt" },
      card(L("按类型", "By type"), Charts.bars(Object.entries(v.counts.type).map(([k, n]) => ({ label: L((v.types[k] || {}).zh || k, (v.types[k] || {}).en || k), value: n })))),
      card(L("按原因标记", "By flag"), Charts.bars(Object.entries(v.counts.flag).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([k, n]) => ({ label: k, value: n })), { labelW: 150 })),
      card(L("类型说明", "Types"), h("div", {}, ...Object.entries(v.types).map(([k, x]) => h("div", { class: "small" }, h("b", {}, L(x.zh, x.en)), "：", L(x.explain_zh, x.explain_en))))));
    return page("violations", q.system || STATE.ppSystem, L(`模式违例（${v.n}）`, `Pattern violations (${v.n})`), filters, counts,
      h("div", { class: "card mt" }, violationsTable(v.violations)));
  }

  // ------------------------------------------------------------ facets
  function facetNode(n, depth) {
    const items = (n.items || []).map((it) => h("li", { class: "small" }, txt(it),
      it.confidence != null ? h("span", { class: "muted" }, ` · ${L("置信", "conf.")} ${num(it.confidence, 2)}`) : "",
      it.source ? h("span", { class: "muted mono" }, ` · ${it.source}`) : ""));
    return h("div", { class: "pp-facet d" + depth }, h("div", { class: "rowline" }, h("b", {}, L(n.name_zh, n.name_en)),
      n.confidence != null ? h("span", { class: "small muted" }, ` ${L("置信", "conf.")} ${num(n.confidence, 2)}`) : ""),
      items.length ? h("ul", { class: "difflist" }, ...items) : "", ...(n.children || []).map((c) => facetNode(c, depth + 1)));
  }
  function facetCard(f, title) {
    return h("div", { class: "card mt" }, h("h3", {}, title),
      h("div", { class: "chips" }, chip(L("刻面", "facets"), f.n_facets), chip(L("条目", "items"), f.n_items), chip(L("来源", "source"), f.source),
        chip(L("已声明刻面", "declared"), (f.registry || []).length)),
      h("div", { class: "grid g2 mt6" }, ...(f.facets || []).map((n) => facetNode(n, 0))));
  }
  async function facetsPage(params, q) {
    await ensureSystems();
    const sys = pickSystem(params);
    const f = q.ip ? await API3.ipFacets(sys, q.ip) : await API3.facets(sys);
    const reg = h("details", { class: "mt" }, h("summary", {}, L("刻面注册表（运行时由引擎声明）", "facet registry (declared at runtime by engines)")),
      h("div", { class: "tablewrap" }, h("table", {}, h("tr", {}, h("th", {}, "id"), h("th", {}, L("名称", "name")), h("th", {}, L("生产者", "producer")), h("th", {}, L("来源模型", "sources")), h("th", {}, L("适用条件", "applicable"))),
        ...(f.registry || []).map((d) => h("tr", {}, h("td", { class: "mono small" }, d.id), h("td", {}, L(d.name_zh, d.name_en)),
          h("td", { class: "mono small" }, d.producer || "—"), h("td", { class: "mono small" }, (d.sources || []).join(", ")), h("td", { class: "small" }, String(d.applicable ?? "—")))))));
    return page("facets", sys, L(`多维刻面 · ${q.ip ? q.ip + " @ " : ""}${sys}`, `Facets · ${q.ip ? q.ip + " @ " : ""}${sys}`),
      h("div", { class: "small muted" }, L("像看一个人有外观、生物学特征、社会属性一样：功能、时间节律、空间/网络、内容、序列、关系、技术、量、身份、风险等多个刻面由不同引擎叠加而成，只显示适用且有内容的刻面。",
        "Like reading a person by appearance, biology and social attributes: functional, temporal, spatial, content, sequential, relational, technical, volume, identity and risk facets are stacked from different engines; only applicable, non-empty facets are shown.")),
      facetCard(f, L("刻面组成", "Facet composition")), reg);
  }

  // ------------------------------------------------------------ strategy
  async function strategyPage(params) {
    await ensureSystems();
    const sys = pickSystem(params);
    const s = await API3.strategy(sys);
    const ch = s.characteristics || {};
    const flat = [];
    Object.entries(ch).forEach(([k, v]) => { if (v != null && typeof v === "object" && !Array.isArray(v)) {
      Object.entries(v).forEach(([k2, v2]) => { if (typeof v2 !== "object") flat.push([`${k}.${k2}`, v2]); }); }
      else if (!Array.isArray(v)) flat.push([k, v]); });
    const chars = h("div", { class: "kvgrid" }, ...flat.slice(0, 60).map(([k, v]) => kvRow(k, typeof v === "number" ? Charts.fmtNum(v) : String(v))));
    const eng = h("table", {}, h("tr", {}, h("th", {}, L("维度", "Dim")), h("th", {}, L("引擎", "Engine")), h("th", {}, L("作用", "Role")), h("th", {}, L("选择", "Choice")), h("th", {}, L("理由", "Reason"))),
      ...(s.engines || []).map((e) => h("tr", {}, h("td", { class: "mono" }, e.dim), h("td", { class: "mono small" }, e.engine), h("td", {}, L(e.name_zh, e.name_en)),
        h("td", {}, h("span", { class: "badge " + (e.on ? "sev-low" : "sev-info") }, e.value)), h("td", { class: "small muted" }, e.reason || "—"))));
    const chosen = h("div", { class: "chips" }, ...Object.entries(s.chosen || {}).filter(([k]) => !/^p\d\d$/.test(k)).map(([k, v]) => chip(k, v)));
    const hints = (s.hints || []).map((x) => h("div", { class: "small", style: "color:var(--medium)" }, "⚠ ", txt(x)));
    const hist = (s.history || []).length ? h("div", { class: "tablewrap" }, genericRows(s.history)) : empty(L("尚无策略切换", "no strategy change yet"));
    return page("strategy", sys, L(`系统特征与策略 · ${sys}`, `Characterisation & strategy · ${sys}`),
      h("div", { class: "small muted" }, L("每个业务系统先被测量（人口规模、IP 流动性、载荷可见性、会话可识别性、NAT 迹象……），再按前置条件和实测的预测增益（扣除 CPU 成本）自动选择启用哪些引擎、按什么粒度看“谁”。",
        "Each system is measured first (population, churn, payload visibility, session identifiability, NAT evidence, …); engines and the who granularity are then chosen from preconditions and the measured predictive gain net of CPU cost.")),
      h("div", { class: "card mt" }, h("div", { class: "rowline wrap" }, h("h3", {}, L("谁的粒度", "Who granularity")),
        h("span", { class: "tag" }, L(s.who.zh || "—", s.who.en || "—")), s.family && s.family.id ? chip(L("系统族", "family"), `${s.family.id} (${(s.family.members || []).join(", ")})`) : "",
        chip(L("版本", "version"), s.version ?? "—"), chip(L("评估日", "day"), s.day ?? "—")), chosen, ...hints),
      h("div", { class: "grid g2 mt" }, card(L("启用的引擎", "Engines switched"), h("div", { class: "tablewrap" }, eng)), card(L("测得的系统特征", "Measured characteristics"), chars)),
      card(L("策略历史", "Strategy history"), hist));
  }
  function genericRows(rows) {
    const cols = Array.from(new Set(rows.flatMap((r) => Object.keys(r || {})))).slice(0, 8);
    return h("table", {}, h("tr", {}, ...cols.map((c) => h("th", {}, c))),
      ...rows.slice(-50).reverse().map((r) => h("tr", {}, ...cols.map((c) => { const v = (r || {})[c];
        return h("td", { class: "small" }, v == null ? "—" : typeof v === "object" ? JSON.stringify(v).slice(0, 80) : typeof v === "number" ? Charts.fmtNum(v) : String(v)); }))));
  }

  // ------------------------------------------------------------ attributes
  async function attributesPage(params) {
    await ensureSystems();
    const sys = pickSystem(params);
    const a = await API3.attributes(sys);
    const ROLE_ZH = { split: "细分", target: "目标", invariant: "不变式", shape: "形态", redundant: "冗余", dropped: "丢弃", probe: "探查中" };
    const search = h("input", { class: "search", placeholder: L("筛选属性…", "filter…"), oninput: (e) => {
      const ql = e.target.value.toLowerCase(); document.querySelectorAll(".pp-attr").forEach((r) => r.classList.toggle("hidden", !!ql && !r.dataset.txt.includes(ql))); } });
    const tb = h("table", {}, h("tr", {}, h("th", {}, L("属性", "Attribute")), h("th", {}, L("类型", "Type")), h("th", {}, L("角色", "Role")),
      h("th", { class: "num" }, L("覆盖", "Coverage")), h("th", { class: "num" }, L("基数", "Card.")), h("th", { class: "num" }, L("熵", "Entropy")),
      h("th", { class: "num" }, L("稳定性", "Stability")), h("th", {}, L("主要取值", "Top values")), h("th", {}, L("首次", "First"))));
    a.attributes.forEach((r) => tb.appendChild(h("tr", { class: "pp-attr", "data-txt": `${r.name} ${r.type} ${r.role}`.toLowerCase() },
      h("td", { class: "mono small" }, r.name, r.state !== "active" ? h("span", { class: "badge sev-medium" }, r.state) : ""),
      h("td", {}, h("span", { class: "tag" }, r.type)), h("td", {}, L(ROLE_ZH[r.role] || r.role, r.role)),
      h("td", { class: "num" }, pctS(Math.min(1, r.coverage || 0))), h("td", { class: "num" }, Charts.fmtNum(r.card)),
      h("td", { class: "num" }, num(r.entropy, 2)), h("td", { class: "num" }, num(r.stability, 2)),
      h("td", { class: "small mono ellip" }, (r.top || []).slice(0, 3).map((x) => `${x[0]} ${pctS(x[1])}`).join(" · ")),
      h("td", { class: "small nowrap" }, fmtShort(r.first_seen)))));
    return page("attributes", sys, L(`指标/属性登记 · ${sys}`, `Attribute registry · ${sys}`),
      h("div", { class: "small muted" }, L("属性不写死：新出现的指标首个周期即登记、自动推断类型，并由数据决定其角色（细分、目标、不变式、冗余或丢弃）。",
        "Nothing is hard-coded: a new metric is registered in its first tick, typed automatically, and its role (split, target, invariant, redundant or dropped) is decided by the data.")),
      h("div", { class: "rowline wrap mt6" }, chip(L("属性", "attributes"), `${a.n}${a.a_max ? " / " + a.a_max : ""}`),
        ...Object.entries(a.by_role || {}).map(([k, v]) => chip(L(ROLE_ZH[k] || k, k), v)),
        ...Object.entries(a.by_type || {}).map(([k, v]) => chip(k, v)), search),
      h("div", { class: "card tablewrap mt" }, a.attributes.length ? tb : empty()));
  }

  // ------------------------------------------------------------ budget
  async function budgetPage() {
    const b = await API3.budget();
    const trees = Object.entries(b.trees || {});
    const tb = h("table", {}, h("tr", {}, h("th", {}, L("模式树", "Tree")), h("th", {}, L("档位", "Tier")), h("th", {}, L("需求", "Demand")),
      h("th", { class: "num" }, "n_max"), h("th", { class: "num" }, L("事件/天", "events/day")), h("th", { class: "num" }, L("内存", "Memory")),
      h("th", { class: "num" }, L("权重", "Weight")), h("th", {}, L("空闲", "Idle"))),
      ...trees.map(([k, c]) => h("tr", {}, h("td", {}, link(sysHref(k), k)), h("td", {}, h("span", { class: "tag" }, c.tier || "—")), h("td", {}, c.demand || "—"),
        h("td", { class: "num" }, String(c.n_max ?? "—")), h("td", { class: "num" }, Charts.fmtNum(c.ev_day)),
        h("td", { class: "num" }, fmtBytes(c.bytes ?? (b.ptree_bytes || {})[k])), h("td", { class: "num" }, Charts.fmtNum(c.weight)), h("td", {}, c.idle ? "✓" : ""))));
    const sysRows = h("table", {}, h("tr", {}, h("th", {}, L("系统", "System")), h("th", { class: "num" }, L("活跃 IP 集", "active set")),
      h("th", { class: "num" }, L("已赢得 IP", "earned")), h("th", { class: "num" }, "E_max")),
      ...Object.entries(b.systems || {}).map(([s, r]) => h("tr", {}, h("td", {}, s), h("td", { class: "num" }, String(r.n_active ?? "—")),
        h("td", { class: "num" }, String(r.n_earned ?? "—")), h("td", { class: "num" }, String(r.e_max ?? "—")))));
    // one chart per measure (never two y-scales): P-core tree bytes per system
    const ser = Object.entries(b.series || {}).map(([s, rows], i) => ({ name: s, ts: rows.map((r) => r.ts), values: rows.map((r) => r.tree_bytes), color: Charts.SERIES[i % Charts.SERIES.length] }));
    const ser2 = Object.entries(b.series || {}).map(([s, rows], i) => ({ name: s, ts: rows.map((r) => r.ts), values: rows.map((r) => r.pcore_ms_h), color: Charts.SERIES[i % Charts.SERIES.length] }));
    const legend = h("div", { class: "legend" }, ...ser.map((s) => h("span", {}, h("i", { class: "dot", style: `background:${s.color}` }), s.name)));
    const eng = Charts.bars((b.engines || []).filter((e) => e.duration_ms != null).sort((x, y) => y.duration_ms - x.duration_ms)
      .map((e) => ({ label: e.engine.replace("behavior.", ""), value: e.duration_ms })), { labelW: 140 });
    const lad = b.ladder || {};
    return page("budget", STATE.ppSystem, L("资源预算（P15）", "Resource budget (P15)"),
      h("div", { class: "small muted" }, L("内存与 CPU 按系统价值与活跃度从全局预算中分配；空闲的模式树降档并最终检查点落盘；超预算时按降级阶梯逐级收缩，而不是遍历全部 IP。",
        "Memory and CPU are allocated from a global budget by each tree's value and activity; idle trees shrink and are checkpointed; over budget, a degradation ladder steps in - never an enumeration of every IP.")),
      h("div", { class: "chips mt6" }, chip(L("降级阶梯", "ladder step"), lad.step ?? 0), chip(L("停止探索", "no explore"), lad.no_explore ? "✓" : "—"),
        chip(L("读时刷新", "refresh on read"), lad.refresh_on_read ? "✓" : "—"), chip(L("跳过载荷解析", "skip body parsing"), lad.skip_body_parsing ? "✓" : "—"),
        ...Object.entries(b.usage || {}).filter(([, v]) => typeof v !== "object").slice(0, 8).map(([k, v]) => chip(k, typeof v === "number" ? Charts.fmtNum(v) : v))),
      h("div", { class: "card tablewrap mt" }, h("h3", {}, L("模式树档位与上限", "Trees: tier and caps")), trees.length ? tb : empty()),
      h("div", { class: "grid g2 mt" },
        card(L("模式树内存（字节）", "Tree memory (bytes)"), Charts.timeSeries(ser, { h: 170, fmtT: (x) => fmtShort(x).slice(0, 5), fmtFull: fmtShort }), legend),
        card(L("P-core CPU（毫秒/小时，按份额）", "P-core CPU (ms / h, by share)"), Charts.timeSeries(ser2, { h: 170, fmtT: (x) => fmtShort(x).slice(0, 5), fmtFull: fmtShort }), legend.cloneNode(true))),
      h("div", { class: "grid g2 mt" }, card(L("各引擎最近一次耗时（毫秒）", "Engine time, last run (ms)"), eng),
        card(L("活跃 / 已赢得 IP 集（有界）", "Active / earned IP sets (bounded)"), h("div", { class: "tablewrap" }, sysRows))));
  }

  // ------------------------------------------------------------ router
  ROUTES.pp = async (params, q) => {
    const sub = params[0] || "";
    const rest = params.slice(1);
    switch (sub) {
      case "": return overview();
      case "system": return systemPage(rest, q || {});
      case "groups": return groupsPage();
      case "group": return groupPage(rest);
      case "ip": return ipPage(rest);
      case "lattice": return latticePage(rest, q || {});
      case "pattern": return patternPage(rest);
      case "violations": return violationsPage(rest, q || {});
      case "facets": return facetsPage(rest, q || {});
      case "strategy": return strategyPage(rest);
      case "attributes": return attributesPage(rest);
      case "budget": return budgetPage();
      default: return overview();
    }
  };
})();
