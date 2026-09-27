// lib-3 v2 pages (docs/lib3/api_ui.md): entity page, class pages, incident
// queue with feedback, system view with detector health, eval report.
// Everything here is defensive: any field the backend returns as null / []
// (a model not fitted yet, B29 explain not registered) renders as "—".
(() => {
  const esc = Charts.esc;
  const DOW_ZH = ["一","二","三","四","五","六","日"];
  const DOW_EN = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];
  const FEAT_ZH = {http_requests:"请求",flows:"连接",bytes_up:"上行字节",bytes_down:"下行字节"};
  const DAYTYPE = {workday:["工作日","workday"],nonworkday:["休息日","non-workday"]};
  const FAMILY_ZH = {intensity:"强度",shape:"形态",peer:"同类",temporal:"时间",categorical:"类别",breadth:"广度",
    exfil:"外传",sequence:"序列",identity:"身份",change:"变化",c2:"C2",xsys:"跨系统"};
  const VERDICTS = [["tp","确认威胁","true positive","sev-critical"],["fp","误报","false positive","sev-low"],
    ["expected_change","预期变更","expected change","sev-info"],["benign_known","已知良性","benign (known)","sev-medium"]];
  const SCOPES = [["this","仅此条","this"],["pattern","同模式","pattern"],["entity","该实体","entity"],["class","该类","class"],["system","全系统","system"]];
  const TTLS = [["", "永久","forever"],["3600","1 小时","1 h"],["86400","1 天","1 d"],["604800","7 天","7 d"],["2592000","30 天","30 d"]];
  const lbl = (zh, en) => t(zh, en);
  const settled = async (p) => { try { return await p; } catch (e) { return null; } };
  const num = (v, d=2) => v==null||!isFinite(v) ? "—" : Number(v).toFixed(d);
  const share = v => v==null ? "—" : (v*100).toFixed(v<0.01?2:1)+"%";

  function chips(items, fmt){ const d=h("div",{class:"chips"});
    (items||[]).forEach(it=>d.appendChild(fmt(it))); if(!(items||[]).length) d.appendChild(empty()); return d; }
  function chip(label, val, title){ return h("span",{class:"chip",title:title||null}, label, val!=null&&val!==""?h("b",{}," "+val):""); }
  function hbar(label, value, max, color, note, href){
    const w = max>0 ? Math.max(1, Math.min(100, 100*value/max)) : 0;
    return h("div",{class:"hbar"}, h("span",{class:"hl"}, href?link(href,label):label),
      h("span",{class:"hb"}, h("i",{style:`width:${w}%;background:${color||Charts.SERIES[0]}`})),
      h("span",{class:"hv"}, note!=null?note:num(value)));
  }
  function genericTable(rows, maxCols=10){
    if(!rows || !rows.length) return empty();
    const cols = Object.keys(rows[0]).filter(k=>typeof rows[0][k]!=="object" || rows[0][k]===null).slice(0,maxCols);
    const tb=h("table",{}, h("tr",{}, ...cols.map(c=>h("th",{},c))));
    rows.forEach(r=>tb.appendChild(h("tr",{}, ...cols.map(c=>{ const v=r[c];
      return h("td",{class:typeof v==="number"?"num":""}, v==null?"—":typeof v==="number"?fmtN(v):typeof v==="boolean"?(v?"✓":"✗"):String(v)); }))));
    return h("div",{class:"tablewrap"}, tb);
  }

  // ================================================================ entity
  ROUTES.entity = async ([sys, ent]) => {
    STATE.system = sys;
    const d = await API.entity(sys, ent);
    const since3d = d.now - 3*86400;
    const [por, tl, sc, idt] = await Promise.all([settled(API.portrait(sys, ent)),
      settled(API.timeline(sys, ent, d.now - 8*86400)), settled(API.scores(sys, ent, since3d)),
      settled(API.identity(sys, ent))]);
    const pj = (por && por.portrait && por.portrait.json) || {};
    const root = h("section",{class:"entity-page"});
    // ---- header
    const pg = d.peer_group || {};
    root.appendChild(h("div",{class:"detail-head"},
      link(`#/entities/${encodeURIComponent(sys)}`, "← ", lbl("实体列表","entities")),
      h("h2",{}, h("span",{class:"mono"}, d.entity)), h("span",{class:"muted"},"@ "+sys),
      pg.class_path ? link(pg.role?`#/class/${encodeURIComponent(sys)}/${encodeURIComponent("class:"+pg.role)}`:"#",
        h("span",{class:"tag",title:pg.role_name||""}, pg.class_path + (pg.prob!=null?` · ${num(pg.prob)}`:""))) : "",
      tierBadge(d.risk && d.risk.tier, d.risk && d.risk.score),
      regimeBadge(d.regime && d.regime.state),
      h("span",{class:"chip"}, lbl("可辨识度 ","identifiability "), h("b",{}, num(d.separability))),
      h("span",{class:"chip"}, lbl("成熟度 ","maturity "), h("b",{}, (d.maturity||{}).stage||"—")),
      d.open_incidents ? link(`#/incidents?entity=${encodeURIComponent(d.entity)}&system=${encodeURIComponent(sys)}`,
        h("span",{class:"badge sev-high"}, lbl(`${d.open_incidents} 个进行中事件`,`${d.open_incidents} open incident(s)`))) : ""));

    // ---- portrait + risk
    root.appendChild(h("div",{class:"grid g2"}, portraitCard(sys, ent, por), riskCard(d)));
    // ---- rhythm + workload
    root.appendChild(h("div",{class:"grid g2 mt"}, rhythmCard(por), workloadCard(pj, por && por.current)));
    // ---- chips
    root.appendChild(h("div",{class:"grid g3 mt"},
      card(lbl("终端栈 · 客户端","Client stacks"), clientBlock(pj.client)),
      card(lbl("常用模板与目的地","Top templates & destinations"), topsBlock(pj)),
      card(lbl("行为序列 n-gram · 显著特征","N-grams · distinctive traits"), ngramBlock(pj), traitsBlock(pj.distinctive))));
    // ---- identity
    root.appendChild(h("div",{class:"grid g2 mt"}, identityCard(idt, pj), attributionCard(idt)));
    // ---- risk timeline
    root.appendChild(h("div",{class:"mt"}, riskTimelineCard(sc, tl, d)));
    // ---- detector families + features
    root.appendChild(h("div",{class:"grid g2 mt"}, familyCard(sc), featuresCard(d.features)));
    // ---- events / matches / explorer
    root.appendChild(h("div",{class:"grid g2 mt"},
      h("div",{class:"card tablewrap scroll"}, h("h3",{},lbl("行为事件","Events")), d.events.length?eventsTable(d.events,true):empty()),
      h("div",{class:"card tablewrap scroll"}, h("h3",{},lbl("行为特征命中（TA在做什么）","Signature matches")), d.matches.length?matchesTable(d.matches):empty())));
    root.appendChild(h("div",{class:"mt"}, await explorerCard(sys, ent)));
    return root;
  };

  function portraitCard(sys, ent, por){
    const c = card(lbl("画像","Portrait"));
    if(!por || !por.portrait || !por.portrait.json){ c.appendChild(empty(lbl("画像尚未生成（B30 每 2 小时刷新）","no portrait yet (B30 refreshes every 2 h)"))); return c; }
    const p = por.portrait;
    const body = h("div",{});
    const draw = (pp, verLabel) => {
      body.innerHTML = "";
      body.appendChild(h("p",{class:"narrative"}, (STATE.lang==="en"?pp.text_en:pp.text_zh) || pp.text_zh || pp.text_en || "—"));
      body.appendChild(h("div",{class:"muted small"}, lbl("版本 ","version "), String(pp.version??"—"), verLabel||"", " · ", lbl("更新于 ","updated "), fmtTs(pp.updated)));
      body.appendChild(diffList(pp.diff, lbl("较上一版变化","changes vs previous version")));
    };
    draw(p);
    const sel = h("select",{class:"search small",onchange:async e=>{
      const v = e.target.value; const r = await settled(API.portrait(sys, ent, v));
      if(r && r.portrait) draw(r.portrait, v==por.current_version?"":lbl("（历史）"," (historic)")); }},
      ...(por.versions||[]).map(v=>h("option",{value:v.version,selected:v.version===por.version?"selected":null}, `v${v.version} · ${fmtShort(v.ts)}`)));
    const cmpOut = h("div",{});
    const cmpBtn = h("button",{class:"mini",onclick:async ()=>{
      const r = await settled(API.portraitDiff(sys, ent, sel.value, por.current_version));
      cmpOut.innerHTML=""; cmpOut.appendChild(r ? diffList(r.diff, lbl(`v${r.from} → v${r.to}`,`v${r.from} → v${r.to}`)) : empty()); }},
      lbl("与当前版对比","diff vs current"));
    c.appendChild(h("div",{class:"rowline wrap",style:"margin-bottom:8px"}, sel, cmpBtn));
    c.appendChild(body); c.appendChild(cmpOut);
    return c;
  }
  function diffList(diff, title){
    if(!diff || !diff.length) return h("div",{class:"muted small mt6"}, lbl("无显著变化","no significant change"));
    const ul = h("ul",{class:"difflist"});
    diff.forEach(x=>{
      const name = (STATE.lang==="en"?x.label_en:x.label_zh) || x.field;
      const dt = x.day_type ? " · "+lbl(...(DAYTYPE[x.day_type]||[x.day_type,x.day_type])) : "";
      const f = x.feature ? " · "+(STATE.lang==="en"?x.feature:(FEAT_ZH[x.feature]||x.feature)) : "";
      const ba = (v) => Array.isArray(v) ? v.map(fmtN).join(" / ") : (v==null?"—":typeof v==="object"?JSON.stringify(v).slice(0,60):String(v));
      ul.appendChild(h("li",{}, h("b",{},name), dt, f, ": ", h("span",{class:"muted"}, ba(x.before)), " → ", ba(x.after),
        x.max_rel_shift!=null?h("span",{class:"muted small"},` (${(x.max_rel_shift*100).toFixed(0)}%)`):"",
        x.jsd!=null?h("span",{class:"muted small"},` (JSD ${num(x.jsd,3)})`):""));
    });
    return h("div",{class:"mt6"}, h("div",{class:"small muted"}, title), ul);
  }

  function riskCard(d){
    const r = d.risk || {};
    const c = card(lbl("风险","Risk"));
    c.appendChild(h("div",{class:"rowline"},
      h("div",{class:"kpi",style:`color:${SEV_COLOR[r.tier]||"inherit"}`}, r.score==null?"—":Math.round(r.score)),
      h("div",{}, tierBadge(r.tier), h("div",{class:"small muted"}, lbl("趋势 ","trend "), r.trend==null?"—":Math.round(r.trend))),
      h("div",{class:"right chips"}, ...(r.stages||[]).map(s=>h("span",{class:"chip"}, s)))));
    const reasons = r.top_reasons || [];
    if(reasons.length){
      c.appendChild(h("div",{class:"small muted mt6"}, lbl("主要原因（占比）","top reasons (share)")));
      const max = Math.max(...reasons.map(x=>x.share||0), 0.01);
      reasons.forEach(x=>{ const b = hbar(`${x.source}:${x.name}`, x.share||0, max, SEV_COLOR[r.tier]||Charts.SERIES[0], share(x.share), null);
        if(x.detail) b.title = x.detail; c.appendChild(b); });
    } else c.appendChild(empty(lbl("暂无风险证据","no risk evidence")));
    const rg = d.regime || {};
    c.appendChild(h("div",{class:"kvgrid mt6"},
      kvRow(lbl("行为状态","regime"), rg.state), kvRow(lbl("类型","type"), rg.type),
      kvRow(lbl("合法概率","p(legit)"), rg.p_legit==null?null:num(rg.p_legit)),
      kvRow(lbl("模型版本","version"), rg.version!=null?`v${rg.version}.${rg.branch||0}`:null),
      kvRow(lbl("回滚次数","rollbacks"), rg.rollbacks), kvRow(lbl("首次出现","first seen"), fmtShort(d.first_seen))));
    return c;
  }

  function rhythmCard(por){
    const c = card(lbl("7×24 活动节律（P活跃，15分钟槽按小时聚合）","7×24 rhythm (P(active) per hour of 15-min slots)"));
    const rh = por && por.rhythm;
    if(!rh || !rh.p_active){ c.appendChild(empty(lbl("节律模型尚未建立","rhythm model not built yet"))); return c; }
    const rows = STATE.lang==="en"?DOW_EN:DOW_ZH.map(x=>"周"+x);
    const hl = rh.now && rh.now.dow!=null ? {r:rh.now.dow, c:rh.now.hour} : null;
    c.appendChild(Charts.heatmap(rh.p_active, {rowLabels:rows, colLabels:[...Array(24).keys()].map(i=>String(i).padStart(2,"0")),
      overlay: rh.observed_7d, highlight: hl, colEvery: 2,
      tipFn:(i,j,v,ov)=>`<b>${esc(rows[i])} ${String(j).padStart(2,"0")}:00</b><br>P(active) ${v==null?"—":(v*100).toFixed(0)+"%"}`+
        `<br>${esc(lbl("近7天实际","observed 7d"))}: ${ov==null?"—":(ov*100).toFixed(0)+"%"}`}));
    const pj = (por.portrait||{}).json||{}; const rj = pj.rhythm || {};
    c.appendChild(h("div",{class:"legend"},
      Charts.seqLegend("P(active)"),
      h("span",{}, h("i",{class:"ring"}), lbl("近 7 天实际活跃（圈越大越多）","observed in last 7 d")),
      h("span",{}, h("i",{class:"nowbox"}), lbl("当前 ","now ") + (rh.now && rh.now.active!=null ? (rh.now.active>0?lbl("活跃","active"):lbl("静默","idle")) : "")),
      rj.workday?h("span",{}, lbl("工作日窗口 ","workday window "), h("b",{}, rj.workday.window||"—")):"",
      rj.nonworkday?h("span",{}, lbl("休息日 ","non-workday "), h("b",{}, rj.nonworkday.window||"—")):"",
      rj.machine_like!=null?h("span",{}, rj.machine_like?lbl("机器式节律","machine-like"):lbl("人工式节律","human-like")):""));
    return c;
  }

  function workloadCard(pj, cur){
    const c = card(lbl("工作负载区间（每15分钟，自然单位）","Workload bands (per 15 min, natural units)"));
    const wl = pj.workload || {};
    const names = Object.keys(wl);
    if(!names.length){ c.appendChild(empty(lbl("负载区间尚未建立","workload bands not built yet"))); return c; }
    const curF = (cur && cur.features) || {};
    names.forEach(n=>{
      const w = wl[n] || {};
      const rows = ["workday","nonworkday"].filter(k=>w[k]).map(k=>({label:lbl(...DAYTYPE[k]), lo:w[k].p10, mid:w[k].p50, hi:w[k].p90}));
      const cv = curF[n] && curF[n].per_15min;
      c.appendChild(h("div",{class:"bandrow"},
        h("div",{class:"small"}, h("b",{}, STATE.lang==="en"?n.replace("_"," "):(FEAT_ZH[n]||n)),
          h("span",{class:"muted"}, ` · p10–p90 · ${lbl("当前","now")} `), h("b",{}, fmtN(cv))),
        Charts.bands(rows, {current: cv, log: n.startsWith("bytes"), unit: w.unit||"per 15 min", qLabel:"p10–p90", w:380})));
    });
    c.appendChild(h("div",{class:"legend"}, h("span",{}, h("i",{class:"dot",style:"background:#3987e566"}), "p10–p90"),
      h("span",{}, h("i",{class:"bar-ink"}), "p50"), h("span",{}, h("i",{class:"dot",style:"background:#ffd166"}), lbl("当前（按当前节拍折算到15分钟）","current (scaled to 15 min)"))));
    return c;
  }

  function clientBlock(cl){
    if(!cl || !(cl.dominant||[]).length) return empty();
    const d = h("div",{});
    (cl.dominant||[]).forEach(s=>d.appendChild(hbar(`${s.ua||"?"} · ${s.os||"?"}`, s.share||0, 1, Charts.SERIES[2], share(s.share))));
    d.appendChild(h("div",{class:"small muted mt6"}, lbl(`${cl.n_stacks||0} 种终端栈 · 熵 ${num(cl.entropy_bits)} bit`,`${cl.n_stacks||0} stacks · entropy ${num(cl.entropy_bits)} bit`)));
    d.appendChild(chips(Object.entries(cl.ua_mix||{}), ([k,v])=>chip(k, share(v))));
    return d;
  }
  function topsBlock(pj){
    const d = h("div",{});
    const tm = pj.top_templates || [];
    if(tm.length){ const mx=Math.max(...tm.map(x=>x.share||0),0.01);
      tm.slice(0,6).forEach(x=>d.appendChild(hbar(h("span",{class:"mono small",title:x.template}, x.template), x.share||0, mx, Charts.SERIES[0], share(x.share)))); }
    else d.appendChild(empty());
    [["SNI",pj.top_sni],[lbl("端口","ports"),pj.top_ports],[lbl("对端","peers"),pj.top_peers]].forEach(([k,arr])=>{
      if(arr && arr.length) d.appendChild(h("div",{class:"mt6"}, h("span",{class:"small muted"},k+" "), chips(arr, x=>chip(x.value, share(x.share))))); });
    return d;
  }
  function ngramBlock(pj){
    const ng = pj.top_ngrams || [];
    return h("div",{}, h("div",{class:"small muted"}, "n-gram"),
      chips(ng.slice(0,6), x=>h("span",{class:"chip mono small",title:`count ${fmtN(x.count)}`}, (x.ngram||[]).map(s=>String(s).replace(/\|2xx$/,"")).join(" → "))));
  }
  function traitsBlock(dist){
    const d = h("div",{class:"mt6"}, h("div",{class:"small muted"}, lbl("显著特征（相对同类）","distinctive traits (vs peers)")));
    const f = (dist && dist.features) || [], v = (dist && dist.vocab) || [];
    if(!f.length && !v.length){ d.appendChild(empty()); return d; }
    d.appendChild(chips(f.slice(0,5), x=>chip(`${x.dir==="higher"?"↑":"↓"} ${x.feature}`, `${fmtN(x.self)} vs ${fmtN(x.peers)}`)));
    d.appendChild(chips(v.slice(0,5), x=>chip(`${x.dim}=${String(x.value).slice(0,40)}`, share(x.share_self), `peers ${share(x.share_peers)}`)));
    return d;
  }

  function identityCard(idt, pj){
    const c = card(lbl("可辨识度（身份模型 B15）","Identifiability (B15)"));
    if(!idt){ c.appendChild(empty()); return c; }
    const val = idt.identifiability ?? idt.separability;
    c.appendChild(h("div",{class:"rowline wrap"},
      Charts.gauge(val, idt.identifiability_ci || (idt.eer_ci ? [Math.max(0,1-2*idt.eer_ci[1]), Math.max(0,1-2*idt.eer_ci[0])] : null), {w:210}),
      h("div",{class:"kvgrid",style:"flex:1;min-width:180px"},
        kvRow("recall@1", idt.recall1==null?null:`${num(idt.recall1)} [${(idt.recall1_ci||[]).map(x=>num(x)).join("–")}]`),
        kvRow("EER_hard", idt.eer_hard==null?null:`${num(idt.eer_hard,3)} [${(idt.eer_ci||[]).map(x=>num(x,3)).join("–")}]`),
        kvRow("T99", idt.t99==null?null:lbl(`${num(idt.t99,0)} 个窗口`,`${num(idt.t99,0)} windows`)),
        kvRow(lbl("留出窗口","held-out windows"), idt.n_windows),
        kvRow(lbl("模型版本","model"), idt.fitted?`v${idt.model_version}`:lbl("未拟合","not fitted")))));
    c.appendChild(h("div",{class:"small muted mt6"}, lbl("易混淆对象","confusable with")));
    const conf = idt.confusion_row || [];
    if(conf.length){ const mx=Math.max(...conf.map(x=>x.share||0),0.01);
      conf.slice(0,6).forEach(x=>c.appendChild(hbar(x.entity, x.share||0, mx, Charts.SERIES[4], share(x.share), entHref(STATE.system, x.entity)))); }
    else c.appendChild(chips(idt.confusable_with||[], x=>chip(x)));
    if(!(conf.length || (idt.confusable_with||[]).length)) c.lastChild.replaceWith(h("div",{class:"small"}, lbl("无（可唯一识别）","none (uniquely identifiable)")));
    c.appendChild(h("div",{class:"small muted mt6"}, lbl("匿名集 ","anonymity set "), (idt.anonymity_set||[]).join(", ")));
    return c;
  }
  function attributionCard(idt){
    const c = card(lbl("归因后验（B16）","Attribution posterior (B16)"));
    const p = idt && idt.posterior;
    if(!p || !p.ts.length){ c.appendChild(empty()); return c; }
    c.appendChild(Charts.timeSeries([
      {name:lbl("本体后验","posterior self"), ts:p.ts, values:p.posterior_self, color:Charts.SERIES[0]},
      {name:lbl("未知身份","p(unknown)"), ts:p.ts, values:p.p_unknown, color:Charts.SERIES[1]}],
      {h:180, yMin:0, yMax:1, yTicks:[0,0.5,1], fmtT:fmtShort, fmtFull:fmtTs}));
    c.appendChild(h("div",{class:"legend"},
      h("span",{}, h("i",{class:"dot",style:`background:${Charts.SERIES[0]}`}), lbl("本体后验","posterior self")),
      h("span",{}, h("i",{class:"dot",style:`background:${Charts.SERIES[1]}`}), lbl("未知身份","p(unknown)"))));
    const a = idt.attribution || {};
    if(a.looks_like) c.appendChild(h("div",{class:"small muted"}, lbl("最相似 ","looks like "), h("b",{},a.looks_like), ` (${num(a.looks_like_posterior)})`));
    return c;
  }

  function riskTimelineCard(sc, tl, d){
    const c = card(lbl("风险时间线 · 事件 / 状态 / 版本 / 回滚标记","Risk timeline · incident / regime / version / rollback markers"));
    const markers = [];
    const items = (tl && tl.items) || [];
    const t0 = sc ? sc.since : (d.now - 3*86400);
    items.forEach(it=>{
      if(it.ts < t0) return;
      if(it.type==="incident") markers.push({ts: it.opened||it.ts, label:`${lbl("事件","incident")} ${it.id} (${it.severity})`, color:SEV_COLOR[it.severity]||"#d03b3b", shape:"tri", row:0});
      else if(it.type==="profile_version") markers.push({ts: it.ts, label:`${lbl("画像版本","portrait")} v${it.version}${(it.diff||[]).length?": "+it.diff.join(", "):""}`, color:"#3987e5", shape:"line", row:1});
      else if(it.type==="event" && it.kind==="regime") markers.push({ts: it.ts, label:`regime: ${it.description||it.state||""}`, color:"#9085e9", shape:"line", row:2});
    });
    ((tl && tl.regime_markers) || []).forEach(m=>{ if(m.ts>=t0) markers.push({ts:m.ts, label:`${m.state==="rollback"?lbl("回滚","rollback"):"regime"} → ${m.state}${m.type?" ("+m.type+")":""}`,
      color: m.state==="rollback"?"#ec835a":"#9085e9", shape:"line", row:m.state==="rollback"?3:2}); });
    const risk = sc && sc.risk;
    if(!risk || !risk.ts.length){ c.appendChild(empty()); return c; }
    c.appendChild(Charts.timeSeries([{name:lbl("风险","risk"), ts:risk.ts, values:risk.values, color:Charts.SERIES[0]}],
      {w:1300, h:250, yMin:0, yMax:100, yTicks:[0,30,60,85,100], markers, t0, t1:d.now, fmtT:fmtShort, fmtFull:fmtTs,
       thresholds:[{y:30,label:lbl("中","medium"),color:SEV_COLOR.medium},{y:60,label:lbl("高","high"),color:SEV_COLOR.high},{y:85,label:lbl("严重","critical"),color:SEV_COLOR.critical}]}));
    c.appendChild(h("div",{class:"legend"},
      h("span",{}, h("i",{class:"dot",style:`background:${Charts.SERIES[0]}`}), lbl("风险 0–100","risk 0–100")),
      h("span",{}, h("i",{class:"tri"}), lbl("事件开启","incident opened")),
      h("span",{}, h("i",{class:"dot",style:"background:#3987e5"}), lbl("画像版本","portrait version")),
      h("span",{}, h("i",{class:"dot",style:"background:#9085e9"}), lbl("状态变化","regime change")),
      h("span",{}, h("i",{class:"dot",style:"background:#ec835a"}), lbl("回滚","rollback"))));
    return c;
  }

  function familyCard(sc){
    const c = card(lbl("检测族证据（最新 p，惊奇度 log10(1/p)）","Detector families (latest p, surprise log10 1/p)"));
    const pf = sc && sc.p_family;
    if(!pf || !pf.ts.length){ c.appendChild(empty()); return c; }
    const rows = Object.entries(pf.values).map(([f, arr])=>{ let v=null; for(let i=arr.length-1;i>=0;i--){ if(arr[i]!=null){ v=arr[i]; break; } }
      return {f, p:v, s: v==null?null:Math.min(40, -Math.log10(Math.max(v,1e-300)))}; }).filter(r=>r.p!=null).sort((a,b)=>b.s-a.s);
    const mx = Math.max(3, ...rows.map(r=>r.s));
    rows.forEach(r=>c.appendChild(hbar(STATE.lang==="en"?r.f:`${FAMILY_ZH[r.f]||r.f} ${r.f}`, r.s, mx, r.s>=6?SEV_COLOR.high:r.s>=3?SEV_COLOR.medium:Charts.SERIES[0], `p=${fmtN(r.p)}`)));
    const eq = sc.e_day && sc.e_day.values; const last = eq ? eq.filter(x=>x!=null).slice(-1)[0] : null;
    c.appendChild(h("div",{class:"small muted mt6"}, "e_day ", h("b",{}, fmtN(last)), " · ", fmtDays(last?1/last:null)));
    return c;
  }
  function featuresCard(features){
    const c = h("div",{class:"card tablewrap scroll"}, h("h3",{}, lbl("特征 · 当前 vs 预测区间 p5–p95（引擎自身 z / zr）","Features · current vs predictive p5–p95 (engine z / zr)")));
    const rows = (features||[]).slice().sort((a,b)=>Math.abs(b.z||0)-Math.abs(a.z||0));
    const tb = h("table",{}, h("tr",{}, h("th",{},lbl("特征","feature")), h("th",{},lbl("当前","current")), h("th",{},"p5–p95"), h("th",{},"z"), h("th",{},"zr")));
    rows.forEach(f=>{
      const out = f.current!=null && f.p5!=null && (f.current<f.p5 || f.current>f.p95);
      tb.appendChild(h("tr",{},
        h("td",{class:"small"}, f.name, h("span",{class:"muted"}," "+(f.group||""))),
        h("td",{class:"num small",style:out?"color:var(--high);font-weight:700":""}, fmtN(f.current)),
        h("td",{class:"num small muted"}, f.p5==null?"—":`${fmtN(f.p5)} – ${fmtN(f.p95)}`),
        h("td",{}, zCell(f.z)), h("td",{}, zCell(f.zr))));
    });
    c.appendChild(tb);
    return c;
  }
  function zCell(z){
    if(z==null) return h("span",{class:"muted small"},"—");
    const zz=Math.max(-8,Math.min(8,z)), w=Math.abs(zz/8)*50, x=zz>=0?50:50-w;
    const col = Math.abs(z)>=3?SEV_COLOR.critical:Math.abs(z)>=2?SEV_COLOR.high:Charts.SERIES[0];
    return h("div",{class:"zcell",title:`z = ${z.toFixed(2)}`}, h("div",{class:"zbar"}, h("div",{class:"mid"}), h("i",{style:`left:${x}%;width:${w}%;background:${col}`})),
      h("span",{class:"small mono",style:`color:${col}`}, (z>=0?"+":"")+z.toFixed(1)));
  }
  async function explorerCard(sys, ent){
    const c = card(lbl("指标时序浏览","Metric explorer"));
    const host = h("div",{style:"margin-top:10px"});
    const ml = await settled(API.metrics(sys, ent));
    const all = ml ? [...ml.raw, ...ml.derived] : [];
    const sel = h("select",{class:"search",onchange:e=>draw(e.target.value)}, ...all.map(m=>h("option",{value:m},m)));
    async function draw(name){ host.innerHTML="";
      const s = await settled(API.series(sys, ent, name)); if(!s) return;
      const pts = s.points.filter(p=>typeof p.value==="number");
      host.appendChild(h("div",{class:"muted small"}, `${name} · ${s.kind} · ${pts.length} ${lbl("点","points")}`));
      host.appendChild(Charts.timeSeries([{name, ts:pts.map(p=>p.ts), values:pts.map(p=>p.value), color:Charts.SERIES[6]}], {h:160, fmtT:fmtShort, fmtFull:fmtTs})); }
    c.appendChild(sel); c.appendChild(host);
    if(all.length){ const pref = all.find(m=>m==="l4.bytes_up")||all[0]; sel.value=pref; draw(pref); }
    return c;
  }

  // ================================================================ classes
  ROUTES.classes = async (params) => {
    if(params[0]) STATE.system = params[0];
    if(!STATE.system) STATE.system = STATE.systems[0];
    const d = await API.classes(STATE.system);
    const root = h("section",{});
    root.appendChild(h("div",{class:"detail-head"}, h("h2",{}, lbl("类画像","Classes")),
      systemSelect(s=>go(`#/classes/${encodeURIComponent(s)}`)),
      h("span",{class:"muted small right"}, lbl(`类模型 v${d.model_version}`,`class model v${d.model_version}`))));
    root.appendChild(h("div",{class:"grid g2"}, treeCard(STATE.system, d.tree, null), classTable(STATE.system, d.classes)));
    return root;
  };
  function treeCard(sys, tree, current){
    const c = card(lbl("角色 / 子类层级","Role / sub-class hierarchy"));
    if(!tree || !tree.length){ c.appendChild(empty(lbl("尚未聚类（B02 冷启动中）","not clustered yet (B02 warming up)"))); return c; }
    const ul = h("ul",{class:"tree"});
    tree.forEach(sp=>{
      const li = h("li",{}, h("b",{}, sp.super==="human"?lbl("人工交互","human"):sp.super==="machine"?lbl("自动化","machine"):sp.super));
      const ul2 = h("ul",{});
      (sp.roles||[]).forEach(r=>{
        const cur = r.key===current ? " current" : "";
        const li2 = h("li",{class:cur}, link(`#/class/${encodeURIComponent(sys)}/${encodeURIComponent(r.key)}`, r.key), " ",
          h("span",{class:"muted small",title:r.name}, String(r.name||"").slice(0,48)), h("span",{class:"chip small"}, `${r.n_members}`));
        if((r.subs||[]).length){ const ul3=h("ul",{});
          r.subs.forEach(s=>ul3.appendChild(h("li",{class:s.key===current?"current":""}, link(`#/class/${encodeURIComponent(sys)}/${encodeURIComponent(s.key)}`, s.key.replace(/^sub:/,"")), " ",
            h("span",{class:"muted small"}, (s.members||[]).join(", ")))));
          li2.appendChild(ul3); }
        ul2.appendChild(li2);
      });
      li.appendChild(ul2); ul.appendChild(li);
    });
    c.appendChild(ul);
    return c;
  }
  function classTable(sys, classes){
    const c = h("div",{class:"card tablewrap"}, h("h3",{}, lbl("全部类","All classes")));
    if(!classes.length){ c.appendChild(empty()); return c; }
    const tb = h("table",{}, h("tr",{}, h("th",{},lbl("类","class")), h("th",{},lbl("类型","kind")), h("th",{},lbl("成员","members")),
      h("th",{style:"min-width:150px"},lbl("风险","risk")), h("th",{},lbl("事件","incidents"))));
    classes.forEach(k=>tb.appendChild(h("tr",{class:"clickable",onclick:()=>go(`#/class/${encodeURIComponent(sys)}/${encodeURIComponent(k.key)}`)},
      h("td",{}, h("b",{class:"mono"}, k.key), h("div",{class:"muted small ellip",title:k.name}, k.name||"")),
      h("td",{}, h("span",{class:"tag"}, k.kind)), h("td",{class:"num"}, String(k.n_members)),
      h("td",{}, k.risk!=null?riskBar(k.risk,k.tier):h("span",{class:"muted small"}, k.member_risk_max!=null?lbl(`成员最高 ${Math.round(k.member_risk_max)}`,`member max ${Math.round(k.member_risk_max)}`):"—")),
      h("td",{class:"num"}, String(k.open_incidents||0)))));
    c.appendChild(tb);
    return c;
  }

  ROUTES.class = async ([sys, key]) => {
    STATE.system = sys;
    const [d, list] = await Promise.all([API.klass(sys, key), settled(API.classes(sys))]);
    const root = h("section",{});
    const pj = (d.portrait && d.portrait.json) || {};
    root.appendChild(h("div",{class:"detail-head"},
      link(`#/classes/${encodeURIComponent(sys)}`, "← ", lbl("类列表","classes")),
      h("h2",{}, h("span",{class:"mono"}, d.key)), h("span",{class:"tag"}, d.kind),
      d.parent?link(`#/class/${encodeURIComponent(sys)}/${encodeURIComponent(d.parent)}`, h("span",{class:"chip"}, "↑ "+d.parent)):"",
      h("span",{class:"muted small ellip",title:d.name}, d.name||""),
      d.risk && d.risk.tier ? tierBadge(d.risk.tier, d.risk.score) : "",
      h("span",{class:"chip"}, lbl("版本 ","version "), h("b",{}, d.version??"—"))));
    root.appendChild(h("div",{class:"grid g2"}, membersCard(sys, d), treeCard(sys, list && list.tree, d.key)));
    const pc = card(lbl("类画像","Class portrait"));
    const txt = STATE.lang==="en" ? d.portrait.text_en : d.portrait.text_zh;
    pc.appendChild(txt ? h("p",{class:"narrative"}, txt) : empty(lbl("该类暂无画像（子类或成员不足）","no portrait for this class")));
    if(d.portrait && d.portrait.version!=null) pc.appendChild(h("div",{class:"muted small"}, `v${d.portrait.version} · ${fmtTs(d.portrait.updated)}`));
    if(d.portrait && d.portrait.diff) pc.appendChild(diffList(d.portrait.diff, lbl("较上一版变化","changes vs previous version")));
    if(pj.cohesion!=null) pc.appendChild(h("div",{class:"kvgrid mt6"}, kvRow(lbl("内聚度","cohesion"), num(pj.cohesion)),
      kvRow(lbl("一致性","coherence"), pj.coherence==null?null:num(pj.coherence)), kvRow(lbl("当前活跃比例","active now"), share(pj.active_frac_now))));
    root.appendChild(h("div",{class:"grid g2 mt"}, pc, classBandsCard(d)));
    root.appendChild(h("div",{class:"grid g2 mt"}, activeFracCard(d), classIdentCard(d)));
    root.appendChild(h("div",{class:"grid g2 mt"}, adoptionCard(d), tokensCard(d)));
    const ic = h("div",{class:"card tablewrap mt"}, h("h3",{}, lbl("类事件","Class incidents")));
    ic.appendChild((d.incidents||[]).length ? incidentMiniTable(d.incidents) : empty());
    root.appendChild(ic);
    return root;
  };
  function membersCard(sys, d){
    const c = card(lbl(`成员（${d.n_members}）· 隶属度`,`Members (${d.n_members}) · membership`));
    if(!(d.members||[]).length){ c.appendChild(empty()); return c; }
    d.members.forEach(m=>{
      c.appendChild(h("div",{class:"member"},
        link(entHref(sys, m.ip), h("span",{class:"mono"}, m.ip)),
        h("span",{class:"hb"}, h("i",{style:`width:${Math.round((m.prob||0)*100)}%;background:${Charts.SERIES[2]}`})),
        h("span",{class:"small mono"}, num(m.prob)),
        m.outlier?h("span",{class:"badge sev-high"}, lbl("离群","outlier")):"",
        tierBadge(m.tier, m.risk),
        m.open_incidents?h("span",{class:"badge sev-medium"}, `${m.open_incidents}`):""));
    });
    return c;
  }
  function classBandsCard(d){
    const c = card(lbl("类聚合区间（B18，p5 / p50 / p95）","Class aggregate bands (B18, p5 / p50 / p95)"));
    const b = d.bands; const feats = b && b.features;
    if(!feats || !Object.keys(feats).length){ c.appendChild(empty()); return c; }
    const tb = h("table",{}, h("tr",{}, h("th",{},lbl("特征","feature")), h("th",{},"p5"), h("th",{},"p50"), h("th",{},"p95"), h("th",{style:"width:45%"})));
    const entries = Object.entries(feats);
    entries.forEach(([f, q])=>{
      const arr = Array.isArray(q) ? q : [q.p5, q.p50, q.p95];
      tb.appendChild(h("tr",{}, h("td",{class:"small"}, f), ...arr.map(v=>h("td",{class:"num small"}, fmtN(v))),
        h("td",{}, Charts.bands([{label:"", lo:arr[0], mid:arr[1], hi:arr[2]}], {w:220, labelW:4, log:f.startsWith("bytes"), qLabel:"p5–p95"}))));
    });
    c.appendChild(h("div",{class:"small muted"}, lbl(`暴露 ${b.exposure_s||900}s · 桶 ${b.bucket??"—"}`,`exposure ${b.exposure_s||900}s · bucket ${b.bucket??"—"}`)));
    c.appendChild(h("div",{class:"tablewrap"}, tb));
    return c;
  }
  function activeFracCard(d){
    const c = card(lbl("活跃成员比例热图","Active-fraction heatmap"));
    const hm = d.active_frac_heatmap;
    if(hm && typeof hm === "object" && Object.keys(hm).length){
      const keys = Object.keys(hm).filter(k=>Array.isArray(hm[k]));
      const rows = keys.map(k=>hm[k]);
      const n = Math.max(...rows.map(r=>r.length));
      c.appendChild(Charts.heatmap(rows.map(r=>{ const x=r.slice(); while(x.length<n) x.push(null); return x; }),
        {rowLabels: keys.map(k=>DAYTYPE[k]?lbl(...DAYTYPE[k]):k), labelW:56, colLabels:[...Array(n).keys()].map(i=>n===24?String(i).padStart(2,"0"):String(i)), colEvery:n>24?4:2,
         tipFn:(i,j,v)=>`<b>${esc(keys[i])} ${j}</b><br>${esc(lbl("活跃比例","active fraction"))} ${v==null?"—":(v*100).toFixed(0)+"%"}`}));
      c.appendChild(Charts.seqLegend(lbl("活跃成员比例","active fraction")));
    } else c.appendChild(empty());
    if(d.rhythm && d.rhythm.p_active){
      c.appendChild(h("div",{class:"small muted mt6"}, lbl("类节律 7×24（P活跃）","class rhythm 7×24")));
      c.appendChild(Charts.heatmap(d.rhythm.p_active, {rowLabels: STATE.lang==="en"?DOW_EN:DOW_ZH.map(x=>"周"+x), colLabels:[...Array(24).keys()].map(i=>String(i).padStart(2,"0")), colEvery:3, cell:18, cellH:14,
        highlight: d.rhythm.now && d.rhythm.now.dow!=null ? {r:d.rhythm.now.dow, c:d.rhythm.now.hour} : null}));
    }
    return c;
  }
  function classIdentCard(d){
    const c = card(lbl("类可辨识度 · 共模","Class identifiability · common mode"));
    const id = d.identifiability || {};
    const v = id.identifiability ?? id.separability;
    c.appendChild(h("div",{class:"rowline wrap"}, Charts.gauge(v==null?null:v, id.identifiability_ci, {w:180}),
      h("div",{class:"kvgrid",style:"flex:1"}, kvRow("recall@1", id.recall1==null?null:num(id.recall1)), kvRow(lbl("窗口","windows"), id.n_windows),
        kvRow(lbl("易混淆","confusable"), (id.confusable_with||[]).join(", ")||lbl("无","none")))));
    const cm = d.common || {};
    if(Object.keys(cm).length){
      c.appendChild(h("div",{class:"small muted mt6"}, lbl("共模状态（B05）","common mode (B05)")));
      c.appendChild(commonTable({[d.key]: cm}));
    }
    return c;
  }
  function adoptionCard(d){
    const c = h("div",{class:"card tablewrap"}, h("h3",{}, lbl("采纳历史（新值首次被成员使用）","Adoption history")));
    const ad = d.adoption || [];
    if(d.adoption_rate!=null) c.appendChild(h("div",{class:"small muted"}, lbl("采纳率 ","adoption rate "), share(d.adoption_rate)));
    if(!ad.length){ c.appendChild(empty()); return c; }
    const tb = h("table",{}, h("tr",{}, h("th",{},lbl("维度","dim")), h("th",{},lbl("值","value")), h("th",{},lbl("成员数","members")), h("th",{},lbl("首次","first")), h("th",{},lbl("风险标记","flags"))));
    ad.forEach(a=>tb.appendChild(h("tr",{class:a.risky?"risky":""},
      h("td",{class:"small"}, a.dim||"—"), h("td",{class:"small mono"}, h("span",{class:"ellip",title:a.value}, a.value||"—")),
      h("td",{class:"num"}, String(a.n_members??"—")), h("td",{class:"small muted nowrap"}, fmtShort(a.first_ts)),
      h("td",{}, ...Object.entries(a.flags||{}).filter(([,v])=>v).map(([k])=>h("span",{class:"badge sev-high"}, k))))));
    c.appendChild(tb);
    return c;
  }
  function tokensCard(d){
    const c = card(lbl("类 vs 系统显著模板 · 共有模板 · 离群成员","Class-vs-system tokens · common tokens · outliers"));
    const dt = d.distinctive_tokens || [];
    if(dt.length){ const mx=Math.max(...dt.map(x=>x.z||0),1);
      dt.slice(0,6).forEach(x=>c.appendChild(hbar(h("span",{class:"mono small",title:x.template}, x.template), x.z||0, mx, Charts.SERIES[6], `z ${num(x.z,1)}`))); }
    else c.appendChild(empty());
    c.appendChild(h("div",{class:"small muted mt6"}, lbl("共有模板（≥50% 成员）","common tokens (≥50% of members)")));
    c.appendChild(chips((d.common_tokens||[]).slice(0,8), x=>chip(x.template, share(x.share))));
    c.appendChild(h("div",{class:"small muted mt6"}, lbl("离群成员","outlier members")));
    c.appendChild(chips(d.outlier_members||[], x=>chip(typeof x==="string"?x:(x.ip||"?"), x.jsd!=null?`JSD ${num(x.jsd)}`:(x.reason||""))));
    return c;
  }
  function incidentMiniTable(list){
    const tb = h("table",{}, h("tr",{}, h("th",{},"id"), h("th",{},lbl("严重度","severity")), h("th",{},lbl("状态","status")), h("th",{},lbl("维度","axes")), h("th",{},lbl("最近","last seen"))));
    list.forEach(i=>tb.appendChild(h("tr",{class:"clickable",onclick:()=>go(`#/incident/${encodeURIComponent(i.id)}`)},
      h("td",{class:"mono small"}, i.id), h("td",{}, sevBadge(i.severity)), h("td",{class:"small"}, i.status),
      h("td",{class:"small"}, (i.axes||[]).join(", ")), h("td",{class:"small muted nowrap"}, fmtTs(i.last_seen)))));
    return tb;
  }

  // ================================================================ incidents
  ROUTES.incidents = async (_params, q) => {
    q = q || {};
    const f = {system:q.system||"", entity:q.entity||"", status:q.status ?? "open,acked", severity:q.severity||"", kind:q.kind||""};
    const [d, lq] = await Promise.all([API.incidents(Object.assign({}, f, {limit:100})), settled(API.labelQueue())]);
    const root = h("section",{class:"no-autorefresh"});
    const setF = (k, v) => { const nf = Object.assign({}, f, {[k]:v}); const qs = Object.entries(nf).filter(([,x])=>x!=="").map(([a,b])=>`${a}=${encodeURIComponent(b)}`).join("&");
      go("#/incidents"+(qs?"?"+qs:"")); };
    const sel = (k, opts) => h("select",{class:"search small",onchange:e=>setF(k, e.target.value)},
      ...opts.map(([v,l])=>h("option",{value:v,selected:String(f[k])===v?"selected":null}, l)));
    root.appendChild(h("div",{class:"detail-head"}, h("h2",{}, lbl("事件队列","Incident queue")),
      sel("system", [["",lbl("全部系统","all systems")], ...STATE.systems.map(s=>[s,s])]),
      sel("status", [["open,acked",lbl("进行中","live")],["open","open"],["acked","acked"],["suppressed","suppressed"],["closed","closed"],["",lbl("全部","all")]]),
      sel("severity", [["",lbl("全部严重度","any severity")],["critical,high","high+"],["medium","medium"],["low","low"]]),
      sel("kind", [["",lbl("实体与类","entity + class")],["entity",lbl("实体","entity")],["class",lbl("类","class")]]),
      f.entity?h("span",{class:"chip"}, f.entity, h("a",{href:"#/incidents",class:"x"}," ✕")):"",
      h("span",{class:"muted small right"}, lbl(`${d.count} 条`,`${d.count} incident(s)`))));
    if(lq && ((lq.queue||[]).length || (lq.governor||[]).length)){
      const c = card(lbl("待标注队列（主动学习）","Label queue (active learning)"));
      (lq.queue||[]).forEach(it=>c.appendChild(h("div",{class:"rowline small"}, h("span",{class:"tag"}, it.reason||"—"),
        it.incident_id?link(`#/incident/${encodeURIComponent(it.incident_id)}`, it.incident_id):"", h("span",{class:"muted"}, `${it.system||""}/${it.entity||""}`),
        h("span",{class:"right"}, lbl("风险 ","risk "), fmtN(it.risk)))));
      (lq.governor||[]).forEach(g=>c.appendChild(h("div",{class:"rowline small"}, h("span",{class:"tag"},"drifting>14d"),
        link(entHref(g.system,g.entity), `${g.system}/${g.entity}`), h("span",{class:"muted"}, g.type||""))));
      root.appendChild(c);
    }
    if(!d.incidents.length){ root.appendChild(card("", empty(lbl("没有符合条件的事件","no incidents match")))); return root; }
    const list = h("div",{class:"inc-list"});
    root.appendChild(list);
    const details = await Promise.all(d.incidents.slice(0, 12).map(i=>settled(API.incident(i.id))));
    d.incidents.forEach((i, k)=>list.appendChild(incidentCard(details[k] || i, !!details[k])));
    return root;
  };
  ROUTES.incident = async ([id]) => {
    const d = await API.incident(id);
    return h("section",{class:"no-autorefresh"}, h("div",{class:"detail-head"}, link("#/incidents","← ",lbl("事件队列","incident queue")), h("h2",{}, d.id)),
      incidentCard(d, true, true));
  };

  function incidentCard(i, full, standalone){
    const c = h("div",{class:"card inc sev-edge-"+(i.severity||"info")});
    const isCls = i.kind==="class";
    c.appendChild(h("div",{class:"rowline wrap"},
      sevBadge(i.severity), link(`#/incident/${encodeURIComponent(i.id)}`, h("b",{class:"mono"}, i.id)),
      h("span",{class:"tag"}, isCls?lbl("类","class"):lbl("实体","entity")),
      link(entHref(i.system, i.entity), h("b",{class:"mono"}, `${i.system}/${i.entity}`)),
      h("span",{class:"badge "+(i.status==="open"?"sev-high":i.status==="closed"?"sev-low":"sev-info")}, i.status + (i.close_reason?` · ${i.close_reason}`:"")),
      h("span",{class:"muted small right"}, `${fmtTs(i.opened)} → ${fmtTs(i.last_seen)}`)));
    c.appendChild(h("div",{class:"rare"},
      h("span",{class:"big"}, fmtDays(i.rare_once_in_days)),
      h("span",{class:"muted small"}, lbl(" — 对该", " — as rare as this for this ") + (isCls?lbl("类","class"):lbl("实体","entity")) + lbl("而言的罕见程度（e_day ",", e_day ") + fmtN(i.e_day_min) + ")"),
      h("span",{class:"right small"}, lbl("风险 ","risk "), h("b",{}, i.risk==null?"—":Math.round(i.risk)))));
    const nar = i.narrative && (STATE.lang==="en" ? (i.narrative.en||i.narrative.zh) : (i.narrative.zh||i.narrative.en));
    if(nar) c.appendChild(h("p",{class:"narrative"}, nar));
    c.appendChild(h("div",{class:"chips"}, ...(i.kinds||[]).map(k=>kindBadge(k)), ...(i.axes||[]).map(a=>h("span",{class:"chip"}, "axis:"+a))));
    if(i.entities && i.entities.length>1) c.appendChild(h("div",{class:"small muted mt6"}, lbl("涉及 ","entities "), i.entities.join(", ")));
    if(!full){ c.appendChild(h("button",{class:"mini mt6",onclick:async ()=>{ const dd = await settled(API.incident(i.id)); if(dd) c.replaceWith(incidentCard(dd, true)); }}, lbl("展开证据与反馈","expand evidence & feedback"))); return c; }
    // evidence bars
    const fam = i.evidence_by_family || {}, ax = i.evidence_by_axis || {};
    const mx = Math.max(3, ...Object.values(fam).map(v=>v.surprise||0), ...Object.values(ax).map(v=>v.surprise||0));
    const col = s => s>=6?SEV_COLOR.critical:s>=3?SEV_COLOR.high:s>=1.5?SEV_COLOR.medium:Charts.SERIES[0];
    const barBox = (title, obj, names) => { const b = h("div",{}, h("div",{class:"small muted"}, title));
      const ents = Object.entries(obj).sort((a,b)=>b[1].surprise-a[1].surprise);
      if(!ents.length) b.appendChild(empty());
      ents.forEach(([k,v])=>b.appendChild(hbar(names?(STATE.lang==="en"?k:`${names[k]||k}`):k, v.surprise, mx, col(v.surprise), `e_day ${fmtN(v.e_day)}`)));
      return b; };
    c.appendChild(h("div",{class:"grid g2 mt6"},
      barBox(lbl("各检测族证据（log10 1/e_day）","per-family evidence (log10 1/e_day)"), fam, FAMILY_ZH),
      barBox(lbl("各维度证据","per-axis evidence"), ax, null)));
    if((i.new_tokens||[]).length) c.appendChild(h("div",{class:"mt6"}, h("span",{class:"small muted"}, lbl("新出现的值 ","new tokens ")),
      chips(i.new_tokens, x=>h("span",{class:"chip mono small"}, x))));
    // attributions (B29)
    if((i.attributions||[]).length){
      c.appendChild(h("div",{class:"small muted mt6"}, lbl("数值归因（B29）","numeric attribution (B29)")));
      c.appendChild(chips(i.attributions.slice(0,6), a=>chip(a.feature||a.name||"?", a.text || (a.share!=null?share(a.share):(a.z!=null?`z ${num(a.z,1)}`:"")))));
    }
    // counterfactual
    const cf = i.counterfactual || {};
    const cfb = h("div",{class:"cf mt6"}, h("span",{class:"small muted"}, lbl("反事实（最小改变集）","counterfactual (minimal set)")," "));
    if(cf.available){
      cfb.appendChild(chips(Array.isArray(cf.set)?cf.set:[], x=>chip(typeof x==="string"?x:(x.feature||x.token||JSON.stringify(x)))));
      cfb.appendChild(h("div",{class:"small"}, lbl("有效性 ","valid "), h("b",{}, cf.valid==null?"—":cf.valid?lbl("✓ 决策翻转","✓ decision flips"):lbl("✗ 未翻转","✗ no flip")),
        lbl(" · 重算范围 "," · scope "), h("span",{class:"mono small"}, Array.isArray(cf.scope)?cf.scope.join(", "):(cf.scope||"—"))));
    } else cfb.appendChild(h("span",{class:"small muted"}, lbl("尚未生成（解释引擎 B29 未产出）","not available (B29 explain has not produced one)")));
    c.appendChild(cfb);
    // links
    const links = [];
    if(i.campaign) links.push(h("span",{}, lbl("战役 ","campaign "), h("b",{}, i.campaign.id), ": ", ...i.campaign.incidents.filter(x=>x.id!==i.id).map(x=>link(`#/incident/${encodeURIComponent(x.id)}`, x.id+" ", h("span",{class:"muted"}, x.entity+" ")))));
    if(i.parent) links.push(h("span",{}, lbl("共模父事件 ","common-mode parent "), link(`#/incident/${encodeURIComponent(i.parent.id)}`, i.parent.id)));
    if((i.children||[]).length) links.push(h("span",{}, lbl("共模子事件 ","common-mode children "), ...i.children.map(x=>link(`#/incident/${encodeURIComponent(x.id)}`, x.id+" "))));
    if(links.length) c.appendChild(h("div",{class:"small mt6 links"}, ...links));
    // feedback
    c.appendChild(feedbackForm(i));
    if((i.labels||[]).length) c.appendChild(h("div",{class:"small muted mt6"}, lbl("已有标注：","labels: "),
      ...i.labels.map(l=>h("span",{class:"chip"}, `${l.verdict} · ${l.scope}`, h("span",{class:"muted"}, " "+fmtShort(l.ts))))));
    if(standalone && (i.evidence||[]).length){
      const tb = h("table",{}, h("tr",{}, h("th",{},lbl("时间","time")), h("th",{},lbl("来源","source")), h("th",{},lbl("内容","detail")), h("th",{},"e_day")));
      i.evidence.slice(-60).reverse().forEach(ev=>tb.appendChild(h("tr",{}, h("td",{class:"small muted nowrap"}, fmtTs(ev.ts)), h("td",{class:"small"}, ev.source||""),
        h("td",{class:"small"}, ev.kind||ev.path||ev.signature_id||ev.state||"", " ", (ev.axes||[]).join(","), ev.families?" ["+ev.families.join(",")+"]":""),
        h("td",{class:"num small"}, ev.e_day==null?"—":fmtN(ev.e_day)))));
      c.appendChild(h("details",{class:"mt6"}, h("summary",{class:"small"}, lbl(`证据明细（${i.evidence.length}）`,`evidence (${i.evidence.length})`)), h("div",{class:"tablewrap"}, tb)));
    }
    return c;
  }
  function feedbackForm(i){
    const scope = h("select",{class:"search small"}, ...SCOPES.map(([v,zh,en])=>h("option",{value:v}, lbl(zh,en))));
    const ttl = h("select",{class:"search small",title:"TTL"}, ...TTLS.map(([v,zh,en])=>h("option",{value:v}, lbl(zh,en))));
    const note = h("input",{class:"search small",placeholder:lbl("备注（可选）","note (optional)")});
    const box = h("div",{class:"feedback mt6"}, h("span",{class:"small muted"}, lbl("研判 ","verdict ")));
    VERDICTS.forEach(([v,zh,en,cls])=>box.appendChild(h("button",{class:"fb "+cls,onclick:async (e)=>{
      e.target.disabled = true;
      try{
        const body = {verdict:v, scope:scope.value, note:note.value};
        if(ttl.value) body.ttl_s = Number(ttl.value);
        const r = await API.incidentFeedback(i.id, body);
        toast(lbl(`已提交标注 ${r.label_id}（${zh}，范围 ${scope.value}）`,`label ${r.label_id} saved (${en}, scope ${scope.value})`));
        box.appendChild(h("span",{class:"chip"}, `✓ ${v} · ${scope.value}`));
      }catch(err){ toast(String(err.message||err), false); }
      finally{ e.target.disabled = false; }
    }}, lbl(zh,en))));
    box.append(h("span",{class:"small muted"}, lbl(" 范围 "," scope ")), scope, ttl, note);
    return box;
  }

  // ================================================================ system view
  ROUTES.system = async (params) => {
    if(params[0]) STATE.system = params[0];
    if(!STATE.system) STATE.system = STATE.systems[0];
    const sys = STATE.system;
    const [d, hl, mem] = await Promise.all([API.systemSummary(sys), settled(API.detectorsHealth()), settled(API.memory())]);
    const root = h("section",{});
    root.appendChild(h("div",{class:"detail-head"}, h("h2",{}, lbl("系统视图","System view")),
      systemSelect(s=>go(`#/system/${encodeURIComponent(s)}`)),
      link(`#/incidents?system=${encodeURIComponent(sys)}`, h("span",{class:"chip"}, lbl("进行中事件 ","live incidents "), h("b",{}, String(d.open_incidents))))));
    root.appendChild(h("div",{class:"grid g4"},
      kpiTile(lbl("系统风险","system risk"), d.risk.score==null?"—":Math.round(d.risk.score), tierBadge(d.risk.tier)),
      kpiTile(lbl("实体","entities"), d.entities, ""),
      kpiTile(lbl("战役","campaigns"), (d.campaigns||[]).length, ""),
      kpiTile(lbl("引擎错误","engine errors"), hl?hl.n_errors:"—", hl&&hl.error_total?h("span",{class:"badge sev-critical"}, `${hl.error_total} total`):h("span",{class:"badge sev-low"}, "ok"))));
    const rc = card(lbl("系统风险（24 小时）","System risk (24 h)"));
    rc.appendChild(d.risk_series.ts.length ? Charts.timeSeries([{name:lbl("系统风险","system risk"), ts:d.risk_series.ts, values:d.risk_series.values, color:Charts.SERIES[0]}],
      {h:220, yMin:0, yMax:100, yTicks:[0,30,60,85,100], fmtT:fmtShort, fmtFull:fmtTs,
       thresholds:[{y:30,label:"medium",color:SEV_COLOR.medium},{y:60,label:"high",color:SEV_COLOR.high},{y:85,label:"critical",color:SEV_COLOR.critical}]}) : empty());
    const cc = card(lbl("共模与同步偏移（B05）","Common mode & coherent shifts (B05)"));
    cc.appendChild(Object.keys(d.common||{}).length ? commonTable(d.common) : empty());
    const hist = d.common_history || {};
    const groups = Object.keys(hist);
    if(groups.length){
      cc.appendChild(h("div",{class:"small muted mt6"}, lbl("系统层同向比例（上行，24 小时）","system tier: share of members moving up (24 h)")));
      cc.appendChild(Charts.timeSeries(groups.map((g,k)=>({name:g, ts:hist[g].ts, values:hist[g].frac_up, color:Charts.SERIES[k]})),
        {h:150, yMin:0, yMax:1, yTicks:[0,0.5,1], fmtT:fmtShort, fmtFull:fmtTs}));
      cc.appendChild(h("div",{class:"legend"}, ...groups.map((g,k)=>h("span",{}, h("i",{class:"dot",style:`background:${Charts.SERIES[k]}`}), g))));
    }
    root.appendChild(h("div",{class:"grid g2 mt"}, rc, cc));
    // campaign graph
    const gc = card(lbl("战役图 · 共模父子","Campaign graph · common-mode parent/child"));
    const nodes = (d.campaign_graph.nodes||[]).map(n=>({id:n.id, label:n.entity, group:n.campaign_id||n.parent_id||"",
      color:SEV_COLOR[n.severity]||Charts.SERIES[0], r: n.kind==="class"?12:8, onclick:()=>go(`#/incident/${encodeURIComponent(n.id)}`),
      tip:`<b>${esc(n.id)}</b> ${esc(n.severity)}<br>${esc(n.system)}/${esc(n.entity)}<br>${esc((n.axes||[]).join(", "))}`}));
    gc.appendChild(Charts.graph(nodes, d.campaign_graph.edges||[], {h:320}));
    gc.appendChild(h("div",{class:"legend"}, h("span",{}, h("i",{class:"line",style:"background:#9085e9"}), lbl("同一战役","same campaign")),
      h("span",{}, h("i",{class:"line dashed",style:"border-color:#199e70"}), lbl("共模父→子","common-mode parent→child")),
      ...["critical","high","medium","low"].map(s=>h("span",{}, h("i",{class:"dot",style:`background:${SEV_COLOR[s]}`}), s))));
    (d.campaigns||[]).forEach(cp=>gc.appendChild(h("div",{class:"small"}, h("b",{}, cp.id), ` · ${cp.entities.join(", ")} · ${cp.axes.join(", ")} · ${fmtShort(cp.opened)}`)));
    const sc = h("div",{class:"card tablewrap scroll"}, h("h3",{}, lbl("同步偏移事件","Coherent shifts")));
    sc.appendChild((d.coherent_shifts||[]).length ? eventsTable(d.coherent_shifts, false) : empty(lbl("24 小时内无系统/类同步偏移","no system / class shift in 24 h")));
    root.appendChild(h("div",{class:"grid g2 mt"}, gc, sc));
    // detector health
    root.appendChild(h("div",{class:"mt"}, healthPanel(hl, sys, mem)));
    return root;
  };
  function kpiTile(title, val, extra){ return h("div",{class:"card"}, h("h3",{}, title), h("div",{class:"rowline"}, h("div",{class:"kpi"}, String(val)), extra||"")); }
  function commonTable(common){
    const tb = h("table",{}, h("tr",{}, h("th",{},lbl("层级","tier")), h("th",{},lbl("组","group")), h("th",{},"L"), h("th",{},lbl("↑比例","up")), h("th",{},lbl("↓比例","down")),
      h("th",{},lbl("方向","dir")), h("th",{},lbl("持续","run")), h("th",{},lbl("参与/成员","scored/members"))));
    Object.entries(common).forEach(([key, groups])=>Object.entries(groups).forEach(([g, v])=>{
      const dir = v.dir>0?"↑":v.dir<0?"↓":"·";
      tb.appendChild(h("tr",{class:v.dir? "active-row":""}, h("td",{class:"small mono"}, key), h("td",{class:"small"}, g), h("td",{class:"num small"}, fmtN(v.L)),
        h("td",{class:"num small"}, share(v.frac_up)), h("td",{class:"num small"}, share(v.frac_down)),
        h("td",{class:"small",style:v.dir?`color:${SEV_COLOR.medium};font-weight:700`:""}, dir), h("td",{class:"num small"}, String(v.run??"—")),
        h("td",{class:"num small"}, `${v.n_scored??"—"}/${v.n_members??"—"}`)));
    }));
    return h("div",{class:"tablewrap"}, tb);
  }
  function healthPanel(hl, sys, mem){
    const c = card(lbl("检测器健康 · 引擎错误与数据陈旧度","Detector health · engine errors & staleness"));
    if(!hl){ c.appendChild(empty()); return c; }
    const tb = h("table",{}, h("tr",{}, h("th",{},lbl("引擎","engine")), h("th",{},lbl("层","layer")), h("th",{},lbl("上次运行","last run")),
      h("th",{},lbl("耗时","ms")), h("th",{},lbl("错误数","errors")), h("th",{},lbl("最近错误","last error")), h("th",{},lbl("最大陈旧度","max staleness"))));
    const staleLimit = 3600;
    hl.engines.forEach(e=>{
      const stale = e.max_staleness_s!=null && e.max_staleness_s > staleLimit;
      tb.appendChild(h("tr",{class:e.last_error?"err-row":stale?"stale-row":""},
        h("td",{class:"mono small"}, e.name), h("td",{class:"small layer-"+e.layer}, e.layer),
        h("td",{class:"small muted nowrap"}, fmtTs(e.last_run)), h("td",{class:"num small"}, fmtN(e.duration_ms)),
        h("td",{class:"num small"}, e.error_count?h("b",{style:"color:var(--critical)"}, String(e.error_count)):"0"),
        h("td",{class:"small",title:e.traceback||""}, e.last_error?`${e.last_error} @ ${fmtTs(e.last_error_ts)}`:"—"),
        h("td",{class:"num small",title:(e.series||[]).map(s=>`${s.name}: ${s.staleness_s==null?"never":s.staleness_s+"s"}`).join("\n")},
          e.max_staleness_s==null?"—":stale?h("b",{style:"color:var(--high)"}, `${fmtN(e.max_staleness_s)} s`):`${fmtN(e.max_staleness_s)} s`)));
    });
    c.appendChild(h("details",{open:hl.n_errors?"open":null}, h("summary",{class:"small"}, lbl(`引擎（${hl.engines.length}，${hl.n_errors} 个有错误）`,`engines (${hl.engines.length}, ${hl.n_errors} with errors)`)),
      h("div",{class:"tablewrap scroll"}, tb)));
    const rows = (hl.detectors||{})[sys] || [];
    const tb2 = h("table",{}, h("tr",{}, h("th",{},lbl("检测器","detector")), h("th",{},lbl("族","family")), h("th",{class:"num"},"KS D"),
      h("th",{class:"num"},lbl("实际/预算告警率","rate / budget")), h("th",{class:"num"},"weight_mult"), h("th",{class:"num"},lbl("降级占比","degraded")), h("th",{class:"num"},"n")));
    rows.forEach(r=>{
      const bad = (r.ks!=null && r.ks>0.3) || (r.rate_ratio!=null && r.rate_ratio>2);
      tb2.appendChild(h("tr",{class:bad?"stale-row":""}, h("td",{class:"mono small"}, r.detector), h("td",{class:"small"}, r.family),
        h("td",{class:"num small"}, fmtN(r.ks)), h("td",{class:"num small"}, fmtN(r.rate_ratio)), h("td",{class:"num small"}, fmtN(r.weight_mult)),
        h("td",{class:"num small"}, r.degraded_share==null?"—":share(r.degraded_share)), h("td",{class:"num small"}, r.n==null?"—":String(r.n))));
    });
    c.appendChild(h("div",{class:"small muted mt6"}, lbl(`检测器校准（${sys}，B24 calib_health）`,`detector calibration (${sys}, B24 calib_health)`)));
    c.appendChild(h("div",{class:"tablewrap scroll"}, tb2));
    if(mem) c.appendChild(h("div",{class:"small muted mt6"}, lbl("存储内存 ","store memory "), h("b",{}, fmtN(mem.approx_bytes)+"B"),
      ` · ${fmtN(mem.bytes_per_entity)}B/${lbl("实体","entity")} · ${mem.vec_series} vec · ${mem.derived_series} derived · ${mem.incidents} incidents · ${mem.labels} labels`));
    return c;
  }

  // ================================================================ eval report
  ROUTES.eval = async () => {
    const root = h("section",{});
    root.appendChild(h("div",{class:"detail-head"}, h("h2",{}, lbl("评估报告","Evaluation report"))));
    let r;
    try{ r = await API.evalReport(); }
    catch(e){
      if(e.status===404){ root.appendChild(card(lbl("尚无评估报告","No evaluation report yet"),
        h("p",{class:"muted"}, lbl("运行评估后刷新本页：","Run the evaluation, then reload:")),
        h("pre",{class:"mono small"}, ".venv/bin/python scripts/evaluate.py --packs A,B,C,D,E --seeds 0,1,2 --out reports"),
        h("div",{class:"muted small"}, ((e.body||{}).searched||[]).join("  ·  ")))); return root; }
      throw e;
    }
    const s = r.summary || {};
    root.appendChild(h("div",{class:"grid g4"},
      kpiTile(lbl("运行数","runs"), s.n_runs??"—", h("span",{class:"muted small"}, (s.packs||[]).join(","))),
      kpiTile(lbl("通过","pass"), s.gates_pass??"—", h("span",{class:"badge sev-low"},"pass")),
      kpiTile(lbl("失败","fail"), s.gates_fail??"—", s.gates_fail?h("span",{class:"badge sev-critical"},"fail"):""),
      kpiTile(lbl("不适用","n/a"), s.gates_na??"—", s.all_pass?h("span",{class:"badge sev-low"}, lbl("全部通过","all pass")):"")));
    const gc = h("div",{class:"card tablewrap mt"}, h("h3",{}, lbl("验收门","Gates"), h("span",{class:"muted small"}, "  "+(r.generated||""))));
    const tb = h("table",{}, h("tr",{}, h("th",{},"#"), h("th",{},lbl("名称","gate")), h("th",{},lbl("结果","result")), h("th",{},lbl("值","value")), h("th",{},lbl("目标","target")), h("th",{},lbl("检查项","checks"))));
    Object.entries(r.gates||{}).forEach(([k,g])=>{
      const checks = ((g.details||{}).checks||[]);
      tb.appendChild(h("tr",{}, h("td",{class:"small mono"}, k.split("_")[0]), h("td",{}, g.name||k), h("td",{}, passPill(g.pass)),
        h("td",{class:"num small"}, fmtN(g.value)), h("td",{class:"num small"}, fmtN(g.target)),
        h("td",{}, checks.length?h("details",{}, h("summary",{class:"small"}, `${checks.filter(c=>c.pass===true).length}/${checks.length}`),
          ...checks.map(ch=>h("div",{class:"small rowline"}, passPill(ch.pass), ch.name, h("span",{class:"muted right"}, `${fmtN(ch.value)} / ${fmtN(ch.target)}`)))):"—")));
    });
    gc.appendChild(tb); root.appendChild(gc);
    [["scenarios",lbl("场景（检测时延）","Scenarios (time to detect)")],["far",lbl("误报率","False-alarm rate")],["legit",lbl("合法变更","Legitimate change")],["runs",lbl("运行","Runs")]].forEach(([k,title])=>{
      root.appendChild(h("div",{class:"card mt"}, h("h3",{}, title), genericTable(r[k]||[], 12)));
    });
    if(r.design_table_md) root.appendChild(h("div",{class:"card mt"}, h("h3",{}, lbl("设计表","Design table")), h("pre",{class:"small"}, r.design_table_md)));
    return root;
  };
  function passPill(p){ return p===true?h("span",{class:"badge sev-low"},"✓ pass"):p===false?h("span",{class:"badge sev-critical"},"✗ fail"):h("span",{class:"badge sev-info"},"n/a"); }
})();
