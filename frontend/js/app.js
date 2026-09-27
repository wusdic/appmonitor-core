// AppMonitor Core — dashboard controller: shared helpers, hash router and the
// v1 views (overview, entity list, events, signatures, catalog, engines).
// The lib-3 v2 pages (entity, class, incident queue, system view, eval) live
// in pages.js.
const SEV = ["info","low","medium","high","critical"];
// status colours (reserved; always shown with a text label)
const SEV_COLOR = {info:"#4da3ff",low:"#0ca30c",medium:"#fab219",high:"#ec835a",critical:"#d03b3b"};
const TIER_ZH = {low:"低",medium:"中",high:"高",critical:"严重"};
const LAYER_CN = {raw:"原始指标库",derived:"次生指标库",behavior:"行为库",signature:"行为特征库"};
const CAT_COLOR = {browse:"#4da3ff",search:"#4da3ff",write:"#7bd88f",api:"#2ec7a6",integration:"#2ec7a6",
  auth:"#f2c14e",admin:"#f2c14e",transfer:"#ff9f43",streaming:"#4da3ff",sync:"#2ec7a6",
  scan:"#ff5d6c",recon:"#ff5d6c",beacon:"#ff5d6c",tunnel:"#ff5d6c",exfil:"#ff5d6c",
  c2:"#ff5d6c",intrusion:"#ff5d6c",posture:"#f2c14e",health:"#f2c14e",maintenance:"#8ba0be",
  composite:"#7c5cff"};
const $ = s => document.querySelector(s);
const h = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const k in attrs) {
    const v = attrs[k];
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  kids.flat().forEach(k => { if (k === null || k === undefined || k === false || k === "") return;
    e.appendChild(typeof k === "string" || typeof k === "number" ? document.createTextNode(String(k)) : k); });
  return e;
};
let STATE = { view:"overview", system:null, systems:[], tz:"Asia/Shanghai", lang:"zh", params:[] };
try { STATE.lang = localStorage.getItem("appmon.lang") || "zh"; } catch (_) { /* storage blocked */ }
const t = (zh, en) => STATE.lang === "en" ? (en ?? zh) : zh;
const sevBadge = s => h("span",{class:"badge sev-"+(s||"info")}, s||"—");
const tierBadge = (tier, score) => tier
  ? h("span",{class:"badge sev-"+tier, title:"risk tier"}, t(TIER_ZH[tier]||tier, tier) + (score!=null?` · ${Math.round(score)}`:""))
  : h("span",{class:"muted small"},"—");
const catBadge = c => h("span",{class:"badge",style:`background:${(CAT_COLOR[c]||"#4da3ff")}22;color:${CAT_COLOR[c]||"#4da3ff"}`}, c);
const pct = v => Math.round((v||0)*100);
const fmtN = v => Charts.fmtNum(v);
function fmtTs(ts, withDate=true){
  if(ts==null) return "—";
  try{
    const o = {timeZone:STATE.tz, hour12:false, hour:"2-digit", minute:"2-digit", second:"2-digit"};
    if(withDate){ o.month="2-digit"; o.day="2-digit"; }
    return new Intl.DateTimeFormat(STATE.lang==="en"?"en-GB":"zh-CN", o).format(new Date(ts*1000));
  }catch(_){ return new Date(ts*1000).toLocaleString(); }
}
const fmtShort = ts => fmtTs(ts, true).replace(/:\d\d$/,"");
function fmtDays(days){
  if(days==null||!isFinite(days)) return "—";
  if(days>=3650) return t("超过万年一遇","rarer than once in 10k years").replace("万年", days>=3.65e6?"千万年":"万年");
  if(days>=365) return t(`约 ${Math.round(days/365)} 年一次`,`once in ~${Math.round(days/365)} years`);
  if(days>=1) return t(`约 ${Math.round(days)} 天一次`,`once in ~${Math.round(days)} days`);
  return t(`每天约 ${(1/days).toFixed(1)} 次`,`~${(1/days).toFixed(1)}× per day`);
}
const link = (href, ...kids) => h("a",{href}, ...kids);
const entHref = (sys, e) => e && e.startsWith("class:") ? `#/class/${encodeURIComponent(sys)}/${encodeURIComponent(e)}`
  : `#/entity/${encodeURIComponent(sys)}/${encodeURIComponent(e)}`;
function card(title, ...kids){ return h("div",{class:"card"}, title?h("h3",{},title):"", ...kids); }
function kvRow(k, v){ return h("div",{class:"kv"}, h("span",{class:"muted"},k), h("b",{},v==null||v===""?"—":v)); }
function empty(msg){ return h("div",{class:"muted small empty"}, msg||t("暂无数据","no data yet")); }
function toast(msg, ok=true){
  const d=h("div",{class:"toast "+(ok?"ok":"bad")}, msg); document.body.appendChild(d);
  setTimeout(()=>d.remove(), 3200);
}
function systemSelect(onChange){
  return h("select",{class:"search",onchange:e=>onChange(e.target.value)},
    ...STATE.systems.map(s=>h("option",{value:s,selected:s===STATE.system?"selected":null},s)));
}
function meter(v){ return h("div",{class:"rowline"}, h("div",{class:"meter",style:"flex:1"},
  h("span",{style:`width:${pct(v)}%`})), h("span",{class:"small muted"}, v==null?"—":pct(v)+"%")); }
function riskBar(score, tier){
  const w = Math.max(0, Math.min(100, score||0));
  return h("div",{class:"rowline"}, h("div",{class:"meter risk",style:"flex:1"},
    h("span",{style:`width:${w}%;background:${SEV_COLOR[tier]||"#4da3ff"}`})), tierBadge(tier, score));
}

// ------------------------------------------------------------------ router
const ROUTES = {};   // view -> render(params)
const TABS = ["overview","entities","classes","incidents","system","events","signatures","catalog","engines","eval"];
function route(){
  const hash = location.hash.replace(/^#\/?/,"") || "overview";
  const [path, query] = hash.split("?");
  const parts = path.split("/").map(decodeURIComponent);
  STATE.view = parts[0] || "overview";
  STATE.params = parts.slice(1);
  STATE.query = Object.fromEntries(new URLSearchParams(query||""));
  const tab = {entity:"entities", class:"classes", incident:"incidents"}[STATE.view] || STATE.view;
  document.querySelectorAll(".tab").forEach(b=>b.classList.toggle("active", b.dataset.view===tab));
  render();
}
async function render(){
  const root = $("#app"); const fn = ROUTES[STATE.view] || ROUTES.overview;
  const token = (STATE.renderToken = (STATE.renderToken||0)+1);
  try{
    const node = await fn(STATE.params, STATE.query);
    if(token !== STATE.renderToken || !node) return;       // a newer route won
    root.innerHTML=""; root.appendChild(node);
  }catch(e){
    console.error(e);
    if(token !== STATE.renderToken) return;
    root.innerHTML="";
    root.appendChild(card(t("加载失败","Failed to load"), h("div",{class:"muted"},
      e.status===404 ? t("未找到 (404)","not found (404)") : String(e.message||e))));
  }
}
const go = (hash) => { if(location.hash===hash) render(); else location.hash = hash; };

// ------------------------------------------------------------------ overview
ROUTES.overview = async () => {
  const o = await API.overview();
  const root = h("section",{});
  root.appendChild(pipelineFlow());
  const sevSeg = SEV.map(s=>({label:s,value:(o.severity_breakdown||{})[s]||0,color:SEV_COLOR[s]})).filter(x=>x.value);
  root.appendChild(h("div",{class:"grid g4"},
    kpi(t("业务系统","Systems"), o.systems.length, "monitored systems"),
    kpi(t("监测实体","Entities"), o.entity_count, "IP / IP-类"),
    kpi(t("行为事件","Events"), o.event_count, "lib-3 findings"),
    kpi(t("特征命中","Matches"), o.match_count, "signature matches")));
  const catData = Object.entries(o.category_breakdown||{}).sort((a,b)=>b[1]-a[1])
    .map(([k,v])=>({label:k,value:v,color:CAT_COLOR[k]||"#4da3ff"}));
  root.appendChild(h("div",{class:"grid g3",style:"margin-top:14px"},
    card(t("系统与实体","Systems"), systemsTable(o.systems)),
    card(t("事件严重度分布","Event severity"),
      sevSeg.length?Charts.donut(sevSeg):empty(),
      h("div",{class:"legend"}, ...sevSeg.map(s=>h("span",{}, h("i",{class:"dot",style:`background:${s.color}`}), `${s.label} ${s.value}`)))),
    card(t("活动类别命中 Top","Top activity categories"), Charts.bars(catData.slice(0,10),{h:Math.max(1,catData.slice(0,10).length)*26+8}))));
  $("#tickinfo").textContent = `tick ${o.tick_count} · live ${o.live_ticks}`;
  return root;
};
function kpi(title,val,sub){ return h("div",{class:"card"}, h("h3",{},title),
  h("div",{class:"kpi"}, String(val), h("small",{},sub))); }
function pipelineFlow(){
  const nodes = [["原始指标库","被动流量 + 主动探测","layer-raw"],
    ["次生指标库","聚合/比率/周期/熵/会话/图/趋势","layer-derived"],
    ["行为库","特征→基线→似然→校准→融合→风险→事件→画像","layer-behavior"],
    ["行为特征库","规则匹配 + 时序关联","layer-signature"]];
  const f=h("div",{class:"flow"});
  nodes.forEach((n,i)=>{ f.appendChild(h("div",{class:"node"},
    h("div",{class:"h "+n[2]},n[0]), h("div",{class:"muted small"},n[1])));
    if(i<nodes.length-1) f.appendChild(h("div",{class:"arrow"},"→")); });
  return f;
}
function systemsTable(systems){
  const tb=h("table",{}, h("tr",{}, h("th",{},t("系统","System")), h("th",{},t("实体数","Entities")), h("th",{})));
  systems.forEach(s=> tb.appendChild(h("tr",{},
    h("td",{}, h("b",{},s.id)), h("td",{class:"num"},String(s.entities)),
    h("td",{class:"links"}, link(`#/entities/${encodeURIComponent(s.id)}`, t("画像","entities")," →"),
      link(`#/system/${encodeURIComponent(s.id)}`, t("系统视图","system")," →")))));
  return h("div",{class:"tablewrap"}, tb);
}

// ------------------------------------------------------------------ entities
// Ranked by behavior.risk (B26); tier + risk replace the raw anomaly %.
ROUTES.entities = async (params) => {
  if(params[0]) STATE.system = params[0];
  if(!STATE.system) STATE.system = STATE.systems[0];
  const data = await API.entities(STATE.system);
  const root = h("section",{});
  root.appendChild(h("div",{class:"detail-head"}, h("h2",{},t("实体行为画像","Entities")),
    systemSelect(s=>go(`#/entities/${encodeURIComponent(s)}`)),
    h("span",{class:"muted small right"},t("按风险排序 · 点击行查看画像","ranked by risk · click a row"))));
  const tb=h("table",{}, h("tr",{},
    h("th",{},t("实体","Entity")), h("th",{},t("类路径","Class path")), h("th",{style:"min-width:170px"},t("风险","Risk")),
    h("th",{},t("趋势","Trend")), h("th",{},t("可辨识度","Identifiability")), h("th",{},t("事件","Incidents")),
    h("th",{},t("状态","Regime")), h("th",{},t("当前活动","Current activity"))));
  data.entities.forEach(e=>{
    tb.appendChild(h("tr",{class:"clickable",onclick:()=>go(entHref(STATE.system,e.entity))},
      h("td",{}, h("b",{class:"mono"},e.entity)),
      h("td",{}, e.class_path?h("span",{class:"tag",title:e.role_name||""},e.class_path):h("span",{class:"muted"},t("学习中","learning"))),
      h("td",{}, riskBar(e.risk, e.tier)),
      h("td",{class:"num"}, e.trend==null?"—":Math.round(e.trend)),
      h("td",{}, meter(e.separability)),
      h("td",{class:"num"}, e.open_incidents?h("span",{class:"badge sev-high"},String(e.open_incidents)):"0"),
      h("td",{}, regimeBadge(e.regime)),
      h("td",{}, e.current_category?catBadge(e.current_category):h("span",{class:"muted"},"—"),
        " ", h("span",{class:"small muted"}, e.current_activity||""))));
  });
  root.appendChild(h("div",{class:"card tablewrap"},tb));
  return root;
};
function regimeBadge(state){
  if(!state) return h("span",{class:"muted small"},"—");
  const c = {normal:"sev-low",suspect:"sev-medium",drifting:"sev-high",accepted:"sev-info",returned:"sev-low",rejected:"sev-critical"}[state]||"sev-info";
  return h("span",{class:"badge "+c}, state);
}

// ------------------------------------------------------------------ events
ROUTES.events = async () => {
  const o = await API.events();
  return h("section",{}, h("div",{class:"card tablewrap scroll"},
    h("h3",{},t("全局行为事件流","Behaviour events")), eventsTable(o.events,false)));
};
function eventsTable(events,compact){
  const tb=h("table",{}, h("tr",{},
    h("th",{},t("时间","Time")), compact?"":h("th",{},t("实体","Entity")), h("th",{},t("类型","Kind")),
    h("th",{},"e_day"), h("th",{},t("严重度","Severity")), h("th",{},t("说明","Description"))));
  events.forEach(e=>{ tb.appendChild(h("tr",{},
    h("td",{class:"small muted nowrap"},fmtTs(e.ts)),
    compact?"":h("td",{class:"mono small"}, link(entHref(e.system,e.entity), e.system+"/"+e.entity)),
    h("td",{}, kindBadge(e.kind)),
    h("td",{class:"num small"}, e.e_day==null?"—":fmtN(e.e_day)),
    h("td",{}, sevBadge(e.severity)),
    h("td",{class:"small"}, e.description))); });
  return tb;
}
function kindBadge(k){
  const c={incident:"#ec835a",regime:"#9085e9",first_seen:"#c98500",unknown_identity:"#d55181",identity_mismatch:"#d55181"}[k]||"#4da3ff";
  return h("span",{class:"badge",style:`background:${c}22;color:${c}`}, k);
}
function matchesTable(matches){
  const tb=h("table",{}, h("tr",{}, h("th",{},t("时间","Time")), h("th",{},t("类别","Category")), h("th",{},t("活动","Activity")), h("th",{},t("置信","Conf."))));
  matches.forEach(m=> tb.appendChild(h("tr",{},
    h("td",{class:"small muted nowrap"},fmtTs(m.ts)),
    h("td",{}, catBadge(m.category)),
    h("td",{class:"small"}, (m.signature_id.startsWith("composite:")?"⛓ ":"")+m.label),
    h("td",{class:"num"}, pct(m.confidence)+"%"))));
  return tb;
}

// ------------------------------------------------------------------ signatures
ROUTES.signatures = async () => {
  const s = await API.signatures();
  const root=h("section",{});
  const search = h("input",{class:"search",placeholder:t("筛选特征…","filter…"),oninput:e=>filterSig(e.target.value)});
  root.appendChild(h("div",{class:"detail-head"}, h("h2",{},t("行为特征库 · 指标组合 → 语义","Signature library")),
    h("span",{class:"chip"},t("原子特征 ","primitives "), h("b",{},String(s.primitives.length))),
    h("span",{class:"chip"},t("组合特征 ","composite "), h("b",{},String(s.composite.length))), search));
  const wrap=h("div",{class:"grid g3"});
  s.primitives.forEach(p=> wrap.appendChild(sigCard(p,false)));
  s.composite.forEach(p=> wrap.appendChild(sigCard(p,true)));
  root.appendChild(wrap);
  return root;
};
function sigCard(p,composite){
  const conds=[];
  (p.all||[]).forEach(c=>conds.push(["ALL",c])); (p.any||[]).forEach(c=>conds.push(["ANY",c]));
  (p.none||[]).forEach(c=>conds.push(["NOT",c]));
  const body = composite
    ? h("div",{class:"small muted"}, (p.sequence?("顺序: "+p.sequence.join(" → ")):("并发: "+(p.cooccur||[]).join(" + ")))+` · 窗口${p.window_s||300}s`)
    : h("div",{}, ...conds.map(([kind,c])=>h("div",{class:"chip",style:"margin:2px 4px 2px 0;display:inline-block"},
        h("span",{class:"muted"},kind+" "), h("b",{},c.metric), " "+(c.op||"")+" "+(Array.isArray(c.value)?("["+c.value.join(",")+"]"):c.value))));
  return h("div",{class:"card sigitem","data-txt":(p.id+" "+p.label+" "+p.category).toLowerCase()},
    h("div",{class:"rowline wrap"}, catBadge(p.category), sevBadge(p.severity),
      composite?h("span",{class:"tag"},"⛓ 组合"):"", h("b",{class:"right"},p.label)),
    h("div",{class:"small mono muted",style:"margin:4px 0"}, p.id),
    body, p.description?h("div",{class:"small",style:"margin-top:6px;color:#b9c7dd"},p.description):"");
}
function filterSig(q){ q=q.toLowerCase(); document.querySelectorAll(".sigitem").forEach(c=>
  c.classList.toggle("hidden", !!q && !c.dataset.txt.includes(q))); }

// ------------------------------------------------------------------ catalog
ROUTES.catalog = async () => {
  const c = await API.catalog();
  const root=h("section",{});
  const search=h("input",{class:"search",placeholder:t("筛选指标…","filter…"),oninput:e=>filterCat(e.target.value)});
  root.appendChild(h("div",{class:"detail-head"}, h("h2",{},t("指标库 · 原始 + 次生","Metric catalog")),
    h("span",{class:"chip"},t("原始指标 ","raw "), h("b",{},String(c.raw.length))),
    h("span",{class:"chip"},t("次生指标 ","derived "), h("b",{},String(c.derived.length))), search));
  root.appendChild(h("div",{class:"card tablewrap"}, h("h3",{},t("原始指标库","Raw metrics")), catTable(c.raw)));
  root.appendChild(h("div",{class:"card tablewrap",style:"margin-top:14px"}, h("h3",{},t("次生指标库","Derived metrics")), catTable(c.derived)));
  return root;
};
function catTable(rows){
  const tb=h("table",{}, h("tr",{},
    h("th",{},"指标"), h("th",{},"分类"), h("th",{},"单位"), h("th",{},"获取方式"), h("th",{},"引擎"), h("th",{},"说明")));
  rows.forEach(r=> tb.appendChild(h("tr",{class:"catrow","data-txt":(r.name+" "+r.category+" "+(r.desc||"")).toLowerCase()},
    h("td",{class:"mono small"}, r.name), h("td",{}, h("span",{class:"tag"},r.category)),
    h("td",{class:"small muted"}, r.unit||""), h("td",{}, methodBadge(r.method)),
    h("td",{class:"small mono muted"}, r.engine||""), h("td",{class:"small"}, r.desc||""))));
  return tb;
}
function methodBadge(m){ const map={passive_span:["被动·镜像解码","#4da3ff"],passive_flow:["被动·流记录","#4da3ff"],
  passive_log:["被动·日志","#4da3ff"],active_probe:["主动·探测","#ff9f43"],active_dns:["主动·DNS","#ff9f43"],
  active_tls:["主动·TLS","#ff9f43"],derived:["派生·计算","#2ec7a6"]};
  const [tx,c]=map[m]||[m,"#8ba0be"]; return h("span",{class:"badge",style:`background:${c}22;color:${c}`}, tx); }
function filterCat(q){ q=q.toLowerCase(); document.querySelectorAll(".catrow").forEach(r=>
  r.classList.toggle("hidden", !!q && !r.dataset.txt.includes(q))); }

// ------------------------------------------------------------------ engines
ROUTES.engines = async () => {
  const e = await API.engines();
  const root=h("section",{}, h("h2",{},t(`引擎拓扑 · 共 ${e.count} 个引擎`,`Engines · ${e.count}`)));
  ["raw","derived","behavior","signature"].forEach(layer=>{
    const list = e.layers[layer]||[];
    const c=h("div",{class:"card tablewrap",style:"margin-bottom:14px"},
      h("h3",{class:"layer-"+layer}, LAYER_CN[layer]+" · "+layer+" ("+list.length+")"));
    const tb=h("table",{}, h("tr",{}, h("th",{},"引擎"), h("th",{},"产出"), h("th",{},"消费"),
      h("th",{},"上一tick产出"), h("th",{},"说明")));
    list.forEach(en=> tb.appendChild(h("tr",{},
      h("td",{}, h("b",{class:"mono"},en.name), en.last_error?h("div",{class:"small",style:"color:var(--critical)"},en.last_error):""),
      h("td",{class:"small mono muted"}, (en.produces||[]).join(", ")),
      h("td",{class:"small mono muted"}, (en.consumes||[]).join(", ")),
      h("td",{class:"num"}, String(en.last_count)),
      h("td",{class:"small"}, en.description))));
    c.appendChild(tb); root.appendChild(c);
  });
  return root;
};

// ------------------------------------------------------------------ boot
async function refreshHealth(){
  try{ const hb=await API.health();
    STATE.tz = hb.tz || STATE.tz;
    $("#status").textContent = hb.warmed?`${t("运行中","live")} · ${fmtTs(hb.now)} · ${STATE.tz}`:t("预热中…","warming up…");
    $("#status").classList.toggle("live",!!hb.warmed);
  }catch(e){ $("#status").textContent=t("后端未就绪","backend not ready"); }
}
function applyLang(){
  document.documentElement.lang = STATE.lang==="en"?"en":"zh-CN";
  document.querySelectorAll("[data-zh]").forEach(n=>{ n.textContent = STATE.lang==="en"?n.dataset.en:n.dataset.zh; });
  const b=$("#langbtn"); if(b) b.textContent = STATE.lang==="en"?"中文":"EN";
}
async function boot(){
  try{ const sys = await API.systems(); STATE.systems=sys.systems; STATE.system=STATE.system||sys.systems[0]; }catch(_){ /* backend down */ }
  await refreshHealth();
  applyLang();
  const lb=$("#langbtn");
  if(lb) lb.addEventListener("click",()=>{ STATE.lang = STATE.lang==="en"?"zh":"en";
    try{ localStorage.setItem("appmon.lang",STATE.lang); }catch(_){ /* ignore */ }
    applyLang(); render(); });
  window.addEventListener("hashchange", route);
  route();
  setInterval(async ()=>{
    await refreshHealth();
    if(["overview","entities","incidents"].includes(STATE.view) && !STATE.params[1] && !document.querySelector(".no-autorefresh")) render();
  }, 8000);
}
document.addEventListener("DOMContentLoaded", boot);
