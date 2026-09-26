// AppMonitor Core — dashboard controller.
const SEV = ["info","low","medium","high","critical"];
const SEV_COLOR = {info:"#4da3ff",low:"#7bd88f",medium:"#f2c14e",high:"#ff9f43",critical:"#ff5d6c"};
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
    if (k === "class") e.className = attrs[k];
    else if (k === "html") e.innerHTML = attrs[k];
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), attrs[k]);
    else e.setAttribute(k, attrs[k]);
  }
  kids.flat().forEach(k => e.appendChild(typeof k === "string" ? document.createTextNode(k) : k));
  return e;
};
const sevBadge = s => h("span",{class:"badge sev-"+s}, s);
const catBadge = c => h("span",{class:"badge",style:`background:${(CAT_COLOR[c]||"#4da3ff")}22;color:${CAT_COLOR[c]||"#4da3ff"}`}, c);
const pct = v => Math.round((v||0)*100);
const fmtTs = t => new Date(t*1000).toLocaleTimeString();

let STATE = { view:"overview", system:null, entity:null, systems:[] };

// ------------------------------------------------------------------ router
function switchView(v){
  STATE.view=v;
  document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("active",t.dataset.view===v));
  document.querySelectorAll(".view").forEach(s=>s.classList.add("hidden"));
  $("#view-"+v).classList.remove("hidden");
  render();
}
document.querySelectorAll(".tab").forEach(t=>t.addEventListener("click",()=>switchView(t.dataset.view)));

async function render(){
  try{
    if(STATE.view==="overview") await renderOverview();
    else if(STATE.view==="entities") await (STATE.entity?renderEntityDetail():renderEntities());
    else if(STATE.view==="events") await renderEvents();
    else if(STATE.view==="signatures") await renderSignatures();
    else if(STATE.view==="catalog") await renderCatalog();
    else if(STATE.view==="engines") await renderEngines();
  }catch(e){ console.error(e); }
}

// ------------------------------------------------------------------ overview
async function renderOverview(){
  const o = await API.overview();
  const root = $("#view-overview"); root.innerHTML="";
  root.appendChild(pipelineFlow());
  const sevSeg = SEV.map(s=>({label:s,value:(o.severity_breakdown||{})[s]||0,color:SEV_COLOR[s]})).filter(x=>x.value);
  const kpis = h("div",{class:"grid g4"},
    kpi("业务系统", o.systems.length, "monitored systems"),
    kpi("监测实体", o.entity_count, "IP / IP-类"),
    kpi("行为事件", o.event_count, "anomaly / drift / sequence"),
    kpi("特征命中", o.match_count, "signature matches"));
  root.appendChild(kpis);
  const catData = Object.entries(o.category_breakdown||{}).sort((a,b)=>b[1]-a[1])
    .map(([k,v])=>({label:k,value:v,color:CAT_COLOR[k]||"#4da3ff"}));
  root.appendChild(h("div",{class:"grid g3",style:"margin-top:14px"},
    h("div",{class:"card"}, h("h3",{},"系统与实体"), systemsTable(o.systems)),
    h("div",{class:"card"}, h("h3",{},"事件严重度分布"),
      sevSeg.length?Charts.donut(sevSeg):h("div",{class:"muted"},"暂无"),
      h("div",{class:"legend"}, ...sevSeg.map(s=>h("span",{}, h("i",{class:"dot",style:`background:${s.color}`}), `${s.label} ${s.value}`)))),
    h("div",{class:"card"}, h("h3",{},"活动类别命中 Top"), Charts.bars(catData.slice(0,10),{h:catData.slice(0,10).length*26+8}))));
  $("#tickinfo").textContent = `tick ${o.tick_count} · live ${o.live_ticks}`;
}

function kpi(title,val,sub){ return h("div",{class:"card"}, h("h3",{},title),
  h("div",{class:"kpi"}, String(val), h("small",{},sub))); }

function pipelineFlow(){
  const nodes = [["原始指标库","被动流量 + 主动探测","layer-raw"],
    ["次生指标库","聚合/比率/周期/熵/会话/图/趋势","layer-derived"],
    ["行为库","特征向量→基线→指纹→聚类→异常/漂移/序列","layer-behavior"],
    ["行为特征库","规则匹配 + 时序关联","layer-signature"]];
  const f=h("div",{class:"flow"});
  nodes.forEach((n,i)=>{ f.appendChild(h("div",{class:"node"},
    h("div",{class:"h "+n[2]},n[0]), h("div",{class:"muted small"},n[1])));
    if(i<nodes.length-1) f.appendChild(h("div",{class:"arrow"},"→")); });
  return f;
}
function systemsTable(systems){
  const t=h("table",{}, h("tr",{}, h("th",{},"系统"), h("th",{},"实体数"), h("th",{})));
  systems.forEach(s=> t.appendChild(h("tr",{},
    h("td",{}, h("b",{},s.id)), h("td",{class:"num"},String(s.entities)),
    h("td",{}, h("span",{class:"tag pill",onclick:()=>{STATE.system=s.id;STATE.entity=null;switchView("entities");}},"查看画像 →")))));
  return t;
}

// ------------------------------------------------------------------ entities
async function renderEntities(){
  const root=$("#view-entities"); root.innerHTML="";
  if(!STATE.system) STATE.system = (await API.systems()).systems[0];
  const sysSel = h("select",{class:"search",onchange:e=>{STATE.system=e.target.value;renderEntities();}},
    ...STATE.systems.map(s=>h("option",{value:s,...(s===STATE.system?{selected:"1"}:{})},s)));
  const data = await API.entities(STATE.system);
  root.appendChild(h("div",{class:"detail-head"}, h("h3",{style:"margin:0"},"实体行为画像"), sysSel,
    h("span",{class:"muted small right"},"按当前异常/漂移评分排序 · 点击行查看画像")));
  const t=h("table",{}, h("tr",{},
    h("th",{},"实体(IP/IP类)"), h("th",{},"行为类型(泛化)"), h("th",{},"可区分度"),
    h("th",{},"当前活动"), h("th",{},"异常"), h("th",{},"漂移"), h("th",{},"基线")));
  data.entities.forEach(e=>{
    const score=Math.max(e.anomaly_score,e.drift_score);
    const row=h("tr",{style:"cursor:pointer",onclick:()=>{STATE.entity=e.entity;renderEntityDetail();}},
      h("td",{}, h("b",{class:"mono"},e.entity)),
      h("td",{}, e.archetype?h("span",{class:"tag"},e.archetype):h("span",{class:"muted"},"学习中")),
      h("td",{}, meter(e.separability)),
      h("td",{}, e.current_category?catBadge(e.current_category):h("span",{class:"muted"},"—"),
        " ", h("span",{class:"small muted"}, e.current_activity||"")),
      h("td",{class:"num",style:scoreColor(e.anomaly_score)}, e.anomaly_score?pct(e.anomaly_score)+"%":"—"),
      h("td",{class:"num",style:scoreColor(e.drift_score)}, e.drift_score?pct(e.drift_score)+"%":"—"),
      h("td",{}, e.stable?h("span",{class:"badge sev-low"},"已建立"):h("span",{class:"badge sev-info"},"n="+e.sample_count)));
    t.appendChild(row);
  });
  root.appendChild(h("div",{class:"card scroll"},t));
}
function meter(v){ return h("div",{class:"rowline"}, h("div",{class:"meter",style:"flex:1"},
  h("span",{style:`width:${pct(v)}%`})), h("span",{class:"small muted"},pct(v)+"%")); }
function scoreColor(s){ if(s>=0.8)return "color:var(--critical);font-weight:700";
  if(s>=0.6)return "color:var(--high);font-weight:700"; if(s>0)return "color:var(--medium)"; return "color:var(--muted)"; }

async function renderEntityDetail(){
  const root=$("#view-entities"); root.innerHTML="";
  const d = await API.entity(STATE.system, STATE.entity);
  root.appendChild(h("div",{class:"detail-head"},
    h("span",{class:"back",onclick:()=>{STATE.entity=null;renderEntities();}},"← 返回列表"),
    h("h3",{style:"margin:0"}, h("span",{class:"mono"},d.entity)),
    h("span",{class:"muted"}, "@ "+d.system),
    d.archetype?h("span",{class:"tag"},"泛化行为类: "+d.archetype+" ("+pct(d.archetype_confidence)+"%)"):"" ,
    h("span",{class:"chip"}, "可区分度 ", h("b",{},pct(d.separability)+"%")),
    d.stable?h("span",{class:"badge sev-low"},"基线已建立"):h("span",{class:"badge sev-info"},"学习中 n="+d.sample_count)));
  // fingerprint feature bars (z vs baseline)
  const feats = d.features.slice().sort((a,b)=>Math.abs(b.z)-Math.abs(a.z));
  const fpCard = h("div",{class:"card"}, h("h3",{},"行为指纹 · 各维度相对基线偏离 (稳健z)"));
  feats.forEach(f=> fpCard.appendChild(zRow(f)));
  // events & matches
  const evCard = h("div",{class:"card scroll"}, h("h3",{},"行为事件"),
    d.events.length?eventsTable(d.events,true):h("div",{class:"muted"},"无"));
  const mCard = h("div",{class:"card scroll"}, h("h3",{},"当前行为特征命中（TA在做什么）"),
    d.matches.length?matchesTable(d.matches):h("div",{class:"muted"},"无"));
  root.appendChild(h("div",{class:"grid g2"}, fpCard, h("div",{class:"grid",style:"gap:14px"}, mCard, evCard)));
  // metric explorer
  const explorer = h("div",{class:"card"}, h("h3",{},"指标时序浏览"));
  const chartHost = h("div",{style:"margin-top:10px"});
  const sel = h("select",{class:"search",onchange:e=>drawSeries(e.target.value,chartHost)});
  // load the entity's available raw/derived metric names
  const ml = await fetch(`/api/systems/${encodeURIComponent(STATE.system)}/entities/${encodeURIComponent(STATE.entity)}/metrics`).then(r=>r.json());
  const allMetrics = [...ml.raw, ...ml.derived];
  allMetrics.forEach(m=> sel.appendChild(h("option",{value:m},m)));
  explorer.appendChild(sel); explorer.appendChild(chartHost);
  root.appendChild(explorer);
  if(allMetrics.length){ const pref = allMetrics.find(m=>m==="l4.bytes_up")||allMetrics[0]; sel.value=pref; drawSeries(pref,chartHost); }
}
function zRow(f){
  const z=Math.max(-12,Math.min(12,f.z)); const left=50+ (z/12)*50;
  const w=Math.abs(z/12)*50; const x=z>=0?50:left;
  const col = Math.abs(z)>=3?"#ff5d6c":Math.abs(z)>=2?"#ff9f43":"#4da3ff";
  return h("div",{class:"feat-bar",style:"margin:4px 0"},
    h("div",{class:"n"},f.name),
    h("div",{class:"zbar"}, h("div",{class:"mid"}), h("i",{style:`left:${x}%;width:${w}%;background:${col}`})),
    h("div",{class:"small mono",style:"width:56px;text-align:right;color:"+col}, (f.z>=0?"+":"")+f.z.toFixed(1)+"σ"));
}
async function drawSeries(name,host){
  host.innerHTML="";
  const s = await API.series(STATE.system, STATE.entity, name);
  const pts = s.points.map(p=>p.value).filter(v=>typeof v==="number");
  host.appendChild(h("div",{class:"muted small"}, name+" · "+s.kind+" · "+pts.length+" 点"));
  host.appendChild(Charts.sparkline(s.points,{w:640,h:120,color:"#7c5cff"}));
}

// ------------------------------------------------------------------ events
async function renderEvents(){
  const root=$("#view-events"); root.innerHTML="";
  const o = await API.events();
  root.appendChild(h("div",{class:"card scroll"}, h("h3",{},"全局行为事件流（异常 / 漂移 / 异常序列）"),
    eventsTable(o.events,false)));
}
function eventsTable(events,compact){
  const t=h("table",{}, h("tr",{},
    h("th",{},"时间"), compact?"":h("th",{},"实体"), h("th",{},"类型"), h("th",{},"评分"),
    h("th",{},"严重度"), h("th",{},"说明")));
  events.forEach(e=>{ t.appendChild(h("tr",{},
    h("td",{class:"small muted"},fmtTs(e.ts)),
    compact?"":h("td",{class:"mono small"}, e.system+"/"+e.entity),
    h("td",{}, kindBadge(e.kind)),
    h("td",{class:"num"}, pct(e.score)+"%"),
    h("td",{}, sevBadge(e.severity)),
    h("td",{class:"small"}, e.description))); });
  return t;
}
function kindBadge(k){ const m={anomaly:"异常",drift:"漂移",sequence:"异常序列",class:"分类"};
  const c={anomaly:"#ff9f43",drift:"#7c5cff",sequence:"#f2c14e"}[k]||"#4da3ff";
  return h("span",{class:"badge",style:`background:${c}22;color:${c}`}, m[k]||k); }
function matchesTable(matches){
  const t=h("table",{}, h("tr",{}, h("th",{},"时间"), h("th",{},"类别"), h("th",{},"活动"), h("th",{},"置信")));
  matches.forEach(m=> t.appendChild(h("tr",{},
    h("td",{class:"small muted"},fmtTs(m.ts)),
    h("td",{}, catBadge(m.category)),
    h("td",{class:"small"}, (m.signature_id.startsWith("composite:")?"⛓ ":"")+m.label),
    h("td",{class:"num"}, pct(m.confidence)+"%"))));
  return t;
}

// ------------------------------------------------------------------ signatures
async function renderSignatures(){
  const root=$("#view-signatures"); root.innerHTML="";
  const s = await API.signatures();
  const search = h("input",{class:"search",placeholder:"筛选特征…",oninput:e=>filterSig(e.target.value)});
  root.appendChild(h("div",{class:"detail-head"}, h("h3",{style:"margin:0"},"行为特征库 · 指标组合 → 语义"),
    h("span",{class:"chip"},"原子特征 ", h("b",{},String(s.primitives.length))),
    h("span",{class:"chip"},"组合特征 ", h("b",{},String(s.composite.length))), search));
  const wrap=h("div",{class:"grid",id:"sigwrap"});
  s.primitives.forEach(p=> wrap.appendChild(sigCard(p,false)));
  s.composite.forEach(p=> wrap.appendChild(sigCard(p,true)));
  root.appendChild(wrap);
}
function sigCard(p,composite){
  const conds=[];
  (p.all||[]).forEach(c=>conds.push(["ALL",c])); (p.any||[]).forEach(c=>conds.push(["ANY",c]));
  (p.none||[]).forEach(c=>conds.push(["NOT",c]));
  const body = composite
    ? h("div",{class:"small muted"}, (p.sequence?("顺序: "+p.sequence.join(" → ")):("并发: "+(p.cooccur||[]).join(" + ")))+` · 窗口${p.window_s||300}s`)
    : h("div",{}, ...conds.map(([kind,c])=>h("div",{class:"chip",style:"margin:2px 4px 2px 0;display:inline-block"},
        h("span",{class:"muted"},kind+" "), h("b",{},c.metric), " "+(c.op||"")+" "+(Array.isArray(c.value)?("["+c.value.join(",")+"]"):c.value))));
  return h("div",{class:"card sigitem","data-txt":(p.id+" "+p.label+" "+p.category).toLowerCase()},
    h("div",{class:"rowline"}, catBadge(p.category), sevBadge(p.severity),
      composite?h("span",{class:"tag"},"⛓ 组合"):"", h("b",{class:"right"},p.label)),
    h("div",{class:"small mono muted",style:"margin:4px 0"}, p.id),
    body, p.description?h("div",{class:"small",style:"margin-top:6px;color:#b9c7dd"},p.description):"");
}
function filterSig(q){ q=q.toLowerCase(); document.querySelectorAll(".sigitem").forEach(c=>
  c.classList.toggle("hidden", q && !c.dataset.txt.includes(q))); }

// ------------------------------------------------------------------ catalog
async function renderCatalog(){
  const root=$("#view-catalog"); root.innerHTML="";
  const c = await API.catalog();
  const search=h("input",{class:"search",placeholder:"筛选指标…",oninput:e=>filterCat(e.target.value)});
  root.appendChild(h("div",{class:"detail-head"}, h("h3",{style:"margin:0"},"指标库 · 原始 + 次生"),
    h("span",{class:"chip"},"原始指标 ", h("b",{},String(c.raw.length))),
    h("span",{class:"chip"},"次生指标 ", h("b",{},String(c.derived.length))), search));
  root.appendChild(h("div",{class:"card"}, h("h3",{},"原始指标库（原始指标 · 获取方式）"), catTable(c.raw)));
  root.appendChild(h("div",{class:"card",style:"margin-top:14px"}, h("h3",{},"次生指标库（组合/分析派生）"), catTable(c.derived)));
}
function catTable(rows){
  const t=h("table",{class:"cattable"}, h("tr",{},
    h("th",{},"指标"), h("th",{},"分类"), h("th",{},"单位"), h("th",{},"获取方式"), h("th",{},"引擎"), h("th",{},"说明")));
  rows.forEach(r=> t.appendChild(h("tr",{class:"catrow","data-txt":(r.name+" "+r.category+" "+(r.desc||"")).toLowerCase()},
    h("td",{class:"mono small"}, r.name),
    h("td",{}, h("span",{class:"tag"},r.category)),
    h("td",{class:"small muted"}, r.unit||""),
    h("td",{}, methodBadge(r.method)),
    h("td",{class:"small mono muted"}, r.engine||""),
    h("td",{class:"small"}, r.desc||""))));
  return t;
}
function methodBadge(m){ const map={passive_span:["被动·镜像解码","#4da3ff"],passive_flow:["被动·流记录","#4da3ff"],
  passive_log:["被动·日志","#4da3ff"],active_probe:["主动·探测","#ff9f43"],active_dns:["主动·DNS","#ff9f43"],
  active_tls:["主动·TLS","#ff9f43"],derived:["派生·计算","#2ec7a6"]};
  const [t,c]=map[m]||[m,"#8ba0be"]; return h("span",{class:"badge",style:`background:${c}22;color:${c}`}, t); }
function filterCat(q){ q=q.toLowerCase(); document.querySelectorAll(".catrow").forEach(r=>
  r.classList.toggle("hidden", q && !r.dataset.txt.includes(q))); }

// ------------------------------------------------------------------ engines
async function renderEngines(){
  const root=$("#view-engines"); root.innerHTML="";
  const e = await API.engines();
  root.appendChild(h("h3",{},`引擎拓扑 · 共 ${e.count} 个引擎（低耦合 · 可复用）`));
  ["raw","derived","behavior","signature"].forEach(layer=>{
    const list = e.layers[layer]||[];
    const card=h("div",{class:"card",style:"margin-bottom:14px"},
      h("h3",{class:"layer-"+layer}, LAYER_CN[layer]+" · "+layer+" ("+list.length+")"));
    const t=h("table",{}, h("tr",{}, h("th",{},"引擎"), h("th",{},"产出"), h("th",{},"消费"),
      h("th",{},"上一tick产出"), h("th",{},"说明")));
    list.forEach(en=> t.appendChild(h("tr",{},
      h("td",{}, h("b",{class:"mono"},en.name), en.last_error?h("div",{class:"small",style:"color:var(--critical)"},en.last_error):""),
      h("td",{class:"small mono muted"}, (en.produces||[]).join(", ")),
      h("td",{class:"small mono muted"}, (en.consumes||[]).join(", ")),
      h("td",{class:"num"}, String(en.last_count)),
      h("td",{class:"small"}, en.description))));
    card.appendChild(t); root.appendChild(card);
  });
}

// ------------------------------------------------------------------ boot
async function boot(){
  try{
    const sys = await API.systems(); STATE.systems=sys.systems; STATE.system=sys.systems[0];
    const hb = await API.health();
    $("#status").textContent = hb.warmed?`运行中 · live ${hb.live_ticks}`:"预热中…";
    $("#status").classList.toggle("live",hb.warmed);
  }catch(e){ $("#status").textContent="后端未就绪"; }
  render();
  setInterval(async ()=>{
    try{ const hb=await API.health();
      $("#status").textContent = hb.warmed?`运行中 · live ${hb.live_ticks}`:"预热中…";
      $("#status").classList.toggle("live",hb.warmed);
      if(STATE.view==="overview"||STATE.view==="events") render();
      if(STATE.view==="entities"&&!STATE.entity) render();
    }catch(_){}
  }, 4000);
}
boot();
