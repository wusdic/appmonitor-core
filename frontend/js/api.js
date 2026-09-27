// Thin API client over the backend REST layer (v1 + v2, docs/lib3/api_ui.md).
const API = (() => {
  const base = "";
  const enc = encodeURIComponent;
  async function get(path) {
    const r = await fetch(base + "/api" + path);
    if (!r.ok) {
      const err = new Error(path + " -> " + r.status);
      err.status = r.status;
      try { err.body = await r.json(); } catch (_) { /* not JSON */ }
      throw err;
    }
    return r.json();
  }
  async function post(path, body) {
    const r = await fetch(base + "/api" + path, {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) { const err = new Error((data && data.detail) || (path + " -> " + r.status));
      err.status = r.status; throw err; }
    return data;
  }
  const qs = (o) => {
    const p = Object.entries(o || {}).filter(([, v]) => v !== undefined && v !== null && v !== "");
    return p.length ? "?" + p.map(([k, v]) => `${k}=${enc(v)}`).join("&") : "";
  };
  const ent = (sys, e) => `/systems/${enc(sys)}/entities/${enc(e)}`;
  return {
    health: () => get("/health"),
    overview: () => get("/overview"),
    engines: () => get("/engines"),
    catalog: () => get("/catalog"),
    signatures: () => get("/signatures"),
    systems: () => get("/systems"),
    entities: (sys) => get(`/systems/${enc(sys)}/entities`),
    entity: (sys, e) => get(ent(sys, e)),
    metrics: (sys, e) => get(ent(sys, e) + "/metrics"),
    series: (sys, e, name) => get(ent(sys, e) + `/series?name=${enc(name)}`),
    events: (sys) => get("/events" + qs({ system: sys })),
    matches: (sys) => get("/matches" + qs({ system: sys })),
    // v2 — entity
    portrait: (sys, e, version) => get(ent(sys, e) + "/portrait" + qs({ version })),
    portraitDiff: (sys, e, from, to) => get(ent(sys, e) + "/portrait/diff" + qs({ from, to })),
    timeline: (sys, e, since) => get(ent(sys, e) + "/timeline" + qs({ since })),
    scores: (sys, e, since) => get(ent(sys, e) + "/scores" + qs({ since })),
    identity: (sys, e, since) => get(ent(sys, e) + "/identity" + qs({ since })),
    // v2 — classes
    classes: (sys) => get(`/systems/${enc(sys)}/classes`),
    klass: (sys, key) => get(`/systems/${enc(sys)}/classes/${key.split("/").map(enc).join("/")}`),
    // v2 — incidents / feedback
    incidents: (f) => get("/incidents" + qs(f)),
    incident: (id) => get(`/incidents/${enc(id)}`),
    incidentFeedback: (id, body) => post(`/incidents/${enc(id)}/feedback`, body),
    eventFeedback: (id, body) => post(`/events/${enc(id)}/feedback`, body),
    labelQueue: (sys) => get("/label-queue" + qs({ system: sys })),
    // v2 — operations
    systemSummary: (sys, since) => get(`/systems/${enc(sys)}/summary` + qs({ since })),
    detectorsHealth: () => get("/detectors/health"),
    evalReport: () => get("/eval/report"),
    memory: () => get("/store/memory"),
  };
})();
