// Thin API client over the backend REST layer.
const API = (() => {
  const base = "";
  async function get(path) {
    const r = await fetch(base + "/api" + path);
    if (!r.ok) throw new Error(path + " -> " + r.status);
    return r.json();
  }
  return {
    health: () => get("/health"),
    overview: () => get("/overview"),
    engines: () => get("/engines"),
    catalog: () => get("/catalog"),
    signatures: () => get("/signatures"),
    systems: () => get("/systems"),
    entities: (sys) => get(`/systems/${encodeURIComponent(sys)}/entities`),
    entity: (sys, e) => get(`/systems/${encodeURIComponent(sys)}/entities/${encodeURIComponent(e)}`),
    series: (sys, e, name) =>
      get(`/systems/${encodeURIComponent(sys)}/entities/${encodeURIComponent(e)}/series?name=${encodeURIComponent(name)}`),
    events: (sys) => get("/events" + (sys ? `?system=${encodeURIComponent(sys)}` : "")),
    matches: (sys) => get("/matches" + (sys ? `?system=${encodeURIComponent(sys)}` : "")),
  };
})();
