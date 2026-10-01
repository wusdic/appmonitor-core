"""Headless render check of the "画像模式" pages (API v3) with the
pre-installed Chromium (never `playwright install`).

Start the backend with the progressive core first, e.g.
    APPMON_PROGRESSIVE=decision APPMON_PROGRESSIVE_DAYS=5 \\
        .venv/bin/python -m uvicorn app.main:app --app-dir backend --port 8099
then run
    .venv/bin/python scripts/render_check_progressive.py

Every page is loaded in zh and en at desktop and phone width; a page fails
on any console error, an uncaught exception, a "加载失败 / Failed to load"
card or horizontal page scroll. One lattice node is clicked to load its
detail panel. Exit status 1 when anything failed.
"""
import glob
import json
import os
import sys
import urllib.request

from playwright.sync_api import sync_playwright

BASE = os.environ.get("APPMON_URL", "http://127.0.0.1:8099")
OUT = os.environ.get("OUT_DIR", "/tmp/claude-0/shots-pp")
_cands = glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome") + \
    glob.glob("/opt/pw-browsers/chromium/chrome-linux/chrome")
EXE = _cands[0] if _cands else "/opt/pw-browsers/chromium/chrome-linux/chrome"


def _json(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read())


def pages():
    st = _json("/api/v3/status")
    systems = [s["system"] for s in st["systems"]] or ["oa"]
    s0 = "oa" if "oa" in systems else systems[0]
    out = ["#/pp", f"#/pp/system/{s0}", "#/pp/groups", f"#/pp/lattice/{s0}", "#/pp/violations",
           f"#/pp/facets/{s0}", f"#/pp/strategy/{s0}", f"#/pp/attributes/{s0}", "#/pp/budget"]
    g = _json("/api/v3/groups")["groups"]
    if g:
        out.append(f"#/pp/group/{g[0]['id']}")
        ips = [m for m in g[0]["members"] if "/" not in m]
        sys_g = next(iter(g[0].get("systems") or {s0: 1}))
        if ips:
            out.append(f"#/pp/ip/{sys_g}/{ips[0]}")
    lat = _json(f"/api/v3/systems/{s0}/lattice?depth=2")
    if lat["nodes"]:
        out.append("#/pp/pattern/" + urllib.request.quote(lat["nodes"][-1]["pattern_id"], safe=""))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    errors = []
    hashes = pages()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=EXE, headless=True, args=["--no-sandbox", "--disable-gpu"])
        for lang in ("zh", "en"):
            for width in (1360, 390):
                ctx = browser.new_context(viewport={"width": width, "height": 900})
                ctx.add_init_script(f"try{{localStorage.setItem('appmon.lang','{lang}')}}catch(e){{}}")
                page = ctx.new_page()
                cur = {"h": ""}
                page.on("console", lambda m, cur=cur: errors.append(f"{cur['h']} console: {m.text}")
                        if m.type == "error" else None)
                page.on("pageerror", lambda e, cur=cur: errors.append(f"{cur['h']} pageerror: {e}"))
                for i, h in enumerate(hashes):
                    cur["h"] = f"{lang}/{width} {h}"
                    page.goto(f"{BASE}/app/{h}", wait_until="networkidle", timeout=60000)
                    page.wait_for_timeout(600)
                    txt = page.inner_text("#app")
                    if "加载失败" in txt or "Failed to load" in txt:
                        errors.append(f"{cur['h']} render failed: {txt[:200]}")
                    sw = page.evaluate("document.documentElement.scrollWidth")
                    if sw > width + 1:
                        errors.append(f"{cur['h']} horizontal scroll: {sw}px")
                    page.screenshot(path=os.path.join(OUT, f"{lang}_{width}_{i:02d}.png"))
                    print(f"{cur['h']}: {len(txt)} chars")
                    if h.startswith("#/pp/lattice/"):
                        nodes = page.locator(".pp-node")
                        if nodes.count() > 1:
                            nodes.nth(1).click()
                            page.wait_for_timeout(1200)
                            if not page.inner_text(".pp-detail").strip():
                                errors.append(f"{cur['h']} lattice detail empty")
                ctx.close()
        browser.close()
    print("ERRORS:", len(errors))
    for e in errors[:40]:
        print("  ", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
