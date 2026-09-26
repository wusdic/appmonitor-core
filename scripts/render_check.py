"""Headless render check of the dashboard using the pre-installed Chromium."""
import os, sys, time
from playwright.sync_api import sync_playwright

BASE = os.environ.get("APPMON_URL", "http://127.0.0.1:8099/app/")
OUT = os.environ.get("OUT_DIR", "/tmp/claude-0/shots")
os.makedirs(OUT, exist_ok=True)
import glob
_cands = glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome") + \
         glob.glob("/opt/pw-browsers/chromium/chrome-linux/chrome")
EXE = _cands[0] if _cands else "/opt/pw-browsers/chromium/chrome-linux/chrome"

def main():
    errors = []
    with sync_playwright() as p:
        launch = {"headless": True, "args": ["--no-sandbox", "--disable-gpu"]}
        browser = p.chromium.launch(executable_path=EXE, **launch)
        page = browser.new_page(viewport={"width": 1360, "height": 900})
        page.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type in ("error",) else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.goto(BASE, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1500)
        tabs = ["overview", "entities", "events", "signatures", "catalog", "engines"]
        for t in tabs:
            page.click(f'.tab[data-view="{t}"]')
            page.wait_for_timeout(900)
            page.screenshot(path=os.path.join(OUT, f"{t}.png"), full_page=True)
            print(f"shot {t} :: cards={page.locator('.card').count()}")
        # drill into an entity
        page.click('.tab[data-view="entities"]'); page.wait_for_timeout(700)
        rows = page.locator("#view-entities table tr")
        if rows.count() > 1:
            rows.nth(1).click(); page.wait_for_timeout(1200)
            page.screenshot(path=os.path.join(OUT, "entity_detail.png"), full_page=True)
            print("shot entity_detail :: feat-bars=", page.locator(".feat-bar").count())
        browser.close()
    print("CONSOLE ERRORS:", len(errors))
    for e in errors[:20]:
        print("  ", e)
    sys.exit(1 if errors else 0)

if __name__ == "__main__":
    main()
