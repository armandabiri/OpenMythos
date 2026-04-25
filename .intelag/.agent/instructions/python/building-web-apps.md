# Intelag Web App Building Playbook

Step-by-step instructions for adding a FastAPI + Jinja web UI to **any**
Intelag project. Produces a branded dashboard with reusable kit components,
ribbon-tab navigation, a system-tray launcher, a Chromium `--app` window,
and platform-appropriate desktop shortcuts. Cross-platform
(Windows / macOS / Linux) with graceful fallbacks.

Applies IN ADDITION to `python.md` (core Python standards). Where the two
conflict, this playbook wins for files under the web-app package.

---

## PLACEHOLDERS

Substitute consistently throughout your project:

| Token                | Meaning                                        | Example                                  |
| -------------------- | ---------------------------------------------- | ---------------------------------------- |
| `{APP}`              | web-app package name (sibling of core)         | `idc_app`, `inventory_app`, `deploy_app` |
| `{CORE}`             | existing business-logic package                | `intelag_data_collection_manager`        |
| `{PROJECT}`          | PyPI project name (pyproject `[project].name`) | `intelag-data-collection-manager`        |
| `{BRAND}`            | company brand slug for assets folder           | `intelag`                                |
| `{APP_TITLE}`        | user-visible product name                      | `IDC Manager`, `Inventory Control`       |
| `{APP_SLUG}`         | kebab-case id (shortcut filename, ws key)      | `idc-manager`                            |
| `{APP_PORT}`         | default dev-server port                        | `8765`                                   |
| `{COPYRIGHT_HOLDER}` | footer copyright line                          | `Intelag Inc.`                           |

`{BRAND}` maps to an images folder: `{APP}/ui/static/images/{BRAND}/`.

---

## HARD RULES (web-app specific)

* File size ≤ 200 lines. Split the moment you hit ~180.
* `{APP}` is a TOP-LEVEL package — sibling of `{CORE}`, never nested inside it.
* Web stack deps live ONLY in `[project.optional-dependencies]`. Core CLI
  users must not install FastAPI to run non-UI commands.
* Every reusable UI component = 3 files: `theme.css` + `macros.html` +
  `macro.js` under their own kit folder. No exceptions.
* Every page = 3 files: `layout.html` + `bootstrap.js` + `page.css` under
  their own page folder. No exceptions.
* Every page shows the `{BRAND}` logo in its header. ALWAYS.
* Every page has a footer with copyright + app version + build link. ALWAYS.
* Subprocess spawning goes through `ProcessManager` — never raw
  `subprocess.Popen` in a route handler.
* Long-running children are spawned through a clean entry module
  (`{CORE}/cli/main_entry.py`) that bypasses any `__main__`-block argv
  overrides and forces line-buffered UTF-8 stdout.
* Windows: pass `-X utf8` to `pythonw.exe` in every desktop shortcut.
  Otherwise emoji in any logger crashes the child silently.
* Shortcut launchers include platform-appropriate icons (`.ico` on Windows,
  `.png` on Linux/macOS), auto-regenerated from the source SVG.
* `-l/--launch` must open a managed standalone app window (`Chromium --app`)
  when a Chromium-family browser is available. `webbrowser.open_*()` is a
  fallback only, never the primary standalone path.
* The launched app window is OWNED by the runtime. `Ctrl+C`, console close,
  tray **Quit**, fatal startup failure, or normal process exit must stop both
  the server and the standalone app window. No orphaned browser processes.
* Package `__init__.py` files must not eagerly import `{APP}.server` or create
  the FastAPI app as an import side effect. Expose `create_app` lazily so
  non-web CLI imports keep the web stack optional.
* Tray mode must not be a hard requirement — always fall back to
  "server-only" on runtime failure so the user keeps a working UI.

---

## STEP 1 — Project layout

```text
<repo-root>/
├── {CORE}/                            # existing business logic, UNCHANGED
│   ├── cli/
│   │   ├── main_entry.py              # NEW — clean subprocess entry
│   │   └── app_cmd.py                 # NEW — CLI group: `{cmd} app`
│   └── services/ …
├── {APP}/                             # NEW — the web app (SIBLING, not nested)
│   ├── __init__.py                    # exports create_app, PACKAGE_ROOT, __version__
│   ├── __main__.py                    # enables `python -m {APP}`
│   ├── server.py                      # FastAPI factory + uvicorn bootstrap
│   ├── api/
│   │   ├── routes/                    # one file per /api/* group
│   │   └── services/                  # ProcessManager, log_stream, bridges
│   └── ui/
│       ├── __init__.py                # STATIC_DIR, TEMPLATE_DIR
│       ├── manifest.py                # KIT_COMPONENTS, PAGES, RIBBON_TABS
│       ├── mount.py                   # mount_ui(app) helper
│       ├── pages_router.py            # renders pages from manifest
│       ├── static/
│       │   ├── css/
│       │   │   ├── app.css            # @layer order + imports
│       │   │   ├── tokens.css         # all CSS variables
│       │   │   ├── reset.css
│       │   │   ├── base.css
│       │   │   ├── utilities.css
│       │   │   ├── kit.css            # imports every kit/<c>/shared/theme.css
│       │   │   ├── kit/<component>/shared/theme.css
│       │   │   └── pages/<page>/page.css
│       │   ├── js/
│       │   │   ├── app-client.js      # shared fetch + helpers
│       │   │   ├── app-boot.js        # per-page bootstrap discovery
│       │   │   ├── kit/<component>/shared/macro.js
│       │   │   └── pages/<page>/bootstrap.js
│       │   └── images/{BRAND}/
│       │       ├── {BRAND}-logo.svg          # source of truth (colour)
│       │       ├── {BRAND}-logo-white.svg    # source (white variant)
│       │       └── …auto-generated {BRAND}-logo.ico/.png…
│       └── templates/
│           ├── base.html              # <html> skeleton, favicon, title
│           ├── shell/dashboard.html   # header + ribbon + page-root + footer
│           ├── kit/<component>/shared/macros.html
│           └── pages/<page>/layout.html
├── pyproject.toml
└── README.md
```

Name `{APP}` as `<project_short>_app` in snake_case.

---

## STEP 2 — pyproject.toml split

Core stays lean; FastAPI/uvicorn/jinja live in `[app]`; tray deps extend
`[app]` via self-reference.

```toml
dependencies = [
    # …only runtime deps needed by {CORE} CLI, no web stack…
]

[project.optional-dependencies]
dev = ["pytest"]

# pip install "{PROJECT}[app]"  — FastAPI web UI
app = [
    "fastapi",
    "jinja2",
    "python-multipart",
    "uvicorn[standard]",
    "websockets",
]

# pip install "{PROJECT}[tray]"  — app + tray + auto icon rebuild
tray = [
    "{PROJECT}[app]",                 # self-reference pulls in [app]
    "pystray>=0.19",
    "Pillow>=9.0",
    "cairosvg>=2.7",
]

all = ["{PROJECT}[app,tray]"]

[tool.setuptools.packages.find]
where = ["."]
include = ["{CORE}*", "{APP}*"]

[tool.setuptools.package-data]
"{APP}.ui" = ["static/**/*", "templates/**/*"]
```

---

## STEP 3 — FastAPI factory (`{APP}/server.py`)

Keep under 120 lines. Only wires routes + lifespan — never business logic.

```python
"""FastAPI application factory for {APP_TITLE}."""

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import system as system_routes
# …one import per route group…
from .api.services.process_manager import ProcessManager
from .ui.mount import mount_ui
from . import __version__


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        pm: Optional[ProcessManager] = getattr(app.state, "process_manager", None)
        if pm is not None:
            await pm.shutdown()


def create_app(config_dir: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="{APP_TITLE}", version=__version__, lifespan=_lifespan)
    app.state.config_dir = config_dir
    app.state.process_manager = ProcessManager()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(Exception)
    async def _unhandled(_request, exc):
        return JSONResponse({"error": str(exc), "type": type(exc).__name__}, status_code=500)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "version": __version__}

    _register_api_routes(app)
    mount_ui(app)
    return app
```

`/healthz` is mandatory — load balancers, docker healthchecks, and the
dashboard's own header status dot all rely on it.

---

## STEP 4 — Route files (one per `/api/*` group)

```python
"""<group> — brief description."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class FooRequest(BaseModel):
    field: Optional[str] = None


@router.post("/action")
async def action(payload: FooRequest, request: Request) -> dict:
    pm = request.app.state.process_manager
    proc = await pm.start("foo", ["foo", "run"],
                          config_dir=request.app.state.config_dir)
    return proc.snapshot()
```

* Shared state lives on `request.app.state.*`. No module globals.
* Background work goes through `ProcessManager.start(key, args, config_dir)`.

---

## STEP 5 — ProcessManager (long-running subprocess registry)

`{APP}/api/services/process_manager.py` — critical infrastructure.

**Mandatory features:**

* Key-based registry; duplicate `start(key, …)` raises.
* `asyncio.create_subprocess_exec` with `stdout=PIPE, stderr=STDOUT`.
* Windows: `creationflags=subprocess.CREATE_NEW_PROCESS_GROUP`.
  POSIX: `start_new_session=True`. Reason: `Ctrl+Break` / signals target
  only the child group, not the UI server.
* Stdout pumped line-by-line into `deque(maxlen=2000)` + fanned out to
  WebSocket subscribers.
* `subscribe(key)` returns an `asyncio.Queue` pre-seeded with the current
  buffer — reconnecting clients see recent history immediately.
* Three-stage `stop(key, timeout=10.0)`:
  1. Graceful: `CTRL_BREAK_EVENT` on Windows / `SIGTERM` on POSIX — wait `timeout/2`.
  2. Terminate: `proc.terminate()` — wait `timeout/2`.
  3. Hard kill: `proc.kill()`.
* `shutdown()` reaps all tracked children on FastAPI lifespan exit.

---

## STEP 6 — Clean subprocess entry module (`{CORE}/cli/main_entry.py`)

NOT optional. Solves three pythonw.exe hazards that *will* bite you:

* Existing `cli/idc.py` (or whatever your CLI entry is) may `cls` the
  terminal or hardcode `sys.argv` for dev ergonomics → calling `-m`
  silently runs the wrong command.
* `pythonw.exe` default encoding is `cp1252` → emoji in logger formats →
  `UnicodeEncodeError` → child dies silently.
* `pythonw` streams have no writable `.buffer`; writes go to NUL.

Template:

```python
"""Subprocess-safe entry point — bypasses CLI __main__ tricks."""

import io, os, sys, tempfile
from pathlib import Path


def _looks_dead(stream) -> bool:
    if stream is None: return True
    if getattr(stream, "closed", False): return True
    name = getattr(stream, "name", "")
    if isinstance(name, str) and name.lower() in ("nul", "/dev/null"): return True
    fileno = getattr(stream, "fileno", None)
    if callable(fileno):
        try:
            fd = fileno()
            if isinstance(fd, int) and fd < 0: return True
        except (OSError, ValueError): return True
    return False

def _setup_log_file_streams() -> None:
    log_path = Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "{APP_SLUG}.log"
    fh = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)
    sys.stdout = fh
    sys.stderr = fh

def _rebind_line_buffered_utf8() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None: continue
        buf = getattr(stream, "buffer", None)
        if buf is None: continue
        detach = getattr(stream, "detach", None)
        if callable(detach):
            try: buf = detach()
            except (ValueError, OSError): pass
        setattr(sys, name, io.TextIOWrapper(buf, encoding="utf-8",
            errors="replace", line_buffering=True, write_through=True))

def _prepare_streams() -> None:
    if _looks_dead(sys.stdout) or _looks_dead(sys.stderr):
        _setup_log_file_streams()
    else:
        _rebind_line_buffered_utf8()


# Import main FIRST so target module's own stdout rebind runs, THEN apply
# ours exactly once. Two rebinds create a TextIOWrapper GC race that
# closes fd 1 (print silently vanishes).
from {CORE}.cli.idc import main  # noqa: E402
_prepare_streams()


if __name__ == "__main__":
    try: main()
    except SystemExit: raise
    except BaseException:
        import traceback; traceback.print_exc()
        raise
```

Spawn children as:

```python
[sys.executable, "-X", "utf8", "-u", "-m", "{CORE}.cli.main_entry", *args]
```

---

## STEP 7 — UI shell (CSS `@layer` order is mandatory)

`{APP}/ui/static/css/app.css`:

```css
@layer tokens, reset, base, utilities, kit, theme, page;

@import url("./tokens.css");
@import url("./reset.css");
@import url("./base.css");
@import url("./utilities.css");
@import url("./kit.css");
@import url("./{BRAND}-theme.css");
```

`tokens.css` is the ONLY file that defines CSS variables. Dark theme is
default; light theme uses `:root[data-theme="light"]` overrides.

Required token families — NO magic numbers anywhere else:

* **Colors**: `--{BRAND}-bg`, `--{BRAND}-panel`, `--{BRAND}-text`,
  `--{BRAND}-accent`, `--{BRAND}-success`, `--{BRAND}-danger`,
  `--{BRAND}-warning`, `--{BRAND}-info`.
* **Spacing**: `--{BRAND}-space-1` through `--{BRAND}-space-7` (0.25rem → 3rem).
* **Radii**: `--{BRAND}-radius-sm|md|lg|pill`.
* **Typography**: `--{BRAND}-font-sans`, `--{BRAND}-font-mono`,
  `--{BRAND}-text-xs|sm|md|lg|xl|2xl`.
* **Shadows**: `--{BRAND}-shadow-sm|md|lg`.
* **Layout**: `--{BRAND}-header-h`, `--{BRAND}-ribbon-h`,
  `--{BRAND}-footer-h`, `--{BRAND}-page-maxw`.

---

## STEP 8 — Kit component anatomy (3 files, always)

```text
static/css/kit/<component>/shared/theme.css    — styles only, one @layer kit block
static/js/kit/<component>/shared/macro.js      — ES module, default export = factory
templates/kit/<component>/shared/macros.html   — Jinja macro(s), no logic
```

**CSS naming (BEM-ish):**

* Base: `.kit-<component>`
* Parts: `.kit-<component>__<part>`
* Variants: `.kit-<component>--<variant>`
* State: `.is-active`, `.is-loading`, `.is-running`, …

**HTML hooks:**

* `data-kit="<component>"` on the root element.
* `data-role="<role>"` on sub-elements the JS factory queries/updates.
* `data-action="<action>"` on clickables for event-delegation targets.

**JS factory shape:**

```javascript
// kit/<component>/shared/macro.js
export default function create<Component>(element) {
  const inner = element.querySelector('[data-role="inner"]');
  return {
    update(value) { inner.textContent = value; },
    destroy() { /* remove listeners, close WS, etc. */ },
  };
}
```

Take ONLY the root element. Options come through `data-*`. Always return
an object with `destroy()` for lifecycle cleanup.

**Jinja macro shape:** every macro accepts a trailing `classes=""` param so
callers can mix in page-level adjustments without template forks.

```jinja
{% macro <component>(label, variant="default", classes="") -%}
<div class="kit-<component>{% if variant != 'default' %} kit-<component>--{{ variant }}{% endif %} {{ classes }}"
     data-kit="<component>">
  <span data-role="inner">{{ label }}</span>
</div>
{%- endmacro %}
```

---

## STEP 9 — Required kit components (every project)

Minimum component set you always build first:

| Component        | Purpose                                                       |
| ---------------- | ------------------------------------------------------------- |
| `header`         | Top bar with logo + title + status indicator + theme toggle   |
| `footer`         | Bottom bar with copyright + version + docs link (see STEP 11) |
| `ribbon`         | Horizontal tab navigation                                     |
| `panel`          | Bordered container with title/subtitle/body/footer slots      |
| `card`           | Stat card (eyebrow + value + label + trend)                   |
| `button`         | Primary/success/danger/warning/ghost variants + icon/sm sizes |
| `badge`          | Status chip (success/danger/warning/info/accent + pulse)      |
| `meter`          | Horizontal progress bar (cpu/mem/disk style)                  |
| `table`          | Scrollable data table with sticky header                      |
| `log-list`       | WebSocket-backed streaming log viewer                         |
| `toast`          | Ephemeral notification (success/danger/warning)               |
| `form` + `input` | Form row + text/select/textarea/switch primitives             |
| `modal`          | Overlay dialog                                                |
| `tabs`           | In-panel sub-navigation                                       |

Optional (add when needed):

| Component      | Purpose                           |
| -------------- | --------------------------------- |
| `oscilloscope` | Live SVG rate chart (see STEP 18) |
| `sidebar`      | Collapsible side navigation       |
| `drawer`       | Slide-in panel                    |
| `tooltip`      | Hover hint                        |

---

## STEP 10 — Page anatomy (3 files, always)

```
templates/pages/<page>/layout.html       — extends shell/dashboard.html
static/css/pages/<page>/page.css         — page-scoped overrides (@layer page)
static/js/pages/<page>/bootstrap.js      — default export = init(root)
```

**`layout.html`** — always wrap the body in
`<section class="page page--<page>" data-page-root="<page>">` with a
`<header class="page__header">` and a grid of `panel` macros.

**`bootstrap.js`** — default export is `async function init(root)`. Use
`import()` to lazy-load heavy modules (charts, codecs).

**Discovery** in `app-boot.js`:

```javascript
const root = document.getElementById("page-root");
const id = root?.dataset.page;
if (id) import(`./pages/${id}/bootstrap.js`).then(m => m?.default?.(root));
```

---

## STEP 11 — BRANDING CONTRACT (mandatory on every page)

Every page, without exception:

### A. Logo in the header

The `header` kit component shows the `{BRAND}` logo + `{APP_TITLE}`. Two
variants are embedded (white + colour); CSS picks one per theme:

```jinja
<span class="kit-header__logo" aria-hidden="true">
  <img src="{{ _assets }}/images/{BRAND}/{BRAND}-logo-white.svg"
       alt="{BRAND}"
       class="kit-header__logo-img kit-header__logo-img--dark" />
  <img src="{{ _assets }}/images/{BRAND}/{BRAND}-logo.svg"
       alt="{BRAND}"
       class="kit-header__logo-img kit-header__logo-img--light" />
</span>
```

```css
.kit-header__logo-img { height: 28px; display: none; }
:root[data-theme="dark"]  .kit-header__logo-img--dark  { display: block; }
:root[data-theme="light"] .kit-header__logo-img--light { display: block; }
```

### B. Favicon in `base.html`

```html
<link rel="icon" type="image/svg+xml"
      href="{{ asset_base }}/images/{BRAND}/{BRAND}-logo-white.svg" />
<link rel="alternate icon" type="image/svg+xml"
      href="{{ asset_base }}/images/{BRAND}/{BRAND}-logo.svg" />
<link rel="apple-touch-icon"
      href="{{ asset_base }}/images/{BRAND}/{BRAND}-logo.svg" />
<meta name="theme-color" content="#08161b" media="(prefers-color-scheme: dark)" />
<meta name="theme-color" content="#eef7fa" media="(prefers-color-scheme: light)" />
<meta name="application-name" content="{APP_TITLE}" />
```

### C. Footer on every page

Build a `kit/footer` component; include it in `shell/dashboard.html` so
every page inherits it.

`templates/kit/footer/shared/macros.html`:

```jinja
{% macro footer(version, year, docs_href="", repo_href="") -%}
<footer class="kit-footer" data-kit="footer">
  <div class="kit-footer__left">
    <span class="kit-footer__copyright">© {{ year }} {COPYRIGHT_HOLDER}. All rights reserved.</span>
  </div>
  <div class="kit-footer__right">
    <span class="kit-footer__version u-mono">{APP_TITLE} v{{ version }}</span>
    {% if docs_href %}<a class="kit-footer__link" href="{{ docs_href }}" target="_blank" rel="noopener">Docs</a>{% endif %}
    {% if repo_href %}<a class="kit-footer__link" href="{{ repo_href }}" target="_blank" rel="noopener">Repo</a>{% endif %}
  </div>
</footer>
{%- endmacro %}
```

`static/css/kit/footer/shared/theme.css`:

```css
@layer kit {
  .kit-footer {
    display: flex; justify-content: space-between; align-items: center;
    gap: var(--{BRAND}-space-3);
    padding: var(--{BRAND}-space-2) var(--{BRAND}-space-5);
    height: var(--{BRAND}-footer-h);
    border-top: 1px solid var(--{BRAND}-border);
    background: var(--{BRAND}-shell-soft);
    color: var(--{BRAND}-text-muted);
    font-size: var(--{BRAND}-text-xs);
    font-family: var(--{BRAND}-font-mono);
  }
  .kit-footer__right { display: flex; gap: var(--{BRAND}-space-3); align-items: center; }
  .kit-footer__link { color: var(--{BRAND}-accent); text-decoration: none; }
  .kit-footer__link:hover { text-decoration: underline; }
}
```

Wire into `shell/dashboard.html` (after the `<main>`):

```jinja
{% from "kit/footer/shared/macros.html" import footer with context %}
…
<main …>{% block content %}{% endblock %}</main>
{{ footer(version=app_version, year=app_year,
          docs_href=docs_url|default(''), repo_href=repo_url|default('')) }}
```

`app_version`, `app_year`, `docs_url`, `repo_url` are injected by
`pages_router.py` into every page context. Defaults: `__version__` from
`{APP}/__init__.py`; `datetime.now().year`.

### D. Logo in README

Every project README opens with the logo:

```markdown
<p align="center">
  <img src="{APP}/ui/static/images/{BRAND}/{BRAND}-logo.svg" alt="{BRAND}" width="160" />
</p>

# {APP_TITLE}

One-line tagline describing what this project does.
```

Relative paths resolve both on GitHub and locally.

---

## STEP 12 — Ribbon navigation (manifest-driven)

`{APP}/ui/manifest.py`:

```python
PAGES: tuple[str, ...] = ("dashboard", "tasks", "logs", "settings", "about")

PAGE_ROUTES: dict[str, str] = {"dashboard": "/", "tasks": "/tasks", …}

PAGE_TITLES: dict[str, dict[str, str]] = {
    "dashboard": {"title": "Dashboard", "subtitle": "Overview"},
    …
}

RIBBON_TABS: tuple[dict[str, str], ...] = tuple(
    {"id": p, "label": PAGE_TITLES[p]["title"], "href": PAGE_ROUTES[p]}
    for p in PAGES
)
```

`pages_router.py` iterates `PAGES`, registers `GET` handlers, passes
`ribbon_tabs=RIBBON_TABS` + `active=page_id` + `app_version` + `app_year`
into every template context.

Adding a new page = one entry in `PAGES` + three files (STEP 10).

---

## STEP 13 — Shell imports with context

Jinja macros DON'T inherit caller context automatically. The shell template
MUST import with context:

```jinja
{% from "kit/header/shared/macros.html" import header with context %}
{% from "kit/footer/shared/macros.html" import footer with context %}
```

Without `with context`, `{{ asset_base }}`, `{{ app_version }}`, etc. are
unresolved inside the macro body.

---

## STEP 14 — Standalone Chromium window (`-l/--launch`)

`{CORE}/cli/app_cmd.py` — launch flow:

1. `_chromium_candidates()` — walk known Chrome/Edge/Brave/Chromium install
   paths on Windows and macOS, use `shutil.which(...)` on Linux. Honour
   `BROWSER_EXECUTABLE` / `CHROME_EXECUTABLE` env overrides.
2. `_launch_app_window(url)` — spawn first found with:

   ```
   --app=<url>
   --user-data-dir=<HOME>/.{APP_SLUG}/app_profile
   --no-first-run
   --no-default-browser-check
   ```

   Detached (`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` on Windows,
   `start_new_session=True` on POSIX).
3. `_wait_for_server(url, timeout=20)` — poll before launching so the
   window doesn't flash "can't reach this site".
4. Wrap the launched process in an `AppWindowController` (or equivalently
   named helper) that owns:
   * the background launch thread
   * the live `subprocess.Popen` handle for the app window
   * `launch(url)` and `stop()` methods
5. `launch(url)` contract:
   * wait for `/healthz` first
   * launch at most one live window at a time
   * no-op if the app is already open and healthy
   * run in a background thread so uvicorn is never blocked
6. `stop()` contract:
   * terminate the managed app-window process if it is still alive
   * be called from EVERY shutdown path: `Ctrl+C`, console close,
     tray **Quit**, outer `finally:` blocks, and fatal startup failure
   * never raise if the window is already gone
7. No Chromium found → fall back to `webbrowser.open_new_tab(url)`.

Plain `webbrowser.open_*()` does **not** make the app standalone. It opens
an unmanaged browser tab/window that the runtime cannot reliably close later.

---

## STEP 15 — System tray (`--tray`)

`pystray` + `Pillow` + (Linux) AppIndicator.

* Import `pystray` INSIDE the function, not module-level. On ImportError,
  print a platform-specific install hint and `sys.exit(2)`. On Linux,
  include:

  ```
  sudo apt install gir1.2-ayatanaappindicator3-0.1
  ```

* `icon.run()` raising at runtime (Wayland without AppIndicator, headless,
  broken D-Bus) must NOT crash. Log a warning, block on the uvicorn
  thread, honour `KeyboardInterrupt`. Server remains reachable via browser.
* `Ctrl+C` MUST still work in tray mode.
* Use one shared shutdown coordinator for:
  * tray **Quit**
  * `SIGINT` / `SIGTERM` / `SIGBREAK`
  * console close / logoff / shutdown events on Windows
  * outer `finally:` cleanup
* The shutdown coordinator must stop in this order:
  1. mark server exit (`server.should_exit = True`)
  2. stop the managed standalone app window
  3. stop the tray icon
* On Windows, signal handlers / console-control handlers should only notify
  the main runtime loop that shutdown was requested. Do NOT do heavy teardown
  directly inside the callback; perform the actual stop sequence on the main
  thread / coordinator loop instead.

Tray menu (minimum):

* **Open {APP_TITLE}** — respawn the `--app` window. `default=True`.
* `URL: http://…` — disabled label.
* **Copy URL** — `clip` / `pbcopy` / `xclip` then `wl-copy`. Silent noop
  if none found.
* **Quit** — invoke the shared shutdown coordinator.

Uvicorn runs in a daemon thread. On POSIX, `icon.run()` on the main thread is
usually fine. On Windows, keep the main thread responsive to terminal signals;
if necessary, run the tray loop on a background thread and block the main
thread in a join/shutdown loop instead.

---

## STEP 16 — Desktop shortcut / installer (`{cmd} app install`)

Three output formats, one per platform:

| OS      | Path                                                                              | Format         | Method                                    |
| ------- | --------------------------------------------------------------------------------- | -------------- | ----------------------------------------- |
| Windows | `%USERPROFILE%\Desktop\{APP_TITLE}.lnk`                                           | `.lnk`         | PowerShell `WScript.Shell.CreateShortcut` |
| macOS   | `~/Desktop/{APP_TITLE}.command` + `~/Applications/{APP_TITLE}.command`            | bash script    | write + `chmod 0755`                      |
| Linux   | `~/.local/share/applications/{APP_SLUG}.desktop` + `~/Desktop/{APP_SLUG}.desktop` | `.desktop` INI | write + `chmod 0755`                      |

Windows `.lnk` target = `pythonw.exe` (silent); arguments =

```
-X utf8 -m {CORE}.cli.main_entry app start -l --tray --host 127.0.0.1 --port {APP_PORT}
```

`-X utf8` is mandatory (see STEP 6). `-l` and `--tray` default ON —
closing the window hides to tray, which is what users expect. Offer
`--no-tray` to opt out.

Regeneration is idempotent — overwrite the existing shortcut each time.

Shortcut install is for the standalone app runtime, not just "open localhost
in a browser". The created artifact must launch the managed `--app` window
path from STEP 14.

---

## STEP 17 — Icon pipeline (SVG → platform format)

Only the SVG is source-controlled. Everything else is auto-built.

```
static/images/{BRAND}/
├── {BRAND}-logo.svg           # source (colour wordmark)
├── {BRAND}-logo-white.svg     # source (white variant)
├── {BRAND}-logo.ico           # auto-built for Windows shortcuts
├── {BRAND}-logo.png           # auto-built for Linux/macOS shortcuts
└── {BRAND}-logo-white.ico     # auto-built (optional dark-context tray)
```

`_resolve_icon_for_platform()` contract:

* `os.name == "nt"` → return path to `.ico`
* otherwise → return path to `.png`
* Rebuild target from SVG if missing OR older than SVG
* Use `cairosvg.svg2png(output_width=256, ...)` + Pillow's
  `save(..., format="ICO", sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])`
* Missing cairosvg/Pillow: log INFO, return whatever exists. Install still
  completes.

**Pillow multi-size ICO gotcha:** render once at 256 px and let Pillow
downscale via `sizes=`. Passing `append_images=[list]` alone produces
a single-size ICO.

---

## STEP 18 — Live charts (oscilloscope pattern)

No chart libraries. Write SVG directly — smaller bundle, easier to theme.

* `<svg viewBox="0 0 600 180" preserveAspectRatio="none">` — resizable.
* Three path elements per series: `.wave`, `.fill`, `.cursor`.
* `<g data-role="grid">` redrawn each frame from `niceMax(yMax * 1.2)`.
* Sample buffer is `[{t, a, b}]`; prune when `t < latest - windowSec - 10`.
* Auto-scale Y with hysteresis (grow > 25%; never shrink unless peak < 50%
  of current max).
* Provide `data-curve-style="step" | "linear" | "smooth"` — default
  `"step"` (zero-order hold) for discrete-event rate data.
* Hydrate from a `/history` endpoint on init, then live-poll — so tab
  switches don't reset the waveform.

For rate metrics over event streams (e.g. "events per minute"):

* Use a **count-based sliding window**, not an exponential kernel.
* Formula: `rate = (count_in_window - 1) / (effective_span / 60)` where
  `effective_span = min(window, max(min_span, t - first_event_ever))`.
  Poisson-MLE unbiased form.
* `instant` uses a smaller window (e.g. 60s) of the same formula.
* Render as step function; never exponential decay (users see the
  sawtooth at low rates and file bugs).

---

## STEP 19 — WebSocket log streaming

Endpoints:

```
GET  /api/logs/files          — list log files (group by session, newest first)
GET  /api/logs/tail?path=…    — last N lines (HTTP, hydration)
WS   /api/logs/ws/file?path=… — tail file, text frames
WS   /api/logs/ws/process/{key} — subscribe to ProcessManager buffer + live
```

Client-side (`kit/log-list/macro.js`):

* `data-source="<url>"` → connect on init, reconnect on close.
* Classify lines into `err / warn / ok / info / exit` via regex on keywords
  and emoji.
* Keep last ~2000 lines; trim from the head.
* Autoscroll checkbox (default ON) — stop auto-scroll when user scrolls up
  so reading history is possible.
* Clear + Copy buttons.

Server-side: `ProcessManager.subscribe(key)` returns an `asyncio.Queue`
pre-seeded with the ring buffer.

---

## STEP 20 — CLI wiring (`{cmd} app`)

Register an `app` subcommand group in the project's CLI entrypoint:

```python
elif name == "app":
    app_sub = created_p.add_subparsers(dest="app_cmd", metavar="SUB")

    app_start_p = app_sub.add_parser("start", …)
    app_start_p.add_argument("--host", default="127.0.0.1")
    app_start_p.add_argument("--port", type=int, default={APP_PORT})
    app_start_p.add_argument("-l", "--launch", action="store_true")
    app_start_p.add_argument("--browser", choices=["app", "tab"], default="app")
    app_start_p.add_argument("--tray", action="store_true")
    app_start_p.add_argument("--reload", action="store_true")

    app_install_p = app_sub.add_parser("install", …)
    app_install_p.add_argument("--name", default="{APP_TITLE}")
    app_install_p.add_argument("--start-menu", action="store_true")
    app_install_p.add_argument("--no-tray", dest="install_tray",
                                action="store_false", default=True)
```

---

## STEP 21 — Version + health + about

### Version module

`{APP}/version.py` — single source of truth for the UI footer:

```python
from importlib.metadata import PackageNotFoundError, version

try:
    APP_VERSION = version("{PROJECT}")
except PackageNotFoundError:
    APP_VERSION = "0.0.0+dev"

BUILD_ID = "dev"  # overridden in CI to commit SHA
```

### Health endpoint

`/healthz` returns `{"ok": True, "version": APP_VERSION}`. Non-negotiable.
Docker healthchecks and the header status dot both consume it.

### About page

Every project ships an `/about` page:

* Logo (the big colour SVG, ~240 px wide)
* Title + tagline
* Version + build id
* Copyright + license link
* Link to repo + docs
* Credits (open-source deps this project leans on)

---

## STEP 22 — Accessibility baseline

Not optional; this keeps keyboard users and screen readers working.

* `<html lang="en">` set explicitly.
* Every `<img>` has `alt=""` (decorative) or meaningful `alt="…"`.
* Every icon-only button has `aria-label="…"`.
* Every navigable `<nav>` has `aria-label="…"`.
* Focus styles visible — `:focus-visible { outline: 2px solid var(--{BRAND}-accent); outline-offset: 2px; }` in `reset.css`.
* Tab order follows visual order (avoid `tabindex` > 0).
* Modals trap focus and restore it on close.
* Active ribbon tab has `aria-current="page"`.
* Disabled buttons use `disabled` attribute, not `.is-disabled` pointer-events hacks.
* Colour contrast ≥ 4.5:1 for text, ≥ 3:1 for UI chrome (verify against
  both themes).

---

## STEP 23 — Cross-platform compatibility checklist

Before every release:

* [ ] Every `os.name == "nt"` / `sys.platform == "win32"` / `"darwin"`
      branch has a verified POSIX counterpart (not a "return None" stub).
* [ ] Subprocess spawning: `CREATE_NEW_PROCESS_GROUP` on Windows /
      `start_new_session=True` on POSIX.
* [ ] Signal handling: `CTRL_BREAK_EVENT` on Windows (requires the group
      flag!), `SIGTERM` on POSIX.
* [ ] Clipboard: `clip` / `pbcopy` / `xclip` then `wl-copy`.
* [ ] Chromium lookup: known install paths on Windows/macOS,
      `shutil.which()` on Linux.
* [ ] Tray: pystray import failure → clear hint; runtime failure →
      fallback to "server only".
* [ ] Icon pipeline: `.ico` on Windows, `.png` on Linux/macOS.
* [ ] Shortcut creation produces a working artifact on every OS — NEVER a
      silent no-op.
* [ ] Paths use `pathlib.Path`, never raw separator concatenation.
* [ ] Env vars: `TEMP` is Windows-only — fall back to
      `tempfile.gettempdir()`.

---

## STEP 24 — File length enforcement

**≤ 200 lines per file.** At ~180, split:

* Routes → group by prefix into a sub-router.
* JS page bootstrap → extract domain helpers to
  `{APP}/ui/static/js/pages/<page>/<module>.js`.
* Service module → one class per file, aggregator `__init__.py`
  re-exports.
* CSS → move variants into a companion file and `@import` from `theme.css`.
* Python module → private helpers to `_private.py` sibling.

Reason: every web-app file is also documentation. Colleagues open the
file to learn a pattern and must see the whole pattern on one screen.

---

## STEP 25 — Smoke test (run before every commit)

```bash
# 1. Imports + FastAPI factory
python -c "from {APP}.server import create_app; print(len(create_app().routes))"

# 2. Routes respond
python -c "
from fastapi.testclient import TestClient
from {APP}.server import create_app
c = TestClient(create_app())
for p in ('/', '/healthz', '/about', '/ui-assets/css/app.css',
          '/ui-assets/images/{BRAND}/{BRAND}-logo.svg'):
    r = c.get(p); assert r.status_code == 200, f'{p} → {r.status_code}'
print('smoke: OK')
"

# 3. CLI help doesn't explode
python -c "
import sys; sys.argv = ['{cmd}', 'app', '--help']
from {CORE}.cli.idc import main
try: main()
except SystemExit: pass
"

# 4. Subprocess entry is hijack-free
python -u -m {CORE}.cli.main_entry --help | head -3

# 5. Branding contract
python -c "
from fastapi.testclient import TestClient
from {APP}.server import create_app
c = TestClient(create_app())
html = c.get('/').text
for token in ('kit-header__logo', 'kit-footer', '© ',
              '{BRAND}-logo', 'rel=\"icon\"'):
    assert token in html, f'missing branding marker: {token}'
print('branding: OK')
"
```

Base five green = safe to continue. If the project ships `-l/--launch`
or tray mode, checks 6 and 7 are also required before commit.

Add these two checks for projects that implement `-l/--launch` and tray mode:

```bash
# 6. Standalone launcher uses Chromium --app when available
python -c "
import os, tempfile, time
from pathlib import Path
from {APP}.standalone import launch_app_window
tmp = Path(tempfile.gettempdir())
fake = tmp / '{APP_SLUG}-fake-browser.cmd'
log = tmp / '{APP_SLUG}-fake-browser.log'
fake.write_text('@echo off\\n> \"%s\" echo %%*\\n' % log, encoding='utf-8')
if log.exists(): log.unlink()
os.environ['BROWSER_EXECUTABLE'] = str(fake)
proc = launch_app_window('http://127.0.0.1:{APP_PORT}')
time.sleep(0.5)
text = log.read_text(encoding='utf-8')
assert '--app=http://127.0.0.1:{APP_PORT}' in text
assert '--user-data-dir=' in text
print('standalone-launch: OK')
"

# 7. Shutdown contract: parent exit also stops the managed app window
# Use a fake browser executable again, start `{cmd} app start -l`,
# wait for `/healthz`, send SIGINT / CTRL_BREAK_EVENT, and assert:
#   - the CLI process exits
#   - the fake browser child does not keep running as an orphan
```

For Windows, prefer `CTRL_BREAK_EVENT` when testing an external child process
group. For in-process smoke tests, `signal.raise_signal(signal.SIGINT)` is
usually sufficient.

---

## STEP 25A — Definition of Done checklist

Do not call the web-app work "done" until every item below is true:

* [ ] `{APP}` is a top-level sibling package of `{CORE}`.
* [ ] Web dependencies live only in `[project.optional-dependencies]`.
* [ ] Importing `{CORE}` or `{APP}` metadata modules does NOT create the
      FastAPI app or eagerly pull web-only dependencies.
* [ ] `python -m {APP}` works.
* [ ] `{cmd} app start` works.
* [ ] `{cmd} app install` works.
* [ ] `-l/--launch` uses Chromium `--app` with an isolated
      `--user-data-dir`.
* [ ] No Chromium available → clean fallback to a regular browser tab.
* [ ] The app-window process is tracked by a dedicated controller/helper.
* [ ] `Ctrl+C` stops the server.
* [ ] `Ctrl+C` also stops the standalone app window.
* [ ] Closing the terminal / IDE run session does NOT leave the app window
      orphaned.
* [ ] Tray **Quit** stops the server and the standalone app window.
* [ ] Tray **Open {APP_TITLE}** respawns the standalone app window.
* [ ] Tray runtime failure falls back to "server only" instead of killing the UI.
* [ ] Desktop shortcut / installer creates a working launch artifact.
* [ ] Windows shortcut uses `pythonw.exe -X utf8`.
* [ ] Platform icon assets are generated from the source SVG.
* [ ] Every page shows the `{BRAND}` logo in the header.
* [ ] Every page has the footer contract.
* [ ] Smoke tests in STEP 25 pass.

---

## STEP 26 — README sections to add

Every project shipping a web UI adds these sections to its `README.md`:

* **Logo at the top** (STEP 11D).
* **Installation** — mention `[app]` / `[tray]` extras explicitly.
* **Web UI (`{cmd} app`)**:
  * Quickstart (`{cmd} app start -l`, `{cmd} app install`).
  * Dashboard feature table.
  * Flag reference table.
  * Pinning/launching per platform.
  * Platform notes for `--tray` (compatibility matrix).
  * Icon handling.
  * Architecture notes.

---

## COMMON PITFALLS

### TextIOWrapper GC race under pythonw

Two `_rebind_line_buffered_utf8()` calls (before and after importing the
target module) create two wrappers both owning fd 1. The first's
`__del__` closes the fd; subsequent writes silently vanish. **Fix:** one
rebind, AFTER the target import. `-X utf8` handles the pre-import
encoding problem.

### Emoji in logger format strings under pythonw

`cp1252` default → `UnicodeEncodeError` on first emit → silent child
death. **Fix:** `-X utf8` in the shortcut command.

### `CTRL_BREAK_EVENT` with no process group

If the child wasn't spawned with `CREATE_NEW_PROCESS_GROUP`, sending
`CTRL_BREAK_EVENT` either raises `OSError 87` or propagates to the whole
console group — potentially killing the UI server. **Fix:** always spawn
with the flag.

### Standalone app window survives `Ctrl+C` / terminal close

If `-l/--launch` uses `webbrowser.open_*()` as the main path, or if you
discard the `Popen` handle for the Chromium `--app` window, the browser
process outlives the server and becomes an orphan. **Fix:** launch through a
dedicated controller (`AppWindowController` or equivalent), keep the live
process handle, and call `stop()` from every shutdown path: `SIGINT`,
console-close handler, tray **Quit**, and outer `finally:` cleanup. Browser
tab openers remain fallback-only.

### Macro can't resolve `asset_base` / `app_version`

Jinja macros DON'T inherit caller context. **Fix:**
`{% from "..." import X with context %}`, OR pass context as macro params.

### ICO has only one size

`Image.save(..., sizes=[...], append_images=[...])` doesn't produce the
multi-resolution ICO you'd expect. **Fix:** render once at 256 px, save
with `sizes=[all sizes]` — Pillow downscales.

### Chromium `.lnk` taskbar icon stuck on Python logo

Explorer caches shortcut icon thumbnails. **Fix:** delete + re-create the
`.lnk`, or run `ie4uinit.exe -show`, or restart Explorer.

### Supervisor subprocess orphaned after "Stop all"

Your UI's "stop everything" button may only reach processes tracked by
the domain dispatcher, not the supervisor subprocess you launched via
`ProcessManager`. **Fix:** cascade-stop every tracked managed process
via ProcessManager FIRST, then fire any domain-specific cleanup command.

### Log files not appearing in `/api/logs/files`

Default glob is usually `*.log`. **Fix:** include `*.csv` + `*.txt` (any
text artifact your project emits) and group results by session folder so
the current run floats to the top.

### Tray fails silently on Linux Wayland

Wayland compositors without AppIndicator can't render a tray icon;
`pystray.run()` either hangs or raises. **Fix:** wrap `icon.run()` in
try/except; on failure log a warning and block on the uvicorn thread so
the server remains reachable.

---

## APPENDIX A — File count estimate

Baseline for a web UI with 6–10 pages and 14–16 kit components (core set
from STEP 9 including `header`, `footer`, `ribbon`, `panel`, `card`,
`button`, `badge`, `meter`, `table`, `log-list`, `toast`, `form`, `input`,
`modal`, `tabs`):

| Kind                                                            | Count              |
| --------------------------------------------------------------- | ------------------ |
| Python (server, routes, services, manifest, mount, page router) | 16–22              |
| Jinja templates (base, shell, kit macros, page layouts)         | 32–42              |
| CSS (tokens, reset, base, utilities, kit, theme, page)          | 25–35              |
| JS (app-boot, app-client, kit factories, page bootstraps)       | 22–32              |
| Static images (source SVGs + generated ICO/PNG)                 | 3–6                |
| **Total**                                                       | **~100–135 files** |

All files ≤ 200 lines → ~20k lines total tops. Every file readable on one
screen.

---

## APPENDIX B — Context injected into every page

`pages_router.py` must merge these into every template context:

| Key                                      | Source                | Used by              |
| ---------------------------------------- | --------------------- | -------------------- |
| `page_id`, `page_title`, `page_subtitle` | manifest              | shell + page headers |
| `ribbon_tabs`, `active`                  | manifest              | ribbon               |
| `asset_base`                             | constant `/ui-assets` | all macros           |
| `app_version`                            | `{APP}/version.py`    | footer + about       |
| `app_year`                               | `datetime.now().year` | footer copyright     |
| `docs_url`, `repo_url`                   | project config        | footer links         |

Forgetting any of these breaks a page silently (macro renders empty).

---

## APPENDIX C — When NOT to use this playbook

* CLI-only tools with no human dashboard consumer.
* Headless services exposing only machine-readable APIs (skip the entire
  `{APP}/ui/` subtree; keep `api/`).
* Single-page tools where the whole UI fits in one 200-line HTML file.
* Prototypes where file-count discipline costs more than it buys.

Apply the full playbook the moment the UI grows past two pages or the
team expands past one contributor.
