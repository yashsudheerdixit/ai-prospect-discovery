import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import Settings, get_settings
from app.schemas import ErrorResponse, ProspectDiscoveryRequest, ProspectDiscoveryResponse
from app.services.prospect_service import ProspectService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Starting %s on %s:%s", settings.app_name, settings.host, settings.port)
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title="AI-Powered Multi-Platform Prospect Discovery API",
    description=(
        "Discovers relevant public prospects across Google, LinkedIn, X, YouTube, "
        "Instagram, WhatsApp, Facebook, Email, and Threads using LLM intent understanding, "
        "authorized search APIs, profile consolidation, and contact validation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def get_prospect_service(settings: Settings = Depends(get_settings)) -> ProspectService:
    """FastAPI dependency provider for ProspectService."""
    return ProspectService(settings=settings)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    messages = "; ".join(err.get("msg", "Invalid input") for err in exc.errors())
    payload = ErrorResponse(
        success=False,
        error="Invalid request payload",
        detail=messages,
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception("Unhandled error during request %s: %s", request.url.path, exc)
    payload = ErrorResponse(
        success=False,
        error="Internal server error",
        detail="An unexpected error occurred while processing the prospect discovery request.",
    )
    return JSONResponse(status_code=500, content=payload.model_dump())


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root_demo_page() -> str:
    """Interactive demo UI for the AI-Powered Multi-Platform Prospect Discovery API."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Prospect Discovery API</title>
  <style>
    :root {
      --bg: #0b1120;
      --surface: #111827;
      --surface-2: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --green: #10b981;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: radial-gradient(circle at top, #1e1b4b 0%, var(--bg) 55%);
      color: var(--text);
      min-height: 100vh;
      padding: 32px 18px 60px;
    }
    .container { max-width: 980px; margin: 0 auto; }
    .hero {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 24px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.78rem;
      font-weight: 600;
      color: #a7f3d0;
      background: rgba(16, 185, 129, 0.14);
      border: 1px solid rgba(16, 185, 129, 0.35);
      padding: 4px 10px;
      border-radius: 999px;
      margin-bottom: 10px;
    }
    h1 { margin: 0 0 6px; font-size: 1.85rem; letter-spacing: -0.02em; }
    .subtitle { margin: 0; color: var(--muted); font-size: 0.95rem; max-width: 640px; }
    .nav-links { display: flex; gap: 10px; flex-wrap: wrap; }
    .nav-links a {
      text-decoration: none;
      color: var(--text);
      background: var(--surface-2);
      border: 1px solid var(--border);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 0.86rem;
      font-weight: 500;
      transition: border-color .15s, background .15s;
    }
    .nav-links a:hover { border-color: var(--accent); background: #263349; }
    .panel {
      background: rgba(17, 24, 39, 0.9);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 22px;
      box-shadow: 0 20px 45px rgba(0,0,0,0.35);
      backdrop-filter: blur(8px);
    }
    .search-row { display: flex; gap: 10px; flex-wrap: wrap; }
    input {
      flex: 1;
      min-width: 240px;
      padding: 13px 16px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #0f172a;
      color: var(--text);
      font-size: 0.98rem;
      outline: none;
    }
    input:focus { border-color: var(--accent); }
    button.primary {
      padding: 13px 22px;
      border: none;
      border-radius: 10px;
      background: var(--accent);
      color: #fff;
      font-weight: 600;
      font-size: 0.95rem;
      cursor: pointer;
      transition: background .15s;
    }
    button.primary:hover { background: var(--accent-hover); }
    button.primary:disabled { opacity: 0.6; cursor: wait; }
    .presets {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
      align-items: center;
    }
    .presets span { font-size: 0.8rem; color: var(--muted); }
    .chip {
      background: var(--surface-2);
      border: 1px solid var(--border);
      color: #cbd5e1;
      border-radius: 999px;
      padding: 5px 11px;
      font-size: 0.78rem;
      cursor: pointer;
    }
    .chip:hover { border-color: var(--accent); color: #fff; }
    .toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      margin: 22px 0 14px;
    }
    .stats { color: var(--muted); font-size: 0.88rem; }
    .tabs { display: inline-flex; background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 3px; }
    .tab {
      border: none;
      background: transparent;
      color: var(--muted);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
    }
    .tab.active { background: var(--accent); color: #fff; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
      gap: 14px;
    }
    .prospect-card {
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .prospect-head { display: flex; align-items: center; gap: 11px; }
    .avatar {
      width: 38px;
      height: 38px;
      border-radius: 10px;
      background: linear-gradient(135deg, #6366f1, #06b6d4);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 0.9rem;
      flex-shrink: 0;
    }
    .p-name { font-weight: 600; font-size: 1rem; line-height: 1.25; }
    .p-loc { font-size: 0.8rem; color: var(--muted); margin-top: 2px; }
    .links-wrap { display: flex; flex-wrap: wrap; gap: 7px; }
    .s-pill {
      text-decoration: none;
      font-size: 0.78rem;
      font-weight: 500;
      padding: 5px 10px;
      border-radius: 7px;
      background: #0f172a;
      color: #e2e8f0;
      border: 1px solid #334155;
      transition: border-color .15s, transform .1s;
    }
    .s-pill:hover { border-color: #818cf8; transform: translateY(-1px); }
    pre {
      background: #020617;
      border: 1px solid var(--border);
      color: #e2e8f0;
      padding: 16px;
      border-radius: 12px;
      overflow-x: auto;
      font-size: 0.84rem;
      line-height: 1.45;
      margin: 0;
    }
    .hidden { display: none; }
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <div>
        <div class="badge">● FastAPI Live • Multi-Platform Prospect Discovery</div>
        <h1>AI Prospect Discovery Engine</h1>
        <p class="subtitle">
          Natural-language intent understanding, 9-platform discovery, profile consolidation, public contact extraction, and relevance ranking.
        </p>
      </div>
      <div class="nav-links">
        <a href="/docs">Swagger Docs (/docs)</a>
        <a href="/health">Health (/health)</a>
        <a href="https://github.com/yashsudheerdixit/ai-prospect-discovery" target="_blank" rel="noreferrer">GitHub Repo</a>
      </div>
    </div>

    <div class="panel">
      <div class="search-row">
        <input id="q" type="text" value="Football lovers in Washington DC in USA" placeholder="e.g. Startup founders in London UK" />
        <button id="btn" class="primary" onclick="runQuery()">Discover Prospects</button>
      </div>

      <div class="presets">
        <span>Quick test queries:</span>
        <button class="chip" onclick="setQuery('Football lovers in Washington DC in USA')">Football lovers · Washington DC</button>
        <button class="chip" onclick="setQuery('Digital marketing agency owners in New York USA')">Marketing agencies · New York</button>
        <button class="chip" onclick="setQuery('Startup founders in London UK')">Startup founders · London</button>
        <button class="chip" onclick="setQuery('Cricket lovers in Ahmedabad India')">Cricket lovers · Ahmedabad</button>
        <button class="chip" onclick="setQuery('Restaurant owners in Dubai UAE')">Restaurant owners · Dubai</button>
      </div>

      <div class="toolbar">
        <div id="stats" class="stats">Click <strong>Discover Prospects</strong> to query <code>POST /prospect-discovery</code>.</div>
        <div class="tabs">
          <button id="tab-cards" class="tab active" onclick="switchView('cards')">Cards</button>
          <button id="tab-json" class="tab" onclick="switchView('json')">Raw JSON</button>
        </div>
      </div>

      <div id="cards-view" class="grid"></div>
      <pre id="json-view" class="hidden">{}</pre>
    </div>
  </div>

  <script>
    let currentView = "cards";

    function setQuery(text) {
      document.getElementById("q").value = text;
      runQuery();
    }

    function switchView(view) {
      currentView = view;
      document.getElementById("tab-cards").classList.toggle("active", view === "cards");
      document.getElementById("tab-json").classList.toggle("active", view === "json");
      document.getElementById("cards-view").classList.toggle("hidden", view !== "cards");
      document.getElementById("json-view").classList.toggle("hidden", view !== "json");
    }

    function initials(name) {
      return (name || "?").split(/\\s+/).slice(0, 2).map(p => p[0] || "").join("").toUpperCase() || "?";
    }

    function escapeHtml(str) {
      return String(str).replace(/[&<>"']/g, m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
    }

    async function runQuery() {
      const btn = document.getElementById("btn");
      const stats = document.getElementById("stats");
      const cardsEl = document.getElementById("cards-view");
      const jsonEl = document.getElementById("json-view");
      const query = document.getElementById("q").value.trim();
      if (!query) return;

      btn.disabled = true;
      stats.textContent = "Discovering & consolidating prospects across platforms...";
      cardsEl.innerHTML = "";
      const started = performance.now();

      try {
        const res = await fetch("/prospect-discovery", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query })
        });
        const data = await res.json();
        const elapsed = ((performance.now() - started) / 1000).toFixed(2);
        jsonEl.textContent = JSON.stringify(data, null, 2);

        if (!res.ok || !data.success) {
          stats.textContent = "Error (" + res.status + "): " + (data.detail || data.error || "Request failed");
          switchView("json");
          return;
        }

        const list = data.prospects || [];
        stats.innerHTML = "Found <strong>" + data.total_prospects + "</strong> verified prospects in <strong>" + elapsed + "s</strong>";

        if (!list.length) {
          cardsEl.innerHTML = "<p style='color:#94a3b8'>No matching verified prospects returned for this query.</p>";
          return;
        }

        cardsEl.innerHTML = list.map(p => {
          const links = (p.social_link || []).map(s =>
            `<a class="s-pill" href="${escapeHtml(s.link)}" target="_blank" rel="noreferrer">${escapeHtml(s.platform)}</a>`
          ).join("");
          return `
            <div class="prospect-card">
              <div class="prospect-head">
                <div class="avatar">${escapeHtml(initials(p.name))}</div>
                <div>
                  <div class="p-name">${escapeHtml(p.name)}</div>
                  <div class="p-loc">${escapeHtml(p.location || "Location not listed")}</div>
                </div>
              </div>
              <div class="links-wrap">${links}</div>
            </div>
          `;
        }).join("");
      } catch (err) {
        stats.textContent = "Request failed: " + err;
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>"""


@app.get("/health", tags=["System"])
async def health_check(settings: Settings = Depends(get_settings)) -> dict:
    """Health check endpoint reporting service status and configured integrations."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "llm_configured": bool(
            settings.gemini_api_key or settings.openai_api_key or settings.anthropic_api_key
        ),
        "search_backends_configured": [
            name
            for name, present in (
                ("serper", bool(settings.serper_api_key)),
                ("brave", bool(settings.brave_search_api_key)),
                ("tavily", bool(settings.tavily_api_key)),
                ("google_cse", bool(settings.google_cse_api_key and settings.google_cse_cx)),
                ("youtube_data_api_v3", bool(settings.youtube_api_key)),
            )
            if present
        ],
    }


@app.post(
    "/prospect-discovery",
    response_model=ProspectDiscoveryResponse,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Prospect Discovery"],
)
async def discover_prospects(
    payload: ProspectDiscoveryRequest,
    service: ProspectService = Depends(get_prospect_service),
) -> ProspectDiscoveryResponse:
    """Discover, consolidate, validate, and rank public prospects from a natural-language query."""
    return await service.discover_prospects(payload.query)


if __name__ == "__main__":
    import uvicorn

    cfg = get_settings()
    uvicorn.run("app.main:app", host=cfg.host, port=cfg.port, reload=True)
