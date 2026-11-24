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
    """Interactive demo landing page for the Prospect Discovery API."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Prospect Discovery API</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 820px; margin: 40px auto; padding: 0 20px; color: #111827; background: #f9fafb; }
    .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    h1 { margin-top: 0; font-size: 1.6rem; }
    .links { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
    .links a { text-decoration: none; background: #eef2ff; color: #3730a3; padding: 6px 12px; border-radius: 6px; font-weight: 500; font-size: 0.9rem; }
    .row { display: flex; gap: 10px; }
    input { flex: 1; padding: 10px 12px; border: 1px solid #d1d5db; border-radius: 8px; font-size: 1rem; }
    button { padding: 10px 18px; border: none; border-radius: 8px; background: #2563eb; color: #fff; font-weight: 600; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: wait; }
    pre { background: #111827; color: #f3f4f6; padding: 16px; border-radius: 8px; overflow-x: auto; margin-top: 16px; font-size: 0.88rem; }
  </style>
</head>
<body>
  <div class="card">
    <h1>AI-Powered Multi-Platform Prospect Discovery API</h1>
    <div class="links">
      <a href="/docs">OpenAPI / Swagger Docs (/docs)</a>
      <a href="/health">Health Check (/health)</a>
    </div>
    <p>Test <code>POST /prospect-discovery</code> live:</p>
    <div class="row">
      <input id="q" type="text" value="Football lovers in Washington DC in USA" />
      <button id="btn" onclick="runQuery()">Discover</button>
    </div>
    <pre id="out">Click Discover to send a request to POST /prospect-discovery...</pre>
  </div>
  <script>
    async function runQuery() {
      const btn = document.getElementById("btn");
      const out = document.getElementById("out");
      const query = document.getElementById("q").value;
      btn.disabled = true;
      out.textContent = "Searching across platforms...";
      try {
        const res = await fetch("/prospect-discovery", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query })
        });
        const data = await res.json();
        out.textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        out.textContent = "Request failed: " + err;
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
