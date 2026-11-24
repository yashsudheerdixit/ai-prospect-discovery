# AI-Powered Multi-Platform Prospect Discovery API

A FastAPI service that interprets natural-language prospect queries with an LLM, searches publicly accessible profiles across multiple platforms via authorized APIs, extracts and validates public contact information (email, phone, WhatsApp), consolidates duplicate profiles across networks, and returns a structured JSON response.

---

## 1. Features & Architecture

- **Natural-Language Query Understanding (`QueryService`)**: Extracts `interest`, `audience`, `location` (`city`, `state`, `country`), `target_type`, domain keywords, and tailored platform queries using structured JSON output from an LLM (with TTL caching and a deterministic fallback parser).
- **Multi-Platform Discovery (`app/platforms/`)**: Independent, modular platform discoverers executed concurrently via `asyncio.gather`:
  1. **Google** (`google.py`) — Discovers publicly indexed individual/business websites and public profiles (filters out search-engine result pages and generic listicles).
  2. **LinkedIn** (`linkedin.py`) — Public `/in/` and `/company/` profiles via authorized search.
  3. **X** (`x_platform.py`) — Public X/Twitter user profiles (collapses status URLs to canonical author profiles).
  4. **YouTube** (`youtube.py`) — Uses the official **YouTube Data API v3** (`YOUTUBE_API_KEY`) when configured, with authorized web search fallback.
  5. **Instagram** (`instagram.py`) — Public creator/business profiles (excludes `/p/`, `/reel/`, `/explore/`).
  6. **Facebook** (`facebook.py`) — Public profiles and pages.
  7. **Threads** (`threads.py`) — Public `@handle` profiles (`threads.net` / `threads.com`).
  8. **WhatsApp** (`whatsapp.py`) — Publicly published `wa.me/` and `api.whatsapp.com/send` contact links.
  9. **Email** (`email_platform.py` + `ProfileExtractor`) — Publicly published email addresses validated with `email-validator` and filtered against `noreply@` / asset / example domains.
- **Public Contact Extraction & Validation (`ProfileExtractor`, `validators.py`, `normalizers.py`)**:
  - Validates emails (`email-validator`), phone numbers (`phonenumbers` -> E.164 `tel:+...`), WhatsApp links (`https://wa.me/<digits>`), and canonical profile URLs.
  - Enriches non-auth-walled personal/business websites (`ENABLE_PAGE_ENRICHMENT=true`) while **never** bypassing login walls, CAPTCHAs, or private profiles.
- **Duplicate Detection & Entity Consolidation (`DeduplicationService`)**:
  - Merges matching profiles belonging to the same individual/business using shared email, shared phone/WhatsApp, shared canonical URL, or matching `@handle` + name similarity.
  - Prevents false merges between namesakes in conflicting geographical locations.
- **Relevance Evaluation & Filtering (`RelevanceService`)**:
  - Combines deterministic evidence scoring (interest keywords, location signals, anti-listicle checks) with a single batched LLM evaluation call.

### Processing Flow

```
User Query
   │
   ▼
Query Understanding (LLM + TTL Cache + Heuristic Fallback)
   │
   ▼
Concurrent Multi-Platform Discovery (Serper / Brave / Tavily / Google CSE / YouTube Data API v3)
   │
   ▼
Profile Extraction & Contact Validation (Email, Phone, WhatsApp, Social Links)
   │
   ▼
Duplicate Detection & Entity Consolidation
   │
   ▼
Relevance Scoring & Filtering (Deterministic + Batched LLM)
   │
   ▼
Validated Pydantic JSON Response
```

---

## 2. Project Structure

```text
ai-prospect-discovery/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── services/
│   │   ├── llm_service.py
│   │   ├── query_service.py
│   │   ├── search_service.py
│   │   ├── profile_extractor.py
│   │   ├── deduplication_service.py
│   │   ├── relevance_service.py
│   │   └── prospect_service.py
│   ├── platforms/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── google.py
│   │   ├── linkedin.py
│   │   ├── x_platform.py
│   │   ├── youtube.py
│   │   ├── instagram.py
│   │   ├── facebook.py
│   │   ├── threads.py
│   │   ├── whatsapp.py
│   │   └── email_platform.py
│   └── utils/
│       ├── normalizers.py
│       └── validators.py
├── tests/
│   └── test_prospect_discovery.py
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 3. Setup & Execution

### Prerequisites
- Python **3.11+**
- At least one LLM API key (`GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`)
- At least one authorized Search API key (`SERPER_API_KEY`, `BRAVE_SEARCH_API_KEY`, `TAVILY_API_KEY`, or legacy `GOOGLE_CSE_API_KEY` + `GOOGLE_CSE_CX`), plus optional `YOUTUBE_API_KEY` for direct YouTube Data API v3 channel discovery.

### Local Installation

```bash
git clone <your-public-repo-url>
cd ai-prospect-discovery

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your API keys
```

### Run the API Server (Port 4000)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 4000 --reload
```

- **API Endpoint**: `POST http://localhost:4000/prospect-discovery`
- **Swagger / OpenAPI Docs**: `http://localhost:4000/docs`
- **Health Check**: `GET http://localhost:4000/health`

### Run Automated Tests

```bash
pytest -v
```

### Docker Deployment

```bash
docker build -t ai-prospect-discovery .
docker run -p 4000:4000 --env-file .env ai-prospect-discovery
```

---

## 4. API Usage & Test Cases

### Request

```bash
curl -X POST http://localhost:4000/prospect-discovery \
  -H "Content-Type: application/json" \
  -d '{"query": "Football lovers in Washington DC in USA"}'
```

### Expected Response Schema

```json
{
  "success": true,
  "query": "Football lovers in Washington DC in USA",
  "total_prospects": 1,
  "prospects": [
    {
      "name": "Alex Rivera",
      "location": "Washington DC, USA",
      "social_link": [
        {
          "platform": "LinkedIn",
          "link": "https://www.linkedin.com/in/alexriveradc"
        },
        {
          "platform": "X",
          "link": "https://x.com/alexriveradc"
        },
        {
          "platform": "Email",
          "link": "mailto:alex@dcfootball.org"
        },
        {
          "platform": "WhatsApp",
          "link": "https://wa.me/12025550142"
        }
      ]
    }
  ]
}
```

### Practical Test Queries Covered

| # | Query | Parsed Audience | Parsed Location | Target Type |
|---|-------|-----------------|-----------------|-------------|
| 1 | `Football lovers in Washington DC in USA` | Football lovers | Washington DC, District of Columbia, USA | `individual` |
| 2 | `Digital marketing agency owners in New York USA` | Digital marketing agency owners | New York, New York, USA | `business` |
| 3 | `Startup founders in London UK` | Startup founders | London, England, UK | `individual` |
| 4 | `Cricket lovers in Ahmedabad India` | Cricket lovers | Ahmedabad, Gujarat, India | `individual` |
| 5 | `Restaurant owners in Dubai UAE` | Restaurant owners | Dubai, Dubai, UAE | `business` |

---

## 5. Discovery Sources & Platform Limitations

| Platform | Discovery Method | Limitations & Safeguards |
|----------|------------------|--------------------------|
| **Google** | Authorized web search APIs (Serper.dev, Brave Search, Tavily, legacy Google CSE) | Filters out search-result URLs and generic listicles; only personal/business sites or profiles are kept. Note: Google Custom Search JSON API is closed to new customers and scheduled for sunset on Jan 1, 2027, so Serper/Brave/Tavily are preferred. |
| **YouTube** | Official **YouTube Data API v3** (`search` + `channels`) + search fallback | Subject to YouTube API daily quota (10,000 units/day default). |
| **LinkedIn, X, Instagram, Facebook, Threads** | Authorized search index queries (`site:...`) | Platforms enforce authentication walls and anti-scraping protections; direct HTML scraping of login-walled profiles is intentionally avoided. Discovery relies strictly on publicly indexed snippets and cross-linked personal websites. |
| **WhatsApp & Email** | Public snippet extraction + non-auth-walled personal/business site parsing | Only publicly published contact links/addresses are extracted; system/noreply addresses and invalid phone numbers are discarded. |

---

## 6. LLM Platform & Model Research — Cost, Speed & Accuracy

### A. LLM Provider & Model Comparison (Current 2026 Pricing)

> **Methodology Note**: Token pricing and context windows reflect published provider rates as of late 2026. Response latency and task accuracy below are **estimated** based on model tier characteristics for short structured JSON extraction/classification tasks (~400–1,000 tokens), as live end-to-end benchmarks depend on network RTT and prompt payload size.

| Provider | Model Name | API Cost (per 1M Tokens: Input / Output) | Context Window | Estimated Response Speed | Accuracy (Intent & Relevance) | JSON Support | Suitability |
|---|---|---|---|---|---|---|---|
| **Google Gemini** | `gemini-2.5-flash-lite` (or `gemini-3.5-flash-lite`) | **$0.10** input / **$0.40** output (cached input ~$0.01) | 1M tokens | **Very Fast** (~0.4s–0.9s est.) | High for structured extraction & classification | Native `responseMimeType: "application/json"` | **Best overall** for high-volume, low-latency JSON pipelines |
| **Google Gemini** | `gemini-3-flash` | **$0.50** input / **$3.00** output | 1M tokens | Fast (~0.7s–1.4s est.) | Very High | Native JSON mode | Strong if deeper reasoning is needed, ~5x–7x costlier output |
| **OpenAI** | `gpt-4.1-nano` | **$0.10** input / **$0.40** output (cached ~$0.025) | ~1.05M tokens | **Very Fast** (~0.4s–0.9s est.) | High | `response_format: {"type": "json_object"}` | **Best fallback**; identical price tier to Flash-Lite |
| **OpenAI** | `gpt-5-nano` | **$0.05** input / **$0.40** output (cached ~$0.005) | 400K tokens | Very Fast (~0.4s–1.0s est.) | High | `response_format: {"type": "json_object"}` | Ultra-cheap input; excellent alternative for large candidate batches |
| **OpenAI** | `gpt-4o-mini` | **$0.15** input / **$0.60** output | 128K tokens | Fast (~0.5s–1.1s est.) | High | `json_object` / `json_schema` | Reliable legacy option, slightly higher cost than nano/lite |
| **Anthropic** | `claude-haiku-5-5` | **$0.10** input (≤100K) / **$0.50** output (cached ~$0.01) | 1M tokens | Fast (~0.6s–1.2s est.) | Very High | Prompted JSON / tool use | Great instruction following; lacks a native strict JSON MIME toggle without tools |
| **Anthropic** | `claude-sonnet-5-5` | **$2.00** input / **$10.00** output | 1M tokens | Moderate (~1.2s–2.5s est.) | Excellent | Prompted JSON / tool use | Overkill cost (20x–25x) for lightweight query parsing & classification |

---

### B. Cost Optimization Strategies Implemented

1. **Two-Call Ceiling per Request**: Only **2 LLM calls** max per API request — one for `QueryService.understand_query` and one **batched** call in `RelevanceService` evaluating up to 25 candidates together instead of 25 separate per-prospect calls (~90% token savings).
2. **Deterministic Pre-Filtering**: URL validation, email/phone regex + `libphonenumber`, generic listicle filtering, and keyword overlap run deterministically in Python *before* sending candidates to the LLM.
3. **TTL Caching (`cachetools.TTLCache`)**: Repeated or identical natural-language queries reuse cached `QueryIntent` and search results without incurring repeat LLM or search API charges.
4. **Compact Prompt Payloads**: Only concise candidate metadata (`name`, `location`, `urls`, 280-char bio snippet) is sent to the LLM — never raw HTML pages.
5. **Small Fast Models**: Using `gemini-2.5-flash-lite` / `gpt-4.1-nano` instead of frontier reasoning models cuts costs by 10x–25x while keeping API latency under ~1–2 seconds for LLM stages.

---

### C. Final LLM Recommendation

- **Recommended Provider**: **Google Gemini**
- **Recommended Model**: **`gemini-2.5-flash-lite`** (or **`gemini-3.5-flash-lite`** where provisioned)
- **Fallback Model**: **OpenAI `gpt-4.1-nano`** (automatically invoked by `LLMService` if Gemini fails or rate-limits)
- **Estimated Token Usage per Discovery Request**:
  - Call 1 (Query Understanding): ~280 input tokens, ~180 output tokens
  - Call 2 (Batched Relevance Classification for ~15–20 candidates): ~1,400 input tokens, ~550 output tokens
  - **Total Estimated Tokens**: **~1,680 input tokens** + **~730 output tokens**
- **Estimated LLM Cost per Discovery Request**:
  - Input: `(1,680 / 1,000,000) * $0.10 = $0.000168`
  - Output: `(730 / 1,000,000) * $0.40 = $0.000292`
  - **Total Estimated LLM Cost**: **~$0.00046 per request** (less than **\$0.50 per 1,000 requests**!)
- **Expected Response Time (Estimated)**:
  - LLM Query Understanding: ~0.4s–0.8s
  - Parallel Multi-Platform Search & Extraction: ~1.0s–2.2s
  - Batched LLM Relevance Classification: ~0.6s–1.2s
  - **End-to-End API Latency**: **~2.0s–4.2s** (with warm cache: <1.5s)
- **Why Selected**: `gemini-2.5-flash-lite` delivers native JSON MIME output (`responseMimeType: "application/json"`), sub-second latency, 1M context window, and \$0.10/\$0.40 per 1M token pricing — achieving high structured-extraction accuracy at negligible cost, paired with `gpt-4.1-nano` as an equally priced, zero-downtime fallback.
