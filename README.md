# AIGIS Agent

A GIS intelligent agent skeleton built with uv + FastAPI.

## Tech Stack

- FastAPI
- AMap POI (Web Service API)
- OSMnx + NetworkX
- OSRM (via routingpy)
- Isochrone placeholder service
- Leaflet + AMap tiles web UI
- LangChain (stable line)
- sqlite (SQLAlchemy)

## First-Wave Features

- Natural-language POI/address query with fuzzy search
- Simple site-selection scoring
- Simple route analysis
- Service-area analysis (isochrone placeholder)

## LangChain Agent Integration

- `POST /v1/agent/query` is now wired to a real LangChain agent.
- It exposes GIS tools for POI search, route analysis, service area, and site selection scoring.
- If `AIGIS_OPENAI_API_KEY` is missing or LLM call fails, it falls back to deterministic keyword routing.

Environment variables for model access are listed in `.env.example`:

- `AIGIS_OPENAI_API_KEY`
- `AIGIS_OPENAI_MODEL` (default `gpt-4o-mini`)
- `AIGIS_OPENAI_BASE_URL` (optional, for OpenAI-compatible endpoints)
- `AIGIS_LLM_TEMPERATURE`
- `AIGIS_LLM_TIMEOUT_S`

## Project Structure

```text
src/aigis_agent/
  api/
    routes/
      agent.py
      health.py
      poi.py
      routing.py
      service_area.py
      site_selection.py
    router.py
  core/
    config.py
  db/
    base.py
    models.py
    session.py
  schemas/
    agent.py
    poi.py
    routing.py
    service_area.py
    site_selection.py
  services/
    amap_service.py
    agent_service.py
    graph_service.py
    isochrone_service.py
    nominatim_service.py
    osrm_service.py
    site_selection_service.py
  utils/
    geo.py
  web/
    routes.py
    static/
      css/map.css
      js/map.js
    templates/map.html
  main.py
```

## Quick Start

1. Install dependencies (already done in this workspace):

```bash
uv sync
```

2. Run API server:

```bash
uv run aigis-agent
```

3. Open browser:

```text
http://127.0.0.1:8000/
```

4. Open API docs:

```text
http://127.0.0.1:8000/docs
```

## Notes

- Most modules are intentionally scaffolded with TODO comments so you can implement production logic incrementally.
- For production use, configure dedicated upstream services, set API key scopes/IP allowlist, and add rate limiting.
