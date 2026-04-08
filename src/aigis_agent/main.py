"""FastAPI application entry for the AIGIS agent."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from aigis_agent.api.router import api_router
from aigis_agent.core.config import settings
from aigis_agent.db.session import init_db
from aigis_agent.services.greeting_service import initialize_startup_greeting
from aigis_agent.web.routes import web_router


def create_app() -> FastAPI:
    """Create and configure the HTTP application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="GIS intelligent agent skeleton for POI, routing, site selection and service area.",
    )

    app.include_router(api_router, prefix=settings.api_prefix)
    app.include_router(web_router)

    # Serve Leaflet-related JS/CSS assets.
    app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")

    @app.on_event("startup")
    async def on_startup() -> None:
        """Initialize local sqlite tables on boot."""
        init_db()
        initialize_startup_greeting()

    return app


app = create_app()


def run() -> None:
    """Run development server via `uv run aigis-agent`."""
    import uvicorn

    uvicorn.run(
        "aigis_agent.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
