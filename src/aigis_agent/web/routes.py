"""Frontend routes for leaflet demo page."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from aigis_agent.core.config import settings
from aigis_agent.services.greeting_service import get_startup_greeting

web_router = APIRouter(tags=["web"])
templates = Jinja2Templates(directory=str(settings.template_dir))


@web_router.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    """Render first-wave GIS debug page."""
    return templates.TemplateResponse(
        request=request,
        name="map.html",
        context={
            "title": settings.app_name,
            "agent_placeholder": get_startup_greeting(),
        },
    )
