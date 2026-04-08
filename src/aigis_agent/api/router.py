"""Top-level API router that composes all feature modules."""

from fastapi import APIRouter

from aigis_agent.api.routes.agent import router as agent_router
from aigis_agent.api.routes.health import router as health_router
from aigis_agent.api.routes.poi import router as poi_router
from aigis_agent.api.routes.routing import router as routing_router
from aigis_agent.api.routes.service_area import router as service_area_router
from aigis_agent.api.routes.site_selection import router as site_selection_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(poi_router)
api_router.include_router(routing_router)
api_router.include_router(service_area_router)
api_router.include_router(site_selection_router)
api_router.include_router(agent_router)
