"""Health endpoint for quick liveness checks."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Return static OK response to verify service boot."""
    return {"status": "ok"}
