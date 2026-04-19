from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.orchestrator import AppOrchestrator
from app.schemas import (
	ChatMessageRequest,
	ChatMessageResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_ROOT / "web"


def create_app() -> FastAPI:
	"""Create minimal app with fullscreen map page and message endpoint."""
	app = FastAPI(title="Wuhan GIS Assistant (Minimal Scaffold)", version="0.1.0")
	orchestrator = AppOrchestrator()

	app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

	@app.get("/")
	async def index() -> FileResponse:
		return FileResponse(str(WEB_DIR / "index.html"))

	@app.post("/api/message", response_model=ChatMessageResponse)
	async def ingest_message(payload: ChatMessageRequest) -> ChatMessageResponse:
		return await orchestrator.handle_user_message(payload.message)

	return app


app = create_app()


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
