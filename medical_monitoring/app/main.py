"""
FastAPI application entry point for the Medical Monitoring web service.

Provides REST API for uploading EDC files, running the pipeline,
downloading results, and serves the frontend SPA.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router as api_router

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[MedMon API] Starting up...")
    print(f"[MedMon API] Frontend: http://localhost:8000")
    print(f"[MedMon API] API Docs: http://localhost:8000/docs")
    yield
    print("[MedMon API] Shutting down...")


app = FastAPI(
    title="Medical Monitoring API",
    description=(
        "AI-enhanced medical monitoring framework for clinical trials. "
        "Supports non-oncology and oncology indications with RECIST 1.1, "
        "dose escalation, hepatotoxicity, and multi-format reporting."
    ),
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the frontend SPA."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return {
        "service": "Medical Monitoring API",
        "version": "1.2.0",
        "docs": "/docs",
        "frontend": "Static files not found. Place index.html in app/static/",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
