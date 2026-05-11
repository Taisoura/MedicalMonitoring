"""
FastAPI application entry point for the Medical Monitoring web service.

Provides REST API for uploading EDC files, running the pipeline,
and downloading results in multiple formats.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[MedMon API] Starting up...")
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


@app.get("/")
async def root():
    return {
        "service": "Medical Monitoring API",
        "version": "1.2.0",
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /api/upload",
            "run": "POST /api/run/{job_id}",
            "status": "GET /api/status/{job_id}",
            "results": "GET /api/results/{job_id}",
            "export": "GET /api/export/{job_id}/{format}",
            "jobs": "GET /api/jobs",
        },
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
