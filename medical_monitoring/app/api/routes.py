"""
REST API routes for the Medical Monitoring web application.

Endpoints:
  POST /api/upload          -- Upload EDC Excel file
  POST /api/run/{job_id}    -- Trigger pipeline execution
  GET  /api/status/{job_id} -- Poll job status
  GET  /api/results/{job_id} -- Get JSON results
  GET  /api/export/{job_id}/{format} -- Download export file
  POST /api/config          -- Update global configuration
  GET  /api/jobs             -- List all jobs
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from .schemas import (
    ExportFormat,
    JobStatus,
    JobStatusResponse,
    ResultSummaryResponse,
    RunRequest,
    UploadResponse,
)
from ..storage import JobStore
from ..worker import PipelineWorker

router = APIRouter(prefix="/api", tags=["Medical Monitoring"])

_store = JobStore()
_worker = PipelineWorker(_store)


@router.post("/upload", response_model=UploadResponse)
async def upload_edc_file(file: UploadFile = File(...)):
    """Upload an EDC Excel file for processing."""
    if not file.filename or not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(400, "Only .xlsx/.xls files are accepted")

    content = await file.read()
    file_size = len(content)

    job_id = _store.create_job(file.filename, file_size)
    upload_path = _store.get_upload_path(job_id) / file.filename

    with open(upload_path, "wb") as f:
        f.write(content)

    return UploadResponse(
        job_id=job_id,
        filename=file.filename,
        upload_time=datetime.utcnow(),
        file_size_bytes=file_size,
    )


@router.post("/run/{job_id}", response_model=JobStatusResponse)
async def run_pipeline(job_id: str, req: RunRequest = RunRequest()):
    """Trigger pipeline execution for an uploaded file."""
    try:
        meta = _store.get_meta(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Job {job_id} not found")

    if meta["status"] == JobStatus.RUNNING.value:
        raise HTTPException(409, "Pipeline is already running for this job")

    config = req.config.model_dump()
    _store.save_config(job_id, config)
    _worker.submit(job_id, config)

    return JobStatusResponse(
        job_id=job_id,
        status=JobStatus.RUNNING,
        progress=0.0,
        message="Pipeline started",
        created_at=meta.get("created_at"),
        started_at=datetime.utcnow().isoformat(),
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    """Get current job status."""
    try:
        meta = _store.get_meta(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Job {job_id} not found")

    return JobStatusResponse(
        job_id=job_id,
        status=JobStatus(meta["status"]),
        progress=meta.get("progress", 0),
        message=meta.get("message", ""),
        created_at=meta.get("created_at"),
        started_at=meta.get("started_at"),
        completed_at=meta.get("completed_at"),
        error=meta.get("error"),
    )


@router.get("/results/{job_id}", response_model=ResultSummaryResponse)
async def get_results(job_id: str):
    """Get pipeline results summary."""
    try:
        meta = _store.get_meta(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Job {job_id} not found")

    if meta["status"] != JobStatus.COMPLETED.value:
        raise HTTPException(409, f"Job is {meta['status']}, results not yet available")

    try:
        results = _store.get_results(job_id)
    except FileNotFoundError:
        raise HTTPException(404, "Results file not found")

    report = results.get("report", {})
    summary = report.get("executive_summary", {})
    exports = _store.list_exports(job_id)

    return ResultSummaryResponse(
        job_id=job_id,
        total_findings=summary.get("total_findings", 0),
        urgent_count=summary.get("urgent_count", 0),
        high_count=summary.get("high_count", 0),
        medium_count=summary.get("medium_count", 0),
        total_actions=report.get("statistics", {}).get("total_actions", 0),
        n_subjects=results.get("n_subjects", 0),
        modules_run=list(results.get("modules", {}).keys()),
        is_oncology=results.get("is_oncology", False),
        executive_summary=summary,
        available_exports=exports,
    )


@router.get("/export/{job_id}/{format}")
async def download_export(job_id: str, format: ExportFormat):
    """Download an export file."""
    try:
        _store.get_meta(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Job {job_id} not found")

    ext_map = {
        ExportFormat.XLSX: "report.xlsx",
        ExportFormat.PDF: "report.pdf",
        ExportFormat.DOCX: "report.docx",
        ExportFormat.JSON: "results.json",
    }

    filename = ext_map.get(format)
    if not filename:
        raise HTTPException(400, f"Unsupported format: {format}")

    try:
        path = _store.get_export_file(job_id, filename)
    except FileNotFoundError:
        raise HTTPException(404, f"{format.value} export not available")

    media_types = {
        ExportFormat.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ExportFormat.PDF: "application/pdf",
        ExportFormat.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ExportFormat.JSON: "application/json",
    }

    return FileResponse(
        path=str(path),
        filename=filename,
        media_type=media_types.get(format, "application/octet-stream"),
    )


@router.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return _store.list_jobs()


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files."""
    try:
        _store.get_meta(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Job {job_id} not found")

    if _worker.is_running(job_id):
        raise HTTPException(409, "Cannot delete a running job")

    _store.cleanup_job(job_id)
    return {"status": "deleted", "job_id": job_id}
