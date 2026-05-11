"""
Job and file storage management for the Medical Monitoring API.

Handles upload storage, job state tracking, and result file management.
Uses local filesystem storage (can be extended to S3/cloud).
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .api.schemas import JobStatus


class JobStore:
    """Manages job lifecycle and file storage on local disk."""

    def __init__(self, base_dir: str | Path = "mm_jobs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, filename: str, file_size: int) -> str:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "uploads").mkdir(exist_ok=True)
        (job_dir / "exports").mkdir(exist_ok=True)

        meta = {
            "job_id": job_id,
            "filename": filename,
            "file_size_bytes": file_size,
            "status": JobStatus.PENDING.value,
            "progress": 0.0,
            "message": "File uploaded, awaiting pipeline run",
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "config": {},
        }
        self._write_meta(job_id, meta)
        return job_id

    def get_upload_path(self, job_id: str) -> Path:
        return self.base_dir / job_id / "uploads"

    def get_export_path(self, job_id: str) -> Path:
        return self.base_dir / job_id / "exports"

    def get_meta(self, job_id: str) -> dict[str, Any]:
        meta_path = self.base_dir / job_id / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Job {job_id} not found")
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: float = 0.0,
        message: str = "",
        error: str | None = None,
    ) -> None:
        meta = self.get_meta(job_id)
        meta["status"] = status.value
        meta["progress"] = progress
        meta["message"] = message

        if status == JobStatus.RUNNING and meta.get("started_at") is None:
            meta["started_at"] = datetime.utcnow().isoformat()
        if status in (JobStatus.COMPLETED, JobStatus.FAILED):
            meta["completed_at"] = datetime.utcnow().isoformat()
        if error:
            meta["error"] = error

        self._write_meta(job_id, meta)

    def save_config(self, job_id: str, config: dict[str, Any]) -> None:
        meta = self.get_meta(job_id)
        meta["config"] = config
        self._write_meta(job_id, meta)

    def save_results(self, job_id: str, results: dict[str, Any]) -> None:
        result_path = self.base_dir / job_id / "results.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, default=str, indent=2)

    def get_results(self, job_id: str) -> dict[str, Any]:
        result_path = self.base_dir / job_id / "results.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Results not found for job {job_id}")
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_exports(self, job_id: str) -> list[str]:
        export_dir = self.get_export_path(job_id)
        if not export_dir.exists():
            return []
        return [f.name for f in export_dir.iterdir() if f.is_file()]

    def get_export_file(self, job_id: str, filename: str) -> Path:
        path = self.get_export_path(job_id) / filename
        if not path.exists():
            raise FileNotFoundError(f"Export file {filename} not found for job {job_id}")
        return path

    def cleanup_job(self, job_id: str) -> None:
        job_dir = self.base_dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs = []
        for d in self.base_dir.iterdir():
            if d.is_dir():
                try:
                    meta = self.get_meta(d.name)
                    jobs.append(meta)
                except FileNotFoundError:
                    continue
        return sorted(jobs, key=lambda x: x.get("created_at", ""), reverse=True)

    def _write_meta(self, job_id: str, meta: dict[str, Any]) -> None:
        meta_path = self.base_dir / job_id / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
