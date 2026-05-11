"""Pydantic request/response models for the Medical Monitoring API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExportFormat(str, Enum):
    XLSX = "xlsx"
    PDF = "pdf"
    DOCX = "docx"
    JSON = "json"


class Indication(str, Enum):
    GENERAL = "general"
    ONCOLOGY = "oncology"
    STROKE = "stroke"
    CARDIOVASCULAR = "cardiovascular"


class DrugContextRequest(BaseModel):
    drug_name: str = ""
    drug_class: str = ""
    drug_indication: str = ""
    drug_mechanism: str = ""
    ib_risks: list[str] = Field(default_factory=list)
    protocol_phase: str = ""
    therapeutic_area: str = ""


class PipelineConfigRequest(BaseModel):
    indication: Indication = Indication.GENERAL
    ai_enabled: bool = False
    drug_context: DrugContextRequest | None = None
    ctcae_file: str | None = None
    visualization: bool = False
    viz_theme: str = "academic_conference"


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    upload_time: datetime
    file_size_bytes: int


class RunRequest(BaseModel):
    config: PipelineConfigRequest = Field(default_factory=PipelineConfigRequest)


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0
    message: str = ""
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class ResultSummaryResponse(BaseModel):
    job_id: str
    total_findings: int = 0
    urgent_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    total_actions: int = 0
    n_subjects: int = 0
    modules_run: list[str] = Field(default_factory=list)
    is_oncology: bool = False
    executive_summary: dict[str, Any] = Field(default_factory=dict)
    available_exports: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
    error_type: str = "unknown"
