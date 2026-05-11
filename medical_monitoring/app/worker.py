"""
Background task runner for pipeline execution.

Runs the MedicalMonitoringPipeline in a background thread,
updating job status via the JobStore.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any

from .api.schemas import JobStatus
from .storage import JobStore


class PipelineWorker:
    """Executes pipeline jobs in background threads."""

    def __init__(self, job_store: JobStore):
        self.store = job_store
        self._threads: dict[str, threading.Thread] = {}

    def submit(self, job_id: str, config: dict[str, Any]) -> None:
        t = threading.Thread(
            target=self._run_pipeline,
            args=(job_id, config),
            daemon=True,
            name=f"pipeline-{job_id}",
        )
        self._threads[job_id] = t
        t.start()

    def is_running(self, job_id: str) -> bool:
        t = self._threads.get(job_id)
        return t is not None and t.is_alive()

    def _run_pipeline(self, job_id: str, config: dict[str, Any]) -> None:
        from ..pipeline import MedicalMonitoringPipeline
        from ..audit.causality import DrugContext

        try:
            self.store.update_status(
                job_id, JobStatus.RUNNING, progress=0.05,
                message="Initializing pipeline...",
            )

            meta = self.store.get_meta(job_id)
            upload_dir = self.store.get_upload_path(job_id)
            export_dir = self.store.get_export_path(job_id)

            edc_files = list(upload_dir.glob("*.xlsx"))
            if not edc_files:
                raise FileNotFoundError("No EDC file found in uploads")
            edc_file = edc_files[0]

            ctcae_file = config.get("ctcae_file")
            indication = config.get("indication", "general")
            ai_enabled = config.get("ai_enabled", False)

            drug_ctx = None
            dc = config.get("drug_context")
            if dc:
                drug_ctx = DrugContext(
                    drug_name=dc.get("drug_name", ""),
                    drug_class=dc.get("drug_class", ""),
                    indication=dc.get("drug_indication", ""),
                    mechanism=dc.get("drug_mechanism", ""),
                    known_risks=dc.get("ib_risks", []),
                    protocol_phase=dc.get("protocol_phase", ""),
                )

            self.store.update_status(
                job_id, JobStatus.RUNNING, progress=0.10,
                message="Parsing EDC data...",
            )

            pipeline = MedicalMonitoringPipeline(
                edc_file=str(edc_file),
                ctcae_file=str(ctcae_file) if ctcae_file else None,
                indication=indication,
                ai_enabled=ai_enabled,
                drug_context=drug_ctx,
            )

            self.store.update_status(
                job_id, JobStatus.RUNNING, progress=0.30,
                message="Running audit modules...",
            )

            results = pipeline.run()

            self.store.update_status(
                job_id, JobStatus.RUNNING, progress=0.70,
                message="Generating exports...",
            )

            xlsx_path = export_dir / "report.xlsx"
            pipeline.export(xlsx_path)

            try:
                pipeline.export_pdf(export_dir / "report.pdf")
            except Exception:
                pass

            try:
                pipeline.export_docx(export_dir / "report.docx")
            except Exception:
                pass

            try:
                pipeline.export_json(export_dir / "results.json")
            except Exception:
                pass

            serializable_results = self._make_serializable(results)
            self.store.save_results(job_id, serializable_results)

            self.store.update_status(
                job_id, JobStatus.COMPLETED, progress=1.0,
                message="Pipeline completed successfully",
            )

        except Exception as e:
            self.store.update_status(
                job_id, JobStatus.FAILED, progress=0.0,
                message=f"Pipeline failed: {str(e)}",
                error=traceback.format_exc(),
            )

    def _make_serializable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        if isinstance(obj, set):
            return list(obj)
        if hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        try:
            import json
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)
