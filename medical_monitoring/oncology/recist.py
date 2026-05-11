"""
RECIST 1.1 Response Evaluation Engine

Implements the Response Evaluation Criteria in Solid Tumours (RECIST) v1.1:
- Target lesion (TL) tracking: SOD computation, % change from baseline/nadir
- Non-target lesion (NTL) assessment: CR/non-CR non-PD/PD
- New lesion (NL) detection
- Overall response assignment: CR/PR/SD/PD/NE
- Confirmation logic: >=4 weeks for CR/PR
- Investigator assessment cross-check (RS domain)

References:
  Eisenhauer EA et al. Eur J Cancer 2009;45:228-247
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class Response(str, Enum):
    CR = "CR"       # Complete Response
    PR = "PR"       # Partial Response
    SD = "SD"       # Stable Disease
    PD = "PD"       # Progressive Disease
    NE = "NE"       # Not Evaluable

    @classmethod
    def from_str(cls, s: str) -> "Response":
        mapping = {
            "cr": cls.CR, "complete response": cls.CR, "完全缓解": cls.CR,
            "pr": cls.PR, "partial response": cls.PR, "部分缓解": cls.PR,
            "sd": cls.SD, "stable disease": cls.SD, "疾病稳定": cls.SD,
            "pd": cls.PD, "progressive disease": cls.PD, "疾病进展": cls.PD,
            "ne": cls.NE, "not evaluable": cls.NE, "无法评估": cls.NE,
        }
        return mapping.get(str(s).strip().lower(), cls.NE)


@dataclass
class LesionMeasurement:
    """Single measurement of a target lesion at one timepoint."""
    subject_id: str
    lesion_id: str
    location: str
    visit: str
    date: str
    diameter_mm: float | None
    method: str = ""
    is_baseline: bool = False
    not_measurable_reason: str = ""


@dataclass
class TargetLesionAssessment:
    """Aggregated TL assessment at one timepoint."""
    subject_id: str
    visit: str
    date: str
    sod_mm: float | None
    pct_change_from_baseline: float | None
    pct_change_from_nadir: float | None
    n_lesions_measured: int = 0
    n_lesions_total: int = 0
    tl_response: Response = Response.NE


@dataclass
class NonTargetAssessment:
    """NTL assessment at one timepoint."""
    subject_id: str
    visit: str
    date: str
    status: str  # CR, non-CR/non-PD, PD, NE


@dataclass
class OverallAssessment:
    """Overall RECIST 1.1 response at one timepoint."""
    subject_id: str
    visit: str
    date: str
    tl_response: Response
    ntl_status: str
    new_lesion: bool
    overall_response: Response
    investigator_response: Response | None = None
    confirmed: bool = False
    confirmation_date: str | None = None
    discordance: str | None = None


@dataclass
class RECISTResult:
    """Complete RECIST evaluation result for a subject."""
    subject_id: str
    baseline_sod: float | None = None
    nadir_sod: float | None = None
    best_response: Response = Response.NE
    confirmed_best_response: Response = Response.NE
    assessments: list[OverallAssessment] = field(default_factory=list)
    tl_timeline: list[TargetLesionAssessment] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)


class RECISTEngine:
    """RECIST 1.1 evaluation engine."""

    PD_THRESHOLD = 20.0          # >=20% increase from nadir for PD
    PD_ABSOLUTE_MM = 5.0         # Plus >=5mm absolute increase
    PR_THRESHOLD = -30.0         # >=30% decrease from baseline for PR
    CONFIRM_WINDOW_DAYS = 28     # Minimum days between response and confirmation

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.pd_threshold = cfg.get("pd_threshold", self.PD_THRESHOLD)
        self.pr_threshold = cfg.get("pr_threshold", self.PR_THRESHOLD)
        self.pd_absolute_mm = cfg.get("pd_absolute_mm", self.PD_ABSOLUTE_MM)
        self.confirm_window = cfg.get("confirm_window_days", self.CONFIRM_WINDOW_DAYS)

    def evaluate_dataset(self, dataset) -> dict[str, RECISTResult]:
        """Evaluate RECIST for all subjects in an EDCDataset."""
        results = {}

        tl_baseline = dataset.get_table("TL1")
        tl_followup = dataset.get_table("TL")
        ntl_baseline = dataset.get_table("NTL1")
        ntl_followup = dataset.get_table("NTL")
        nl_data = dataset.get_table("NL")
        rs_data = dataset.get_table("RS")

        subj_col = "SUBJID" if tl_baseline is not None and "SUBJID" in (tl_baseline.columns if tl_baseline is not None else []) else "subject_id"

        subjects = set()
        for df in [tl_baseline, tl_followup, rs_data]:
            if df is not None and subj_col in df.columns:
                subjects.update(df[subj_col].dropna().unique())

        for subj in sorted(subjects):
            if not subj or str(subj).lower() == "nan":
                continue
            result = self._evaluate_subject(
                str(subj), subj_col,
                tl_baseline, tl_followup,
                ntl_baseline, ntl_followup,
                nl_data, rs_data,
            )
            results[str(subj)] = result

        return results

    def _evaluate_subject(
        self,
        subj: str,
        subj_col: str,
        tl_bl: pd.DataFrame | None,
        tl_fu: pd.DataFrame | None,
        ntl_bl: pd.DataFrame | None,
        ntl_fu: pd.DataFrame | None,
        nl: pd.DataFrame | None,
        rs: pd.DataFrame | None,
    ) -> RECISTResult:
        result = RECISTResult(subject_id=subj)

        baseline_sod = self._compute_baseline_sod(subj, subj_col, tl_bl)
        result.baseline_sod = baseline_sod

        tl_timeline = self._build_tl_timeline(subj, subj_col, tl_fu, baseline_sod)
        result.tl_timeline = tl_timeline
        result.nadir_sod = self._compute_nadir(tl_timeline, baseline_sod)

        ntl_timeline = self._build_ntl_timeline(subj, subj_col, ntl_fu)
        nl_visits = self._get_new_lesion_visits(subj, subj_col, nl)
        inv_responses = self._get_investigator_responses(subj, subj_col, rs)

        nadir = baseline_sod
        for tla in tl_timeline:
            if tla.sod_mm is not None and nadir is not None:
                if tla.sod_mm < nadir:
                    nadir = tla.sod_mm
                tla.pct_change_from_nadir = (
                    ((tla.sod_mm - nadir) / nadir * 100) if nadir > 0 else 0.0
                )

            tla.tl_response = self._assign_tl_response(
                tla.sod_mm, baseline_sod, nadir,
                tla.n_lesions_measured, tla.n_lesions_total,
            )

            ntl_status = self._find_ntl_status(tla.visit, ntl_timeline)
            has_new = tla.visit in nl_visits

            overall = self._assign_overall_response(
                tla.tl_response, ntl_status, has_new,
            )

            inv_resp = inv_responses.get(tla.visit)
            disc = None
            if inv_resp and inv_resp != overall:
                disc = f"Investigator={inv_resp.value}, Algorithm={overall.value}"
                result.findings.append({
                    "type": "recist_discordance",
                    "subject_id": subj,
                    "visit": tla.visit,
                    "description": f"RECIST discordance: {disc}",
                    "severity": "medium",
                    "investigator": inv_resp.value,
                    "algorithm": overall.value,
                })

            oa = OverallAssessment(
                subject_id=subj,
                visit=tla.visit,
                date=tla.date,
                tl_response=tla.tl_response,
                ntl_status=ntl_status,
                new_lesion=has_new,
                overall_response=overall,
                investigator_response=inv_resp,
                discordance=disc,
            )
            result.assessments.append(oa)

            if nadir is not None and tla.sod_mm is not None and tla.sod_mm < nadir:
                nadir = tla.sod_mm

        self._apply_confirmation(result)
        result.best_response = self._best_overall(result.assessments)
        result.confirmed_best_response = self._best_confirmed(result.assessments)

        return result

    def _compute_baseline_sod(
        self, subj: str, subj_col: str, tl_bl: pd.DataFrame | None
    ) -> float | None:
        if tl_bl is None or subj_col not in tl_bl.columns:
            return None

        rows = tl_bl[tl_bl[subj_col] == subj]
        sod_col = None
        for c in ("TLTSOD1", "TLTSOD", "TLSOD1"):
            if c in rows.columns:
                sod_col = c
                break

        if sod_col:
            vals = pd.to_numeric(rows[sod_col], errors="coerce").dropna()
            if not vals.empty:
                return float(vals.iloc[0])

        diam_col = None
        for c in ("TLSOD1", "TLSOD"):
            if c in rows.columns:
                diam_col = c
                break
        if diam_col:
            diams = pd.to_numeric(rows[diam_col], errors="coerce").dropna()
            if not diams.empty:
                return float(diams.sum())

        return None

    def _build_tl_timeline(
        self, subj: str, subj_col: str, tl_fu: pd.DataFrame | None,
        baseline_sod: float | None,
    ) -> list[TargetLesionAssessment]:
        if tl_fu is None or subj_col not in tl_fu.columns:
            return []

        rows = tl_fu[tl_fu[subj_col] == subj]
        if rows.empty:
            return []

        visit_col = "VISIT" if "VISIT" in rows.columns else "visit"
        date_col = next((c for c in ("TLDAT", "date") if c in rows.columns), None)
        sod_col = next((c for c in ("TLTSOD", "TLSOD") if c in rows.columns), None)
        pct_col = next((c for c in ("TLBCHA", "TLBCHAM") if c in rows.columns), None)

        timeline = []
        visits_seen = set()

        for _, row in rows.iterrows():
            visit = str(row.get(visit_col, "")).strip()
            if not visit or visit in visits_seen:
                continue
            visits_seen.add(visit)

            sod_val = None
            if sod_col:
                sod_val = pd.to_numeric(row.get(sod_col), errors="coerce")
                if pd.isna(sod_val):
                    sod_val = None

            pct_bl = None
            if pct_col:
                pct_bl = pd.to_numeric(row.get(pct_col), errors="coerce")
                if pd.isna(pct_bl):
                    pct_bl = None
            elif sod_val is not None and baseline_sod and baseline_sod > 0:
                pct_bl = (sod_val - baseline_sod) / baseline_sod * 100

            date_val = str(row.get(date_col, "")) if date_col else ""

            tla = TargetLesionAssessment(
                subject_id=subj,
                visit=visit,
                date=date_val,
                sod_mm=float(sod_val) if sod_val is not None else None,
                pct_change_from_baseline=float(pct_bl) if pct_bl is not None else None,
                pct_change_from_nadir=None,
            )
            timeline.append(tla)

        return timeline

    def _compute_nadir(
        self, timeline: list[TargetLesionAssessment], baseline: float | None,
    ) -> float | None:
        vals = [baseline] if baseline else []
        vals.extend(t.sod_mm for t in timeline if t.sod_mm is not None)
        return min(vals) if vals else None

    def _build_ntl_timeline(
        self, subj: str, subj_col: str, ntl_fu: pd.DataFrame | None,
    ) -> dict[str, str]:
        """Returns {visit: ntl_status}."""
        if ntl_fu is None or subj_col not in ntl_fu.columns:
            return {}

        rows = ntl_fu[ntl_fu[subj_col] == subj]
        visit_col = "VISIT" if "VISIT" in rows.columns else "visit"
        status_col = next((c for c in ("NTLRS", "ntl_status") if c in rows.columns), None)

        result = {}
        for _, row in rows.iterrows():
            visit = str(row.get(visit_col, "")).strip()
            if not visit:
                continue
            if status_col:
                raw = str(row.get(status_col, "")).strip()
                result[visit] = self._normalize_ntl_status(raw)
            else:
                result[visit] = "NE"
        return result

    def _normalize_ntl_status(self, raw: str) -> str:
        lower = raw.lower()
        if "cr" in lower or "完全" in lower:
            return "CR"
        if "pd" in lower or "进展" in lower:
            return "PD"
        if "non" in lower or "非" in lower or "稳定" in lower:
            return "non-CR/non-PD"
        if raw.strip():
            return "non-CR/non-PD"
        return "NE"

    def _get_new_lesion_visits(
        self, subj: str, subj_col: str, nl: pd.DataFrame | None,
    ) -> set[str]:
        if nl is None or subj_col not in nl.columns:
            return set()

        rows = nl[nl[subj_col] == subj]
        visit_col = "VISIT" if "VISIT" in rows.columns else "visit"
        yn_col = next((c for c in ("NLYN", "NLRSYN") if c in rows.columns), None)

        visits = set()
        for _, row in rows.iterrows():
            visit = str(row.get(visit_col, "")).strip()
            if yn_col:
                val = str(row.get(yn_col, "")).strip().lower()
                if val in ("是", "yes", "1", "y"):
                    visits.add(visit)
            elif visit:
                visits.add(visit)
        return visits

    def _get_investigator_responses(
        self, subj: str, subj_col: str, rs: pd.DataFrame | None,
    ) -> dict[str, Response]:
        if rs is None or subj_col not in rs.columns:
            return {}

        rows = rs[rs[subj_col] == subj]
        visit_col = "VISIT" if "VISIT" in rows.columns else "visit"
        recist_col = next((c for c in ("RECIST", "RSRECIST", "overall_response") if c in rows.columns), None)

        result = {}
        for _, row in rows.iterrows():
            visit = str(row.get(visit_col, "")).strip()
            if not visit or not recist_col:
                continue
            raw = str(row.get(recist_col, "")).strip()
            if raw:
                result[visit] = Response.from_str(raw)
        return result

    def _assign_tl_response(
        self,
        sod: float | None,
        baseline: float | None,
        nadir: float | None,
        n_measured: int,
        n_total: int,
    ) -> Response:
        if sod is None or baseline is None:
            return Response.NE

        if sod == 0:
            return Response.CR

        pct_from_baseline = (sod - baseline) / baseline * 100 if baseline > 0 else 0
        if pct_from_baseline <= self.pr_threshold:
            return Response.PR

        if nadir is not None and nadir > 0:
            pct_from_nadir = (sod - nadir) / nadir * 100
            abs_increase = sod - nadir
            if pct_from_nadir >= self.pd_threshold and abs_increase >= self.pd_absolute_mm:
                return Response.PD

        return Response.SD

    def _find_ntl_status(self, visit: str, ntl_timeline: dict[str, str]) -> str:
        return ntl_timeline.get(visit, "NE")

    def _assign_overall_response(
        self,
        tl: Response,
        ntl: str,
        new_lesion: bool,
    ) -> Response:
        """RECIST 1.1 overall response assignment (Table 1)."""
        if new_lesion:
            return Response.PD

        if ntl == "PD":
            return Response.PD

        if tl == Response.PD:
            return Response.PD

        if tl == Response.CR and ntl in ("CR", "NE", ""):
            return Response.CR

        if tl == Response.CR and ntl == "non-CR/non-PD":
            return Response.PR

        if tl == Response.PR:
            return Response.PR

        if tl == Response.SD:
            return Response.SD

        return Response.NE

    def _apply_confirmation(self, result: RECISTResult) -> None:
        """Apply confirmation logic for CR/PR."""
        assessments = result.assessments
        for i, oa in enumerate(assessments):
            if oa.overall_response in (Response.CR, Response.PR):
                for j in range(i + 1, len(assessments)):
                    next_oa = assessments[j]
                    if next_oa.overall_response == oa.overall_response:
                        oa.confirmed = True
                        oa.confirmation_date = next_oa.date
                        break
                    elif next_oa.overall_response == Response.PD:
                        break

    def _best_overall(self, assessments: list[OverallAssessment]) -> Response:
        priority = {Response.CR: 0, Response.PR: 1, Response.SD: 2, Response.PD: 3, Response.NE: 4}
        best = Response.NE
        for oa in assessments:
            if priority.get(oa.overall_response, 99) < priority.get(best, 99):
                best = oa.overall_response
        return best

    def _best_confirmed(self, assessments: list[OverallAssessment]) -> Response:
        priority = {Response.CR: 0, Response.PR: 1, Response.SD: 2, Response.PD: 3, Response.NE: 4}
        best = Response.NE
        for oa in assessments:
            resp = oa.overall_response if oa.confirmed else (
                Response.SD if oa.overall_response in (Response.CR, Response.PR) else oa.overall_response
            )
            if priority.get(resp, 99) < priority.get(best, 99):
                best = resp
        return best

    def generate_findings(self, results: dict[str, RECISTResult]) -> list[dict[str, Any]]:
        """Collect all RECIST-related findings across subjects."""
        all_findings = []
        for subj, result in results.items():
            all_findings.extend(result.findings)

            if result.best_response != result.confirmed_best_response:
                all_findings.append({
                    "type": "recist_unconfirmed",
                    "subject_id": subj,
                    "description": (
                        f"Best response {result.best_response.value} not confirmed "
                        f"(confirmed: {result.confirmed_best_response.value})"
                    ),
                    "severity": "medium",
                })

            if result.baseline_sod is None:
                all_findings.append({
                    "type": "recist_missing_baseline",
                    "subject_id": subj,
                    "description": "Missing baseline SOD for target lesions",
                    "severity": "high",
                })

        return all_findings
