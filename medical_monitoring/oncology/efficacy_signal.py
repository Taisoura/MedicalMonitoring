"""
Efficacy Signal Detection Module

Detects early efficacy signals in oncology trials:
- Waterfall plot data (best % change from baseline SOD)
- Spider plot data (% change at each assessment timepoint)
- Response rate computation with exact binomial CI
- Dose-response relationship exploration
- Duration of response (DOR) calculation
- Swimmer plot data (treatment timeline per subject)

References:
  Eisenhauer EA et al. RECIST 1.1 (Eur J Cancer 2009)
  Clopper-Pearson exact binomial CI
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class WaterfallEntry:
    """Best % change from baseline SOD for one subject."""
    subject_id: str
    best_pct_change: float
    best_response: str
    dose_level: str = ""
    confirmed: bool = False


@dataclass
class SpiderEntry:
    """% change from baseline SOD across timepoints for one subject."""
    subject_id: str
    dose_level: str = ""
    timepoints: list[str] = field(default_factory=list)
    pct_changes: list[float] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)


@dataclass
class ResponseRateResult:
    """Response rate with confidence interval."""
    evaluable_n: int
    responders: int
    response_rate: float
    ci_lower: float
    ci_upper: float
    ci_method: str = "Clopper-Pearson"
    confidence_level: float = 0.95
    by_response: dict[str, int] = field(default_factory=dict)
    by_dose: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class DOREntry:
    """Duration of response for one subject."""
    subject_id: str
    response_date: str
    response_type: str
    end_date: str
    end_reason: str  # progression, death, censored
    duration_days: int | None = None
    censored: bool = False


@dataclass
class SwimmerEntry:
    """Swimmer plot data for one subject."""
    subject_id: str
    dose_level: str = ""
    treatment_start: str = ""
    treatment_end: str = ""
    duration_days: int | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    best_response: str = "NE"
    ongoing: bool = False


@dataclass
class EfficacyResult:
    """Complete efficacy signal analysis result."""
    waterfall_data: list[WaterfallEntry] = field(default_factory=list)
    spider_data: list[SpiderEntry] = field(default_factory=list)
    response_rate: ResponseRateResult | None = None
    dor_data: list[DOREntry] = field(default_factory=list)
    swimmer_data: list[SwimmerEntry] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class EfficacySignalDetector:
    """Oncology efficacy signal detection engine."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.confidence_level = cfg.get("confidence_level", 0.95)

    def evaluate_from_recist(
        self,
        recist_results: dict[str, Any],
        dataset=None,
    ) -> EfficacyResult:
        """Build efficacy analysis from RECIST evaluation results."""
        result = EfficacyResult()

        for subj, rr in recist_results.items():
            wf = self._build_waterfall_entry(subj, rr)
            if wf:
                result.waterfall_data.append(wf)

            sp = self._build_spider_entry(subj, rr)
            if sp:
                result.spider_data.append(sp)

            dor = self._compute_dor(subj, rr)
            if dor:
                result.dor_data.append(dor)

        result.waterfall_data.sort(key=lambda x: x.best_pct_change)
        result.response_rate = self._compute_response_rate(recist_results)
        result.findings = self._generate_findings(result)
        result.summary = self._build_summary(result)

        return result

    def evaluate_dataset(self, dataset) -> EfficacyResult:
        """Evaluate efficacy directly from dataset (without prior RECIST run)."""
        from .recist import RECISTEngine
        engine = RECISTEngine()
        recist_results = engine.evaluate_dataset(dataset)
        return self.evaluate_from_recist(recist_results, dataset)

    def _build_waterfall_entry(self, subj: str, rr) -> WaterfallEntry | None:
        timeline = getattr(rr, "tl_timeline", [])
        if not timeline:
            return None

        pct_changes = [
            t.pct_change_from_baseline for t in timeline
            if t.pct_change_from_baseline is not None
        ]
        if not pct_changes:
            return None

        best_pct = min(pct_changes)
        best_resp = getattr(rr, "best_response", None)
        confirmed = getattr(rr, "confirmed_best_response", None)

        return WaterfallEntry(
            subject_id=subj,
            best_pct_change=best_pct,
            best_response=best_resp.value if best_resp else "NE",
            dose_level=getattr(rr, "dose_level", ""),
            confirmed=confirmed is not None and confirmed == best_resp,
        )

    def _build_spider_entry(self, subj: str, rr) -> SpiderEntry | None:
        timeline = getattr(rr, "tl_timeline", [])
        if not timeline:
            return None

        timepoints = []
        pct_changes = []
        dates = []

        for t in timeline:
            if t.pct_change_from_baseline is not None:
                timepoints.append(t.visit)
                pct_changes.append(t.pct_change_from_baseline)
                dates.append(t.date)

        if not pct_changes:
            return None

        return SpiderEntry(
            subject_id=subj,
            timepoints=timepoints,
            pct_changes=pct_changes,
            dates=dates,
        )

    def _compute_response_rate(
        self, recist_results: dict[str, Any],
    ) -> ResponseRateResult:
        by_response: dict[str, int] = {}
        n_evaluable = 0
        n_responders = 0

        for subj, rr in recist_results.items():
            br = getattr(rr, "best_response", None)
            if br is None:
                continue
            resp_str = br.value if hasattr(br, "value") else str(br)
            by_response[resp_str] = by_response.get(resp_str, 0) + 1
            n_evaluable += 1
            if resp_str in ("CR", "PR"):
                n_responders += 1

        rate = n_responders / n_evaluable if n_evaluable > 0 else 0.0
        ci_lo, ci_hi = self._clopper_pearson(n_responders, n_evaluable)

        return ResponseRateResult(
            evaluable_n=n_evaluable,
            responders=n_responders,
            response_rate=rate,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            by_response=by_response,
        )

    def _clopper_pearson(
        self, k: int, n: int, alpha: float | None = None,
    ) -> tuple[float, float]:
        if alpha is None:
            alpha = 1 - self.confidence_level
        if n == 0:
            return (0.0, 1.0)
        lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
        hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
        return (float(lo), float(hi))

    def _compute_dor(self, subj: str, rr) -> DOREntry | None:
        assessments = getattr(rr, "assessments", [])
        if not assessments:
            return None

        first_response = None
        for oa in assessments:
            resp = oa.overall_response
            resp_val = resp.value if hasattr(resp, "value") else str(resp)
            if resp_val in ("CR", "PR"):
                first_response = oa
                break

        if first_response is None:
            return None

        end_oa = None
        for oa in assessments:
            resp_val = oa.overall_response.value if hasattr(oa.overall_response, "value") else str(oa.overall_response)
            if resp_val == "PD":
                end_oa = oa
                break

        if end_oa:
            try:
                d1 = pd.to_datetime(first_response.date, errors="coerce")
                d2 = pd.to_datetime(end_oa.date, errors="coerce")
                dur = (d2 - d1).days if pd.notna(d1) and pd.notna(d2) else None
            except Exception:
                dur = None
            return DOREntry(
                subject_id=subj,
                response_date=first_response.date,
                response_type=first_response.overall_response.value,
                end_date=end_oa.date,
                end_reason="progression",
                duration_days=dur,
                censored=False,
            )
        else:
            last = assessments[-1]
            return DOREntry(
                subject_id=subj,
                response_date=first_response.date,
                response_type=first_response.overall_response.value,
                end_date=last.date,
                end_reason="censored",
                duration_days=None,
                censored=True,
            )

    def _generate_findings(self, result: EfficacyResult) -> list[dict[str, Any]]:
        findings = []

        rr = result.response_rate
        if rr and rr.response_rate > 0:
            findings.append({
                "type": "efficacy_response_rate",
                "description": (
                    f"ORR: {rr.response_rate:.1%} ({rr.responders}/{rr.evaluable_n}), "
                    f"95% CI [{rr.ci_lower:.1%}, {rr.ci_upper:.1%}]"
                ),
                "severity": "info",
                "orr": rr.response_rate,
                "ci": [rr.ci_lower, rr.ci_upper],
            })

        deep_responders = [w for w in result.waterfall_data if w.best_pct_change <= -50]
        if deep_responders:
            findings.append({
                "type": "deep_response_signal",
                "description": f"{len(deep_responders)} subject(s) with >=50% tumor shrinkage",
                "severity": "info",
                "subjects": [d.subject_id for d in deep_responders],
            })

        return findings

    def _build_summary(self, result: EfficacyResult) -> dict[str, Any]:
        rr = result.response_rate
        return {
            "n_evaluable": rr.evaluable_n if rr else 0,
            "orr": rr.response_rate if rr else None,
            "orr_ci": [rr.ci_lower, rr.ci_upper] if rr else None,
            "dcr": (
                sum(1 for r, c in (rr.by_response or {}).items() if r in ("CR", "PR", "SD"))
                / rr.evaluable_n if rr and rr.evaluable_n > 0 else None
            ),
            "by_response": rr.by_response if rr else {},
            "n_waterfall": len(result.waterfall_data),
            "n_dor": len(result.dor_data),
            "median_best_change": (
                float(np.median([w.best_pct_change for w in result.waterfall_data]))
                if result.waterfall_data else None
            ),
        }
