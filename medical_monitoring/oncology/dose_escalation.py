"""
Dose Escalation Monitoring Module

Supports Phase I oncology trial dose escalation with:
- DLT (Dose-Limiting Toxicity) tracking and evaluation
- BOIN design decision boundaries (escalate / stay / de-escalate)
- 3+3 design support
- Dose modification tracking (interruptions, reductions)
- SMC (Safety Monitoring Committee) decision support summaries

References:
  Liu S, Yuan Y. Bayesian Optimal Interval Designs. 2015.
  CTCAE v5.0 for DLT grading criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DLTEvent:
    """A single DLT event."""
    subject_id: str
    dose_level: str
    cohort: str
    visit: str
    date: str
    dlt_type: str  # hematologic, non-hematologic, death
    ae_reference: str = ""
    description: str = ""


@dataclass
class DoseCohort:
    """Summary of a dose cohort."""
    dose_level: str
    dose_mg: float | None
    n_enrolled: int = 0
    n_evaluable: int = 0
    n_dlt: int = 0
    dlt_rate: float = 0.0
    dlt_events: list[DLTEvent] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    status: str = "open"  # open, closed, mtd


@dataclass
class BOINDecision:
    """BOIN design decision for a dose cohort."""
    dose_level: str
    n_treated: int
    n_dlt: int
    observed_rate: float
    lambda_e: float
    lambda_d: float
    decision: str  # escalate, stay, de-escalate, stop
    target_rate: float


@dataclass
class DoseModification:
    """Dose modification event."""
    subject_id: str
    visit: str
    date: str
    planned_dose: float | None
    actual_dose: float | None
    action: str  # interruption, reduction, delay
    reason: str = ""


@dataclass
class DoseEscalationResult:
    """Complete dose escalation analysis result."""
    cohorts: dict[str, DoseCohort] = field(default_factory=dict)
    boin_decisions: list[BOINDecision] = field(default_factory=list)
    dose_modifications: list[DoseModification] = field(default_factory=list)
    mtd_estimate: str | None = None
    findings: list[dict[str, Any]] = field(default_factory=list)
    smc_summary: dict[str, Any] = field(default_factory=dict)


class DoseEscalationMonitor:
    """Phase I dose escalation monitoring engine."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.target_dlt_rate = cfg.get("target_dlt_rate", 0.30)
        self.max_dlt_rate = cfg.get("max_dlt_rate", 0.40)
        self.cohort_size = cfg.get("cohort_size", 3)
        self.max_sample = cfg.get("max_sample_per_dose", 12)
        self.dlt_criteria = cfg.get("dlt_criteria", {})

        self._lambda_e, self._lambda_d = self._compute_boin_boundaries()

    def _compute_boin_boundaries(self) -> tuple[float, float]:
        """Compute BOIN escalation/de-escalation boundaries."""
        p_target = self.target_dlt_rate
        p1 = max(0.05, p_target - 0.10)
        p2 = min(0.95, p_target + 0.10)

        lambda_e = np.log(p_target / p1) / np.log(
            p_target * (1 - p1) / (p1 * (1 - p_target))
        )
        lambda_d = np.log(p2 / p_target) / np.log(
            p2 * (1 - p_target) / (p_target * (1 - p2))
        )
        return float(lambda_e), float(lambda_d)

    def evaluate_dataset(self, dataset) -> DoseEscalationResult:
        """Run dose escalation analysis on an EDCDataset."""
        result = DoseEscalationResult()

        dlt_df = dataset.get_table("DLT")
        ex_df = dataset.get_table("EX")
        ran_df = dataset.get_table("RAN")

        subj_col = "SUBJID" if dlt_df is not None and "SUBJID" in (dlt_df.columns if dlt_df is not None else []) else "subject_id"

        cohort_map = self._build_cohort_map(ran_df, subj_col)
        result.cohorts = self._build_cohorts(cohort_map, dlt_df, subj_col)
        self._extract_dlt_events(result, dlt_df, subj_col, cohort_map)
        result.dose_modifications = self._extract_dose_modifications(ex_df, subj_col)

        for dose_level, cohort in result.cohorts.items():
            if cohort.n_evaluable > 0:
                cohort.dlt_rate = cohort.n_dlt / cohort.n_evaluable
            decision = self._boin_decision(cohort)
            result.boin_decisions.append(decision)

        result.mtd_estimate = self._estimate_mtd(result.cohorts)
        result.findings = self._generate_findings(result)
        result.smc_summary = self._build_smc_summary(result)

        return result

    def _build_cohort_map(
        self, ran_df: pd.DataFrame | None, subj_col: str,
    ) -> dict[str, dict[str, str]]:
        """Map subject -> {dose_level, cohort, arm}."""
        if ran_df is None:
            return {}

        mapping = {}
        grp_col = next((c for c in ("RANGRP", "dose_group", "RANARM") if c in ran_df.columns), None)
        arm_col = next((c for c in ("RANARM", "RANSPHA") if c in ran_df.columns), None)

        for _, row in ran_df.iterrows():
            subj = str(row.get(subj_col, "")).strip()
            if not subj or subj.lower() == "nan":
                continue
            dose = str(row.get(grp_col, "unknown")).strip() if grp_col else "unknown"
            arm = str(row.get(arm_col, "")).strip() if arm_col else ""
            mapping[subj] = {"dose_level": dose, "cohort": arm}
        return mapping

    def _build_cohorts(
        self,
        cohort_map: dict[str, dict[str, str]],
        dlt_df: pd.DataFrame | None,
        subj_col: str,
    ) -> dict[str, DoseCohort]:
        cohorts: dict[str, DoseCohort] = {}

        for subj, info in cohort_map.items():
            dl = info["dose_level"]
            if dl not in cohorts:
                dose_mg = self._parse_dose_mg(dl)
                cohorts[dl] = DoseCohort(dose_level=dl, dose_mg=dose_mg)
            cohorts[dl].n_enrolled += 1
            cohorts[dl].subjects.append(subj)

        if dlt_df is not None and subj_col in dlt_df.columns:
            eval_col = next(
                (c for c in ("DLTFYN", "dlt_evaluable") if c in dlt_df.columns), None
            )
            for _, row in dlt_df.iterrows():
                subj = str(row.get(subj_col, "")).strip()
                if subj in cohort_map:
                    dl = cohort_map[subj]["dose_level"]
                    if dl in cohorts:
                        if eval_col:
                            val = str(row.get(eval_col, "")).strip().lower()
                            if val in ("是", "yes", "1", "y"):
                                cohorts[dl].n_evaluable += 1
                        else:
                            cohorts[dl].n_evaluable += 1

        for c in cohorts.values():
            if c.n_evaluable == 0:
                c.n_evaluable = c.n_enrolled

        return cohorts

    def _extract_dlt_events(
        self,
        result: DoseEscalationResult,
        dlt_df: pd.DataFrame | None,
        subj_col: str,
        cohort_map: dict[str, dict[str, str]],
    ) -> None:
        if dlt_df is None:
            return

        yn_col = next((c for c in ("DLTYN", "dlt_occurred") if c in dlt_df.columns), None)
        date_col = next((c for c in ("DLTDAT", "dlt_date") if c in dlt_df.columns), None)
        visit_col = "VISIT" if "VISIT" in dlt_df.columns else "visit"

        heme_col = next((c for c in ("DLTRES1",) if c in dlt_df.columns), None)
        nonheme_col = next((c for c in ("DLTRES2",) if c in dlt_df.columns), None)
        death_col = next((c for c in ("DLTRES3",) if c in dlt_df.columns), None)

        for _, row in dlt_df.iterrows():
            subj = str(row.get(subj_col, "")).strip()
            if not subj or subj.lower() == "nan":
                continue

            if yn_col:
                val = str(row.get(yn_col, "")).strip().lower()
                if val not in ("是", "yes", "1", "y"):
                    continue

            info = cohort_map.get(subj, {"dose_level": "unknown", "cohort": ""})

            dlt_types = []
            if heme_col and str(row.get(heme_col, "")).strip().lower() in ("是", "yes", "1"):
                dlt_types.append("hematologic")
            if nonheme_col and str(row.get(nonheme_col, "")).strip().lower() in ("是", "yes", "1"):
                dlt_types.append("non-hematologic")
            if death_col and str(row.get(death_col, "")).strip().lower() in ("是", "yes", "1"):
                dlt_types.append("death")
            if not dlt_types:
                dlt_types = ["unspecified"]

            for dt in dlt_types:
                event = DLTEvent(
                    subject_id=subj,
                    dose_level=info["dose_level"],
                    cohort=info.get("cohort", ""),
                    visit=str(row.get(visit_col, "")),
                    date=str(row.get(date_col, "")),
                    dlt_type=dt,
                )
                dl = info["dose_level"]
                if dl in result.cohorts:
                    result.cohorts[dl].n_dlt += 1
                    result.cohorts[dl].dlt_events.append(event)

    def _extract_dose_modifications(
        self, ex_df: pd.DataFrame | None, subj_col: str,
    ) -> list[DoseModification]:
        if ex_df is None:
            return []

        mods = []
        adj_col = next((c for c in ("EXADJYN", "dose_adjusted") if c in ex_df.columns), None)
        act_col = next((c for c in ("EXADJACN", "adjustment_action") if c in ex_df.columns), None)
        plan_col = next((c for c in ("EXPDOSE", "planned_dose") if c in ex_df.columns), None)
        actual_col = next((c for c in ("EXDOSE", "actual_dose") if c in ex_df.columns), None)
        reason_col = next((c for c in ("ADJREAS", "adjustment_reason") if c in ex_df.columns), None)
        date_col = next((c for c in ("EXSTDAT", "start_date") if c in ex_df.columns), None)
        visit_col = "VISIT" if "VISIT" in ex_df.columns else "visit"

        for _, row in ex_df.iterrows():
            subj = str(row.get(subj_col, "")).strip()
            if not subj or subj.lower() == "nan":
                continue

            if adj_col:
                val = str(row.get(adj_col, "")).strip().lower()
                if val not in ("是", "yes", "1", "y"):
                    continue

            planned = pd.to_numeric(row.get(plan_col, None), errors="coerce") if plan_col else None
            actual = pd.to_numeric(row.get(actual_col, None), errors="coerce") if actual_col else None
            action_raw = str(row.get(act_col, "")).strip() if act_col else ""

            action = "modification"
            lower = action_raw.lower()
            if "中断" in lower or "interrupt" in lower:
                action = "interruption"
            elif "减量" in lower or "reduc" in lower:
                action = "reduction"
            elif "延迟" in lower or "delay" in lower:
                action = "delay"

            mods.append(DoseModification(
                subject_id=subj,
                visit=str(row.get(visit_col, "")),
                date=str(row.get(date_col, "")),
                planned_dose=float(planned) if planned is not None and not pd.isna(planned) else None,
                actual_dose=float(actual) if actual is not None and not pd.isna(actual) else None,
                action=action,
                reason=str(row.get(reason_col, "")) if reason_col else "",
            ))

        return mods

    def _boin_decision(self, cohort: DoseCohort) -> BOINDecision:
        n = cohort.n_evaluable
        k = cohort.n_dlt
        rate = k / n if n > 0 else 0.0

        if n == 0:
            decision = "stay"
        elif rate <= self._lambda_e:
            decision = "escalate"
        elif rate >= self._lambda_d:
            if self._should_stop(n, k):
                decision = "stop"
            else:
                decision = "de-escalate"
        else:
            decision = "stay"

        return BOINDecision(
            dose_level=cohort.dose_level,
            n_treated=n,
            n_dlt=k,
            observed_rate=rate,
            lambda_e=self._lambda_e,
            lambda_d=self._lambda_d,
            decision=decision,
            target_rate=self.target_dlt_rate,
        )

    def _should_stop(self, n: int, k: int) -> bool:
        """Posterior Pr(p > target | k, n) > 0.95 => stop for safety."""
        a, b = k + 1, n - k + 1
        prob_exceed = 1 - stats.beta.cdf(self.target_dlt_rate, a, b)
        return prob_exceed > 0.95

    def _estimate_mtd(self, cohorts: dict[str, DoseCohort]) -> str | None:
        """Estimate MTD as highest dose with DLT rate closest to target."""
        evaluable = [(dl, c) for dl, c in cohorts.items() if c.n_evaluable >= self.cohort_size]
        if not evaluable:
            return None

        evaluable.sort(key=lambda x: x[1].dose_mg or 0)

        best_dl = None
        best_diff = float("inf")
        for dl, c in evaluable:
            if c.dlt_rate <= self.max_dlt_rate:
                diff = abs(c.dlt_rate - self.target_dlt_rate)
                if diff < best_diff:
                    best_diff = diff
                    best_dl = dl
        return best_dl

    def _parse_dose_mg(self, dose_str: str) -> float | None:
        import re
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mg)?", dose_str, re.IGNORECASE)
        return float(match.group(1)) if match else None

    def _generate_findings(self, result: DoseEscalationResult) -> list[dict[str, Any]]:
        findings = []

        for dl, cohort in result.cohorts.items():
            if cohort.dlt_rate > self.max_dlt_rate:
                findings.append({
                    "type": "dlt_rate_exceeded",
                    "dose_level": dl,
                    "description": f"DLT rate {cohort.dlt_rate:.0%} exceeds maximum {self.max_dlt_rate:.0%} at {dl}",
                    "severity": "critical",
                    "dlt_rate": cohort.dlt_rate,
                    "n_dlt": cohort.n_dlt,
                    "n_evaluable": cohort.n_evaluable,
                })

        for bd in result.boin_decisions:
            if bd.decision == "stop":
                findings.append({
                    "type": "boin_stop",
                    "dose_level": bd.dose_level,
                    "description": f"BOIN recommends STOP at {bd.dose_level} (rate={bd.observed_rate:.0%})",
                    "severity": "critical",
                })

        n_mods = len(result.dose_modifications)
        if n_mods > 0:
            by_type = {}
            for m in result.dose_modifications:
                by_type.setdefault(m.action, []).append(m)
            for action, mods in by_type.items():
                if len(mods) >= 3:
                    findings.append({
                        "type": "frequent_dose_modification",
                        "description": f"{len(mods)} dose {action}s observed across subjects",
                        "severity": "high" if action == "interruption" else "medium",
                        "count": len(mods),
                        "action": action,
                    })

        return findings

    def _build_smc_summary(self, result: DoseEscalationResult) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "total_enrolled": sum(c.n_enrolled for c in result.cohorts.values()),
            "total_evaluable": sum(c.n_evaluable for c in result.cohorts.values()),
            "total_dlt": sum(c.n_dlt for c in result.cohorts.values()),
            "mtd_estimate": result.mtd_estimate,
            "cohort_summary": [],
        }

        for dl in sorted(result.cohorts.keys(), key=lambda x: result.cohorts[x].dose_mg or 0):
            c = result.cohorts[dl]
            bd = next((b for b in result.boin_decisions if b.dose_level == dl), None)
            summary["cohort_summary"].append({
                "dose_level": dl,
                "dose_mg": c.dose_mg,
                "enrolled": c.n_enrolled,
                "evaluable": c.n_evaluable,
                "dlt": c.n_dlt,
                "dlt_rate": f"{c.dlt_rate:.0%}",
                "boin_decision": bd.decision if bd else "N/A",
                "dlt_types": [e.dlt_type for e in c.dlt_events],
            })

        n_mods = len(result.dose_modifications)
        summary["dose_modifications"] = {
            "total": n_mods,
            "interruptions": sum(1 for m in result.dose_modifications if m.action == "interruption"),
            "reductions": sum(1 for m in result.dose_modifications if m.action == "reduction"),
            "delays": sum(1 for m in result.dose_modifications if m.action == "delay"),
        }

        return summary
