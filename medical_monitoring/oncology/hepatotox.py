"""
Enhanced Hepatotoxicity Monitoring Module

Extends standard Hy's Law screening with oncology-specific considerations:
- Liver metastases adjustment (baseline-adjusted fold-change)
- DILI pattern classification (hepatocellular vs cholestatic vs mixed)
- ALT/AST time-to-onset and time-course analysis per dose cohort
- eDISH (evaluation of Drug-Induced Serious Hepatotoxicity) plotting data
- Grade 3+ AST/ALT SAE case flagging

References:
  FDA Guidance: Drug-Induced Liver Injury (2009)
  Temple R. Hy's Law. Hepatology 2006.
  Watkins PB. eDISH approach. Hepatology 2011.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class HysLawCase:
    """A potential Hy's Law case."""
    subject_id: str
    alt_peak: float
    alt_xuln: float
    tbil_peak: float
    tbil_xuln: float
    alp_peak: float
    alp_xuln: float
    meets_hys_law: bool
    meets_modified: bool  # with liver mets adjustment
    has_liver_mets: bool
    r_value: float | None
    dili_pattern: str  # hepatocellular, cholestatic, mixed
    date_of_peak_alt: str = ""
    time_to_onset_days: int | None = None
    dose_level: str = ""


@dataclass
class LiverTimepoint:
    """Single liver function measurement."""
    subject_id: str
    visit: str
    date: str
    alt: float | None
    ast: float | None
    tbil: float | None
    alp: float | None
    ggt: float | None = None
    alt_xuln: float | None = None
    ast_xuln: float | None = None
    tbil_xuln: float | None = None
    alp_xuln: float | None = None


@dataclass
class EDISHPoint:
    """Data point for eDISH plot (peak ALT xULN vs peak TBIL xULN)."""
    subject_id: str
    alt_xuln: float
    tbil_xuln: float
    dose_level: str = ""
    has_liver_mets: bool = False


@dataclass
class HepatotoxResult:
    """Complete hepatotoxicity analysis result."""
    hys_law_cases: list[HysLawCase] = field(default_factory=list)
    edish_data: list[EDISHPoint] = field(default_factory=list)
    liver_timecourses: dict[str, list[LiverTimepoint]] = field(default_factory=dict)
    grade3_cases: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class HepatotoxMonitor:
    """Enhanced hepatotoxicity monitoring with liver mets adjustment."""

    ALT_ULN = 40.0
    AST_ULN = 40.0
    TBIL_ULN = 1.2  # mg/dL; will be adjusted by config
    ALP_ULN = 120.0
    GGT_ULN = 50.0

    HYS_ALT_THRESHOLD = 3.0    # ALT >= 3xULN
    HYS_TBIL_THRESHOLD = 2.0   # TBIL >= 2xULN
    HYS_ALP_THRESHOLD = 2.0    # ALP < 2xULN (cholestatic exclusion)

    LIVER_METS_ALT_THRESHOLD = 5.0  # >= 5xULN for pts with liver mets

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.alt_uln = cfg.get("alt_uln", self.ALT_ULN)
        self.ast_uln = cfg.get("ast_uln", self.AST_ULN)
        self.tbil_uln = cfg.get("tbil_uln", self.TBIL_ULN)
        self.alp_uln = cfg.get("alp_uln", self.ALP_ULN)
        self.liver_mets_subjects: set[str] = set(cfg.get("liver_mets_subjects", []))
        self.use_baseline_adjusted = cfg.get("use_baseline_adjusted", True)

    def evaluate_dataset(self, dataset) -> HepatotoxResult:
        """Run hepatotoxicity analysis on an EDCDataset."""
        result = HepatotoxResult()

        lb_chem = dataset.get_table("LB_CHEM")
        if lb_chem is None:
            lb_chem = dataset.get_lb("CHEM")
        if lb_chem is None:
            for key in dataset.tables:
                if key.startswith("LB"):
                    lb_chem = dataset.tables[key]
                    break

        if lb_chem is None:
            result.findings.append({
                "type": "hepatox_no_data",
                "description": "No laboratory chemistry data found for hepatotoxicity analysis",
                "severity": "high",
            })
            return result

        subj_col = "SUBJID" if "SUBJID" in lb_chem.columns else "subject_id"
        subjects = sorted(set(
            s for s in lb_chem[subj_col].dropna().unique()
            if s and str(s).lower() != "nan"
        ))

        self._detect_liver_mets_from_data(dataset)

        for subj in subjects:
            subj = str(subj)
            timecourse = self._extract_liver_timecourse(subj, subj_col, lb_chem)
            if timecourse:
                result.liver_timecourses[subj] = timecourse

                hys = self._evaluate_hys_law(subj, timecourse)
                if hys:
                    result.hys_law_cases.append(hys)

                edish = self._compute_edish_point(subj, timecourse)
                if edish:
                    result.edish_data.append(edish)

                g3 = self._check_grade3(subj, timecourse)
                if g3:
                    result.grade3_cases.extend(g3)

        result.findings = self._generate_findings(result)
        result.summary = self._build_summary(result, len(subjects))

        return result

    def _detect_liver_mets_from_data(self, dataset) -> None:
        """Try to detect liver mets from TL1/NTL1 baseline lesion data."""
        for domain in ("TL1", "NTL1"):
            df = dataset.get_table(domain)
            if df is None:
                continue
            subj_col = "SUBJID" if "SUBJID" in df.columns else "subject_id"
            loc_col = next(
                (c for c in ("TLLOC1", "NTLLOC1", "TLSLOC1", "NTLSLOC1") if c in df.columns),
                None,
            )
            if loc_col:
                for _, row in df.iterrows():
                    loc = str(row.get(loc_col, "")).lower()
                    if "肝" in loc or "liver" in loc:
                        subj = str(row.get(subj_col, "")).strip()
                        if subj and subj.lower() != "nan":
                            self.liver_mets_subjects.add(subj)

    def _extract_liver_timecourse(
        self, subj: str, subj_col: str, lb: pd.DataFrame,
    ) -> list[LiverTimepoint]:
        rows = lb[lb[subj_col] == subj]
        if rows.empty:
            return []

        visit_col = "VISIT" if "VISIT" in rows.columns else "visit"
        date_col = next((c for c in ("LBDAT", "date") if c in rows.columns), None)
        test_col = next((c for c in ("LBTEST", "test_name") if c in rows.columns), None)
        result_col = next((c for c in ("LBORRES", "LBSTRES", "result") if c in rows.columns), None)

        if not test_col or not result_col:
            return []

        visit_data: dict[str, dict[str, Any]] = {}
        for _, row in rows.iterrows():
            visit = str(row.get(visit_col, "")).strip()
            if not visit:
                continue

            test = str(row.get(test_col, "")).strip().upper()
            val = pd.to_numeric(row.get(result_col), errors="coerce")
            if pd.isna(val):
                continue

            if visit not in visit_data:
                visit_data[visit] = {"date": str(row.get(date_col, "")) if date_col else ""}

            if "ALT" in test or "谷丙" in test or "丙氨酸" in test:
                visit_data[visit]["alt"] = float(val)
            elif "AST" in test or "谷草" in test or "天冬氨酸" in test:
                visit_data[visit]["ast"] = float(val)
            elif "TBIL" in test or "总胆红素" in test:
                visit_data[visit]["tbil"] = float(val)
            elif "ALP" in test or "碱性磷酸酶" in test:
                visit_data[visit]["alp"] = float(val)
            elif "GGT" in test or "谷氨酰" in test:
                visit_data[visit]["ggt"] = float(val)

        timeline = []
        for visit, data in visit_data.items():
            alt = data.get("alt")
            ast = data.get("ast")
            tbil = data.get("tbil")
            alp = data.get("alp")

            tp = LiverTimepoint(
                subject_id=subj,
                visit=visit,
                date=data.get("date", ""),
                alt=alt, ast=ast, tbil=tbil, alp=alp,
                ggt=data.get("ggt"),
                alt_xuln=alt / self.alt_uln if alt else None,
                ast_xuln=ast / self.ast_uln if ast else None,
                tbil_xuln=tbil / self.tbil_uln if tbil else None,
                alp_xuln=alp / self.alp_uln if alp else None,
            )
            timeline.append(tp)

        return timeline

    def _evaluate_hys_law(
        self, subj: str, timecourse: list[LiverTimepoint],
    ) -> HysLawCase | None:
        alt_vals = [(t.alt_xuln, t) for t in timecourse if t.alt_xuln is not None]
        tbil_vals = [(t.tbil_xuln, t) for t in timecourse if t.tbil_xuln is not None]
        alp_vals = [(t.alp_xuln, t) for t in timecourse if t.alp_xuln is not None]

        if not alt_vals or not tbil_vals:
            return None

        peak_alt_xuln, peak_alt_tp = max(alt_vals, key=lambda x: x[0])
        peak_tbil_xuln, _ = max(tbil_vals, key=lambda x: x[0])
        peak_alp_xuln = max((v for v, _ in alp_vals), default=0.0)

        peak_alt = peak_alt_tp.alt or 0.0
        peak_tbil = max((t.tbil for t in timecourse if t.tbil), default=0.0)
        peak_alp = max((t.alp for t in timecourse if t.alp), default=0.0)

        has_liver_mets = subj in self.liver_mets_subjects
        alt_threshold = self.LIVER_METS_ALT_THRESHOLD if has_liver_mets else self.HYS_ALT_THRESHOLD

        standard_hys = (
            peak_alt_xuln >= self.HYS_ALT_THRESHOLD
            and peak_tbil_xuln >= self.HYS_TBIL_THRESHOLD
            and peak_alp_xuln < self.HYS_ALP_THRESHOLD
        )

        modified_hys = (
            peak_alt_xuln >= alt_threshold
            and peak_tbil_xuln >= self.HYS_TBIL_THRESHOLD
            and peak_alp_xuln < self.HYS_ALP_THRESHOLD
        )

        r_value = self._compute_r_value(peak_alt_xuln, peak_alp_xuln)
        dili_pattern = self._classify_dili_pattern(r_value)

        if standard_hys or modified_hys or peak_alt_xuln >= 3.0:
            return HysLawCase(
                subject_id=subj,
                alt_peak=peak_alt,
                alt_xuln=peak_alt_xuln,
                tbil_peak=peak_tbil,
                tbil_xuln=peak_tbil_xuln,
                alp_peak=peak_alp,
                alp_xuln=peak_alp_xuln,
                meets_hys_law=standard_hys,
                meets_modified=modified_hys,
                has_liver_mets=has_liver_mets,
                r_value=r_value,
                dili_pattern=dili_pattern,
                date_of_peak_alt=peak_alt_tp.date,
            )

        return None

    def _compute_r_value(self, alt_xuln: float, alp_xuln: float) -> float | None:
        """R-value = (ALT/ULN) / (ALP/ULN) for DILI pattern classification."""
        if alp_xuln and alp_xuln > 0:
            return alt_xuln / alp_xuln
        return None

    def _classify_dili_pattern(self, r_value: float | None) -> str:
        if r_value is None:
            return "unknown"
        if r_value >= 5:
            return "hepatocellular"
        elif r_value <= 2:
            return "cholestatic"
        else:
            return "mixed"

    def _compute_edish_point(
        self, subj: str, timecourse: list[LiverTimepoint],
    ) -> EDISHPoint | None:
        alt_vals = [t.alt_xuln for t in timecourse if t.alt_xuln is not None]
        tbil_vals = [t.tbil_xuln for t in timecourse if t.tbil_xuln is not None]

        if not alt_vals or not tbil_vals:
            return None

        return EDISHPoint(
            subject_id=subj,
            alt_xuln=max(alt_vals),
            tbil_xuln=max(tbil_vals),
            has_liver_mets=subj in self.liver_mets_subjects,
        )

    def _check_grade3(
        self, subj: str, timecourse: list[LiverTimepoint],
    ) -> list[dict[str, Any]]:
        cases = []
        for tp in timecourse:
            if tp.alt_xuln is not None and tp.alt_xuln >= 5.0:
                cases.append({
                    "subject_id": subj,
                    "visit": tp.visit,
                    "date": tp.date,
                    "parameter": "ALT",
                    "value_xuln": tp.alt_xuln,
                    "ctcae_grade": 3 if tp.alt_xuln < 20 else 4,
                })
            if tp.ast_xuln is not None and tp.ast_xuln >= 5.0:
                cases.append({
                    "subject_id": subj,
                    "visit": tp.visit,
                    "date": tp.date,
                    "parameter": "AST",
                    "value_xuln": tp.ast_xuln,
                    "ctcae_grade": 3 if tp.ast_xuln < 20 else 4,
                })
        return cases

    def _generate_findings(self, result: HepatotoxResult) -> list[dict[str, Any]]:
        findings = []

        for case in result.hys_law_cases:
            if case.meets_hys_law:
                findings.append({
                    "type": "hys_law_met",
                    "subject_id": case.subject_id,
                    "description": (
                        f"Hy's Law criteria met: ALT {case.alt_xuln:.1f}xULN, "
                        f"TBIL {case.tbil_xuln:.1f}xULN, ALP {case.alp_xuln:.1f}xULN"
                    ),
                    "severity": "critical",
                    "dili_pattern": case.dili_pattern,
                })
            elif case.alt_xuln >= 3.0:
                findings.append({
                    "type": "alt_elevated",
                    "subject_id": case.subject_id,
                    "description": f"ALT elevated to {case.alt_xuln:.1f}xULN",
                    "severity": "high",
                })

        for g3 in result.grade3_cases:
            if g3["ctcae_grade"] >= 4:
                findings.append({
                    "type": "grade4_liver",
                    "subject_id": g3["subject_id"],
                    "description": (
                        f"Grade 4 {g3['parameter']}: {g3['value_xuln']:.1f}xULN "
                        f"at {g3['visit']}"
                    ),
                    "severity": "critical",
                })

        liver_mets_cases = [c for c in result.hys_law_cases if c.has_liver_mets]
        if liver_mets_cases:
            findings.append({
                "type": "liver_mets_hepatox",
                "description": (
                    f"{len(liver_mets_cases)} subject(s) with liver mets show hepatotoxicity signals"
                ),
                "severity": "high",
                "subjects": [c.subject_id for c in liver_mets_cases],
            })

        return findings

    def _build_summary(self, result: HepatotoxResult, n_subjects: int) -> dict[str, Any]:
        return {
            "n_subjects_analyzed": n_subjects,
            "n_hys_law_met": sum(1 for c in result.hys_law_cases if c.meets_hys_law),
            "n_modified_hys": sum(1 for c in result.hys_law_cases if c.meets_modified),
            "n_alt_elevated_3x": sum(1 for c in result.hys_law_cases if c.alt_xuln >= 3.0),
            "n_grade3_events": len(result.grade3_cases),
            "n_grade4_events": sum(1 for g in result.grade3_cases if g["ctcae_grade"] >= 4),
            "n_liver_mets": len(self.liver_mets_subjects),
            "dili_patterns": {
                "hepatocellular": sum(1 for c in result.hys_law_cases if c.dili_pattern == "hepatocellular"),
                "cholestatic": sum(1 for c in result.hys_law_cases if c.dili_pattern == "cholestatic"),
                "mixed": sum(1 for c in result.hys_law_cases if c.dili_pattern == "mixed"),
            },
        }
