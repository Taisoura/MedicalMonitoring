"""
Module F2: CTCAE Systematic Grading Engine

Uses CTCAE v5.0 Clean.xlsx as the lookup table to:
1. Auto-grade lab results against CTCAE thresholds
2. Detect grading inconsistencies (CTCAE > AE recorded grade)
3. Identify Grade 3+ without AE records (safety data gaps)
4. Hy's Law screening for drug-induced liver injury
5. Grade 2+ clinically significant without AE statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..parsers.edc_parser import EDCDataset
from ..utils.helpers import Finding, normalize_subject_id, safe_float


@dataclass
class CTCAEGradeResult:
    """Result of CTCAE auto-grading for a single lab record."""
    subject_id: str
    visit: str
    test_code: str
    test_name: str
    result: float
    unit: str
    lower_limit: float | None
    upper_limit: float | None
    ctcae_grade: int
    ctcae_term: str
    calculation_basis: str
    investigator_assessment: str
    matched_ae: str
    ae_grade: int | None
    grade_discrepancy: int  # ctcae_grade - ae_grade (positive = underreported)


@dataclass
class HysLawResult:
    """Hy's Law screening result for a subject."""
    subject_id: str
    alt_max: float | None
    alt_fold: float | None
    ast_max: float | None
    ast_fold: float | None
    tbil_max: float | None
    tbil_fold: float | None
    alp_max: float | None
    alp_fold: float | None
    meets_hys_law: bool | None  # None = cannot evaluate (missing data)
    missing_data: list[str] = field(default_factory=list)
    clinical_note: str = ""


# Pre-coded CTCAE v5.0 laboratory thresholds (Investigations SOC)
# Format: (direction, [G1_threshold, G2_threshold, G3_threshold, G4_threshold])
# "high_fold" = result/ULN; "low_fold" = result/LLN; "absolute" = absolute values
LAB_CTCAE_THRESHOLDS: dict[str, dict[str, Any]] = {
    "ALT": {
        "direction": "high_fold",
        "ctcae_term": "Alanine aminotransferase increased",
        "meddra_code": "10001551",
        "thresholds": [1.0, 3.0, 5.0, 20.0],  # >ULN-3x=G1, >3-5x=G2, >5-20x=G3, >20x=G4
    },
    "AST": {
        "direction": "high_fold",
        "ctcae_term": "Aspartate aminotransferase increased",
        "meddra_code": "10003481",
        "thresholds": [1.0, 3.0, 5.0, 20.0],
    },
    "ALP": {
        "direction": "high_fold",
        "ctcae_term": "Alkaline phosphatase increased",
        "meddra_code": "10001675",
        "thresholds": [1.0, 2.5, 5.0, 20.0],
    },
    "GGT": {
        "direction": "high_fold",
        "ctcae_term": "GGT increased",
        "meddra_code": "10056910",
        "thresholds": [1.0, 2.5, 5.0, 20.0],
    },
    "TBIL": {
        "direction": "high_fold",
        "ctcae_term": "Blood bilirubin increased",
        "meddra_code": "10005364",
        "thresholds": [1.0, 1.5, 3.0, 10.0],
    },
    "CREAT": {
        "direction": "high_fold",
        "ctcae_term": "Creatinine increased",
        "meddra_code": "10011368",
        "thresholds": [1.0, 1.5, 3.0, 6.0],
    },
    "CK": {
        "direction": "high_fold",
        "ctcae_term": "CPK increased",
        "meddra_code": "10011411",
        "thresholds": [1.0, 2.5, 5.0, 10.0],
    },
    "LYM": {
        "direction": "low_absolute",
        "ctcae_term": "Lymphocyte count decreased",
        "meddra_code": "10025256",
        "thresholds": [0.8, 0.5, 0.2, 0.2],  # <LLN-0.8=G1, <0.8-0.5=G2, <0.5-0.2=G3, <0.2=G4
        "unit": "10E9/L",
    },
    "WBC": {
        "direction": "low_absolute",
        "ctcae_term": "White blood cell decreased",
        "meddra_code": "10047942",
        "thresholds": [3.0, 2.0, 1.0, 1.0],
        "unit": "10E9/L",
    },
    "NEUT": {
        "direction": "low_absolute",
        "ctcae_term": "Neutrophil count decreased",
        "meddra_code": "10029366",
        "thresholds": [1.5, 1.0, 0.5, 0.5],
        "unit": "10E9/L",
    },
    "PLT": {
        "direction": "low_absolute",
        "ctcae_term": "Platelet count decreased",
        "meddra_code": "10035528",
        "thresholds": [75.0, 50.0, 25.0, 25.0],
        "unit": "10E9/L",
    },
    "HGB": {
        "direction": "low_absolute",
        "ctcae_term": "Anemia",
        "meddra_code": "10002272",
        "thresholds": [100.0, 80.0, 80.0, 0.0],  # G3 needs transfusion criterion
        "unit": "g/L",
    },
    "K_LOW": {
        "direction": "low_absolute",
        "ctcae_term": "Hypokalemia",
        "meddra_code": "10021018",
        "thresholds": [3.0, 3.0, 2.5, 2.5],
        "unit": "mmol/L",
    },
    "K_HIGH": {
        "direction": "high_absolute",
        "ctcae_term": "Hyperkalemia",
        "meddra_code": "10020639",
        "thresholds": [5.5, 6.0, 7.0, 7.0],
        "unit": "mmol/L",
    },
    "NA_LOW": {
        "direction": "low_absolute",
        "ctcae_term": "Hyponatremia",
        "meddra_code": "10021036",
        "thresholds": [130.0, 125.0, 120.0, 120.0],
        "unit": "mmol/L",
    },
    "ALB": {
        "direction": "low_absolute",
        "ctcae_term": "Hypoalbuminemia",
        "meddra_code": "10020943",
        "thresholds": [30.0, 20.0, 20.0, 0.0],
        "unit": "g/L",
    },
    "FIBRINO": {
        "direction": "low_fold",
        "ctcae_term": "Fibrinogen decreased",
        "meddra_code": "10016596",
        "thresholds": [0.75, 0.5, 0.25, 0.25],  # xLLN: <1.0-0.75=G1, etc.
    },
    "INR": {
        "direction": "high_absolute",
        "ctcae_term": "INR increased",
        "meddra_code": "10022402",
        "thresholds": [1.2, 1.5, 2.5, float("inf")],
    },
    "APTT": {
        "direction": "high_fold",
        "ctcae_term": "Activated partial thromboplastin time prolonged",
        "meddra_code": "10000636",
        "thresholds": [1.0, 1.5, 2.5, float("inf")],
    },
}


class CTCAEGradingEngine:
    """CTCAE v5.0 auto-grading engine with Hy's Law screening."""

    def __init__(self, ctcae_reference_path: str | Path | None = None, ai_engine=None):
        self.reference_df: pd.DataFrame | None = None
        self.ai_engine = ai_engine
        self._ai_matcher = None
        if ai_engine is not None:
            from ..ai.matcher import SemanticMatcher
            self._ai_matcher = SemanticMatcher(ai_engine)
        if ctcae_reference_path:
            self._load_reference(ctcae_reference_path)

    def _load_reference(self, path: str | Path) -> None:
        """Load the CTCAE v5.0 Clean.xlsx reference table."""
        self.reference_df = pd.read_excel(path, engine="openpyxl")

    def grade_lab_result(
        self,
        test_code: str,
        result: float,
        lower_limit: float | None = None,
        upper_limit: float | None = None,
    ) -> tuple[int, str, str]:
        """
        Auto-grade a single lab result against CTCAE thresholds.
        Returns: (grade, ctcae_term, calculation_basis)
        """
        code_upper = test_code.upper().strip()

        # Handle potassium direction
        if code_upper == "K":
            if upper_limit and result > upper_limit:
                code_upper = "K_HIGH"
            elif lower_limit and result < lower_limit:
                code_upper = "K_LOW"
            else:
                return (0, "", "Within normal range")

        # Handle sodium
        if code_upper == "NA":
            if lower_limit and result < lower_limit:
                code_upper = "NA_LOW"
            else:
                return (0, "", "Within normal range")

        threshold_info = LAB_CTCAE_THRESHOLDS.get(code_upper)
        if not threshold_info:
            return (0, "", f"No CTCAE threshold defined for {code_upper}")

        direction = threshold_info["direction"]
        thresholds = threshold_info["thresholds"]
        ctcae_term = threshold_info["ctcae_term"]

        grade = 0
        basis = ""

        if direction == "high_fold":
            if upper_limit and upper_limit > 0:
                fold = result / upper_limit
                if fold > thresholds[3]:
                    grade = 4
                    basis = f"{fold:.1f}×ULN (>{thresholds[3]}×)"
                elif fold > thresholds[2]:
                    grade = 3
                    basis = f"{fold:.1f}×ULN (>{thresholds[2]}-{thresholds[3]}×)"
                elif fold > thresholds[1]:
                    grade = 2
                    basis = f"{fold:.1f}×ULN (>{thresholds[1]}-{thresholds[2]}×)"
                elif fold > thresholds[0]:
                    grade = 1
                    basis = f"{fold:.1f}×ULN (>{thresholds[0]}-{thresholds[1]}×)"

        elif direction == "low_fold":
            if lower_limit and lower_limit > 0:
                fold = result / lower_limit
                if fold < thresholds[3]:
                    grade = 4
                    basis = f"{fold:.2f}×LLN (<{thresholds[3]}×)"
                elif fold < thresholds[2]:
                    grade = 3
                    basis = f"{fold:.2f}×LLN (<{thresholds[2]}×)"
                elif fold < thresholds[1]:
                    grade = 2
                    basis = f"{fold:.2f}×LLN (<{thresholds[1]}×)"
                elif fold < thresholds[0]:
                    grade = 1
                    basis = f"{fold:.2f}×LLN (<{thresholds[0]}×)"

        elif direction == "low_absolute":
            if result < thresholds[3]:
                grade = 4
                basis = f"{result} {threshold_info.get('unit', '')} (<{thresholds[3]})"
            elif result < thresholds[2]:
                grade = 3
                basis = f"{result} {threshold_info.get('unit', '')} (<{thresholds[2]})"
            elif result < thresholds[1]:
                grade = 2
                basis = f"{result} {threshold_info.get('unit', '')} (<{thresholds[1]})"
            elif lower_limit and result < lower_limit:
                grade = 1
                basis = f"{result} {threshold_info.get('unit', '')} (<LLN)"

        elif direction == "high_absolute":
            if result > thresholds[3]:
                grade = 4
                basis = f"{result} (>{thresholds[3]})"
            elif result > thresholds[2]:
                grade = 3
                basis = f"{result} (>{thresholds[2]})"
            elif result > thresholds[1]:
                grade = 2
                basis = f"{result} (>{thresholds[1]})"
            elif result > thresholds[0]:
                grade = 1
                basis = f"{result} (>{thresholds[0]})"

        return (grade, ctcae_term, basis)

    def run_full_grading(self, dataset: EDCDataset) -> dict[str, Any]:
        """Run complete CTCAE grading audit on all lab data."""
        results = {
            "graded_records": [],
            "grade_discrepancies": [],
            "g3_plus_no_ae": [],
            "g2_plus_cs_no_ae": [],
            "hys_law_screening": [],
            "ncs_disputes": [],
            "summary": {},
        }

        ae_df = dataset.get_table("AE")
        all_graded = []

        for lb_name in ["LB1", "LB2", "LB3"]:
            lb_df = dataset.get_table(lb_name)
            if lb_df is None:
                continue

            for _, row in lb_df.iterrows():
                graded = self._grade_single_record(row, ae_df, lb_name)
                if graded and graded.ctcae_grade > 0:
                    all_graded.append(graded)

        results["graded_records"] = all_graded
        results["grade_discrepancies"] = [
            r for r in all_graded if r.grade_discrepancy >= 1
        ]
        results["g3_plus_no_ae"] = [
            r for r in all_graded if r.ctcae_grade >= 3 and not r.matched_ae
        ]
        results["g2_plus_cs_no_ae"] = [
            r for r in all_graded
            if r.ctcae_grade >= 2 and "临床意义" in r.investigator_assessment and not r.matched_ae
        ]
        results["hys_law_screening"] = self._screen_hys_law(dataset)
        results["ncs_disputes"] = [
            r for r in all_graded
            if r.ctcae_grade >= 2 and "无临床意义" in r.investigator_assessment
        ]

        g_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for r in all_graded:
            g_counts[min(r.ctcae_grade, 4)] += 1

        results["summary"] = {
            "total_records_graded": len(all_graded),
            "grade_distribution": g_counts,
            "discrepancies_count": len(results["grade_discrepancies"]),
            "g3_plus_no_ae_count": len(results["g3_plus_no_ae"]),
            "g2_plus_cs_no_ae_count": len(results["g2_plus_cs_no_ae"]),
            "ncs_disputes_count": len(results["ncs_disputes"]),
            "hys_law_cases": len([h for h in results["hys_law_screening"]
                                  if h.meets_hys_law is True]),
        }
        return results

    def _grade_single_record(
        self, row: pd.Series, ae_df: pd.DataFrame | None, source: str
    ) -> CTCAEGradeResult | None:
        """Grade a single lab record and compare with AE data."""
        subj = normalize_subject_id(row.get("subject_id"))
        test_code = str(row.get("field_1", "")).strip()
        test_name = str(row.get("field_0", "")).strip()
        result_val = safe_float(row.get("field_2"))
        unit = str(row.get("field_3", "")).strip()
        lower = safe_float(row.get("field_4"))
        upper = safe_float(row.get("field_5"))
        cs_assessment = str(row.get("field_6", "")).strip()
        visit = str(row.get("visit", "")).strip()

        if result_val is None:
            return None

        grade, ctcae_term, basis = self.grade_lab_result(test_code, result_val, lower, upper)
        if grade == 0:
            return None

        matched_ae = ""
        ae_grade = None
        if ae_df is not None:
            matched_ae, ae_grade = self._find_matching_ae(subj, test_code, ae_df)

        discrepancy = grade - (ae_grade or 0) if ae_grade is not None else grade

        return CTCAEGradeResult(
            subject_id=subj,
            visit=visit,
            test_code=test_code,
            test_name=test_name,
            result=result_val,
            unit=unit,
            lower_limit=lower,
            upper_limit=upper,
            ctcae_grade=grade,
            ctcae_term=ctcae_term,
            calculation_basis=basis,
            investigator_assessment=cs_assessment,
            matched_ae=matched_ae,
            ae_grade=ae_grade,
            grade_discrepancy=discrepancy,
        )

    def _find_matching_ae(
        self, subject_id: str, test_code: str, ae_df: pd.DataFrame
    ) -> tuple[str, int | None]:
        """Find an AE record matching the lab abnormality for a subject."""
        import re as _re

        subj_aes = ae_df[ae_df["subject_id"].apply(normalize_subject_id) == subject_id]
        if subj_aes.empty:
            return ("", None)

        lab_ae_patterns = {
            "ALT": r"ALT|转氨酶|肝功能|aminotransferase|liver",
            "AST": r"AST|转氨酶|肝功能|aminotransferase|liver",
            "LYM": r"淋巴|lymph",
            "WBC": r"白细胞|leukocyte|white blood",
            "PLT": r"血小板|platelet|thrombocyto",
            "HGB": r"贫血|anemia|anaemia|hemoglobin",
            "K": r"钾|potassium|kalemia",
            "FIBRINO": r"纤维蛋白原|fibrinogen|凝血",
            "APTT": r"APTT|凝血|coagul",
            "CK": r"CK|CPK|肌酸激酶|creatine kinase",
            "CREAT": r"肌酐|creatinine|肾功能|renal",
        }

        pattern = lab_ae_patterns.get(test_code.upper(), test_code)

        for _, ae_row in subj_aes.iterrows():
            ae_name = str(ae_row.get("field_0", "")).strip()
            if _re.search(pattern, ae_name, _re.IGNORECASE):
                severity_str = str(ae_row.get("field_2", "")).strip()
                try:
                    ae_g = int(_re.search(r"\d", severity_str).group())
                except (AttributeError, ValueError):
                    ae_g = None
                return (ae_name, ae_g)

        if self._ai_matcher:
            for _, ae_row in subj_aes.iterrows():
                ae_name = str(ae_row.get("field_0", "")).strip()
                if not ae_name:
                    continue
                result = self._ai_matcher.ctcae_matches_ae(
                    ctcae_term=test_code, lab_abnormality=test_code, ae_name=ae_name,
                )
                if result.matches and result.confidence >= 0.7:
                    severity_str = str(ae_row.get("field_2", "")).strip()
                    try:
                        ae_g = int(_re.search(r"\d", severity_str).group())
                    except (AttributeError, ValueError):
                        ae_g = None
                    return (ae_name, ae_g)

        return ("", None)

    def _screen_hys_law(self, dataset: EDCDataset) -> list[HysLawResult]:
        """Screen all subjects for Hy's Law criteria."""
        results = []
        subjects = dataset.subject_ids

        for subj in subjects:
            alt_max, alt_fold = self._get_max_fold(dataset, subj, "ALT")
            ast_max, ast_fold = self._get_max_fold(dataset, subj, "AST")

            if (alt_fold or 0) < 3 and (ast_fold or 0) < 3:
                continue

            tbil_max, tbil_fold = self._get_max_fold(dataset, subj, "TBIL")
            alp_max, alp_fold = self._get_max_fold(dataset, subj, "ALP")

            missing = []
            if tbil_max is None:
                missing.append("TBIL")
            if alp_max is None:
                missing.append("ALP")

            meets = None
            note = ""
            if tbil_fold is not None and alp_fold is not None:
                meets = (
                    ((alt_fold or 0) > 3 or (ast_fold or 0) > 3)
                    and tbil_fold > 2
                    and alp_fold <= 2
                )
                if meets:
                    note = "MEETS Hy's Law: high mortality risk (10-50%)"
                else:
                    note = "Does NOT meet Hy's Law criteria"
            else:
                note = f"Cannot evaluate: missing {', '.join(missing)}"

            results.append(HysLawResult(
                subject_id=subj,
                alt_max=alt_max,
                alt_fold=alt_fold,
                ast_max=ast_max,
                ast_fold=ast_fold,
                tbil_max=tbil_max,
                tbil_fold=tbil_fold,
                alp_max=alp_max,
                alp_fold=alp_fold,
                meets_hys_law=meets,
                missing_data=missing,
                clinical_note=note,
            ))

        return results

    def _get_max_fold(
        self, dataset: EDCDataset, subject_id: str, test_code: str
    ) -> tuple[float | None, float | None]:
        """Get maximum value and fold-change for a test in a subject."""
        max_val = None
        max_fold = None

        for lb_name in ["LB1", "LB2", "LB3", "LB6"]:
            lb_df = dataset.get_table(lb_name)
            if lb_df is None:
                continue

            subj_rows = lb_df[lb_df["subject_id"].apply(normalize_subject_id) == subject_id]
            for _, row in subj_rows.iterrows():
                code = str(row.get("field_1", "")).strip().upper()
                if code != test_code.upper():
                    continue

                val = safe_float(row.get("field_2"))
                upper = safe_float(row.get("field_5"))
                if val is not None:
                    if max_val is None or val > max_val:
                        max_val = val
                    if upper and upper > 0:
                        fold = val / upper
                        if max_fold is None or fold > max_fold:
                            max_fold = fold

        return (max_val, max_fold)

    def generate_findings(self, grading_results: dict) -> list[Finding]:
        """Convert grading results into Finding objects for reporting."""
        findings = []

        for r in grading_results.get("grade_discrepancies", []):
            severity = Finding.URGENT if r.grade_discrepancy >= 2 else Finding.HIGH
            findings.append(Finding(
                category="CTCAE Grade Discrepancy",
                subject_id=r.subject_id,
                description=(
                    f"{r.test_code} = {r.result} ({r.calculation_basis}): "
                    f"CTCAE G{r.ctcae_grade} but AE recorded as G{r.ae_grade}"
                ),
                severity=severity,
                source_table="LB/AE",
                recommendation=f"Upgrade AE severity to Grade {r.ctcae_grade}",
                regulatory_basis="CTCAE v5.0",
            ))

        for r in grading_results.get("g3_plus_no_ae", []):
            findings.append(Finding(
                category="G3+ No AE Record",
                subject_id=r.subject_id,
                description=(
                    f"{r.test_code} = {r.result} ({r.calculation_basis}): "
                    f"CTCAE Grade {r.ctcae_grade} but no AE recorded"
                ),
                severity=Finding.HIGH,
                source_table="LB",
                recommendation="Record AE or provide NCS justification",
                regulatory_basis="ICH E6(R2) Section 12.2",
            ))

        for h in grading_results.get("hys_law_screening", []):
            if h.meets_hys_law is True:
                findings.append(Finding(
                    category="Hy's Law",
                    subject_id=h.subject_id,
                    description=f"MEETS Hy's Law: ALT/AST>3xULN + TBIL>2xULN. {h.clinical_note}",
                    severity=Finding.URGENT,
                    source_table="LB",
                    recommendation="Urgent DILI evaluation; potential 10-50% mortality",
                    regulatory_basis="FDA Guidance on DILI",
                ))
            elif h.missing_data:
                findings.append(Finding(
                    category="Hy's Law - Incomplete Data",
                    subject_id=h.subject_id,
                    description=f"Cannot evaluate Hy's Law: missing {', '.join(h.missing_data)}",
                    severity=Finding.HIGH,
                    source_table="LB",
                    recommendation=f"Query: test {', '.join(h.missing_data)} when ALT/AST elevated",
                    regulatory_basis="FDA Guidance on DILI",
                ))

        return findings
