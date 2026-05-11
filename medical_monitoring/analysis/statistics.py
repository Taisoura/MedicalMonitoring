"""
Module D: Statistical Analysis Toolbox

Provides statistical methods for Medical Monitoring, with automatic
method recommendation based on sample size and data characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ORResult:
    """Odds Ratio calculation result."""
    predictor: str
    outcome: str
    exposed_events: int
    exposed_total: int
    unexposed_events: int
    unexposed_total: int
    odds_ratio: float
    ci_lower: float
    ci_upper: float
    chi2: float
    p_value: float
    interpretation: str


@dataclass
class SurvivalResult:
    """Kaplan-Meier / survival analysis result."""
    group: str
    n_subjects: int
    n_events: int
    median_time: float | None
    survival_at_28d: float | None


@dataclass
class ConsistencyFinding:
    """Score consistency audit finding."""
    subject_id: str
    discrepancy_type: str
    detail: str
    severity: str  # "critical" or "warning"
    recommendation: str


class StatisticalToolbox:
    """Statistical analysis methods for medical monitoring."""

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level

    # --- Sample Size Recommendation ---

    def recommend_methods(self, n: int, data_type: str = "binary") -> list[str]:
        """Recommend appropriate statistical methods based on sample size."""
        methods = []
        if data_type == "binary":
            if n < 30:
                methods = ["Fisher exact test", "Descriptive OR (no inference)"]
            elif n < 100:
                methods = ["Fisher exact test", "OR with 95% CI",
                           "Firth logistic regression", "Kaplan-Meier"]
            elif n < 500:
                methods = ["Chi-square test", "Multivariable logistic regression",
                           "Cox PH model", "Propensity score matching"]
            else:
                methods = ["LASSO regression", "Random forest", "Bayesian network",
                           "Latent class analysis", "Cox PH model"]
        elif data_type == "continuous":
            if n < 30:
                methods = ["Mann-Whitney U test", "Wilcoxon signed-rank"]
            elif n < 100:
                methods = ["Mann-Whitney U", "t-test (if normal)",
                           "Mixed-effects model (longitudinal)"]
            else:
                methods = ["ANOVA", "Linear mixed model", "GEE"]
        return methods

    # --- Odds Ratio Calculation ---

    def compute_or(
        self,
        df: pd.DataFrame,
        predictor_col: str,
        outcome_col: str,
        predictor_name: str = "",
    ) -> ORResult:
        """Compute OR from a 2x2 table with Fisher exact test."""
        table = pd.crosstab(df[predictor_col], df[outcome_col])

        if table.shape != (2, 2):
            return ORResult(
                predictor=predictor_name or predictor_col,
                outcome=outcome_col,
                exposed_events=0, exposed_total=0,
                unexposed_events=0, unexposed_total=0,
                odds_ratio=float("nan"), ci_lower=0, ci_upper=0,
                chi2=0, p_value=1.0, interpretation="Insufficient data for 2x2 table",
            )

        a = table.iloc[1, 1]  # exposed + event
        b = table.iloc[1, 0]  # exposed + no event
        c = table.iloc[0, 1]  # unexposed + event
        d = table.iloc[0, 0]  # unexposed + no event

        if b == 0 or c == 0:
            or_val = float("inf") if a > 0 and d > 0 else float("nan")
            ci_l, ci_u = 0.0, float("inf")
        else:
            or_val = (a * d) / (b * c) if (b * c) > 0 else float("nan")
            log_or = np.log(or_val) if or_val > 0 else 0
            se = np.sqrt(1/max(a, 0.5) + 1/max(b, 0.5) + 1/max(c, 0.5) + 1/max(d, 0.5))
            ci_l = np.exp(log_or - 1.96 * se)
            ci_u = np.exp(log_or + 1.96 * se)

        _, p_fisher = stats.fisher_exact(table.values)
        chi2_stat, _, _, _ = stats.chi2_contingency(table.values, correction=True)

        if or_val > 2:
            interp = "Strong positive association"
        elif or_val > 1.5:
            interp = "Moderate positive association"
        elif or_val > 1:
            interp = "Weak positive association"
        elif or_val == 1:
            interp = "No association"
        else:
            interp = "Negative association (protective?)"

        return ORResult(
            predictor=predictor_name or predictor_col,
            outcome=outcome_col,
            exposed_events=int(a),
            exposed_total=int(a + b),
            unexposed_events=int(c),
            unexposed_total=int(c + d),
            odds_ratio=round(or_val, 3),
            ci_lower=round(ci_l, 3),
            ci_upper=round(ci_u, 3),
            chi2=round(chi2_stat, 3),
            p_value=round(p_fisher, 4),
            interpretation=interp,
        )

    def compute_all_or(
        self,
        df: pd.DataFrame,
        predictor_cols: list[str],
        outcome_col: str,
    ) -> list[ORResult]:
        """Compute OR for multiple predictors against a single outcome."""
        results = []
        for col in predictor_cols:
            if col in df.columns and df[col].nunique() == 2:
                result = self.compute_or(df, col, outcome_col, predictor_name=col)
                results.append(result)
        return results

    # --- Score Trajectory Analysis ---

    def analyze_trajectories(
        self,
        df: pd.DataFrame,
        score_cols: list[str],
        threshold: float = 10.0,
    ) -> pd.DataFrame:
        """Identify subjects with dramatic score worsening between visits."""
        if len(score_cols) < 2:
            return pd.DataFrame()

        results = []
        for _, row in df.iterrows():
            subj = row.get("subject_id", "")
            baseline = row.get(score_cols[0])
            for i in range(1, len(score_cols)):
                followup = row.get(score_cols[i])
                if pd.notna(baseline) and pd.notna(followup):
                    delta = followup - baseline
                    if abs(delta) >= threshold:
                        results.append({
                            "subject_id": subj,
                            "baseline_score": baseline,
                            "followup_score": followup,
                            "delta": delta,
                            "baseline_visit": score_cols[0],
                            "followup_visit": score_cols[i],
                            "direction": "worsening" if delta > 0 else "improvement",
                        })
        return pd.DataFrame(results)

    # --- Scoring Consistency Audit ---

    def audit_score_consistency(
        self,
        df: pd.DataFrame,
        rules: list[dict[str, Any]] | None = None,
    ) -> list[ConsistencyFinding]:
        """
        Check cross-score consistency (e.g., NIHSS worsening should correlate
        with mRS increase; death should have mRS=6).
        """
        if rules is None:
            rules = self._default_consistency_rules()

        findings = []
        for _, row in df.iterrows():
            subj = str(row.get("subject_id", ""))
            for rule in rules:
                finding = self._apply_consistency_rule(subj, row, rule)
                if finding:
                    findings.append(finding)
        return findings

    def _default_consistency_rules(self) -> list[dict]:
        """Default scoring consistency rules for stroke trials."""
        return [
            {
                "name": "death_mrs_6",
                "condition": lambda r: r.get("death") == 1,
                "check": lambda r: r.get("mrs_v4") == 6 or r.get("mrs_v5") == 6,
                "severity": "warning",
                "description": "Death but mRS != 6",
                "recommendation": "Verify mRS at final visit or record mRS=6 post-mortem",
            },
            {
                "name": "nihss_worsening_no_endpoint",
                "condition": lambda r: (
                    pd.notna(r.get("delta_nihss_v1v2")) and r.get("delta_nihss_v1v2", 0) >= 10
                ),
                "check": lambda r: r.get("death") == 1 or r.get("ce1", 0) == 1,
                "severity": "critical",
                "description": "NIHSS worsening >=10 without death or CE1 event",
                "recommendation": "Verify NIHSS scoring accuracy and check for unreported events",
            },
            {
                "name": "high_aspects_death",
                "condition": lambda r: (
                    pd.notna(r.get("aspects")) and r.get("aspects", 0) >= 9
                    and r.get("death") == 1
                ),
                "check": lambda r: False,  # always flag
                "severity": "warning",
                "description": "High ASPECTS (>=9) but death occurred",
                "recommendation": "Investigate new vascular event, re-occlusion, or complications",
            },
        ]

    def _apply_consistency_rule(
        self, subj: str, row: pd.Series, rule: dict
    ) -> ConsistencyFinding | None:
        try:
            if rule["condition"](row):
                if not rule["check"](row):
                    detail_parts = []
                    for col in ["nihss_v1", "nihss_v2", "aspects", "mrs_v4", "death"]:
                        if col in row.index and pd.notna(row[col]):
                            detail_parts.append(f"{col}={row[col]}")
                    return ConsistencyFinding(
                        subject_id=subj,
                        discrepancy_type=rule["name"],
                        detail=f"{rule['description']} ({', '.join(detail_parts)})",
                        severity=rule["severity"],
                        recommendation=rule["recommendation"],
                    )
        except (KeyError, TypeError):
            pass
        return None

    # --- Safety Signal Detection ---

    def compute_prr(
        self,
        ae_df: pd.DataFrame,
        target_ae: str,
        ae_name_col: str = "ae_name",
        drug_col: str = "drug_related",
    ) -> dict[str, float]:
        """
        Proportional Reporting Ratio for safety signal detection.
        PRR = (a/(a+b)) / (c/(c+d))
        """
        target_mask = ae_df[ae_name_col].str.contains(target_ae, case=False, na=False)
        related_mask = ae_df[drug_col].astype(str).str.contains("有关|related", case=False, na=False)

        a = (target_mask & related_mask).sum()
        b = (target_mask & ~related_mask).sum()
        c = (~target_mask & related_mask).sum()
        d = (~target_mask & ~related_mask).sum()

        if (a + b) == 0 or (c + d) == 0 or c == 0:
            return {"prr": float("nan"), "chi2": 0, "signal": False}

        prr = (a / (a + b)) / (c / (c + d))
        n = a + b + c + d
        chi2 = (n * (a * d - b * c) ** 2) / ((a + b) * (c + d) * (a + c) * (b + d)) if n > 0 else 0
        signal = prr >= 2 and chi2 >= 4 and a >= 3

        return {"prr": round(prr, 3), "chi2": round(chi2, 3), "signal": signal}

    # --- Shift Table ---

    def generate_shift_table(
        self,
        baseline_values: pd.Series,
        postbaseline_values: pd.Series,
        categories: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate a shift table comparing baseline to post-baseline categories."""
        if categories is None:
            categories = ["Normal", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]

        baseline_cat = pd.Categorical(baseline_values, categories=categories, ordered=True)
        post_cat = pd.Categorical(postbaseline_values, categories=categories, ordered=True)

        shift = pd.crosstab(
            baseline_cat, post_cat,
            rownames=["Baseline"], colnames=["Post-Baseline"],
            dropna=False,
        )
        return shift
