"""
Medical Monitoring Pipeline

End-to-end orchestration of the complete medical monitoring workflow:
Phase 0.5 (AI pre-processing) → Phase 1 → Phase 6 progressive analysis.

Usage:
    from medical_monitoring.pipeline import MedicalMonitoringPipeline

    pipeline = MedicalMonitoringPipeline(
        edc_file="path/to/edc_export.xlsx",
        ctcae_file="path/to/CTCAE v5.0 Clean.xlsx",
    )
    report = pipeline.run()
    pipeline.export("output_report.xlsx")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .analysis.statistics import StatisticalToolbox
from .analysis.subject_db import SubjectDatabaseBuilder, SubjectDBConfig
from .audit.ae_quality import AEQualityAuditor
from .audit.causality import CausalityAuditor, CausalityConfig, DrugContext
from .audit.cross_domain import CrossDomainAuditor
from .audit.ctcae_grading import CTCAEGradingEngine
from .parsers.edc_parser import EDCDataset, EDCParser
from .reporting.report_generator import MedicalMonitoringReportGenerator

logger = logging.getLogger(__name__)


class MedicalMonitoringPipeline:
    """
    Complete Medical Monitoring pipeline orchestrator.

    Executes the progressive analysis:
      Phase 0.5 (optional): AI pre-processing (normalization, cleaning, annotation)
      Phase 1: Data integrity & baseline overview
      Phase 2: Cross-domain auditing
      Phase 3: Regression & consistency analysis
      Phase 4: Multi-domain deep audit
      Phase 5: Drug causality audit
      Phase 6: Reverse verification & scoring
    """

    def __init__(
        self,
        edc_file: str | Path,
        ctcae_file: str | Path | None = None,
        audit_rules_config: str | Path | None = None,
        subject_db_config: SubjectDBConfig | None = None,
        causality_config: CausalityConfig | None = None,
        drug_context: DrugContext | None = None,
        ai_config_path: str | Path | None = None,
        ai_enabled: bool = False,
    ):
        self.edc_file = Path(edc_file)
        self.ctcae_file = Path(ctcae_file) if ctcae_file else None
        self.audit_rules_config = Path(audit_rules_config) if audit_rules_config else None

        # AI engine setup
        self.ai_enabled = ai_enabled
        self.ai_engine = None
        self._ai_config = None
        if ai_enabled:
            self._init_ai(ai_config_path)

        # Wire DrugContext into CausalityConfig if provided separately
        if drug_context and causality_config:
            causality_config.drug_context = drug_context
            if drug_context.drug_name and not causality_config.drug_name:
                causality_config.drug_name = drug_context.drug_name
        elif drug_context and not causality_config:
            causality_config = CausalityConfig(
                drug_name=drug_context.drug_name,
                drug_context=drug_context,
            )

        self.drug_context = drug_context

        self.parser = EDCParser()
        self.cross_domain = CrossDomainAuditor(self.audit_rules_config, ai_engine=self.ai_engine)
        self.subject_db_builder = SubjectDatabaseBuilder(subject_db_config)
        self.stats = StatisticalToolbox()
        self.ctcae_engine = CTCAEGradingEngine(self.ctcae_file, ai_engine=self.ai_engine)
        self.ae_quality = AEQualityAuditor(ai_engine=self.ai_engine)
        self.causality = CausalityAuditor(causality_config, ai_engine=self.ai_engine)
        self.report_gen = MedicalMonitoringReportGenerator()

        self.dataset: EDCDataset | None = None
        self.subject_db: pd.DataFrame | None = None
        self.results: dict[str, Any] = {}

    def _init_ai(self, config_path: str | Path | None = None) -> None:
        """Initialize the AI engine from configuration."""
        try:
            from .ai.engine import AIConfig, LLMEngine

            if config_path:
                self._ai_config = AIConfig.from_yaml(config_path)
            else:
                self._ai_config = AIConfig.default()

            if self._ai_config.enabled:
                self.ai_engine = LLMEngine(self._ai_config)
                logger.info("AI engine initialized (providers: %s)",
                            list(self._ai_config.providers.keys()))
            else:
                logger.info("AI is disabled in configuration")
                self.ai_enabled = False
        except Exception as e:
            logger.warning("AI initialization failed, continuing without AI: %s", e)
            self.ai_enabled = False
            self.ai_engine = None

    def run(self) -> dict[str, Any]:
        """Execute the complete medical monitoring pipeline."""
        # Phase 1: Parse and build baseline
        self.dataset = self._phase1_parse()

        # Phase 0.5: AI pre-processing (between parse and audit)
        if self.ai_enabled and self.ai_engine and self.dataset:
            self.results["ai_preprocessing"] = self._phase0_5_ai_preprocess()

        # Phase 2: Cross-domain auditing
        self.results["cross_domain"] = self._phase2_cross_domain()

        # Phase 3: Subject database + regression + consistency
        self.subject_db = self._phase3_analysis()
        self.results["statistics"] = self._phase3_statistics()

        # Phase 4: CTCAE grading (multi-domain deep audit)
        self.results["ctcae_grading"] = self._phase4_ctcae()

        # Phase 5: Drug causality
        self.results["causality"] = self._phase5_causality()

        # Phase 6: AE quality audit (includes reverse verification)
        self.results["ae_quality"] = self._phase6_ae_quality()

        # Generate report
        report = self._generate_report()
        self.results["report"] = report

        return self.results

    def _phase0_5_ai_preprocess(self) -> dict[str, Any]:
        """Phase 0.5: AI pre-processing -- clean, normalize, annotate."""
        ai_results: dict[str, Any] = {}

        if self.dataset is None or self.ai_engine is None:
            return ai_results

        # 1. Field annotation for key forms
        try:
            from .ai.annotator import FieldAnnotator
            annotator = FieldAnnotator(self.ai_engine)
            for form_code, df in self.dataset.tables.items():
                ann = annotator.annotate(df, form_code,
                                         self.dataset.form_mapping.get(form_code, ""))
                ai_results.setdefault("annotations", {})[form_code] = {
                    "mapping": ann.annotations,
                    "confidence": ann.confidence,
                    "source": ann.source,
                }
            logger.info("AI field annotation complete for %d forms", len(self.dataset.tables))
        except Exception as e:
            logger.warning("Field annotation failed: %s", e)

        # 2. Data cleaning on AE table
        ae_df = self.dataset.get_table("AE")
        if ae_df is not None:
            try:
                from .ai.cleaner import DataCleaner
                cleaner = DataCleaner(self.ai_engine)
                ae_names = ae_df["field_0"].dropna().astype(str).tolist()
                if ae_names:
                    clean_results = cleaner.clean_batch(ae_names, field_type="ae_name")
                    corrections = [r for r in clean_results if r.correction_type != "none"]
                    ai_results["ae_cleaning"] = {
                        "total_processed": len(ae_names),
                        "corrections": len(corrections),
                        "types": {r.correction_type for r in corrections},
                    }
                    logger.info("AI cleaned %d/%d AE names", len(corrections), len(ae_names))
            except Exception as e:
                logger.warning("AE data cleaning failed: %s", e)

            # 3. AE term normalization
            try:
                from .ai.normalizer import MedTermNormalizer
                normalizer = MedTermNormalizer(self.ai_engine)
                ae_names = ae_df["field_0"].dropna().astype(str).tolist()
                unique_names = list(set(n.strip() for n in ae_names if n.strip()))
                if unique_names:
                    normalized = normalizer.normalize_terms(unique_names)
                    high_conf = [n for n in normalized if n.confidence >= 0.7]
                    ai_results["ae_normalization"] = {
                        "unique_terms": len(unique_names),
                        "normalized": len(high_conf),
                        "variant_groups": len(normalizer.build_variant_groups(normalized)),
                    }
                    logger.info("AI normalized %d/%d AE terms",
                                len(high_conf), len(unique_names))
            except Exception as e:
                logger.warning("AE normalization failed: %s", e)

        return ai_results

    def export(self, output_path: str | Path) -> None:
        """Export results to Excel."""
        if "report" in self.results:
            self.report_gen.export_to_excel(self.results["report"], output_path)

    def export_pdf(self, output_path: str | Path) -> Path:
        """Export results to a formal PDF report."""
        from .reporting.export_pdf import PDFReportExporter
        exporter = PDFReportExporter(output_path)
        report = self.results.get("report", {})
        findings = self.report_gen.all_findings if self.report_gen else None
        return exporter.export(report, findings)

    def export_docx(self, output_path: str | Path) -> Path:
        """Export results to a Word (.docx) report."""
        from .reporting.export_docx import DocxReportExporter
        exporter = DocxReportExporter(output_path)
        report = self.results.get("report", {})
        findings = self.report_gen.all_findings if self.report_gen else None
        return exporter.export(report, findings)

    def export_json(self, output_path: str | Path, minimal: bool = False) -> Path:
        """Export results to structured JSON."""
        from .reporting.export_json import JSONReportExporter
        exporter = JSONReportExporter(output_path)
        report = self.results.get("report")
        findings = self.report_gen.all_findings if self.report_gen else None
        if minimal:
            exporter.export_minimal(self.results)
        else:
            exporter.export(self.results, report, findings, self.subject_db)
        return Path(output_path)

    def launch_dashboard(self, json_path: str | Path | None = None, port: int = 8501):
        """Launch the Streamlit web dashboard."""
        if json_path is None:
            json_path = Path("mm_results.json")
            self.export_json(json_path)
        from .reporting.dashboard_app import launch_dashboard as _launch
        _launch(json_path, port)

    def _phase1_parse(self) -> EDCDataset:
        """Phase 1: Parse EDC data and establish data landscape."""
        dataset = self.parser.parse(self.edc_file)
        self.report_gen.project_info = {
            "project_code": dataset.project_code,
            "n_subjects": str(dataset.n_subjects),
            "n_tables": str(dataset.n_tables),
            "source_file": str(self.edc_file.name),
        }
        return dataset

    def _phase2_cross_domain(self) -> dict[str, list]:
        """Phase 2: Run all cross-domain audit rules."""
        if self.dataset is None:
            return {}
        return self.cross_domain.run_all(self.dataset)

    def _phase3_analysis(self) -> pd.DataFrame:
        """Phase 3: Build subject-level database."""
        if self.dataset is None:
            return pd.DataFrame()
        return self.subject_db_builder.build(self.dataset)

    def _phase3_statistics(self) -> dict[str, Any]:
        """Phase 3: Run statistical analyses on subject database."""
        if self.subject_db is None or self.subject_db.empty:
            return {}

        results = {}

        # OR analysis for risk factors
        binary_predictors = [
            col for col in self.subject_db.columns
            if self.subject_db[col].nunique() == 2 and col not in ["subject_id", "sex", "death"]
        ]

        if "death" in self.subject_db.columns and binary_predictors:
            results["mortality_or"] = [
                r.__dict__ for r in
                self.stats.compute_all_or(self.subject_db, binary_predictors, "death")
            ]

        # Score trajectory
        nihss_cols = [c for c in self.subject_db.columns if c.startswith("nihss_v")]
        if len(nihss_cols) >= 2:
            trajectories = self.stats.analyze_trajectories(self.subject_db, nihss_cols)
            results["score_trajectories"] = trajectories

        # Consistency audit
        consistency = self.stats.audit_score_consistency(self.subject_db)
        results["consistency"] = consistency

        # Method recommendations
        n = len(self.subject_db)
        results["recommended_methods"] = self.stats.recommend_methods(n)

        return results

    def _phase4_ctcae(self) -> dict[str, Any]:
        """Phase 4: Run CTCAE systematic grading."""
        if self.dataset is None:
            return {}
        return self.ctcae_engine.run_full_grading(self.dataset)

    def _phase5_causality(self) -> dict[str, Any]:
        """Phase 5: Run drug causality audit."""
        if self.dataset is None:
            return {}
        return self.causality.run_full_audit(self.dataset)

    def _phase6_ae_quality(self) -> dict[str, Any]:
        """Phase 6: Run AE quality audit."""
        if self.dataset is None:
            return {}
        return self.ae_quality.run_full_audit(self.dataset)

    def visualize(
        self,
        output_dir: str | Path,
        theme: str = "fda_regulatory",
        interactive: bool = True,
    ) -> dict[str, list[Path]]:
        """
        Generate all visualizations from pipeline results.

        Args:
            output_dir: Directory for saved figures.
            theme: One of 'fda_regulatory', 'academic_conference', 'pharma_dashboard'.
            interactive: If True, also produce Plotly HTML charts.

        Returns:
            Dict mapping chart names to lists of saved file paths.
        """
        from .visualization.viz_overview import OverviewVisualizer
        from .visualization.viz_audit import AuditVisualizer
        from .visualization.viz_subject import SubjectVisualizer
        from .visualization.viz_statistics import StatisticsVisualizer
        from .visualization.viz_causality import CausalityVisualizer
        from .visualization.viz_ae_quality import AEQualityVisualizer
        from .visualization.viz_ctcae import CTCAEVisualizer
        from .visualization.viz_dashboard import DashboardVisualizer

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: dict[str, list[Path]] = {}

        # Module A: Overview
        if self.dataset:
            viz_a = OverviewVisualizer(theme)
            try:
                viz_a.plot_crf_heatmap(self.dataset, output_dir=out, interactive=interactive)
                viz_a.plot_enrollment(self.dataset, output_dir=out, interactive=interactive)
                viz_a.plot_consort(self.dataset, output_dir=out, interactive=interactive)
            except Exception:
                pass

        # Module B: Audit
        cross = self.results.get("cross_domain")
        if cross:
            viz_b = AuditVisualizer(theme)
            try:
                viz_b.plot_findings_bar(cross, output_dir=out, interactive=interactive)
                viz_b.plot_findings_treemap(cross, output_dir=out)
                viz_b.plot_subject_rule_heatmap(cross, output_dir=out, interactive=interactive)
            except Exception:
                pass

        # Module C: Subject
        if self.subject_db is not None and not self.subject_db.empty:
            viz_c = SubjectVisualizer(theme)
            try:
                viz_c.plot_demographics_pyramid(self.subject_db, output_dir=out, interactive=interactive)
                score_cols = [c for c in self.subject_db.columns if c.startswith("nihss_v")]
                if len(score_cols) >= 2:
                    viz_c.plot_spaghetti(self.subject_db, score_cols, score_name="NIHSS",
                                         output_dir=out, interactive=interactive)
                event_cols = [c for c in self.subject_db.columns if c.startswith("ce")]
                viz_c.plot_swimmer(self.subject_db, event_cols=event_cols,
                                    output_dir=out, interactive=interactive)
            except Exception:
                pass

        # Module D: Statistics
        stats = self.results.get("statistics", {})
        viz_d = StatisticsVisualizer(theme)
        or_data = stats.get("mortality_or", [])
        if or_data:
            try:
                viz_d.plot_forest(or_data, output_dir=out, interactive=interactive)
            except Exception:
                pass

        # Module E: Causality
        caus = self.results.get("causality")
        if caus:
            viz_e = CausalityVisualizer(theme)
            try:
                viz_e.plot_bidirectional_waterfall(caus, output_dir=out, interactive=interactive)
                viz_e.plot_consistency_heatmap(caus, output_dir=out, interactive=interactive)
            except Exception:
                pass

        # Module F1: AE Quality
        aeq = self.results.get("ae_quality")
        if aeq and "error" not in aeq:
            viz_f1 = AEQualityVisualizer(theme)
            try:
                viz_f1.plot_quality_radar(aeq, output_dir=out, interactive=interactive)
                viz_f1.plot_severity_bar(ae_results=aeq, output_dir=out, interactive=interactive)
                actions = aeq.get("10_action_plan", [])
                if actions:
                    viz_f1.plot_action_gantt(actions, output_dir=out, interactive=interactive)
                cross_refs = aeq.get("9_cross_ref", [])
                if cross_refs:
                    viz_f1.plot_cross_ref_matrix(cross_refs, output_dir=out, interactive=interactive)
            except Exception:
                pass

        # Module F2: CTCAE
        ctcae = self.results.get("ctcae_grading")
        if ctcae:
            viz_f2 = CTCAEVisualizer(theme)
            try:
                hys = ctcae.get("hys_law_screening", [])
                if hys:
                    viz_f2.plot_edish(hys, output_dir=out, interactive=interactive)
                graded = ctcae.get("graded_records", [])
                if graded:
                    viz_f2.plot_grade_waterfall(graded, output_dir=out, interactive=interactive)
                disc = ctcae.get("grade_discrepancies", [])
                if disc:
                    viz_f2.plot_discrepancy_lollipop(disc, output_dir=out, interactive=interactive)
                ncs = ctcae.get("ncs_disputes", [])
                if ncs:
                    viz_f2.plot_ncs_dot(ncs, output_dir=out, interactive=interactive)
                g_dist = ctcae.get("summary", {}).get("grade_distribution")
                if g_dist:
                    viz_f2.plot_grade_waffle(g_dist, output_dir=out)
            except Exception:
                pass

        # Executive Dashboard
        viz_dash = DashboardVisualizer(theme)
        try:
            viz_dash.generate_dashboard(
                self.results, self.subject_db,
                output_dir=out, interactive=interactive,
            )
        except Exception:
            pass

        return saved

    def _generate_report(self) -> dict[str, Any]:
        """Generate the final medical monitoring report."""
        return self.report_gen.generate_report(self.results, self.subject_db)
