"""
Medical Monitoring Framework for Non-Oncology Clinical Trials.

A systematic, reusable framework for EDC data medical monitoring,
distilled from ICH E6/E9, FDA Guidance, and EMA Scientific Advice.

Modules:
    A - EDC Data Parser (parsers.edc_parser)
    B - Cross-Domain Audit Engine (audit.cross_domain)
    C - Subject-Level Database Builder (analysis.subject_db)
    D - Statistical Analysis Toolbox (analysis.statistics)
    E - Drug Causality Audit (audit.causality)
    F1 - AE Quality Audit (audit.ae_quality)
    F2 - CTCAE Grading Engine (audit.ctcae_grading)
    Report Generator (reporting.report_generator)
    Visualization (visualization.*) -- FDA/Academic/Pharma themed charts
    AI Semantic Layer (ai.*) -- LLM-powered normalization, matching, reasoning
"""

__version__ = "1.2.0"
