"""
AI semantic enhancement layer for Medical Monitoring.

Provides LLM-powered capabilities for:
    - Medical terminology normalization (MedDRA PT alignment)
    - Cross-domain semantic matching (CM-AE, LB-AE, CTCAE-AE)
    - Clinical reasoning support (causality, SAE, severity)
    - EDC field annotation (column semantic role detection)
    - Data cleaning (typo correction, format unification, negation detection)

Supports Qwen (DashScope) and GPT (OpenAI) backends with A/B testing.
"""
