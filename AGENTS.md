# AGENTS.md -- Medical Monitoring Framework

## Project Context

This is a clinical trial medical monitoring framework that automates safety data review, cross-domain auditing, and regulatory report generation.

## Code Conventions

- Python 3.11+ with type hints
- Conventional Commits: `feat(module):`, `fix(audit):`, `docs:`, `refactor:`, `test:`
- Docstrings: Google style for public APIs
- Config: YAML for rules/prompts, dataclasses for internal state
- Error handling: graceful degradation (never crash on missing data)

## Architecture Rules

- `pipeline.py` is the single orchestration entry point
- All audit modules accept `EDCDataset` and return `list[Finding]` or `dict`
- AI modules are optional -- framework must work without LLM access
- Oncology modules live under `oncology/` and follow same interfaces
- Visualization modules inherit from `VizBase` and support 3 themes
- Export modules produce standalone files (no external dependencies at runtime)

## Data Sensitivity

- NEVER commit .xlsx data files (patient data)
- Subject IDs must be de-identified before LLM calls
- API keys go in config YAML or env vars, never hardcoded
- Generated reports may contain PHI -- handle per protocol

## Key Interfaces

```python
# Pipeline
pipeline = MedicalMonitoringPipeline(edc_file, ctcae_file, ...)
results = pipeline.run()
pipeline.export("report.xlsx")
pipeline.export_pdf("report.pdf")
pipeline.export_json("results.json")

# Findings
Finding(category, subject_id, description, severity, ...)

# AI Engine
engine = LLMEngine(AIConfig.default())
resp = engine.complete(system, user, task_type, provider)
```

## Module Dependencies

```
parsers → audit → analysis → reporting
           ↑                      ↑
           ai (optional)    visualization
           ↑
       oncology (extends audit)
```
