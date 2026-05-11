# Medical Monitoring Framework

A comprehensive, AI-enhanced medical monitoring system for clinical trials. Supports both **non-oncology** (e.g., stroke, cardiovascular) and **oncology** (dose-escalation, RECIST, combination therapy) indications.

## Architecture

```
medical_monitoring/
├── ai/              # LLM layer (Qwen/GPT, A/B test, caching)
├── analysis/        # Statistical toolbox, subject database
├── app/             # FastAPI backend + Streamlit dashboard
├── audit/           # Cross-domain, causality, AE quality, CTCAE
├── config/          # YAML rules and AI configuration
├── oncology/        # Oncology-specific modules (RECIST, DLT, etc.)
├── parsers/         # EDC data parsing (generic + CDISC)
├── reporting/       # Export: Excel, PDF, Word, JSON, Web Dashboard
├── utils/           # Shared helpers
└── visualization/   # Charts (FDA, academic, pharma themes)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline (non-oncology)
python run_pipeline.py <edc_file.xlsx> \
    --ctcae "CTCAE v5.0 Clean.xlsx" \
    --indication stroke \
    --ai \
    --output report.xlsx \
    --pdf report.pdf \
    --docx report.docx \
    --json results.json

# Run pipeline (oncology)
python run_pipeline.py <edc_file.xlsx> \
    --indication oncology \
    --drug-name "ZN-F-6418" \
    --drug-class "KRAS G12D inhibitor" \
    --ai \
    --output report.xlsx

# Launch web dashboard
python run_pipeline.py <edc_file.xlsx> --json results.json --dashboard

# Launch API server
python run_server.py
```

## Features

### Core Modules

| Module | Description |
|--------|-------------|
| EDC Parser | Auto-detect CDISC/generic formats, multi-domain parsing |
| Cross-Domain Audit | CM-AE temporal, MH-CM reasonability, LB-AE under-reporting |
| Causality Audit | Bidirectional (forward/reverse), IB-adaptive (FULL/PARTIAL/BLIND) |
| CTCAE Grading | Auto-grade labs, Hy's Law screening, NCS dispute detection |
| AE Quality | 10-dimension audit with naming standardization |
| Statistical Analysis | OR, Chi-square, trajectory, consistency |

### Oncology Extensions

| Module | Description |
|--------|-------------|
| RECIST 1.1 Engine | Target/non-target lesion tracking, response assignment |
| Dose Escalation | BOIN/3+3 monitoring, DLT evaluation, SMC decision support |
| Hepatotoxicity | Hy's Law with liver mets adjustment, DILI pattern |
| Efficacy Signal | Waterfall, spider, response rate CI, dose-response |
| Combination Safety | Multi-drug interaction monitoring |

### AI/LLM Integration

- Dual-provider: Qwen (qwen-max) + GPT (gpt-5.5) with A/B testing
- Medical terminology normalization (MedDRA PT alignment)
- Semantic cross-domain matching (CM-AE, LB-AE, CTCAE-AE)
- Clinical reasoning (causality, SAE screening, severity)
- Data cleaning (typo correction, negation detection)

### Output Formats

- Excel (.xlsx) -- multi-sheet structured report
- PDF (.pdf) -- formal regulatory-grade document
- Word (.docx) -- CRO/Sponsor-ready template
- JSON (.json) -- API-friendly structured export
- Web Dashboard -- Streamlit interactive visualization
- Visualizations -- matplotlib (static) + Plotly (interactive)

## Configuration

### Drug Context (IB/Protocol flexibility)

The framework adapts to available information:

| Level | Provided | Capabilities |
|-------|----------|-------------|
| FULL | Drug name + IB risks + mechanism | All audits including forward causality |
| PARTIAL | Drug name or class only | Reverse audit + consistency + AI inference |
| BLIND | Nothing (confidential) | Data-driven only, AI-assisted |

### Visualization Themes

- `fda_regulatory` -- Conservative, black/white/blue
- `academic_conference` -- ASCO/ESMO color schemes
- `pharma_dashboard` -- Modern dashboard aesthetics

## Tech Stack

- Python 3.11+
- pandas, numpy, scipy (data analysis)
- openai, dashscope (LLM integration)
- reportlab, python-docx (document generation)
- matplotlib, seaborn, plotly (visualization)
- FastAPI, Streamlit (web layer)

## License

Proprietary -- Internal Use Only
