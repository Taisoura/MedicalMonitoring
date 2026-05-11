"""
Visualization module for Medical Monitoring Framework.

Provides publication-ready static figures (matplotlib/seaborn) and
interactive HTML charts (plotly) for all monitoring modules.

Submodules:
    themes          -- 3 preset themes (FDA/Academic/Pharma) + custom registration
    base            -- VizBase shared base class
    viz_overview    -- Module A: CONSORT / CRF heatmap / enrollment / quality radar
    viz_audit       -- Module B: stacked bar / treemap / cross-heatmap / Gantt
    viz_subject     -- Module C: pyramid / patient profile / UpSet / spaghetti / swimmer
    viz_statistics  -- Module D: forest / KM / shift / volcano / ROC / Bland-Altman
    viz_causality   -- Module E: sankey / bidirectional waterfall / radar / heatmap
    viz_ae_quality  -- Module F1: 10-dim radar / sunburst / action Gantt / matrix / bar
    viz_ctcae       -- Module F2: eDISH / waterfall / lollipop / xULN trend / waffle / dot
    viz_dashboard   -- Executive one-page dashboard
"""
