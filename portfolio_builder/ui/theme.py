"""Gradio theme, CSS and shared colors for the robo-advisor dashboard."""

from __future__ import annotations

import gradio as gr

RULE_BASED_COLOR = "#4F46E5"     # indigo
MEAN_VARIANCE_COLOR = "#0D9488"  # teal
BENCHMARK_COLOR = "#64748B"      # slate
TARGET_COLOR = "#F59E0B"         # amber
MUTED_TEXT = "#64748B"
GRID_COLOR = "rgba(148, 163, 184, 0.25)"

METHOD_COLORS = {"rule_based": RULE_BASED_COLOR, "mean_variance": MEAN_VARIANCE_COLOR, "benchmark": BENCHMARK_COLOR}
METHOD_LABELS = {"rule_based": "Rule-based", "mean_variance": "Mean-variance", "benchmark": "S&P 500 (SPY)"}

ASSET_CLASS_COLORS = {
    "US large-cap stocks": "#2563EB",
    "International developed stocks": "#7C3AED",
    "Emerging market stocks": "#DB2777",
    "Real estate (REITs)": "#F59E0B",
    "US Aggregate bonds": "#10B981",
    "Treasury inflation-protected securities": "#06B6D4",
    "Cash and Money Markets": "#94A3B8",
}
ASSET_CLASS_SHORT = {
    "US large-cap stocks": "US large-cap",
    "International developed stocks": "Intl developed",
    "Emerging market stocks": "Emerging markets",
    "Real estate (REITs)": "REITs",
    "US Aggregate bonds": "US bonds",
    "Treasury inflation-protected securities": "TIPS",
    "Cash and Money Markets": "Cash",
}

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="teal",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_shadow="0 1px 2px rgba(15, 23, 42, 0.06)",
    block_title_text_weight="600",
    button_primary_shadow="none",
)

CSS = """
#app-header h1 { font-size: 1.65rem; font-weight: 700; margin: 0; letter-spacing: -0.01em; }
#app-header p { margin: .25rem 0 0; color: var(--body-text-color-subdued); }
.section-title h3 { margin: .35rem 0 .1rem !important; font-size: 1.02rem; font-weight: 650; }
.section-caption p { margin: 0 0 .2rem; color: var(--body-text-color-subdued); font-size: .85rem; }
.sb-section h3 { margin: .9rem 0 .1rem !important; font-size: .78rem; letter-spacing: .06em;
                 text-transform: uppercase; color: var(--body-text-color-subdued); }
.sb-title h2 { margin: 0 0 .2rem !important; font-size: 1.15rem; }

.chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .1rem 0 .2rem; }
.chip { display: inline-flex; align-items: center; gap: .3rem; padding: .22rem .65rem; border-radius: 999px;
        font-size: .82rem; background: var(--block-background-fill); border: 1px solid var(--border-color-primary); }
.chip b { font-weight: 650; }

.portfolio-card { background: var(--block-background-fill); border: 1px solid var(--border-color-primary) !important;
                  border-radius: 14px !important; padding: .85rem 1rem .6rem !important; gap: .35rem !important;
                  border-top: 4px solid var(--card-accent) !important; }
.portfolio-card-rule_based { --card-accent: #4F46E5; }
.portfolio-card-mean_variance { --card-accent: #0D9488; }
.portfolio-card { justify-content: flex-start !important; }
.portfolio-card > .block { flex-grow: 0 !important; }
.card-subtitle p { margin: .35rem 0 0 !important; font-size: .82rem; color: var(--body-text-color-subdued); }
.holdings-table table, .holdings-table th, .holdings-table td, .holdings-table span, .holdings-table input {
  font-family: var(--font) !important; font-size: .8rem !important; white-space: nowrap !important; }
.holdings-table th, .holdings-table td { padding: .28rem .3rem !important; }
.holdings-table .cell-wrap { padding: 0 .1rem !important; }
.holdings-table .cell-menu-button { display: none !important; }
.holdings-table td:nth-child(3), .holdings-table td:nth-child(4),
.holdings-table th:nth-child(3), .holdings-table th:nth-child(4) { text-align: right !important; }
.holdings-table td:nth-child(1) { font-weight: 650; }
.card-head { display: flex; align-items: baseline; justify-content: space-between; gap: .5rem; margin-bottom: .6rem; }
.card-head .name { font-weight: 700; font-size: 1rem; }
.card-head .desc { color: var(--body-text-color-subdued); font-size: .8rem; }
.tiles { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .5rem; }
@media (max-width: 1100px) { .tiles { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 640px) { .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.tile { border-radius: 10px; padding: .5rem .5rem; min-width: 0; background: var(--background-fill-secondary); }
.tile .label { font-size: .68rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--body-text-color-subdued); letter-spacing: .01em; font-weight: 550; }
.tile .value { font-size: 1.3rem; white-space: nowrap; font-weight: 700; margin-top: .1rem; line-height: 1.2; }
.tile .sub { font-size: .72rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--body-text-color-subdued); margin-top: .1rem; }
.tile.good .value { color: #059669; } .tile.fair .value { color: #D97706; } .tile.poor .value { color: #DC2626; }
.tile.neg .value { color: #DC2626; }

.status { border-radius: 10px; padding: .6rem .75rem; font-size: .84rem; margin-top: .6rem; line-height: 1.35; }
.status ul { margin: .25rem 0 0 1rem; padding: 0; }
.status.ok { background: rgba(16, 185, 129, .10); border: 1px solid rgba(16, 185, 129, .35); }
.status.warn { background: rgba(245, 158, 11, .10); border: 1px solid rgba(245, 158, 11, .40); }
.status.error { background: rgba(220, 38, 38, .08); border: 1px solid rgba(220, 38, 38, .40); }
.status .title { font-weight: 650; }

.chart-card { background: var(--block-background-fill); border: 1px solid var(--border-color-primary) !important;
              border-radius: 14px !important; padding: .4rem .7rem .2rem !important; }
footer { display: none !important; }
"""
