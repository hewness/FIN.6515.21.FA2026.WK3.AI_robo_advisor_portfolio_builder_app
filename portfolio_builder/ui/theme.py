"""Gradio theme, CSS and shared colors for the robo-advisor dashboard."""

from __future__ import annotations

import gradio as gr

from ..universe import get_tickers

RULE_BASED_COLOR = "#4F46E5"     # indigo
MEAN_VARIANCE_COLOR = "#0D9488"  # teal
RESEARCH_INFORMED_COLOR = "#9333EA"  # violet
BENCHMARK_COLOR = "#64748B"      # slate
TARGET_COLOR = "#F59E0B"         # amber
# Sky blue: the most distinct option from every portfolio, asset-class and status color that stays visible in
# light and dark themes (min CIEDE2000 distance ~22 to any other dashboard color).
FRONTIER_COLOR = "#0EA5E9"
MUTED_TEXT = "#64748B"
GRID_COLOR = "rgba(148, 163, 184, 0.25)"

# Portfolio models, in card order (left to right).
METHODS = ("rule_based", "mean_variance", "research_informed")
METHOD_COLORS = {"rule_based": RULE_BASED_COLOR, "mean_variance": MEAN_VARIANCE_COLOR,
                 "research_informed": RESEARCH_INFORMED_COLOR, "benchmark": BENCHMARK_COLOR}
METHOD_LABELS = {"rule_based": "Rule-based", "mean_variance": "Mean-variance",
                 "research_informed": "Research-informed", "benchmark": "S&P 500 (SPY)"}
METHOD_TITLES = {
    "rule_based": "Rule-based Lifecycle Portfolio",
    "mean_variance": "Mean-variance Optimized Portfolio",
    "research_informed": "Research-informed Portfolio",
    "benchmark": "S&P 500 (SPY)",
}
# Popular rules of thumb shown next to the portfolios in the research card (neutral, non-portfolio grays).
POPULAR_RULE_COLOR = "#94A3B8"

# Portfolio colors (indigo / teal / violet / slate) are reserved for portfolios and the benchmark, so the
# asset-class palette avoids those hues: warm tones for stocks, greens for bonds, warm gray for cash.
ASSET_CLASS_COLORS = {
    "US large-cap stocks": "#EA580C",
    "International developed stocks": "#CA8A04",
    "Emerging market stocks": "#DB2777",
    "Real estate (REITs)": "#92400E",
    "US Aggregate bonds": "#16A34A",
    "Treasury inflation-protected securities": "#65A30D",
    "Cash and Money Markets": "#A8A29E",
}
ASSET_CLASS_SHORT = {
    "US large-cap stocks": "US large-cap",
    "International developed stocks": "Intl developed",
    "Emerging market stocks": "Emerging mkts",
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
.portfolio-card-research_informed { --card-accent: #9333EA; }
.portfolio-card { justify-content: flex-start !important; }
.portfolio-card > .block { flex-grow: 0 !important; }
.card-subtitle p { margin: .35rem 0 0 !important; font-size: .82rem; color: var(--body-text-color-subdued); }
.holdings-table table, .holdings-table th, .holdings-table td, .holdings-table span, .holdings-table input {
  font-family: var(--font) !important; font-size: .75rem !important; white-space: nowrap !important; }
.holdings-table th, .holdings-table td { padding: .28rem .18rem !important; }
.holdings-table .cell-wrap { padding: 0 .1rem !important; }
.holdings-table .cell-menu-button { display: none !important; }
/* Gradio 6 body cells are div[role=gridcell][data-col]; header cells are th[data-heading]. */
.holdings-table [data-col="3"] .cell-wrap, .holdings-table [data-col="4"] .cell-wrap,
.holdings-table [data-col="5"] .cell-wrap, .holdings-table th[data-heading="3"] .cell-wrap,
.holdings-table th[data-heading="4"] .cell-wrap, .holdings-table th[data-heading="5"] .cell-wrap {
  justify-content: flex-end !important; text-align: right !important; }
.holdings-table [data-col="1"] span { font-weight: 650; }
.holdings-table [data-col="0"] .cell-wrap { justify-content: center !important; padding: 0 !important; }
.holdings-table [data-col="0"] .cell-wrap > span { display: flex !important; align-items: center; justify-content: center;
                                                  width: 100%; }
.asset-dot { display: inline-block; width: .72rem; height: .72rem; border-radius: 3px; vertical-align: middle;
             box-shadow: inset 0 0 0 1px rgba(0, 0, 0, .08); }
.holdings-table .virtual-row [role=gridcell] { transition: opacity .15s ease, background-color .15s ease; }
.holdings-table .virtual-row.row-active [role=gridcell] {
  background: color-mix(in srgb, var(--row-accent) 20%, transparent) !important; }
.holdings-table .virtual-row.row-active [role=gridcell] span { font-weight: 700; }
.holdings-table .virtual-row.row-dim [role=gridcell] { opacity: .32; }
.card-subtitle p { font-weight: 600; }
.donut-plot .modebar-container, .wealth-plot .modebar-container { display: none !important; }

.comparison-card { background: linear-gradient(90deg, #4F46E5 0 33.34%, #0D9488 33.34% 66.67%, #9333EA 66.67% 100%)
                               top / 100% 4px no-repeat,
                               var(--block-background-fill) !important;
                   border: 1px solid var(--border-color-primary) !important; overflow: hidden;
                   border-radius: 14px !important; padding: 1rem 1rem .6rem !important; gap: .5rem !important; }
.comparison-card > .block { flex-grow: 0 !important; }
.comparison-head .title { font-size: 1.12rem; font-weight: 700; line-height: 1.35; }
.comparison-head .title .vs { color: var(--body-text-color-subdued); font-weight: 500; margin: 0 .4rem; }
.comparison-head .sub { color: var(--body-text-color-subdued); font-size: .85rem; margin-top: .15rem; }
.swatch { display: inline-block; width: 1.1rem; height: .28rem; border-radius: 2px; vertical-align: middle;
          margin: 0 .3rem 0 .15rem; }
.swatch.dotted { background: repeating-linear-gradient(90deg, currentColor 0 .22rem, transparent .22rem .4rem) !important; }
.comparison-card .chart-card { background: var(--background-fill-secondary); }
.comparison-card .section-caption { min-height: 2.9rem; }
.card-head { display: flex; flex-direction: column; gap: .1rem; margin-bottom: .6rem; }
.card-head .name { font-weight: 700; font-size: 1.12rem; line-height: 1.3; }
.card-head .desc { color: var(--body-text-color-subdued); font-size: .8rem; line-height: 1.35;
                   height: calc(2 * 1.35em); overflow: hidden; display: -webkit-box; -webkit-box-orient: vertical;
                   -webkit-line-clamp: 2; }  /* exactly two lines in both cards keeps sections aligned */
.card-head .basis { font-size: .8rem; font-weight: 600; line-height: 1.35; margin-top: .15rem; white-space: nowrap;
                    overflow: hidden; text-overflow: ellipsis; color: var(--card-accent); }
.tiles { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .5rem; }
/* Cards sit three across, so tiles go three over two, each row filling the card width. */
.portfolio-card .tiles { grid-template-columns: repeat(6, minmax(0, 1fr)); }
.portfolio-card .tiles .tile:nth-child(-n+3) { grid-column: span 2; }
.portfolio-card .tiles .tile:nth-child(n+4) { grid-column: span 3; }
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

/* Research vs. popular wisdom */
.insight-card { background: linear-gradient(90deg, #9333EA, rgba(147, 51, 234, .25)) top / 100% 4px no-repeat,
                            var(--block-background-fill) !important;
                border: 1px solid var(--border-color-primary) !important; border-radius: 14px !important;
                padding: 1rem 1.1rem .8rem !important; overflow: hidden; }
.insight-card > .block, .insight-card .html-container { border: none !important; background: transparent !important;
                                                        padding: 0 !important; box-shadow: none !important; }
.insight { display: grid; grid-template-columns: minmax(260px, 1fr) minmax(0, 1.9fr); gap: 1.4rem; align-items: start; }
@media (max-width: 1000px) { .insight { grid-template-columns: 1fr; } }
.insight .head { grid-column: 1 / -1; }
.insight .head .title { font-size: 1.12rem; font-weight: 700; color: #9333EA; }
.insight .head .sub { color: var(--body-text-color-subdued); font-size: .88rem; margin-top: .15rem; line-height: 1.4; }
.insight .bars-title { font-size: .78rem; font-weight: 650; letter-spacing: .05em; text-transform: uppercase;
                       color: var(--body-text-color-subdued); margin-bottom: .45rem; }
.eq-bar { display: grid; grid-template-columns: 9.5rem minmax(0, 1fr) 3rem; align-items: center; gap: .5rem;
          font-size: .82rem; margin: .32rem 0; }
.eq-bar .lbl { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.eq-bar .track { height: .7rem; border-radius: 999px; background: var(--background-fill-secondary); overflow: hidden; }
.eq-bar .fill { height: 100%; border-radius: 999px; }
.eq-bar .val { text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; }
.eq-bar.main .lbl, .eq-bar.main .val { font-weight: 750; }
.eq-bar.main .track { height: .95rem; }
.eq-bar.popular .lbl { color: var(--body-text-color-subdued); }
.insight .legend-note { font-size: .74rem; color: var(--body-text-color-subdued); margin-top: .5rem; line-height: 1.4; }
.insight ul.points { margin: 0 !important; padding: 0 !important; list-style: none !important; display: grid; gap: .55rem; }
/* Only the violet dot: Gradio's prose styles would otherwise add a default list marker too. */
.insight ul.points li { position: relative; padding-left: 1.15rem !important; margin: 0 !important; font-size: .86rem;
                        line-height: 1.45; list-style: none !important; }
.insight ul.points li::marker { content: none; }
.insight ul.points li::before { content: ""; position: absolute; left: 0; top: .5em; width: .5rem; height: .5rem;
                                border-radius: 50%; background: #9333EA; }
.insight .sources { font-size: .72rem; color: var(--body-text-color-subdued); margin-top: .7rem; line-height: 1.4; }

.chart-card { background: var(--block-background-fill); border: 1px solid var(--border-color-primary) !important;
              border-radius: 14px !important; padding: .4rem .7rem .2rem !important; }
footer { display: none !important; }

/* Sidebar: help text appears as a tooltip above an input's label on hover (so it never covers the input). */
[id^="in-"], .form:has(> [id^="in-"]), .form:has([id^="in-"]) { overflow: visible !important; }
[id^="in-"]:hover, .form:has([id^="in-"]:hover) { z-index: 60 !important; position: relative; }
.has-tip { cursor: help; position: relative; }
.has-tip::after { content: "i"; display: inline-flex; align-items: center; justify-content: center;
                  width: .95rem; height: .95rem; margin-left: .35rem; border-radius: 50%; font-size: .62rem;
                  font-weight: 700; font-style: italic; font-family: Georgia, serif; vertical-align: .08em;
                  color: var(--body-text-color-subdued); border: 1px solid currentColor; opacity: .65; }
.has-tip::before { content: attr(data-tip); position: absolute; left: 0; bottom: calc(100% + 8px); z-index: 1000;
                   width: max-content; max-width: 290px; white-space: normal; text-align: left;
                   padding: .55rem .7rem; border-radius: 10px; background: #1E293B; color: #F1F5F9;
                   border: 1px solid rgba(148, 163, 184, .25); box-shadow: 0 10px 28px rgba(15, 23, 42, .28);
                   font-size: .78rem; font-weight: 450; line-height: 1.45; letter-spacing: normal; text-transform: none;
                   opacity: 0; transform: translateY(3px); pointer-events: none; transition: opacity .14s, transform .14s; }
.has-tip:hover::before { opacity: 1; transform: translateY(0); }
.has-tip:hover::after { opacity: 1; }
.money-input input { font-variant-numeric: tabular-nums; }

/* Compact sidebar fields: label and input on one row, with every label in the same-width column so the inputs line
   up. Sliders keep their track on a second, full-width row (the sidebar is too narrow for label + track + number). */
:root { --inline-label-width: 10.8rem; }
.inline-field, .inline-slider { padding-top: .45rem !important; padding-bottom: .45rem !important; }
.inline-field [data-testid="block-info"], .inline-slider [data-testid="block-info"] {
  font-size: 12px !important; padding-left: 5px !important; padding-right: 5px !important; white-space: nowrap;
  box-sizing: border-box; }
.inline-field > .container { display: flex !important; flex-direction: row !important; align-items: center !important;
                             gap: .6rem !important; }
.inline-field > .container > [data-testid="block-info"] { flex: 0 0 var(--inline-label-width);
                                                          width: var(--inline-label-width); margin: 0 !important; }
.inline-field > .container > input, .inline-field > .container > .input-container,
.inline-field > .container > .wrap { flex: 1 1 auto; min-width: 0; margin: 0 !important; }
.inline-slider .head { display: grid !important; grid-template-columns: var(--inline-label-width) minmax(0, 1fr);
                       align-items: center; column-gap: .6rem; }
.inline-slider .head > label { min-width: 0; margin: 0 !important; }
.inline-slider .head > label > [data-testid="block-info"] { display: block; width: var(--inline-label-width); }
.inline-slider .head > .tab-like-container { display: flex; width: 100%; min-width: 0; }
.inline-slider .head > .tab-like-container > input[type=number] { flex: 1 1 auto; width: auto; min-width: 0; }
"""

# Every holdings table reserves room for the largest possible portfolio (each universe fund + cash; rendered
# rows are 36px plus 87px of header and toolbar) so the wealth charts below line up across the three cards.
HOLDINGS_MIN_HEIGHT = 87 + 36 * (len(get_tickers()) + 1)
CSS += f"\n.holdings-table {{ min-height: {HOLDINGS_MIN_HEIGHT}px; }}\n"
