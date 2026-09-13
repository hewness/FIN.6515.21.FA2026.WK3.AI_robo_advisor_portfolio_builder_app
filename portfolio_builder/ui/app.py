"""Gradio dashboard: sidebar inputs, summary cards and charts for the three recommended portfolios."""

from __future__ import annotations

import logging
from typing import Any

import gradio as gr

from ..service import (
    RISK_LABELS,
    InputValidationError,
    PortfolioService,
    PortfolioServiceError,
    get_form_options,
    get_portfolio_service,
)
from ..optimization.glide_path import HumpGlidePath
from ..universe import get_tickers
from . import charts, components
from .formatting import format_money, parse_money
from .interactions import build_head
from .theme import CSS, METHODS, THEME

logger = logging.getLogger(__name__)

INPUT_FIELDS = ("risk_tolerance", "horizon_years", "initial_investment", "monthly_contribution", "goal",
                "target_amount", "age", "backtest_years", "rebalance", "hump_glide_path",
                "annual_income", "retirement_income", "retirement_age")
CARD_PARTS = ("header", "donut", "table", "wealth")
OUTPUT_KEYS = (
    "status", "chips",
    *(f"{part}_{method}" for method in METHODS for part in CARD_PARTS),
    "insight", "scatter", "backtest_caption", "backtest", "notes",
)
MONEY_FIELDS = ("initial_investment", "monthly_contribution", "target_amount", "annual_income", "retirement_income")
# Tall enough for the largest possible portfolio (every universe fund + cash) at 36px rows + header,
# so tables never scroll.
MAX_HOLDINGS = len(get_tickers()) + 1
HOLDINGS_TABLE_HEIGHT = 45 + 36 * MAX_HOLDINGS + 40
PRESETS = {level.label: score for level, score in RISK_LABELS.items()}
CUSTOM = "Custom"


def sidebar_tooltips() -> dict[str, str]:
    """Hover help for each sidebar input, keyed by the component's elem_id."""
    opts = get_form_options()
    presets = ", ".join(f"{label} {score:g}" for label, score in PRESETS.items())
    goal_defaults = ", ".join(f"{c['label']} {format_money(c['default_target'])}" for c in opts["goal"]["choices"])
    return {
        "in-age": ("Your current age (18–80). Sets the lifecycle glide path, and how many years of future income "
                   "the research-informed model counts."),
        "in-income": ("Your yearly earnings before retirement ($0 – $5,000,000); enter $0 if retired. The "
                      "research-informed model treats future earnings as a bond-like asset (human capital): the "
                      "more of it you have relative to savings, the more stock your portfolio can hold."),
        "in-retire-income": (f"Expected Social Security and pensions per year in retirement ($0 – $1,000,000). "
                             f"Leave blank to use {opts['retirement_income']['default_rate']:.0%} of annual income. "
                             "Counts toward human capital, so retirees with pension income can hold more stock."),
        "in-retire-age": "Age when retirement income replaces your earnings (50–75).",
        "in-goal": "What you are investing for. It adds context to the recommendation and sets a default goal target.",
        "in-target": ("What you want the portfolio to be worth at the end of your horizon ($1,000 – $100,000,000). "
                      f"Used for the goal probability. Defaults: {goal_defaults}."),
        "in-glide-path": (f"On (default): equity follows a hump-shaped glide path, {HumpGlidePath().describe()}. "
                          "Risk tolerance shifts it by up to ±20 points, and the mean-variance portfolio keeps equity "
                          "funds within ±5 points of it. Off: the classic 110 − age rule."),
        "in-risk-preset": "Drives the equity vs. bond allocation. Pick a preset, or fine-tune the risk score below.",
        "in-risk": f"1 = most conservative, 10 = most aggressive. Presets: {presets}.",
        "in-initial": "Starting portfolio value ($1,000 – $10,000,000).",
        "in-monthly": "Ongoing savings added each month ($0 – $50,000).",
        "in-horizon": ("How long you plan to invest (1–30 years). Longer horizons allow more risk: under 3 years "
                       "caps risk at 3, 3–4 years −2, 5–9 years −1, 20+ years +1."),
        "in-lookback": "How many years of history (10–20) to test all three allocations against the S&P 500.",
        "in-rebalance": "How often the backtest resets holdings to their target weights.",
    }


HEAD = build_head(sidebar_tooltips())


def preset_for(score: float | None) -> str:
    return next((label for label, value in PRESETS.items() if score is not None and float(score) == value), CUSTOM)


def default_values() -> dict[str, Any]:
    opts = get_form_options()
    return {name: opts[name]["default"] for name in INPUT_FIELDS}


class Dashboard:
    """Holds the service and turns widget values into dashboard outputs."""

    def __init__(self, service: PortfolioService | None = None) -> None:
        self._service = service

    @property
    def service(self) -> PortfolioService:
        if self._service is None:
            self._service = get_portfolio_service()
        return self._service

    def update(self, *values: Any) -> tuple:
        """Returns one value per ``OUTPUT_KEYS`` entry, in that order."""
        request = dict(zip(INPUT_FIELDS, values))
        for field in MONEY_FIELDS:
            request[field] = parse_money(request[field])
        skip = tuple(gr.skip() for _ in OUTPUT_KEYS[1:])
        try:
            resp = self.service.build_portfolio(request)
        except InputValidationError as exc:
            gr.Warning(next(iter(exc.field_errors.values())))
            return (components.status_panel(errors=exc.field_errors), *skip)
        except PortfolioServiceError as exc:
            logger.exception("Portfolio build failed")
            gr.Warning(str(exc))
            return (components.status_panel(message=str(exc)), *skip)

        values_by_key = {
            "status": components.status_panel(warnings=resp.warnings),
            "chips": components.profile_chips(resp),
            "scatter": charts.risk_return_scatter(resp),
            "backtest_caption": components.backtest_caption(resp),
            "backtest": charts.backtest_chart(resp),
            "notes": components.notes_markdown(resp),
        }
        y_max = charts.wealth_y_max(resp)
        for method in METHODS:
            values_by_key[f"header_{method}"] = components.summary_header(resp, method)
            values_by_key[f"donut_{method}"] = charts.allocation_donut(resp, method)
            values_by_key[f"table_{method}"] = components.holdings_table(resp, method)
            values_by_key[f"wealth_{method}"] = charts.wealth_projection(resp, method, y_max)
        values_by_key["insight"] = components.research_insight_card(resp)
        return tuple(values_by_key[key] for key in OUTPUT_KEYS)


def build_demo(service: PortfolioService | None = None) -> gr.Blocks:
    dashboard = Dashboard(service)
    opts = get_form_options()
    defaults = default_values()

    with gr.Blocks(title="AI Robo-Advisor Portfolio Builder", fill_width=True) as demo:
        # ---------------- Sidebar: grouped inputs ----------------
        # Help text lives in hover tooltips on each label (see SIDEBAR_TOOLTIPS / interactions.py).
        with gr.Sidebar(width=360, open=True, label="Inputs"):
            gr.Markdown("## 🧭 Build your plan", elem_classes="sb-title")

            gr.Markdown("### About you", elem_classes="sb-section")
            age = gr.Number(label="Age", value=defaults["age"], precision=0, elem_id="in-age",
                            elem_classes="inline-field")
            goal = gr.Dropdown(label="Financial goal", value=defaults["goal"], elem_id="in-goal",
                               elem_classes="inline-field",
                               choices=[(c["label"], c["value"]) for c in opts["goal"]["choices"]])
            target = gr.Textbox(label="Goal target", value=format_money(defaults["target_amount"]), max_lines=1,
                                placeholder="$1,500,000", elem_id="in-target",
                                elem_classes=["money-input", "inline-field"])

            gr.Markdown("### Income", elem_classes="sb-section")
            income = gr.Textbox(label="Annual income", value=format_money(defaults["annual_income"]), max_lines=1,
                                placeholder="$85,000", elem_id="in-income", elem_classes=["money-input", "inline-field"])
            retire_income = gr.Textbox(label="Retirement income", value=format_money(defaults["retirement_income"]),
                                       max_lines=1, placeholder="Auto (40%)", elem_id="in-retire-income",
                                       elem_classes=["money-input", "inline-field"])
            retire_age = gr.Number(label="Retirement age", value=defaults["retirement_age"], precision=0,
                                   minimum=50, maximum=75, elem_id="in-retire-age", elem_classes="inline-field")

            gr.Markdown("### Risk profile", elem_classes="sb-section")
            glide = gr.Checkbox(label="Hump-shaped equity glide path", value=defaults["hump_glide_path"],
                                elem_id="in-glide-path", elem_classes="glide-toggle")
            risk_preset = gr.Dropdown(label="Risk tolerance", choices=[*PRESETS, CUSTOM],
                                      value=preset_for(defaults["risk_tolerance"]), elem_id="in-risk-preset",
                                      elem_classes="inline-field")
            risk = gr.Slider(label="Risk score", minimum=opts["risk_tolerance"]["minimum"],
                             maximum=opts["risk_tolerance"]["maximum"], step=opts["risk_tolerance"]["step"],
                             value=defaults["risk_tolerance"], elem_id="in-risk")

            gr.Markdown("### Investment plan", elem_classes="sb-section")
            initial = gr.Textbox(label="Initial investment", value=format_money(defaults["initial_investment"]),
                                 max_lines=1, placeholder="$50,000", elem_id="in-initial", elem_classes="money-input")
            monthly = gr.Textbox(label="Monthly contribution", value=format_money(defaults["monthly_contribution"]),
                                 max_lines=1, placeholder="$1,000", elem_id="in-monthly", elem_classes="money-input")
            horizon = gr.Slider(label="Investment horizon (years)", minimum=1, maximum=30, step=1,
                                value=defaults["horizon_years"], elem_id="in-horizon")

            with gr.Accordion("⚙️ Backtest settings", open=False):
                backtest_years = gr.Slider(label="Lookback (years)", minimum=10, maximum=20, step=1,
                                           value=defaults["backtest_years"], elem_id="in-lookback")
                rebalance = gr.Radio(label="Rebalancing", value=defaults["rebalance"], elem_id="in-rebalance",
                                     choices=[(c["label"], c["value"]) for c in opts["rebalance"]["choices"]])

            status = gr.HTML(components.status_panel())
            reset = gr.Button("↺ Reset to defaults", variant="secondary", size="sm")

        # ---------------- Main panel: results ----------------
        gr.HTML(
            "<div id='app-header'><h1>AI Robo-Advisor Portfolio Builder</h1>"
            "<p>Three portfolios built from "
            f"{len(get_tickers())} ETFs (one per asset class) plus cash and compared side by side: a rule-based "
            "lifecycle portfolio, a mean-variance optimized portfolio, and a research-informed portfolio whose equity "
            "share depends on your income and savings, not just your age. Adjust the inputs in the sidebar; the "
            "dashboard updates automatically.</p></div>"
        )
        chips = gr.HTML()

        # One card per portfolio, side by side; inside each card the panels stack: KPI tiles, allocation donut,
        # holdings, projected wealth. Cards wrap under each other on narrow screens.
        cards: dict[str, dict[str, Any]] = {}
        with gr.Row(equal_height=True):
            for method in METHODS:
                with gr.Column(elem_classes=["portfolio-card", f"portfolio-card-{method}"], min_width=340):
                    header = gr.HTML()
                    gr.Markdown("Allocation by Asset Class", elem_classes="card-subtitle")
                    donut = gr.Plot(show_label=False, container=False, elem_classes="donut-plot")
                    gr.Markdown("Holdings", elem_classes="card-subtitle")
                    table = gr.Dataframe(show_label=False, interactive=False, max_height=HOLDINGS_TABLE_HEIGHT,
                                         datatype=components.HOLDINGS_DATATYPES,
                                         column_widths=["7%", "17%", "34%", "19%", "23%"],
                                         elem_classes="holdings-table")
                    gr.Markdown("Projected Wealth", elem_classes="card-subtitle")
                    gr.Markdown("Expected (mean), optimistic (75th pct) and pessimistic (25th pct) value from "
                                "5,000 simulations, including monthly contributions. Same scale in all cards.",
                                elem_classes="section-caption")
                    wealth = gr.Plot(show_label=False, container=False, elem_classes="wealth-plot")
                cards[method] = {"header": header, "donut": donut, "table": table, "wealth": wealth}

        # Charts that plot all portfolios together.
        with gr.Column(elem_classes="comparison-card"):
            gr.HTML(components.comparison_header())
            with gr.Row(equal_height=True):
                with gr.Column(elem_classes="chart-card", min_width=460):
                    gr.Markdown("### Risk vs. Return", elem_classes="section-title")
                    gr.Markdown("Annualized expected return vs. volatility for each asset class, cash and all "
                                "three portfolios, with the efficient frontier.", elem_classes="section-caption")
                    scatter = gr.Plot(show_label=False, container=False)
                with gr.Column(elem_classes="chart-card", min_width=460):
                    gr.Markdown("### Historical Backtest vs. S&P 500", elem_classes="section-title")
                    bt_caption = gr.Markdown(elem_classes="section-caption")
                    backtest = gr.Plot(show_label=False, container=False)

        # Where the research-informed model departs from popular rules of thumb, and why.
        with gr.Column(elem_classes="insight-card"):
            insight = gr.HTML()

        with gr.Accordion("Notes & assumptions", open=False):
            notes = gr.Markdown()

        # ---------------- Events ----------------
        inputs = [risk, horizon, initial, monthly, goal, target, age, backtest_years, rebalance, glide,
                  income, retire_income, retire_age]
        components_by_key = {
            "status": status, "chips": chips,
            **{f"{part}_{m}": cards[m][part] for m in METHODS for part in CARD_PARTS},
            "insight": insight, "scatter": scatter, "backtest_caption": bt_caption, "backtest": backtest,
            "notes": notes,
        }
        outputs = [components_by_key[key] for key in OUTPUT_KEYS]
        run = dict(fn=dashboard.update, inputs=inputs, outputs=outputs, show_progress="minimal")

        gr.on(
            triggers=[risk.release, horizon.release, backtest_years.release, rebalance.input, glide.input,
                      initial.blur, initial.submit, monthly.blur, monthly.submit,
                      target.blur, target.submit, age.blur, age.submit, income.blur, income.submit,
                      retire_income.blur, retire_income.submit, retire_age.blur, retire_age.submit],
            trigger_mode="always_last",
            **run,
        )
        # Normalize money text ("50k", "50000") to "$50,000" when the field is left; typing is formatted in the browser.
        for money_box in (initial, monthly, target, income, retire_income):
            gr.on([money_box.blur, money_box.submit], format_money, inputs=money_box, outputs=money_box,
                  show_progress="hidden", queue=False)
        risk.release(preset_for, inputs=risk, outputs=risk_preset, show_progress="hidden")
        risk_preset.input(
            lambda label: PRESETS.get(label, gr.skip()), inputs=risk_preset, outputs=risk, show_progress="hidden"
        ).then(**run)
        goal_targets = {c["value"]: format_money(c["default_target"]) for c in opts["goal"]["choices"]}
        goal.input(lambda g: goal_targets.get(g, gr.skip()), inputs=goal, outputs=target,
                   show_progress="hidden").then(**run)

        def reset_values() -> tuple:
            d = default_values()
            return (preset_for(d["risk_tolerance"]), d["risk_tolerance"], d["horizon_years"],
                    format_money(d["initial_investment"]), format_money(d["monthly_contribution"]), d["goal"],
                    format_money(d["target_amount"]), d["age"], d["backtest_years"], d["rebalance"],
                    d["hump_glide_path"], format_money(d["annual_income"]), format_money(d["retirement_income"]),
                    d["retirement_age"])

        reset.click(reset_values, outputs=[risk_preset, *inputs],
                    show_progress="hidden").then(**run)
        demo.load(**run)

    return demo


def launch(**kwargs: Any) -> None:
    build_demo().launch(theme=THEME, css=CSS, head=HEAD, **kwargs)
