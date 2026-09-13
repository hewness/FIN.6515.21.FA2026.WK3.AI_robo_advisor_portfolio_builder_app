"""Gradio dashboard: sidebar inputs, summary cards and charts for both recommended portfolios."""

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
from . import charts, components
from .theme import CSS, THEME

logger = logging.getLogger(__name__)

INPUT_FIELDS = ("risk_tolerance", "horizon_years", "initial_investment", "monthly_contribution", "goal",
                "target_amount", "age", "backtest_years", "rebalance")
PRESETS = {level.label: score for level, score in RISK_LABELS.items()}
CUSTOM = "Custom"


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
        request = dict(zip(INPUT_FIELDS, values))
        skip = tuple(gr.skip() for _ in range(10))
        try:
            resp = self.service.build_portfolio(request)
        except InputValidationError as exc:
            gr.Warning(next(iter(exc.field_errors.values())))
            return (components.status_panel(errors=exc.field_errors), *skip)
        except PortfolioServiceError as exc:
            logger.exception("Portfolio build failed")
            gr.Warning(str(exc))
            return (components.status_panel(message=str(exc)), *skip)

        return (
            components.status_panel(warnings=resp.warnings),
            components.profile_chips(resp),
            components.summary_cards(resp),
            charts.allocation_donuts(resp),
            charts.risk_return_scatter(resp),
            charts.wealth_projection(resp),
            components.backtest_caption(resp),
            charts.backtest_chart(resp),
            components.holdings_table(resp, "rule_based"),
            components.holdings_table(resp, "mean_variance"),
            components.notes_markdown(resp),
        )


def build_demo(service: PortfolioService | None = None) -> gr.Blocks:
    dashboard = Dashboard(service)
    opts = get_form_options()
    defaults = default_values()

    def info(name: str) -> str:
        return opts[name]["help"]

    with gr.Blocks(title="AI Robo-Advisor Portfolio Builder", fill_width=True) as demo:
        # ---------------- Sidebar: grouped inputs ----------------
        with gr.Sidebar(width=360, open=True, label="Inputs"):
            gr.Markdown("## 🧭 Build your plan", elem_classes="sb-title")

            gr.Markdown("### About you", elem_classes="sb-section")
            age = gr.Number(label="Age", value=defaults["age"], precision=0,
                            info=f"{opts['age']['minimum']}–{opts['age']['maximum']} · {info('age')}")
            goal = gr.Dropdown(label="Financial goal", value=defaults["goal"], info=info("goal"),
                               choices=[(c["label"], c["value"]) for c in opts["goal"]["choices"]])
            target = gr.Number(label="Goal target ($)", value=defaults["target_amount"], precision=0,
                               info="What you want the portfolio to be worth at the end of the horizon")

            gr.Markdown("### Risk profile", elem_classes="sb-section")
            risk_preset = gr.Radio(label="Risk tolerance", choices=[*PRESETS, CUSTOM],
                                   value=preset_for(defaults["risk_tolerance"]), info=info("risk_tolerance"))
            risk = gr.Slider(label="Risk score", info="1 = most conservative · 10 = most aggressive",
                             minimum=opts["risk_tolerance"]["minimum"], maximum=opts["risk_tolerance"]["maximum"],
                             step=opts["risk_tolerance"]["step"], value=defaults["risk_tolerance"])

            gr.Markdown("### Investment plan", elem_classes="sb-section")
            initial = gr.Number(label="Initial investment ($)", value=defaults["initial_investment"], precision=0,
                                info="$1,000 – $10,000,000 · starting portfolio value")
            monthly = gr.Number(label="Monthly contribution ($)", value=defaults["monthly_contribution"], precision=0,
                                info="$0 – $50,000 · ongoing savings")
            horizon = gr.Slider(label="Investment horizon (years)", minimum=1, maximum=30, step=1,
                                value=defaults["horizon_years"], info=info("horizon_years"))

            with gr.Accordion("⚙️ Backtest settings", open=False):
                backtest_years = gr.Slider(label="Lookback (years)", minimum=10, maximum=20, step=1,
                                           value=defaults["backtest_years"], info=info("backtest_years"))
                rebalance = gr.Radio(label="Rebalancing", value=defaults["rebalance"], info=info("rebalance"),
                                     choices=[(c["label"], c["value"]) for c in opts["rebalance"]["choices"]])

            status = gr.HTML(components.status_panel())
            reset = gr.Button("↺ Reset to defaults", variant="secondary", size="sm")

        # ---------------- Main panel: results ----------------
        gr.HTML(
            "<div id='app-header'><h1>AI Robo-Advisor Portfolio Builder</h1>"
            "<p>A rule-based lifecycle portfolio and a mean-variance optimized portfolio, built from 11 ETFs "
            "and compared side by side. Adjust the inputs in the sidebar; the dashboard updates automatically.</p></div>"
        )
        chips = gr.HTML()
        summary = gr.HTML()

        with gr.Row(equal_height=True):
            with gr.Column(scale=5, elem_classes="chart-card"):
                gr.Markdown("### Allocation by asset class", elem_classes="section-title")
                gr.Markdown("Recommended weights for each approach; hover for dollar amounts.",
                            elem_classes="section-caption")
                pie = gr.Plot(show_label=False)
            with gr.Column(scale=6, elem_classes="chart-card"):
                gr.Markdown("### Risk vs. return", elem_classes="section-title")
                gr.Markdown("Asset classes, the efficient frontier and both portfolios (annualized estimates).",
                            elem_classes="section-caption")
                scatter = gr.Plot(show_label=False)

        with gr.Column(elem_classes="chart-card"):
            gr.Markdown("### Projected wealth", elem_classes="section-title")
            gr.Markdown("Initial investment plus monthly contributions over your horizon: expected, optimistic "
                        "(75th percentile) and pessimistic (25th percentile) scenarios from 5,000 simulations.",
                        elem_classes="section-caption")
            wealth = gr.Plot(show_label=False)

        with gr.Column(elem_classes="chart-card"):
            gr.Markdown("### Historical backtest vs. S&P 500", elem_classes="section-title")
            bt_caption = gr.Markdown(elem_classes="section-caption")
            backtest = gr.Plot(show_label=False)

        with gr.Column(elem_classes="chart-card"):
            gr.Markdown("### Holdings", elem_classes="section-title")
            with gr.Row():
                rb_table = gr.Dataframe(label="Rule-based", interactive=False, wrap=True, max_height=460,
                                        column_widths=["11%", "12%", "15%", "20%", "42%"])
                mvo_table = gr.Dataframe(label="Mean-variance", interactive=False, wrap=True, max_height=460,
                                         column_widths=["11%", "12%", "15%", "20%", "42%"])

        with gr.Accordion("Notes & assumptions", open=False):
            notes = gr.Markdown()

        # ---------------- Events ----------------
        inputs = [risk, horizon, initial, monthly, goal, target, age, backtest_years, rebalance]
        outputs = [status, chips, summary, pie, scatter, wealth, bt_caption, backtest, rb_table, mvo_table, notes]
        run = dict(fn=dashboard.update, inputs=inputs, outputs=outputs, show_progress="minimal")

        gr.on(
            triggers=[risk.release, horizon.release, backtest_years.release, rebalance.input,
                      initial.blur, initial.submit, monthly.blur, monthly.submit,
                      target.blur, target.submit, age.blur, age.submit],
            trigger_mode="always_last",
            **run,
        )
        risk.release(preset_for, inputs=risk, outputs=risk_preset, show_progress="hidden")
        risk_preset.input(
            lambda label: PRESETS.get(label, gr.skip()), inputs=risk_preset, outputs=risk, show_progress="hidden"
        ).then(**run)
        goal_targets = {c["value"]: c["default_target"] for c in opts["goal"]["choices"]}
        goal.input(lambda g: goal_targets.get(g, gr.skip()), inputs=goal, outputs=target,
                   show_progress="hidden").then(**run)

        def reset_values() -> tuple:
            d = default_values()
            return (preset_for(d["risk_tolerance"]), d["risk_tolerance"], d["horizon_years"],
                    d["initial_investment"], d["monthly_contribution"], d["goal"], d["target_amount"], d["age"],
                    d["backtest_years"], d["rebalance"])

        reset.click(reset_values, outputs=[risk_preset, *inputs[:5], target, age, backtest_years, rebalance],
                    show_progress="hidden").then(**run)
        demo.load(**run)

    return demo


def launch(**kwargs: Any) -> None:
    build_demo().launch(theme=THEME, css=CSS, **kwargs)
