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
from .interactions import HEAD
from .theme import CSS, THEME

logger = logging.getLogger(__name__)

INPUT_FIELDS = ("risk_tolerance", "horizon_years", "initial_investment", "monthly_contribution", "goal",
                "target_amount", "age", "backtest_years", "rebalance")
METHODS = ("rule_based", "mean_variance")
CARD_PARTS = ("header", "donut", "table", "wealth")
OUTPUT_KEYS = (
    "status", "chips",
    *(f"{part}_{method}" for method in METHODS for part in CARD_PARTS),
    "scatter", "backtest_caption", "backtest", "notes",
)
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
        """Returns one value per ``OUTPUT_KEYS`` entry, in that order."""
        request = dict(zip(INPUT_FIELDS, values))
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
        return tuple(values_by_key[key] for key in OUTPUT_KEYS)


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

        # One card per portfolio: KPI tiles; allocation donut beside holdings; projected wealth.
        cards: dict[str, dict[str, Any]] = {}
        with gr.Row(equal_height=True):
            for method in METHODS:
                with gr.Column(elem_classes=["portfolio-card", f"portfolio-card-{method}"], min_width=520):
                    header = gr.HTML()
                    with gr.Row():
                        with gr.Column(scale=5, min_width=230):
                            gr.Markdown("Allocation by Asset Class", elem_classes="card-subtitle")
                            donut = gr.Plot(show_label=False, container=False, elem_classes="donut-plot")
                        with gr.Column(scale=6, min_width=270):
                            gr.Markdown("Holdings", elem_classes="card-subtitle")
                            table = gr.Dataframe(show_label=False, interactive=False, max_height=400,
                                                 datatype=components.HOLDINGS_DATATYPES,
                                                 column_widths=["6%", "18%", "32%", "21%", "23%"],
                                                 elem_classes="holdings-table")
                    gr.Markdown("Projected Wealth", elem_classes="card-subtitle")
                    gr.Markdown("Expected (mean), optimistic (75th pct) and pessimistic (25th pct) value from "
                                "5,000 simulations, including monthly contributions. Same scale in both cards.",
                                elem_classes="section-caption")
                    wealth = gr.Plot(show_label=False, container=False, elem_classes="wealth-plot")
                cards[method] = {"header": header, "donut": donut, "table": table, "wealth": wealth}

        # Charts that plot both portfolios together.
        with gr.Column(elem_classes="comparison-card"):
            gr.HTML(components.comparison_header())
            with gr.Row(equal_height=True):
                with gr.Column(elem_classes="chart-card", min_width=460):
                    gr.Markdown("### Risk vs. Return", elem_classes="section-title")
                    gr.Markdown("Annualized expected return vs. volatility for each asset class, cash and both "
                                "portfolios, with the efficient frontier.", elem_classes="section-caption")
                    scatter = gr.Plot(show_label=False, container=False)
                with gr.Column(elem_classes="chart-card", min_width=460):
                    gr.Markdown("### Historical Backtest vs. S&P 500", elem_classes="section-title")
                    bt_caption = gr.Markdown(elem_classes="section-caption")
                    backtest = gr.Plot(show_label=False, container=False)

        with gr.Accordion("Notes & assumptions", open=False):
            notes = gr.Markdown()

        # ---------------- Events ----------------
        inputs = [risk, horizon, initial, monthly, goal, target, age, backtest_years, rebalance]
        components_by_key = {
            "status": status, "chips": chips,
            **{f"{part}_{m}": cards[m][part] for m in METHODS for part in CARD_PARTS},
            "scatter": scatter, "backtest_caption": bt_caption, "backtest": backtest, "notes": notes,
        }
        outputs = [components_by_key[key] for key in OUTPUT_KEYS]
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
    build_demo().launch(theme=THEME, css=CSS, head=HEAD, **kwargs)
