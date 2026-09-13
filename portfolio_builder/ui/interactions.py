"""Client-side interactions injected into the page <head> (Gradio has no server-side hover or keystroke hooks)."""

from __future__ import annotations

import json

HOVER_HIGHLIGHT_JS = """
(() => {
  // Hovering (or tapping) a slice of a portfolio's allocation donut highlights that asset class's rows in the
  // holdings table of the same card and dims the others.
  // Gradio 6 renders table bodies as a virtualized grid: div.virtual-row > [role=gridcell][data-col].
  const assetClassColumn = (table) => {
    const th = [...table.querySelectorAll("thead th")].find((h) => h.textContent.includes("Asset class"));
    return th ? th.getAttribute("data-heading") : null;
  };

  const highlight = (card, label) => {
    const table = card.querySelector(".holdings-table");
    if (!table) return;
    const col = assetClassColumn(table);
    if (col === null) return;
    table.querySelectorAll(".virtual-row").forEach((row) => {
      const cell = row.querySelector(`[data-col="${col}"]`);
      const value = cell ? cell.textContent.trim() : null;
      const dot = row.querySelector(".asset-dot");
      if (dot) row.style.setProperty("--row-accent", dot.style.backgroundColor);
      row.classList.toggle("row-active", label !== null && value === label);
      row.classList.toggle("row-dim", label !== null && value !== label);
    });
  };

  const bind = (card) => {
    const plot = card.querySelector(".donut-plot .js-plotly-plot");
    if (!plot || plot.__holdingsHover || typeof plot.on !== "function") return;
    plot.__holdingsHover = true;
    const onPoint = (event) => highlight(card, event && event.points && event.points.length ? event.points[0].label : null);
    plot.on("plotly_hover", onPoint);
    plot.on("plotly_click", onPoint);
    plot.on("plotly_unhover", () => highlight(card, null));
  };

  const scan = () => document.querySelectorAll(".portfolio-card").forEach(bind);
  new MutationObserver(scan).observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", scan);
})();
"""

MONEY_INPUT_JS = """
(() => {
  // Format money text boxes as "$#,000" while typing, keeping the caret after the same digit.
  // Runs in the capture phase so Gradio's own input handler already sees the formatted value.
  const MAX_DIGITS = 12;
  const format = (el) => {
    const raw = el.value;
    if (/[kKmM]\\s*$/.test(raw.trim())) return;  // allow shorthand like "50k"; normalized when the field is left
    const caret = el.selectionStart ?? raw.length;
    const digitsBeforeCaret = raw.slice(0, caret).replace(/\\D/g, "").length;
    const digits = raw.split(".")[0].replace(/\\D/g, "").replace(/^0+(?=\\d)/, "").slice(0, MAX_DIGITS);
    const formatted = digits ? "$" + Number(digits).toLocaleString("en-US") : "";
    if (formatted === raw) return;
    el.value = formatted;
    let pos = digitsBeforeCaret === 0 ? Math.min(1, formatted.length) : formatted.length;
    for (let i = 0, seen = 0; i < formatted.length && digitsBeforeCaret > 0; i++) {
      if (/\\d/.test(formatted[i]) && ++seen === digitsBeforeCaret) { pos = i + 1; break; }
    }
    try { el.setSelectionRange(pos, pos); } catch (_) { /* not focusable */ }
  };
  document.addEventListener("input", (event) => {
    const el = event.target;
    if (el && el.closest && el.closest(".money-input") && /^(INPUT|TEXTAREA)$/.test(el.tagName)) format(el);
  }, true);
})();
"""

TOOLTIP_JS = """
(() => {
  // Show each sidebar input's help text as a tooltip when hovering its label.
  const TIPS = __TOOLTIPS__;
  const apply = () => {
    for (const [id, text] of Object.entries(TIPS)) {
      const block = document.getElementById(id);
      if (!block) continue;
      const label = block.querySelector('[data-testid="block-info"], .label-text');  // .label-text: checkboxes
      if (!label || label.dataset.tip === text) continue;
      label.dataset.tip = text;
      label.classList.add("has-tip");
      const input = block.querySelector("input, textarea");
      if (input) input.setAttribute("aria-description", text);  // screen readers still get the help text
    }
  };
  new MutationObserver(apply).observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", apply);
})();
"""


def build_head(tooltips: dict[str, str]) -> str:
    """<head> HTML with all client-side interactions; ``tooltips`` maps an input's elem_id to its help text."""
    tooltip_js = TOOLTIP_JS.replace("__TOOLTIPS__", json.dumps(tooltips))
    return f"<script>{HOVER_HIGHLIGHT_JS}</script>\n<script>{MONEY_INPUT_JS}</script>\n<script>{tooltip_js}</script>"
