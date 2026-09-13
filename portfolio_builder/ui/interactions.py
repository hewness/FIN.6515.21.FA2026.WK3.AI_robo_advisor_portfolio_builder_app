"""Client-side interactions injected into the page <head> (Gradio has no server-side plot hover events)."""

HOVER_HIGHLIGHT_JS = """
<script>
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
</script>
"""

HEAD = HOVER_HIGHLIGHT_JS
