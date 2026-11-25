// Native PES charts have separate Y scales. Keep their X navigation linked.
window.vmiSyncPesX = (event, source, target) => {
  if (!event || !Object.keys(event).some(key => /^xaxis\.(range(\[\d\])?|autorange)$/.test(key))) return;
  const range = source?.$el?._fullLayout?.xaxis?.range;
  const chart = target?.$el;
  const other = chart?._fullLayout?.xaxis?.range;
  if (!range || !other || range.every((value, index) => value === other[index])) return;
  target.Plotly.relayout(chart, {"xaxis.range": [...range], "xaxis.autorange": false});
};

// Keep the axis values in data units so zoom, selection, hover, and export
// all use the original coordinates.
window.vmiUpdatePesAxes = (component) => {
  const chart = component?.$el;
  if (!chart?.isConnected || !chart._fullLayout || chart.vmiUpdatingAxes) return;
  const full = chart._fullLayout;
  const updates = {};
  const annotations = [];
  for (const [name, axis] of Object.entries(full)) {
    if (!/^[xy]axis\d*$/.test(name) || axis.type !== "linear" || !axis.showticklabels) continue;
    const [low, high] = [...axis.range].sort((a, b) => a - b);
    const horizontal = name.startsWith("x");
    const intervals = Math.max(3, Math.round(axis._length / (horizontal ? 84 : 44)));
    const spacing = (high - low) / intervals;
    if (!(spacing > 0) || !Number.isFinite(spacing)) continue;

    const stepScale = Math.pow(10, Math.floor(Math.log10(spacing)));
    const step = [1, 2, 2.5, 5, 10].find(value => value * stepScale >= spacing / Math.SQRT2) * stepScale;
    const exponent = Math.floor(Math.log10(Math.max(Math.abs(low), Math.abs(high))));
    const scale = Math.pow(10, exponent);
    if (!(step > 0) || !(scale > 0)) continue;
    const values = [];
    const labels = [];
    const start = Math.ceil(low / step - 1e-10);
    const stop = Math.floor(high / step + 1e-10);
    const decimals = Math.min(16, Math.max(0, Math.ceil(-Math.log10(step / scale)) + 1));
    for (let i = 0; i <= stop - start; i++) {
      const value = (start + i) * step;
      values.push(value);
      labels.push(String(Number((value / scale).toFixed(decimals))).replace("-", "−"));
    }
    if (JSON.stringify(chart.layout[name]?.tickvals) !== JSON.stringify(values)
        || JSON.stringify(chart.layout[name]?.ticktext) !== JSON.stringify(labels)) {
      updates[`${name}.tickmode`] = "array";
      updates[`${name}.tickvals`] = values;
      updates[`${name}.ticktext`] = labels;
    }

    if (exponent !== 0) {
      const suffix = name.slice(5);
      const free = axis.anchor === "free";
      const baseAxis = free ? full[`yaxis${axis.overlaying?.slice(1) ?? ""}`] : axis;
      const xref = horizontal ? `x${suffix}` : baseAxis?.anchor;
      const yref = horizontal ? axis.anchor : `y${suffix}`;
      annotations.push({
        xref: `${xref} domain`, yref: `${yref} domain`,
        x: horizontal ? 1 : 0, y: horizontal ? 0 : 1,
        xanchor: horizontal ? "right" : "left",
        yanchor: horizontal ? "top" : "bottom",
        xshift: free ? axis.shift : 0,
        yshift: horizontal ? -24 : 3,
        text: `×10<sup>${String(exponent).replace("-", "−")}</sup>`,
        font: {color: axis.tickfont.color, size: axis.tickfont.size},
        showarrow: false,
      });
    }
  }
  if (JSON.stringify(chart.layout.annotations ?? []) !== JSON.stringify(annotations)) {
    updates.annotations = annotations;
  }
  if (Object.keys(updates).length) {
    chart.vmiUpdatingAxes = true;
    // Plotly fires afterplot again after applying the labels.
    component.Plotly.relayout(chart, updates).finally(() => {
      chart.vmiUpdatingAxes = false;
      requestAnimationFrame(() => window.vmiUpdatePesAxes(component));
    });
  }
};
