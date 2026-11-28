// Resize only displayed Julia plots. The default Plotly observer also fires
// while a tab or a results container is being removed.
window.vmiObserveJuliaPesPlot = (component) => {
  const graph = component?.$el;
  if (!graph?.isConnected || graph.vmiJuliaResizeObserver) return;
  let previousWidth = graph._fullLayout?.width;
  const observer = new ResizeObserver(() => {
    if (!graph.isConnected) {
      observer.disconnect();
      delete graph.vmiJuliaResizeObserver;
      return;
    }
    const width = graph.clientWidth;
    if (!width || width === previousWidth) return;
    previousWidth = width;
    component.Plotly.relayout(graph, {width});
  });
  graph.vmiJuliaResizeObserver = observer;
  observer.observe(graph);
};
