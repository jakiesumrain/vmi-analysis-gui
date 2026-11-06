// Run against a Centroiding page showing both interactive result figures.
async page => {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({width: 1440, height: 1000});
  await page.reload();
  await page.waitForFunction(() => [...document.querySelectorAll('.js-plotly-plot')]
    .filter(g => g._fullLayout && g._fullData).length === 2, null, {timeout: 60000});
  const graphs = page.locator('.js-plotly-plot');
  check(await graphs.count() === 2, 'Both tuning and final summary render');
  const tuning = graphs.nth(0), summary = graphs.nth(1);
  check(await tuning.evaluate(g => g._fullData.length === 5), 'Tuning frame, detections and three histograms');
  check(await summary.evaluate(g => g._fullData.filter(t => t.type === 'heatmap').length === 2), 'Raw and Abel heatmaps');
  check(await summary.evaluate(g => g._fullLayout.yaxis.scaleanchor === 'x' && g._fullLayout.yaxis2.scaleanchor === 'x2'),
        'Image pixels retain equal aspect ratio');
  check(await summary.evaluate(g => {
    const annotation = g._fullLayout.annotations.find(a => a.text.startsWith('Frames:'));
    return annotation?.xref === 'paper' && annotation.x > .5 && annotation.y < .5;
  }), 'Run summary stays in its own lower-right panel');

  // Drag within the photon histogram, then pan and reset through Plotly controls.
  const histogram = tuning.locator('.draglayer .x2y2 .nsewdrag');
  await histogram.scrollIntoViewIfNeeded();
  const rect = await histogram.boundingBox();
  const before = await tuning.evaluate(g => [...g._fullLayout.xaxis2.range]);
  await page.mouse.move(rect.x + rect.width * .2, rect.y + rect.height * .2);
  await page.mouse.down();
  await page.mouse.move(rect.x + rect.width * .7, rect.y + rect.height * .8, {steps: 10});
  await page.mouse.up();
  await page.waitForFunction(({before}) => {
    const g = document.querySelectorAll('.js-plotly-plot')[0];
    const range = g._fullLayout.xaxis2.range;
    return range[1] - range[0] < (before[1] - before[0]) * .8;
  }, {before});
  const zoom = await tuning.evaluate(g => [...g._fullLayout.xaxis2.range]);
  await tuning.locator('.modebar-btn[data-title="Pan"]').click();
  await page.mouse.move(rect.x + rect.width * .4, rect.y + rect.height * .5);
  await page.mouse.down();
  await page.mouse.move(rect.x + rect.width * .6, rect.y + rect.height * .5, {steps: 8});
  await page.mouse.up();
  const pan = await tuning.evaluate(g => [...g._fullLayout.xaxis2.range]);
  check(Math.abs(pan[0] - zoom[0]) > 1e-3, 'Panning moves the visible range');
  check(Math.abs((pan[1] - pan[0]) - (zoom[1] - zoom[0])) < 1e-6, 'Panning preserves zoom scale');
  await tuning.locator('.modebar-btn[data-title="Reset axes"]').click();
  await tuning.locator('.modebar-btn[data-title="Zoom"]').click();
  const reset = await tuning.evaluate(g => [...g._fullLayout.xaxis2.range]);
  check(Math.abs(reset[0] - before[0]) < 1e-5 && Math.abs(reset[1] - before[1]) < 1e-5, 'Reset restores histogram range');

  const endpoints = {
    Viridis: ['#440154', '#fde725'], Cividis: ['#00224e', '#fee838'],
    Plasma: ['#0d0887', '#f0f921'], Inferno: ['#000004', '#fcffa4'],
    Magma: ['#000004', '#fcfdbf'], Turbo: ['#30123b', '#7a0402'],
    Gray: ['rgb(0, 0, 0)', 'rgb(254, 254, 253)'], Hot: ['rgb(0,0,0)', 'rgb(255,255,255)']
  };
  for (const [label, graphIndex, traceIndex, initial] of [
    ['Tuning colormap', 0, 0, 'Gray'], ['Raw VMI colormap', 1, 0, 'Turbo'],
    ['Abel VMI colormap', 1, 1, 'Turbo']
  ]) {
    const prior = await graphs.nth(graphIndex).evaluate(g => [...g._fullLayout.xaxis.range]);
    for (const [name, expected] of Object.entries(endpoints)) {
      await page.getByRole('combobox', {name: label, exact: true}).click();
      await page.getByRole('option', {name, exact: true}).click();
      await page.waitForFunction(({graphIndex, traceIndex, expected}) => {
        const scale = document.querySelectorAll('.js-plotly-plot')[graphIndex]._fullData[traceIndex].colorscale;
        return scale[0][1] === expected[0] && scale.at(-1)[1] === expected[1];
      }, {graphIndex, traceIndex, expected});
    }
    check(await graphs.nth(graphIndex).evaluate(g => [...g._fullLayout.xaxis.range])
      .then(range => range.every((v, i) => v === prior[i])), `${label} preserves axis range`);
    await page.getByRole('combobox', {name: label, exact: true}).click();
    await page.getByRole('option', {name: initial, exact: true}).click();
  }

  // Hover an actual positive raw-image pixel and verify the tooltip uses counts,
  // not the logarithm used for coloring.
  await summary.locator('.draglayer .xy .nsewdrag').scrollIntoViewIfNeeded();
  const point = await summary.evaluate(g => {
    const values = g._fullData[0].customdata;
    let row = 0, col = 0, maximum = -Infinity;
    for (let y = 0; y < values.length; y++) for (let x = 0; x < values[y].length; x++) {
      if (values[y][x] > maximum) { maximum = values[y][x]; row = y; col = x; }
    }
    const box = g.getBoundingClientRect();
    return {x: box.left + g._fullLayout.xaxis._offset + g._fullLayout.xaxis.l2p(col),
            y: box.top + g._fullLayout.yaxis._offset + g._fullLayout.yaxis.l2p(row), maximum};
  });
  await page.mouse.move(point.x, point.y);
  await page.waitForFunction(() => document.querySelectorAll('.js-plotly-plot')[1]
    .querySelector('.hoverlayer')?.textContent.includes('Counts:'));
  const tooltip = await summary.locator('.hoverlayer').textContent();
  const hovered = await summary.evaluate(g => {
    const p = g._hoverdata[0];
    return {count: p.customdata, original: g._fullData[0].customdata[p.y][p.x]};
  });
  // At overview scale several detector pixels share a screen pixel. Check
  // whichever source pixel Plotly actually resolved, not a neighboring maximum.
  check(hovered.count === hovered.original, 'Hover uses original raw counts');
  const displayed = Number(tooltip.match(/Counts:\s*([\d.e+-]+)/)?.[1]);
  check(Math.abs(displayed - hovered.original) <= Math.max(1, Math.abs(hovered.original)) * 1e-5,
        'Tooltip shows counts rather than log(counts)');
  check(await summary.evaluate(g => g._fullLayout.xaxis3.title.text.includes('eV') &&
        g._fullLayout.yaxis3.title.text === 'Intensity (arb. units)'), 'Calibrated spectrum units are retained');
  await page.mouse.move(20, 20);
  await tuning.screenshot({path: '.cache/vmi-analysis/centroid_plotly_preview/tuning_preview.png'});
  await summary.screenshot({path: '.cache/vmi-analysis/centroid_plotly_preview/summary_preview.png'});
  check(errors.length === 0, `No JavaScript exceptions: ${errors.join('; ')}`);
  return {passed: true, hoverCounts: hovered.count,
          checks: ['both figures', 'eight colormaps on all three images', 'histogram zoom/pan/reset', 'image aspect/orientation', 'original-count hover', 'spectrum units', 'summary layout']};
}
