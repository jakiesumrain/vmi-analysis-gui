// Run against a saved Accumulate result displayed in the GUI.
async page => {
  const check = (ok, message) => { if (!ok) throw new Error(message); };
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({width: 1440, height: 1100});
  await page.waitForFunction(() => document.querySelector('.js-plotly-plot')?._fullData?.length === 1);
  const graph = page.locator('.js-plotly-plot');
  check(await graph.evaluate(g => g._fullLayout.yaxis.scaleanchor === 'x'), 'Equal image pixel aspect');
  const drag = graph.locator('.draglayer .xy .nsewdrag');
  await drag.scrollIntoViewIfNeeded();
  const box = await drag.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.waitForFunction(() => document.querySelector('.js-plotly-plot')._hoverdata?.length);
  const counts = await graph.evaluate(g => {
    const p = g._hoverdata[0];
    return {hover: p.customdata, original: g._fullData[0].customdata[p.y][p.x]};
  });
  check(counts.hover === counts.original, 'Hover preserves original accumulated counts');
  const before = await graph.evaluate(g => [...g._fullLayout.xaxis.range]);
  await page.mouse.move(box.x + box.width * .2, box.y + box.height * .2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * .7, box.y + box.height * .7, {steps: 10});
  await page.mouse.up();
  await page.waitForFunction(() => {
    const g = document.querySelector('.js-plotly-plot');
    return g._fullLayout.xaxis.range[1] - g._fullLayout.xaxis.range[0] < 800;
  });
  const zoomed = await graph.evaluate(g => [...g._fullLayout.xaxis.range]);
  const originalColor = await graph.evaluate(g => g._fullData[0].colorscale[0][1]);
  const target = await page.getByRole('combobox', {name: 'Colormap'}).inputValue() === 'Plasma' ? 'Viridis' : 'Plasma';
  await page.getByRole('combobox', {name: 'Colormap'}).click();
  await page.getByRole('option', {name: target, exact: true}).click();
  await page.waitForFunction(color => document.querySelector('.js-plotly-plot')._fullData[0].colorscale[0][1] !== color, originalColor);
  check(await graph.evaluate(g => [...g._fullLayout.xaxis.range]).then(range =>
    range.every((v, i) => v === zoomed[i])), 'Changing colormap preserves zoom');
  const endpoints = {
    Viridis: ['#440154', '#fde725'], Cividis: ['#00224e', '#fee838'],
    Plasma: ['#0d0887', '#f0f921'], Inferno: ['#000004', '#fcffa4'],
    Magma: ['#000004', '#fcfdbf'], Turbo: ['#30123b', '#7a0402'],
    Gray: ['rgb(0, 0, 0)', 'rgb(254, 254, 253)'], Hot: ['rgb(0,0,0)', 'rgb(255,255,255)']
  };
  for (const [name, expected] of Object.entries(endpoints)) {
    await page.getByRole('combobox', {name: 'Colormap'}).click();
    await page.getByRole('option', {name, exact: true}).click();
    await page.waitForFunction(({expected}) => {
      const scale = document.querySelector('.js-plotly-plot')._fullData[0].colorscale;
      return scale[0][1] === expected[0] && scale.at(-1)[1] === expected[1];
    }, {expected});
    check(await graph.evaluate(g => [...g._fullLayout.xaxis.range]).then(range =>
      range.every((v, i) => v === zoomed[i])), `${name} preserves zoom`);
  }
  await page.getByRole('combobox', {name: 'Colormap'}).click();
  await page.getByRole('option', {name: target, exact: true}).click();
  check(await graph.evaluate(g => g._fullData[0].customdata[512][512]) !== undefined, 'Counts retained');
  await graph.locator('.modebar-btn[data-title="Reset axes"]').click();
  const reset = await graph.evaluate(g => [...g._fullLayout.xaxis.range]);
  check(Math.abs(reset[0] - before[0]) < 1e-5 && Math.abs(reset[1] - before[1]) < 1e-5, 'Reset restores image');
  await page.reload();
  await page.waitForFunction(() => document.querySelector('.js-plotly-plot')?._fullData?.length === 1);
  check(await page.getByRole('combobox', {name: 'Colormap'}).inputValue() === target, 'Colormap survives reload');
  await page.mouse.move(10, 10);
  await graph.screenshot({path: '.cache/vmi-analysis/centroid_plotly_preview/accumulate_preview.png'});
  check(!errors.length, errors.join('; '));
  return {passed: true, checks: ['image aspect', 'original-count hover', 'rectangle zoom', 'all eight colormaps', 'colormap preserves zoom', 'remembered colormap', 'reset'], counts};
}
