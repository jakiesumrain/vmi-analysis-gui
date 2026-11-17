// Read-only smoke check against the full running application; does not submit jobs.
async page => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const names = ['GMD Map', 'Movie', 'Dark Frame', 'Accumulate', 'Centroiding',
                 'Binning', 'Trig Calibration', 'Circularize', 'PES', 'Pipeline', 'TPX3 Dump'];
  const titles = {'Movie': 'Frame Movie', 'Accumulate': 'Fast Accumulate',
                  'Centroiding': 'Picasso Centroiding', 'Binning': 'GMD Binning'};
  const runLabels = {'Pipeline': 'Run pipeline', 'Trig Calibration': 'Load image & remap',
                     'PES': 'Generate spectra'};
  const show = async name => {
    await page.getByRole('tab', {name, exact: true}).click();
    await page.waitForFunction(title => {
      const panels = document.querySelectorAll('[role="tabpanel"]');
      return panels.length === 1 && panels[0].querySelector('.vmi-title')?.textContent.trim() === title &&
        !document.querySelector('[class*="enter-active"], [class*="leave-active"]');
    }, titles[name] || name);
  };
  await page.setViewportSize({width: 1440, height: 1000});
  await page.waitForFunction(() => typeof window.vmiZoomOpen === 'function');
  for (const group of await page.locator('.vmi-sidebar-disclosure').all()) {
    if (!await group.evaluate(el => el.open)) await group.locator('summary').click();
  }
  const tabNames = await page.getByRole('tab').allTextContents();
  if (tabNames.length !== names.length || !names.every((name, i) => tabNames[i].endsWith(name)))
    throw new Error('Incorrect workspace navigation');
  for (const name of names) {
    await show(name);
    const panel = page.getByRole('tabpanel');
    await panel.waitFor({state: 'visible'});
    if (!await panel.getByRole('button', {name: runLabels[name] || 'Run', exact: true}).isVisible()) {
      throw new Error(`${name}: missing Run control`);
    }
    if (await panel.getByText('Internal Server Error', {exact: true}).count()) throw new Error(`${name}: failed to render`);
  }
  await show('GMD Map');
  const mapPanel = page.getByRole('tabpanel');
  if (!await mapPanel.getByLabel('HDF5 pattern', {exact: true}).isVisible()) throw new Error('Missing HDF5 pattern');
  const browse = mapPanel.getByRole('button', {name: 'Browse', exact: true}).first();
  await browse.press('Enter');
  const picker = page.getByRole('dialog');
  await picker.waitFor();
  await picker.getByRole('button', {name: 'Cancel', exact: true}).click();
  await picker.waitFor({state: 'hidden'});
  await browse.press('Space');
  await picker.waitFor();
  await picker.getByRole('button', {name: 'Cancel', exact: true}).click();
  await picker.waitFor({state: 'hidden'});
  await show('Binning');
  const bins = page.getByRole('tabpanel');
  await bins.getByText('GMD histograms', {exact: true}).click();
  await bins.getByRole('button', {name: 'Histogram GMD values', exact: true}).waitFor();
  await show('Centroiding');
  await page.getByRole('button', {name: 'View saved results', exact: true}).click();
  await page.getByRole('tabpanel').locator('.vmi-results:visible').waitFor();
  await page.screenshot({path: '.cache/vmi-analysis/layout_centroiding.png'});
  await show('Accumulate');
  await page.getByRole('button', {name: 'View saved result', exact: true}).click();
  await page.getByRole('tabpanel').locator('.vmi-results:visible').waitFor();
  await page.screenshot({path: '.cache/vmi-analysis/layout_accumulate.png'});
  if (errors.length) throw new Error(errors.join('; '));
  return {passed: true, panels: names, overlayLoaded: true};
}
