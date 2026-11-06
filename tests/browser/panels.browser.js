// Read-only smoke check against the full running application; does not submit jobs.
async page => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const names = ['GMD Map', 'Movie', 'Dark Frame', 'Accumulate', 'Centroiding',
                 'Binning', 'Circularize', 'PES', 'Pipeline'];
  const titles = {'Movie': 'Frame Movie', 'Accumulate': 'Fast Accumulate',
                  'Centroiding': 'Picasso Centroiding', 'Binning': 'GMD Binning', 'PES': 'PES / Abel Inversion'};
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
  for (const name of names) {
    await show(name);
    const panel = page.getByRole('tabpanel');
    await panel.waitFor({state: 'visible'});
    if (!await panel.getByRole('button', {name: name === 'Pipeline' ? 'Run pipeline' : 'Run', exact: true}).isVisible()) {
      throw new Error(`${name}: missing Run control`);
    }
    if (await panel.getByText('Internal Server Error', {exact: true}).count()) throw new Error(`${name}: failed to render`);
  }
  await show('Centroiding');
  await page.screenshot({path: '.cache/vmi-analysis/layout_centroiding.png'});
  await show('Accumulate');
  await page.screenshot({path: '.cache/vmi-analysis/layout_accumulate.png'});
  if (errors.length) throw new Error(errors.join('; '));
  return {passed: true, panels: names, overlayLoaded: true};
}
