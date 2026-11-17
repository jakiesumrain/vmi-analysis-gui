// Checks disclosure state and panel navigation in a fresh browser session.
async page => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const group = name => page.locator(`[data-workspace-group="${name}"]`);
  const electron = group('Electron');
  const ion = group('Ion');
  const tab = name => page.getByRole('tab', {name, exact: true});
  const panel = title => page.getByRole('tabpanel').getByText(title, {exact: true});
  const isOpen = section => section.evaluate(element => element.open);

  await page.evaluate(() => {
    localStorage.removeItem('vmi-workspace.Electron.open');
    localStorage.removeItem('vmi-workspace.Ion.open');
  });
  await page.reload();
  await electron.locator('summary').waitFor();
  await page.locator('[data-workspace-ready]').last().waitFor();
  if (await isOpen(electron) || await isOpen(ion)) throw new Error('Both groups should start collapsed');
  await electron.locator('summary').press('Enter');
  await ion.locator('summary').press('Enter');

  await tab('TPX3 Dump').click();
  await panel('TPX3 Dump').waitFor();
  await ion.locator('summary').press('Space');
  await tab('TPX3 Dump').waitFor({state: 'hidden'});
  await panel('TPX3 Dump').waitFor();
  if (!await isOpen(electron)) throw new Error('Collapsing Ion also collapsed Electron');

  await page.waitForFunction(() => localStorage.getItem('vmi-workspace.Ion.open') === 'false');
  await page.reload();
  await page.locator('[data-workspace-ready]').last().waitFor();
  if (await isOpen(ion)) throw new Error('Ion collapse state was lost on reload');
  if (!await isOpen(electron)) throw new Error('Electron open state was lost on reload');
  await ion.locator('summary').press('Enter');
  await tab('TPX3 Dump').click();
  await panel('TPX3 Dump').waitFor();
  await tab('Circularize').click();
  await panel('Circularize').waitFor();

  await page.setViewportSize({width: 390, height: 844});
  if (!await ion.locator('summary').isVisible()) throw new Error('Ion heading is hidden on narrow screens');
  const strip = electron.getByRole('tablist');
  if (!await strip.evaluate(element => element.scrollWidth > element.clientWidth)) {
    throw new Error('Electron tools do not have their own horizontal scroll area');
  }
  await tab('PES').click();
  await panel('PES').waitFor();
  await page.getByRole('tab', {name: 'PES', exact: true, selected: true}).waitFor();
  if (errors.length) throw new Error(errors.join('\n'));
  return {passed: true};
}
