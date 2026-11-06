// Run against concurrency_fixture.py, never the user's normal server.
async page => {
  const check = (ok, message) => { if (!ok) throw new Error(message); };
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const endpoint = path => page.url().split('/').slice(0, 3).join('/') + path;
  const state = async () => (await page.request.get(endpoint('/_test/state'))).json();
  const waitState = async predicate => {
    for (let i = 0; i < 150; i++) {
      const value = await state();
      if (predicate(value)) return value;
      await page.waitForTimeout(50);
    }
    throw new Error('Scheduler state timed out: ' + JSON.stringify(await state()));
  };
  const status = (s, name) => s.jobs.filter(j => j.name === name).at(-1)?.status;
  const show = async (tab, title) => {
    await page.getByRole('tab', {name: tab, exact: true}).click();
    await page.waitForFunction(title => {
      const panels = document.querySelectorAll('[role=tabpanel]');
      return panels.length === 1 && panels[0].querySelector('.vmi-title')?.textContent.trim() === title;
    }, title);
  };
  const stop = async (tab, title, name) => {
    await show(tab, title);
    await page.getByRole('button', {name: 'Stop', exact: true}).click();
    await waitState(s => status(s, name) === 'cancelled');
    await page.getByRole('button', {name: 'Run', exact: true}).waitFor();
  };
  await page.setViewportSize({width: 1440, height: 1000});
  const limit = page.getByRole('spinbutton', {name: 'Maximum concurrent jobs'});
  await limit.fill('3');
  await limit.press('Tab');
  await waitState(s => s.limit === 3);
  await page.request.post(endpoint('/_test/start'));
  await waitState(s => s.jobs.filter(j => j.status === 'running').length === 3);
  await show('Accumulate', 'Fast Accumulate');
  await page.getByRole('button', {name: 'Cancel queued', exact: true}).waitFor();
  check(await page.getByText('Waiting for Dark frame:', {exact: false}).isVisible(), 'File conflict reason is visible');
  await page.reload();
  await limit.waitFor();
  check(await limit.inputValue() === '3', 'Concurrency limit survives reload');
  await stop('Movie', 'Frame Movie', 'Frame movie');
  let s = await state();
  check(status(s, 'Dark frame') === 'running' && status(s, 'Centroiding') === 'running', 'Other running panels survive Stop');
  check(status(s, 'Accumulate') === 'queued', 'Reader still waits for writer despite a free slot');
  await stop('Dark Frame', 'Dark Frame', 'Dark frame');
  await waitState(s => status(s, 'Accumulate') === 'running');
  await show('Accumulate', 'Fast Accumulate');
  await page.getByRole('button', {name: 'Stop', exact: true}).waitFor();
  await limit.fill('1');
  await limit.press('Tab');
  await waitState(s => s.limit === 1);
  await page.request.post(endpoint('/_test/add-map'));
  s = await state();
  check(status(s, 'Centroiding') === 'running' && status(s, 'Accumulate') === 'running', 'Lowering limit does not cancel existing work');
  check(status(s, 'GMD map') === 'queued', 'New work respects lowered limit');
  await stop('Accumulate', 'Fast Accumulate', 'Accumulate');
  check(status(await state(), 'GMD map') === 'queued', 'Limit one waits for the last running job');
  await stop('Centroiding', 'Picasso Centroiding', 'Centroiding');
  await waitState(s => status(s, 'GMD map') === 'running');
  await stop('GMD Map', 'GMD Map', 'GMD map');
  await page.getByText('idle', {exact: true}).waitFor();
  await page.screenshot({path: '.cache/vmi-analysis/concurrency_controls.png'});
  check(errors.length === 0, errors.join('; '));
  return {passed: true, checks: ['three simultaneous panels', 'file-conflict waiting reason', 'independent Stop',
    'remembered limit', 'lower limit preserves running jobs', 'queued work starts when safe']};
}
