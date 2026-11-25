// Checks disclosure state and panel navigation in a fresh browser session.
async page => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const group = name => page.locator(`[data-workspace-group="${name}"]`);
  const electron = group('Electron');
  const workbench = group('Workbench');
  const ion = group('Ion');
  const ionSummary = ion.locator(':scope > summary');
  const fallback = ion.locator('[data-workspace-group="Fallback"]');
  const tab = name => page.getByRole('tab', {name, exact: true});
  const panel = title => page.getByRole('tabpanel').getByText(title, {exact: true});
  const isOpen = section => section.evaluate(element => element.open);

  await page.evaluate(() => {
    localStorage.removeItem('vmi-workspace.Electron.open');
    localStorage.removeItem('vmi-workspace.Workbench.open');
    localStorage.removeItem('vmi-workspace.Ion.open');
    localStorage.removeItem('vmi-workspace.Fallback.open');
    localStorage.removeItem('vmi-workspace.sidebar.collapsed');
  });
  await page.reload();
  await electron.locator('summary').waitFor();
  await page.locator('[data-sidebar-ready]').waitFor();
  if (!await isOpen(workbench) || await isOpen(electron) || await isOpen(ion)) {
    throw new Error('Workbench should start open with Electron and Ion folded');
  }
  if (await workbench.locator(':scope > [data-workspace-group]').count() !== 2) {
    throw new Error('Electron and Ion should both be inside Workbench');
  }
  await electron.locator('summary').press('Enter');
  await ionSummary.press('Enter');

  await fallback.locator('summary').waitFor();
  if (await isOpen(fallback)) throw new Error('Fallback tools should start collapsed');
  await fallback.locator('summary').press('Enter');
  for (const name of ['TPX3 Dump (fallback)', 'Ion TOF (fallback)', 'Ion Pipeline (fallback)']) {
    await tab(name).click();
    await panel(name).waitFor();
  }
  if (await tab('TPX3 Analysis').count()) throw new Error('Temporary tool is still present');
  await fallback.locator('summary').press('Space');
  await tab('Ion Pipeline (fallback)').waitFor({state: 'hidden'});

  await tab('TPX3 Dump').click();
  await panel('TPX3 Dump').waitFor();
  await ionSummary.press('Space');
  await tab('TPX3 Dump').waitFor({state: 'hidden'});
  await panel('TPX3 Dump').waitFor();
  if (!await isOpen(electron)) throw new Error('Collapsing Ion also collapsed Electron');

  await page.waitForFunction(() => localStorage.getItem('vmi-workspace.Ion.open') === 'false');
  await page.reload();
  await page.locator('[data-sidebar-ready]').waitFor();
  if (await isOpen(ion)) throw new Error('Ion collapse state was lost on reload');
  if (!await isOpen(electron)) throw new Error('Electron open state was lost on reload');
  await ionSummary.press('Enter');
  await tab('TPX3 Dump').click();
  await panel('TPX3 Dump').waitFor();
  await tab('Circularize').click();
  await panel('Circularize').waitFor();

  const sidebar = page.locator('.vmi-sidebar');
  const settings = sidebar.getByRole('button', {name: 'Settings', exact: true});
  const agent = sidebar.locator(':scope > .vmi-sidebar-agent').getByRole('tab', {name: 'Agent', exact: true});
  const collapse = sidebar.getByRole('button', {name: 'Collapse sidebar', exact: true});
  if ((await agent.boundingBox()).y >= (await workbench.boundingBox()).y ||
      await workbench.getByRole('tab', {name: 'Agent', exact: true}).count()) {
    throw new Error('Agent should be a top-level item above Workbench');
  }
  const [agentBox, collapseBox] = await Promise.all([agent.boundingBox(), collapse.boundingBox()]);
  if (Math.abs(agentBox.y + agentBox.height / 2 - collapseBox.y - collapseBox.height / 2) > 1) {
    throw new Error('The collapse button should stay beside Agent on desktop');
  }
  await agent.click();
  await panel('New chat').waitFor();
  await page.getByRole('tab', {name: 'Agent', exact: true, selected: true}).waitFor();
  const connection = page.locator('.vmi-agent-connection');
  const connectionButton = page.getByRole('button', {name: 'Connection', exact: true});
  const baseUrl = connection.getByRole('textbox', {name: 'Base URL', exact: true});
  if (await baseUrl.isVisible()) throw new Error('Agent connection settings should start collapsed');
  await connectionButton.click();
  await baseUrl.waitFor({state: 'visible'});
  await connectionButton.click();
  await baseUrl.waitFor({state: 'hidden'});
  if (await page.locator('.vmi-header').getByRole('button', {name: 'Settings', exact: true}).count()) {
    throw new Error('Settings is still in the header');
  }
  if (await sidebar.getByText(/Local workspace|Private workspace|13 tools/).count()) {
    throw new Error('The old sidebar footer is still shown');
  }
  await collapse.press('Enter');
  await page.waitForFunction(() => document.querySelector('.vmi-sidebar--collapsed'));
  const width = (await sidebar.boundingBox()).width;
  if (width > 80) throw new Error('Icon rail is not narrow');
  if (await settings.locator('.block').isVisible()) {
    throw new Error('Settings label text is shown in the icon rail');
  }
  for (const name of ['Circularize', 'TPX3 Dump', 'Ion Pipeline', 'Agent']) {
    await tab(name).click();
    await panel(name === 'Agent' ? 'New chat' : name).waitFor();
  }
  if (!await settings.isVisible()) throw new Error('Settings is hidden in the icon rail');
  await settings.click();
  await page.getByRole('dialog').waitFor();
  await page.getByRole('dialog').getByRole('button', {name: 'Close', exact: true}).click();
  await page.reload();
  await page.locator('[data-sidebar-ready]').waitFor();
  if (!(await sidebar.getAttribute('class')).includes('vmi-sidebar--collapsed')) {
    throw new Error('Collapsed rail preference was lost on reload');
  }
  await sidebar.getByRole('button', {name: 'Expand sidebar', exact: true}).press('Space');
  await page.waitForFunction(() => !document.querySelector('.vmi-sidebar--collapsed'));
  if ((await sidebar.boundingBox()).width <= width) throw new Error('Sidebar did not expand');
  if (!await settings.locator('.block').isVisible()) {
    throw new Error('Settings label text did not return when the sidebar expanded');
  }

  await page.setViewportSize({width: 390, height: 844});
  if (!await settings.isVisible()) throw new Error('Settings is hidden on narrow screens');
  if (!await ionSummary.isVisible()) throw new Error('Ion heading is hidden on narrow screens');
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
