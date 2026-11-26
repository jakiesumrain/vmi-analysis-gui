// Checks disclosure state and panel navigation in a fresh browser session.
async page => {
  await page.setViewportSize({width: 1440, height: 1000});
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
    localStorage.removeItem('vmi-workspace.sidebar.width');
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
  const sidebarFoot = sidebar.locator('.vmi-sidebar-foot');
  const settingsBox = await settings.boundingBox();
  const footBox = await sidebarFoot.boundingBox();
  if (Math.abs(settingsBox.x + settingsBox.width - footBox.x - footBox.width) > 1 ||
      await settings.locator('.block').count() || await sidebarFoot.locator('.vmi-sidebar-account').count()) {
    throw new Error('The local sidebar footer should have only a right-aligned Settings icon');
  }
  const agent = sidebar.locator(':scope > .vmi-sidebar-agent').getByRole('button', {name: 'Agent', exact: true});
  const header = page.locator('.vmi-header');
  const collapse = header.getByRole('button', {name: 'Collapse sidebar', exact: true});
  const workbenchHeading = workbench.locator(':scope > summary');
  if ((await agent.boundingBox()).y >= (await workbench.boundingBox()).y ||
      await sidebar.getByRole('tab', {name: 'Agent', exact: true}).count()) {
    throw new Error('Agent should be a top-level item above Workbench');
  }
  const typography = item => item.evaluate(element => {
    const style = getComputedStyle(element);
    return [style.fontFamily, style.fontSize, style.fontWeight];
  });
  if (String(await typography(agent)) !== String(await typography(workbenchHeading)) ||
      (await agent.getByText('Agent', {exact: true}).boundingBox()).x !==
      (await workbenchHeading.getByText('Workbench', {exact: true}).boundingBox()).x) {
    throw new Error('Agent and Workbench headings should have matching typography and alignment');
  }
  const title = header.getByText('Agent-native VMI analysis', {exact: true});
  if ((await collapse.boundingBox()).x >= (await title.boundingBox()).x) {
    throw new Error('The sidebar toggle should be to the left of the app title');
  }
  const divider = page.getByRole('separator', {name: 'Resize sidebar'});
  const startWidth = (await sidebar.boundingBox()).width;
  const {x, y, width: hitWidth} = await divider.boundingBox();
  await page.mouse.move(x + hitWidth / 2, y + 100);
  await page.mouse.down();
  await page.mouse.move(x + hitWidth / 2 + 72, y + 100);
  await page.mouse.up();
  await page.waitForFunction(width =>
    document.querySelector('.vmi-sidebar').getBoundingClientRect().width === width, startWidth + 72);
  await divider.press('ArrowLeft');
  const resizedWidth = startWidth + 56;
  if ((await sidebar.boundingBox()).width !== resizedWidth ||
      await divider.getAttribute('aria-valuenow') !== String(resizedWidth)) {
    throw new Error('The resize handle did not shrink the sidebar or update its accessible value');
  }
  const tabPanels = page.locator('.vmi-tab-panels');
  await tabPanels.evaluate(element => {
    new MutationObserver(records => {
      if (records.some(record => /q-transition--slide-(left|right)/.test(
        `${record.oldValue ?? ''} ${record.target.getAttribute('class') ?? ''}`))) {
        element.dataset.sawHorizontalSlide = 'true';
      }
    }).observe(element, {subtree: true, attributes: true, attributeFilter: ['class'], attributeOldValue: true});
  });
  await agent.click();
  await panel('New chat').waitFor();
  await agent.and(page.locator('[aria-current="page"]')).waitFor();
  const connection = page.locator('.vmi-agent-connection');
  const connectionButton = page.getByRole('button', {name: 'Connection', exact: true});
  const agentActions = sidebar.locator('.vmi-sidebar-agent-actions');
  const agentDisclosure = sidebar.getByRole('button', {name: 'Agent actions', exact: true});
  await agentDisclosure.press('Enter');
  await agentActions.waitFor({state: 'hidden'});
  if (await agentDisclosure.getAttribute('aria-expanded') !== 'false') {
    throw new Error('Agent actions do not announce their collapsed state');
  }
  await collapse.press('Enter');
  await header.getByRole('button', {name: 'Expand sidebar', exact: true}).press('Enter');
  await agentDisclosure.waitFor();
  if (await agentActions.isVisible()) {
    throw new Error('Expanding the sidebar changed the Agent actions disclosure');
  }
  await agentDisclosure.press('Space');
  await agentActions.waitFor();
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
  if (await divider.isVisible()) throw new Error('Resize handle is visible in the icon rail');
  if (await agentActions.isVisible() || await agentDisclosure.isVisible() ||
      await agent.locator('.vmi-sidebar-agent-label').isVisible() ||
      !await agent.locator('.vmi-sidebar-agent-icon').isVisible()) {
    throw new Error('The collapsed sidebar should show only the Agent icon');
  }
  for (const name of ['Circularize', 'TPX3 Dump', 'Ion Pipeline', 'Agent']) {
    await (name === 'Agent' ? agent : tab(name)).click();
    await panel(name === 'Agent' ? 'New chat' : name).waitFor();
  }
  if (await tabPanels.getAttribute('data-saw-horizontal-slide')) throw new Error('Panels still slide horizontally');
  if (!await settings.isVisible()) throw new Error('Settings is hidden in the icon rail');
  await settings.click();
  await page.getByRole('dialog').waitFor();
  await page.getByRole('dialog').getByRole('button', {name: 'Close', exact: true}).click();
  await page.reload();
  await page.locator('[data-sidebar-ready]').waitFor();
  if (!(await sidebar.getAttribute('class')).includes('vmi-sidebar--collapsed')) {
    throw new Error('Collapsed rail preference was lost on reload');
  }
  await header.getByRole('button', {name: 'Expand sidebar', exact: true}).press('Space');
  await page.waitForFunction(() => !document.querySelector('.vmi-sidebar--collapsed'));
  if ((await sidebar.boundingBox()).width !== resizedWidth) throw new Error('Resized width was lost on reload');
  if (!await settings.isVisible() || !await agentActions.isVisible()) {
    throw new Error('Settings and expanded Agent actions should be visible in the wide sidebar');
  }

  await page.setViewportSize({width: 390, height: 844});
  if (await divider.isVisible()) throw new Error('Resize handle is visible on narrow screens');
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
