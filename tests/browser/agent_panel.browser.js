// Read-only Agent layout check against the full application; no session is started.
async page => {
  const check = (ok, message) => { if (!ok) throw new Error(message); };
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({width: 1440, height: 1000});
  await page.getByRole('tab', {name: 'Agent', exact: true}).click();
  const panel = page.getByRole('tabpanel');
  await panel.locator('.vmi-title').filter({hasText: /^Agent$/}).waitFor();
  await page.waitForFunction(() => !document.querySelector('[class*="enter-active"], [class*="leave-active"]'));

  const header = panel.locator('.vmi-panel-head').filter({hasText: /^Agent/});
  const title = header.locator('.vmi-title');
  const connectionButton = header.getByRole('button', {name: 'Connection', exact: true});
  const titleBox = await title.boundingBox();
  const connectionBox = await connectionButton.boundingBox();
  const headerBox = await header.boundingBox();
  check(connectionBox.x > titleBox.x + titleBox.width, 'Connection is not to the right of the title');
  check(Math.abs(connectionBox.x + connectionBox.width - headerBox.x - headerBox.width) < 2,
    'Connection is not aligned to the title row trailing edge');

  const drawer = panel.locator('.vmi-agent-connection');
  const baseUrl = drawer.getByRole('textbox', {name: 'Base URL', exact: true});
  check(!await baseUrl.isVisible(), 'Connection drawer should start closed');
  await connectionButton.press('Enter');
  await baseUrl.waitFor({state: 'visible'});
  check(await connectionButton.getAttribute('aria-expanded') === 'true', 'Drawer state is not announced');
  await connectionButton.press('Space');
  await baseUrl.waitFor({state: 'hidden'});

  const composer = panel.locator('.vmi-agent-composer');
  const prompt = composer.getByRole('textbox', {name: 'Instruction', exact: true});
  const send = composer.getByRole('button', {name: 'Send', exact: true});
  check(await composer.evaluate(element => getComputedStyle(element).position) === 'sticky',
    'Composer is not sticky');
  check(await prompt.isVisible() && await send.isVisible(), 'Composer controls are missing');
  check(await prompt.evaluate(element => getComputedStyle(element, '::placeholder').color) !== 'rgba(0, 0, 0, 0)',
    'Composer placeholder is hidden');
  check(!await send.locator('.block').isVisible(), 'Icon-only Send button still shows its text label');
  check(!await send.isEnabled(), 'Send should be disabled before a session starts');
  check(await panel.locator('.vmi-agent-empty').isVisible(), 'Conversation empty state is missing');
  check(await panel.locator('.vmi-agent-empty').evaluate(element => getComputedStyle(element).borderWidth) === '0px',
    'Conversation still uses the old boxed empty state');
  await page.screenshot({path: '.cache/vmi-analysis/agent_panel_desktop.png'});

  await panel.locator('.vmi-agent-entries').evaluate(entries => {
    entries.replaceChildren();
    for (let i = 0; i < 30; i++) {
      const entry = document.createElement('div');
      entry.className = `vmi-agent-entry vmi-agent-entry-${i % 2 ? 'assistant' : 'user'} w-full`;
      const message = document.createElement('div');
      message.className = `vmi-agent-message ${i % 2 ? '' : 'vmi-agent-user'}`;
      message.textContent = `${i % 2 ? 'Agent response' : 'User message'} ${i}: conversation scrolling test`;
      entry.appendChild(message);
      entries.appendChild(entry);
    }
    const scroller = document.querySelector('.vmi-agent-transcript .q-scrollarea__container');
    scroller.scrollTop = 0;
    scroller.dispatchEvent(new Event('scroll'));
  });
  const latest = panel.getByRole('button', {name: 'Latest messages', exact: true});
  await page.waitForFunction(() => document.querySelector('.vmi-agent-latest')?.getAttribute('aria-hidden') === 'false');
  await latest.click();
  await page.waitForFunction(() => {
    const scroller = document.querySelector('.vmi-agent-transcript .q-scrollarea__container');
    return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 100;
  });
  await page.waitForFunction(() => document.querySelector('.vmi-agent-latest')?.getAttribute('aria-hidden') === 'true');

  await page.setViewportSize({width: 390, height: 844});
  const composerBox = await composer.boundingBox();
  check(composerBox.x >= 0 && composerBox.x + composerBox.width <= 390, 'Composer overflows the narrow viewport');
  check(await connectionButton.isVisible(), 'Connection is hidden on narrow screens');
  check(!await panel.locator('.vmi-agent-keyboard-hint').isVisible(), 'Keyboard hint crowds the narrow composer');
  await connectionButton.click();
  await baseUrl.waitFor({state: 'visible'});
  const drawerBox = await drawer.boundingBox();
  check(drawerBox.x >= 0 && drawerBox.x + drawerBox.width <= 390, 'Connection drawer overflows the narrow viewport');
  const gridColumns = await drawer.locator('.vmi-field-grid').evaluateAll(elements =>
    elements.map(element => getComputedStyle(element).gridTemplateColumns));
  check(gridColumns.every(columns => !columns.trim().includes(' ')),
    'Connection fields do not collapse to one column on narrow screens');
  await page.screenshot({path: '.cache/vmi-analysis/agent_panel_mobile.png'});
  check(errors.length === 0, errors.join('; '));
  return {passed: true, checks: ['title action', 'keyboard drawer', 'sticky composer', 'latest messages', 'narrow layout']};
}
