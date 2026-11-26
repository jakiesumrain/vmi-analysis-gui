// Agent layout and attachment checks in an isolated workspace; no session or job is started.
async page => {
  const check = (ok, message) => { if (!ok) throw new Error(message); };
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({width: 1440, height: 1000});
  await page.getByRole('button', {name: 'Agent', exact: true}).click();
  const panel = page.getByRole('tabpanel');
  await panel.getByRole('heading', {name: 'New chat', exact: true}).waitFor();
  await page.waitForFunction(() => !document.querySelector('[class*="enter-active"], [class*="leave-active"]'));

  const actions = page.locator('.vmi-sidebar-agent-actions');
  const connectionButton = actions.getByRole('button', {name: 'Connection', exact: true});
  await actions.getByRole('button', {name: 'Analysis runs', exact: true}).waitFor();
  check(!await panel.getByText('idle', {exact: true}).count(), 'Chat still shows an idle badge');

  const drawer = panel.locator('.vmi-agent-connection');
  const baseUrl = drawer.getByRole('textbox', {name: 'Base URL', exact: true});
  const diagnostics = drawer.locator('.vmi-agent-diagnostics');
  check(!await diagnostics.isVisible(), 'Diagnostics is visible with Connection closed');
  check(!await baseUrl.isVisible(), 'Connection drawer should start closed');
  await connectionButton.press('Enter');
  await baseUrl.waitFor({state: 'visible'});
  await diagnostics.waitFor({state: 'visible'});
  await drawer.getByRole('textbox', {name: 'Model IDs', exact: true}).fill('test-model');
  check(await connectionButton.getAttribute('aria-expanded') === 'true', 'Drawer state is not announced');
  await connectionButton.press('Space');
  await baseUrl.waitFor({state: 'hidden'});
  await diagnostics.waitFor({state: 'hidden'});

  const composer = panel.locator('.vmi-agent-composer');
  const prompt = composer.getByRole('textbox', {name: 'Instruction', exact: true});
  const send = composer.getByRole('button', {name: 'Send', exact: true});
  check(await composer.evaluate(element => element.getBoundingClientRect().bottom <= innerHeight),
    'Composer starts below the viewport');
  const modelSelector = composer.locator('.vmi-agent-model');
  const modelBox = await modelSelector.boundingBox();
  const sendBox = await send.boundingBox();
  check(modelBox.x + modelBox.width <= sendBox.x && sendBox.x - modelBox.x - modelBox.width < 16,
    'The model selector is too far from Send');
  const modelText = modelSelector.getByText('test-model', {exact: true});
  const checkToolbarAlignment = async () => {
    await modelText.waitFor();
    const icons = [composer.locator('.vmi-agent-attach .q-icon'), send.locator('.q-icon')];
    const centers = await Promise.all([modelText, ...icons].map(async control => {
      const box = await control.boundingBox();
      return box.y + box.height / 2;
    }));
    check(Math.max(...centers) - Math.min(...centers) <= 1, 'Composer controls are vertically misaligned');
  };
  await checkToolbarAlignment();
  check(!await modelSelector.locator('.q-select__dropdown-icon').isVisible(), 'Model selector shows a chevron');
  check(await prompt.evaluate(element => getComputedStyle(element).cursor) === 'text',
    'The instruction field does not show a text cursor');
  check(await prompt.isVisible() && await send.isVisible(), 'Composer controls are missing');
  check(await prompt.evaluate(element => getComputedStyle(element, '::placeholder').color) !== 'rgba(0, 0, 0, 0)',
    'Composer placeholder is hidden');
  check(!await send.locator('.block').isVisible(), 'Icon-only Send button still shows its text label');
  check(!await send.isEnabled(), 'Send should be disabled before a session starts');
  const welcome = panel.locator('.vmi-agent-empty');
  check(await welcome.getByText('What would you like to analyze?', {exact: true}).isVisible(),
    'Conversation welcome is missing');
  await page.waitForFunction(() => document.querySelector('.vmi-agent-empty img')?.naturalWidth > 0);
  check(await welcome.evaluate(element => {
    const icon = element.querySelector('img').getBoundingClientRect();
    const area = element.getBoundingClientRect();
    const text = element.lastElementChild.getBoundingClientRect();
    return Math.abs(icon.x + icon.width / 2 - area.x - area.width / 2) < 1 &&
      icon.bottom < text.top && Math.abs((icon.top + text.bottom) / 2 - (area.top + area.bottom) / 2) < 1 &&
      area.height > 300;
  }), 'Welcome icon and prompt should be centered in the available chat space');
  check(await welcome.evaluate(element => getComputedStyle(element).borderWidth) === '0px',
    'Conversation still uses the old boxed empty state');
  check((await composer.boundingBox()).height <= 115, 'Empty composer is taller than expected');
  const attachmentButton = composer.getByRole('button', {name: 'Attach file or folder', exact: true});
  await attachmentButton.click();
  const attachFile = page.getByText('Attach file', {exact: true});
  await attachFile.waitFor();
  check(await page.getByText('Attach folder', {exact: true}).isVisible(), 'Attach folder is missing');
  check(!await page.getByText('Link a run', {exact: true}).count(), 'Run picker is still in the attachment menu');
  const attachmentBox = await attachmentButton.boundingBox();
  const menuItemBox = await attachFile.boundingBox();
  check(Math.abs(attachmentBox.x - menuItemBox.x) < 36, 'Attachment menu opened far from its button');
  await page.keyboard.press('Escape');
  await attachFile.waitFor({state: 'detached'});

  await prompt.evaluate(element => {
    const transfer = new DataTransfer();
    transfer.items.add(new File([new Uint8Array([137, 80, 78, 71])], 'image.png', {type: 'image/png'}));
    const event = new Event('paste', {bubbles: true, cancelable: true});
    Object.defineProperty(event, 'clipboardData', {value: transfer});
    element.dispatchEvent(event);
  });
  const screenshotChip = composer.locator('.vmi-agent-attachment').filter({hasText: /screenshot-.*\.png/});
  await screenshotChip.waitFor();
  check(await screenshotChip.count() === 1, 'Pasted screenshot was not attached exactly once');

  const dropTransfer = await page.evaluateHandle(() => {
    const transfer = new DataTransfer();
    transfer.items.add(new File(['drop contents'], 'dropped.txt', {type: 'text/plain'}));
    return transfer;
  });
  await composer.dispatchEvent('dragenter', {dataTransfer: dropTransfer});
  check(await composer.evaluate(element => element.classList.contains('vmi-agent-composer-receiving')),
    'Dragging files does not show the drop target');
  await composer.dispatchEvent('drop', {dataTransfer: dropTransfer});
  await dropTransfer.dispose();
  await composer.locator('.vmi-agent-attachment').filter({hasText: 'dropped.txt'}).waitFor();
  check(!await composer.evaluate(element => element.classList.contains('vmi-agent-composer-receiving')),
    'Drop target remains active after the drop');
  await page.waitForFunction(() => {
    const element = document.querySelector('.vmi-agent-composer');
    return element?.getAttribute('aria-busy') === 'false' && !document.querySelector('.vmi-agent-upload-status')?.checkVisibility();
  });
  check(await composer.locator('.vmi-agent-attachment').count() === 2, 'Uploaded attachments are missing');
  await page.screenshot({path: '.cache/vmi-analysis/agent_panel_desktop.png'});

  const suggestions = composer.getByRole('listbox', {name: 'Suggestions'});
  await prompt.fill('/mo');
  await suggestions.getByRole('option').filter({has: page.getByText('/model', {exact: true})}).waitFor();
  await prompt.press('Tab');
  check(await prompt.inputValue() === '/model ', 'Slash command did not complete with Tab');
  await suggestions.waitFor({state: 'hidden'});
  await prompt.fill('/');
  await suggestions.waitFor({state: 'visible'});
  await prompt.press('Escape');
  await suggestions.waitFor({state: 'hidden'});
  await prompt.fill('');

  const timeline = panel.locator('.vmi-agent-timeline');
  const tabPanel = panel.locator('.vmi-agent-page');
  check(await timeline.evaluate(element => element.scrollHeight <= element.clientHeight + 1),
    'The transcript has a nested scrollbar');
  check(Math.abs((await timeline.boundingBox()).width - (await tabPanel.boundingBox()).width) < 2,
    'The conversation does not fill the work area');

  const addMessages = async count => panel.locator('.vmi-agent-entries').evaluate((entries, count) => {
    for (let i = 0; i < count; i++) {
      const entry = document.createElement('div');
      entry.className = 'vmi-agent-entry';
      entry.style.minHeight = '42px';
      entry.textContent = 'Conversation scrolling test';
      entries.appendChild(entry);
    }
  }, count);
  const atBottom = () => {
    const scroller = document.scrollingElement;
    return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 5;
  };

  await addMessages(40);
  await page.waitForFunction(atBottom);
  await page.evaluate(() => scrollBy(0, -400));
  const latest = panel.getByRole('button', {name: 'Latest messages', exact: true});
  await latest.waitFor({state: 'visible'});
  const anchored = await page.evaluate(() => scrollY);
  const before = await composer.boundingBox();
  check(before.y + before.height <= 1000, 'Composer is clipped while reading earlier messages');
  await addMessages(10);
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  check(Math.abs((await page.evaluate(() => scrollY)) - anchored) <= 1,
    'New messages move the page while reading earlier messages');
  await latest.press('Enter');
  await page.waitForFunction(atBottom);
  await latest.waitFor({state: 'hidden'});
  check(await prompt.evaluate(element => document.activeElement === element),
    'Keyboard focus was lost after using Latest messages');

  await page.setViewportSize({width: 390, height: 844});
  const composerBox = await composer.boundingBox();
  check(composerBox.x >= 0 && composerBox.x + composerBox.width <= 390, 'Composer overflows the narrow viewport');
  check(await connectionButton.isVisible(), 'Connection is hidden on narrow screens');
  check(!await composer.getByText('Enter to send', {exact: false}).isVisible(), 'Keyboard hint crowds the composer');
  check(await composer.getByRole('button', {name: 'Attach file or folder', exact: true}).isVisible(),
    'Attachment control is missing');
  await checkToolbarAlignment();
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
  return {passed: true, checks: ['title action', 'keyboard drawer', 'visible composer', 'screenshot paste', 'file drop', 'page scroll', 'latest messages', 'narrow layout']};
}
