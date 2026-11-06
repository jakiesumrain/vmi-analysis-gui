// Run against picker_fixture.py, which uses an isolated temporary directory.
async page => {
  const check = (ok, message) => { if (!ok) throw new Error(message); };
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  const suffix = String(Date.now());
  const created = 'Analysis ' + suffix;
  await page.getByRole('button', {name: 'Choose folder', exact: true}).click();
  await page.getByRole('button', {name: 'New folder', exact: true}).click();
  const input = page.getByRole('textbox', {name: 'Folder name', exact: true});
  await input.fill('../outside');
  await page.getByRole('button', {name: 'Create', exact: true}).click();
  await page.getByText('Could not create folder:', {exact: false}).waitFor();
  check(await input.isVisible(), 'Invalid input keeps creation dialog open');
  await input.fill('Existing');
  await page.getByRole('button', {name: 'Create', exact: true}).click();
  await page.getByText('A file or folder with this name already exists.', {exact: true}).waitFor();
  await input.fill(created);
  await input.press('Enter');
  await input.waitFor({state: 'hidden'});
  await page.getByRole('button', {name: 'Select this folder', exact: true}).click();
  await page.waitForFunction(name => document.querySelector('#picker-result').textContent.includes(name), created);
  const selection = await page.locator('#picker-result').textContent();
  check(selection.replaceAll('\\', '/').endsWith('/' + created), 'New folder is opened and selectable');

  // Cancelling creation leaves the outer picker open and its location unchanged.
  await page.getByRole('button', {name: 'Choose folder', exact: true}).click();
  await page.getByRole('button', {name: 'New folder', exact: true}).click();
  await input.fill('not-created');
  await page.getByRole('button', {name: 'Cancel', exact: true}).last().click();
  await input.waitFor({state: 'hidden'});
  await page.getByRole('button', {name: 'Select this folder', exact: true}).waitFor();
  await page.getByRole('button', {name: 'Cancel', exact: true}).click();

  // Folder creation is also available when choosing a file or saving a result.
  for (const [button, name] of [['Choose file', 'File inputs '], ['Save file', 'Saved results ']]) {
    await page.getByRole('button', {name: button, exact: true}).click();
    await page.getByRole('button', {name: 'New folder', exact: true}).click();
    await input.fill(name + suffix);
    await page.getByRole('button', {name: 'Create', exact: true}).click();
    await input.waitFor({state: 'hidden'});
    await page.getByRole('button', {name: 'Cancel', exact: true}).click();
  }
  await page.getByRole('button', {name: 'Choose folder', exact: true}).click();
  await page.screenshot({path: '.cache/vmi-analysis/picker_new_folder.png'});
  await page.getByRole('button', {name: 'Cancel', exact: true}).click();
  check(!errors.length, errors.join('; '));
  return {passed: true, checks: ['create and select', 'Enter key', 'invalid name', 'existing-name collision', 'cancel', 'file/save modes']};
}
