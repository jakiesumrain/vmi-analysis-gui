// playwright-cli run-code --filename=tests/zoom_overlay.browser.js
// First open the generated zoom_check.html fixture served over localhost.
async page => {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  const near = (actual, expected, message) => check(Math.abs(actual - expected) < 2, `${message}: ${actual} vs ${expected}`);
  const image = page.locator('#vmi-zoom-img');
  const overlay = page.locator('#vmi-zoom-overlay');
  const select = page.getByRole('button', {name: 'Select area', exact: true});
  const reset = page.getByRole('button', {name: 'reset', exact: true});
  const visible = () => overlay.isVisible();
  const box = () => image.boundingBox();
  const drag = async (x1, y1, x2, y2) => {
    await page.mouse.move(x1, y1);
    await page.mouse.down();
    await page.mouse.move(x2, y2, {steps: 8});
    await page.mouse.up();
  };
  const centerCheck = (before, after, x, y) => {
    const fractionX = (x - before.x) / before.width, fractionY = (y - before.y) / before.height;
    const viewport = page.viewportSize();
    near(after.x + fractionX * after.width, viewport.width / 2, 'Selected image X stays centered');
    near(after.y + fractionY * after.height, viewport.height / 2, 'Selected image Y stays centered');
  };
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({width: 1200, height: 900});
  await page.reload();
  await page.waitForFunction(() => document.getElementById('vmi-zoom-img').naturalWidth === 1000);
  const initial = await box();

  // Normal click on the image must not close the overlay after pointer capture.
  await page.mouse.click(600, 450);
  check(await visible(), 'Image click keeps overlay open');
  await select.click();
  const x1 = initial.x + initial.width * .2, y1 = initial.y + initial.height * .1875;
  const x2 = initial.x + initial.width * .45, y2 = initial.y + initial.height * .4375;
  await page.mouse.move(x1, y1); await page.mouse.down();
  await page.mouse.move(x2, y2, {steps: 8});
  check(await page.locator('#vmi-zoom-selection').isVisible(), 'Selection rectangle appears while dragging');
  await page.mouse.up();
  const zoomed = await box();
  check(await visible(), 'Selection release must not close overlay');
  check(zoomed.width > initial.width * 3, 'Selection magnifies the region');
  centerCheck(initial, zoomed, (x1 + x2) / 2, (y1 + y2) / 2);
  check(await page.locator('#vmi-zoom-pan').getAttribute('aria-pressed') === 'true', 'Returns to Pan after selection');
  check(!await page.locator('#vmi-zoom-selection').isVisible(), 'Preview clears after selection');

  // Pan after zoom, then select backwards in the transformed image.
  await drag(600, 450, 640, 480);
  const panned = await box();
  near(panned.x, zoomed.x + 40, 'Pan X'); near(panned.y, zoomed.y + 30, 'Pan Y');
  check(await visible(), 'Pan release must not close overlay');
  await select.click();
  await drag(720, 560, 480, 320);
  const repeated = await box();
  check(repeated.width > panned.width, 'Second selection magnifies an already zoomed image');
  centerCheck(panned, repeated, 600, 440);
  await page.mouse.move(600, 450); await page.mouse.wheel(0, -120);
  await page.waitForFunction(width => document.getElementById('vmi-zoom-img').getBoundingClientRect().width > width, repeated.width);

  // Tiny selections are ignored; Escape cancels mode before closing the overlay.
  await reset.click();
  await select.click();
  const beforeTiny = await box();
  await drag(600, 450, 602, 452);
  near((await box()).width, beforeTiny.width, 'Tiny selection does not zoom');
  check(await visible(), 'Tiny selection does not close overlay');
  await page.keyboard.press('Escape');
  check(await visible(), 'First Escape cancels selection');
  await page.keyboard.press('Escape');
  check(!await visible(), 'Second Escape closes overlay');
  await page.evaluate(() => openFixture());
  await page.waitForFunction(() => document.getElementById('vmi-zoom-img').naturalWidth === 1000);
  near((await box()).width, initial.width, 'Reopening resets the image');

  // Rectangle dragged beyond the image is clamped to image pixels.
  await select.click();
  await page.mouse.move(300, 200); await page.mouse.down();
  await page.mouse.move(20, 30, {steps: 8});
  const preview = await page.locator('#vmi-zoom-selection').boundingBox();
  near(preview.x, initial.x, 'Selection clamps to image left');
  near(preview.y, initial.y, 'Selection clamps to image top');
  await page.mouse.up();
  check(await visible(), 'Clamped selection remains open');
  await reset.click();

  // Escape during an in-progress drag discards the rectangle.
  await select.click();
  await page.mouse.move(300, 200); await page.mouse.down();
  await page.mouse.move(550, 400, {steps: 8});
  await page.keyboard.press('Escape');
  await page.mouse.up();
  check(await visible(), 'Cancelling drag does not close overlay');
  near((await box()).width, initial.width, 'Cancelled drag does not zoom');
  check(!await page.locator('#vmi-zoom-selection').isVisible(), 'Cancelled preview clears');

  await page.getByRole('button', {name: '+', exact: true}).click();
  check((await box()).width > initial.width, 'Toolbar zoom still works');
  await reset.click();
  near((await box()).width, initial.width, 'Reset restores full image');
  check(errors.length === 0, `No JavaScript errors: ${errors.join('; ')}`);
  await page.screenshot({path: 'gui/generated/zoom_check.png'});
  console.log('Passed image overlay checks: rectangle/reverse/repeated selection, pan, wheel, tiny/out-of-bounds/cancelled selection, reset, reopen, and no JavaScript errors.');
}
