// playwright-cli run-code --filename=tests/browser/zoom_overlay.browser.js
// First open the generated zoom_check.html fixture served over localhost.
async page => {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  const near = (actual, expected, message) => check(Math.abs(actual - expected) < 2, `${message}: ${actual} vs ${expected}`);
  const image = page.locator('#vmi-zoom-img');
  const overlay = page.locator('#vmi-zoom-overlay');
  const select = page.getByRole('button', {name: 'Crop area', exact: true});
  const reset = page.getByRole('button', {name: 'reset', exact: true});
  const visible = () => overlay.isVisible();
  const box = () => image.boundingBox();
  const drag = async (x1, y1, x2, y2) => {
    await page.mouse.move(x1, y1);
    await page.mouse.down();
    await page.mouse.move(x2, y2, {steps: 8});
    await page.mouse.up();
  };
  const dimensions = () => image.evaluate(img => ({width: img.naturalWidth, height: img.naturalHeight, source: img.src}));
  const waitForImage = (width, height) => page.waitForFunction(({width, height}) => {
    const img = document.getElementById('vmi-zoom-img');
    return img.complete && img.naturalWidth === width && img.naturalHeight === height;
  }, {width, height});
  const resetFull = async () => { await reset.click(); await waitForImage(1000, 800); };
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({width: 1200, height: 900});
  await page.reload();
  await page.waitForFunction(() => document.getElementById('vmi-zoom-img').naturalWidth === 1000);
  const initial = await box();
  const originalSource = (await dimensions()).source;

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
  await waitForImage(250, 200);
  const cropped = await box();
  const firstCrop = await dimensions();
  check(await visible(), 'Crop release must not close overlay');
  check(firstCrop.source.startsWith('data:image/png'), 'Selection creates a cropped bitmap');
  near(cropped.x + cropped.width / 2, 600, 'Crop centered X');
  near(cropped.y + cropped.height / 2, 450, 'Crop centered Y');
  check(await page.locator('#vmi-zoom-pan').getAttribute('aria-pressed') === 'true', 'Returns to Pan after crop');
  check(!await page.locator('#vmi-zoom-selection').isVisible(), 'Preview clears after crop');
  const pixel = await image.evaluate(img => {
    const canvas = document.createElement('canvas'); canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const context = canvas.getContext('2d'); context.drawImage(img, 0, 0);
    return [...context.getImageData(0, 0, 1, 1).data];
  });
  check(pixel.join(',') === '204,120,92,255', 'Crop contains the selected orange region');
  await page.screenshot({path: '.cache/vmi-analysis/crop_check.png'});

  // Panning and zooming out must never reveal the excluded pixels again.
  await drag(600, 450, 640, 480);
  const panned = await box();
  near(panned.x, cropped.x + 40, 'Pan X'); near(panned.y, cropped.y + 30, 'Pan Y');
  check(await visible(), 'Pan release must not close overlay');
  await select.click();
  await drag(720, 560, 480, 320);
  await waitForImage(60, 60);
  check((await dimensions()).source !== firstCrop.source, 'Repeated crop creates a smaller bitmap');
  const repeated = await box();
  near(repeated.x + repeated.width / 2, 600, 'Repeated crop centered X');
  near(repeated.y + repeated.height / 2, 450, 'Repeated crop centered Y');
  await page.mouse.move(600, 450); await page.mouse.wheel(0, -120);
  await page.waitForFunction(width => document.getElementById('vmi-zoom-img').getBoundingClientRect().width > width, repeated.width);
  const secondSource = (await dimensions()).source;
  for (let i = 0; i < 8; i++) await page.mouse.wheel(0, 120);
  await page.waitForFunction(width => document.getElementById('vmi-zoom-img').getBoundingClientRect().width < width, repeated.width);
  check((await dimensions()).width === 60 && (await dimensions()).source === secondSource,
        'Zooming out keeps only the crop; outside pixels do not return');

  // Tiny selections are ignored; Escape cancels mode before closing the overlay.
  await resetFull();
  check((await dimensions()).source === originalSource, 'Reset after repeated crop restores original source');
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
  check((await dimensions()).source === originalSource, 'Reopening uses untouched original source');

  // Rectangle dragged beyond the image is clamped to image pixels.
  await select.click();
  await page.mouse.move(300, 200); await page.mouse.down();
  await page.mouse.move(20, 30, {steps: 8});
  const preview = await page.locator('#vmi-zoom-selection').boundingBox();
  near(preview.x, initial.x, 'Selection clamps to image left');
  near(preview.y, initial.y, 'Selection clamps to image top');
  await page.mouse.up();
  check(await visible(), 'Clamped selection remains open');
  await waitForImage(200, 150);
  await resetFull();

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
  await resetFull();
  near((await box()).width, initial.width, 'Reset restores full image');
  check((await dimensions()).source === originalSource, 'Reset restores original pixels and source');
  check(errors.length === 0, `No JavaScript errors: ${errors.join('; ')}`);
  await page.screenshot({path: '.cache/vmi-analysis/zoom_check.png'});
  console.log('Passed crop inspection checks: selected bitmap dimensions/pixels, reverse/repeated crop, outside pixels stay excluded when zooming out, pan, wheel, tiny/out-of-bounds/cancelled selection, Reset, and original-source restoration.');
}
