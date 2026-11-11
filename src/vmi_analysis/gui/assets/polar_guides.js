(() => {
  const positions = new WeakMap();
  const groupSelector = '.polar-guide-group[data-polar-view]';
  const guideSelector = '[data-polar-guide]';
  const clamp = (value, low, high) => Math.max(low, Math.min(value, high));
  let drag = null;

  function update(group, change) {
    const [left, right, top, bottom, radialStep, angularStep] = group.dataset.polarView.split(' ').map(Number);
    const center = {x: (left + right) / 2, y: (top + bottom) / 2};
    const position = positions.get(group) ?? center;
    if (position.x < left || position.x > right) position.x = center.x;
    if (position.y < top || position.y > bottom) position.y = center.y;
    change?.(position, {left, right, top, bottom});
    position.x = clamp(position.x, left + 0.5, right - 0.5);
    position.y = clamp(position.y, top + 0.5, bottom - 0.5);
    positions.set(group, position);
    group.style.setProperty('--polar-guide-x', `${100 * (position.x - left) / (right - left)}%`);
    group.style.setProperty('--polar-guide-y', `${100 * (position.y - top) / (bottom - top)}%`);
    // Sample centers are half a bin from the image edge.
    const radius = (position.x - 0.5) * radialStep;
    const angle = (position.y - 0.5) * angularStep;
    group.querySelector('[data-polar-guide-readout]').textContent =
      `r ${radius.toFixed(3)} px · θ ${angle.toFixed(2)}°`;
  }

  document.addEventListener('pointerdown', event => {
    const guide = event.target.closest(guideSelector);
    const group = guide?.closest(groupSelector);
    if (!group || event.button !== 0 || drag) return;
    const line = guide.getBoundingClientRect();
    drag = {guide, group, id: event.pointerId,
      offsetX: event.clientX - (line.left + line.width / 2),
      offsetY: event.clientY - (line.top + line.height / 2)};
    guide.setPointerCapture(event.pointerId);
    event.preventDefault();
    guide.focus({preventScroll: true});
  });

  document.addEventListener('pointermove', event => {
    if (!drag || event.pointerId !== drag.id) return;
    const {guide, group, offsetX, offsetY} = drag;
    const bounds = guide.parentElement.getBoundingClientRect();
    const axis = guide.dataset.polarGuide;
    update(group, (position, {left, right, top, bottom}) => {
      if (axis.includes('x')) {
        const fraction = (event.clientX - offsetX - bounds.left) / bounds.width;
        position.x = left + fraction * (right - left);
      }
      if (axis.includes('y')) {
        const fraction = (event.clientY - offsetY - bounds.top) / bounds.height;
        position.y = top + fraction * (bottom - top);
      }
    });
  });

  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    document.addEventListener(type, event => {
      if (drag?.id === event.pointerId) drag = null;
    });
  }

  document.addEventListener('keydown', event => {
    const guide = event.target.closest(guideSelector);
    const group = guide?.closest(groupSelector);
    if (!group) return;
    const axis = guide.dataset.polarGuide;
    const dx = axis.includes('x') ? Number(event.key === 'ArrowRight') - Number(event.key === 'ArrowLeft') : 0;
    const dy = axis.includes('y') ? Number(event.key === 'ArrowDown') - Number(event.key === 'ArrowUp') : 0;
    if (!dx && !dy) return;
    event.preventDefault();
    update(group, position => {
      position.x += dx;
      position.y += dy;
    });
  });

  new MutationObserver(changes => {
    for (const {target} of changes) {
      if (target.matches(groupSelector)) update(target);
    }
  }).observe(document.body, {subtree: true, attributes: true, attributeFilter: ['data-polar-view']});
})();
