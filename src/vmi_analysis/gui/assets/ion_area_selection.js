(() => {
  let drag;
  const clamp = (value, size) => Math.max(0, Math.min(size, value));
  const point = event => [
    clamp(event.clientX - drag.left, drag.width),
    clamp(event.clientY - drag.top, drag.height),
  ];

  document.addEventListener('pointerdown', event => {
    const root = event.target.closest?.('[data-ion-area="on"]');
    if (!root || !event.target.closest('.nsewdrag') || drag || event.button !== 0) return;
    const {xaxis: x, yaxis: y} = root._fullLayout ?? {};
    if (!x || !y || !x._length || !y._length) return;
    const bounds = root.getBoundingClientRect();
    const draft = document.createElement('div');
    draft.className = 'ion-area-draft';
    document.body.append(draft);
    drag = {
      root, id: event.pointerId, draft, x, y,
      left: bounds.left + x._offset, top: bounds.top + y._offset,
      width: x._length, height: y._length,
    };
    drag.start = point(event);
    root.setPointerCapture(event.pointerId);
    event.preventDefault();
    event.stopPropagation();
  }, true);

  document.addEventListener('pointermove', event => {
    if (drag?.id !== event.pointerId) return;
    drag.end = point(event);
    event.stopPropagation();
    if (drag.frame) return;
    drag.frame = requestAnimationFrame(() => {
      const {draft, start: [x0, y0], end: [x1, y1], left, top} = drag;
      drag.frame = null;
      Object.assign(draft.style, {
        left: `${left + Math.min(x0, x1)}px`,
        top: `${top + Math.min(y0, y1)}px`,
        width: `${Math.abs(x1 - x0)}px`,
        height: `${Math.abs(y1 - y0)}px`,
      });
    });
  }, true);

  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    document.addEventListener(type, event => {
      if (drag?.id !== event.pointerId) return;
      const [x1, y1] = point(event);
      const {root, x, y, draft, frame, start: [x0, y0]} = drag;
      drag = null;
      if (frame) cancelAnimationFrame(frame);
      draft.remove();
      if (type === 'pointerup' && root.dataset.ionArea === 'on'
          && Math.abs(x1 - x0) >= 2 && Math.abs(y1 - y0) >= 2) {
        root.dispatchEvent(new CustomEvent('ion-area-selected', {
          detail: {x: [x.p2d(x0), x.p2d(x1)], y: [y.p2d(y0), y.p2d(y1)]},
        }));
      }
    }, true);
  }
})();
