(() => {
  let drag = null;
  const lastHover = new WeakMap();
  const box = event => event.target instanceof HTMLImageElement && event.target.closest('[data-polar-drag]');
  const point = (event, image) => {
    const bounds = image.getBoundingClientRect();
    const clamp = value => Math.max(0, Math.min(1, value));
    return [clamp((event.clientX - bounds.left) / bounds.width), clamp((event.clientY - bounds.top) / bounds.height)];
  };
  const emit = (root, detail) => root.dispatchEvent(new CustomEvent('polar-pointer', {detail}));

  document.addEventListener('pointerdown', event => {
    const root = box(event);
    if (!root || root.dataset.polarDrag === 'off' || event.button !== 0 || drag || !event.target.naturalWidth) return;
    const image = event.target;
    const group = root.closest('[data-polar-drag-group]');
    const targets = group ? group.querySelectorAll('[data-polar-drag]') : [root];
    const drafts = Array.from(targets, target => {
      const draft = document.createElement('div');
      draft.className = 'polar-drag-draft';
      target.append(draft);
      return draft;
    });
    drag = {root, image, id: event.pointerId, mode: root.dataset.polarDrag,
      revision: root.dataset.polarRevision, start: point(event, image), drafts};
    image.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  document.addEventListener('pointermove', event => {
    if (drag) {
      if (event.pointerId !== drag.id) return;
      drag.end = point(event, drag.image);
      if (drag.frame) return;
      drag.frame = requestAnimationFrame(() => {
        const {mode, start: [x0, y0], end: [x1, y1], drafts} = drag;
        drag.frame = null;
        for (const draft of drafts) Object.assign(draft.style, {
          left: `${Math.min(x0, x1) * 100}%`, width: `${Math.abs(x1 - x0) * 100}%`,
          top: mode === 'full' ? '0' : `${Math.min(y0, y1) * 100}%`,
          height: mode === 'full' ? '100%' : `${Math.abs(y1 - y0) * 100}%`,
        });
      });
      return;
    }
    const root = box(event);
    if (!root || root.dataset.polarDrag === 'off') return;
    const now = performance.now();
    if (now - (lastHover.get(root) ?? -Infinity) >= 75) {
      lastHover.set(root, now);
      emit(root, {kind: 'hover', point: point(event, event.target)});
    }
  });

  document.addEventListener('pointerout', event => {
    const root = box(event);
    if (root && !drag) emit(root, {kind: 'leave'});
  });

  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    document.addEventListener(type, event => {
      if (!drag || drag.id !== event.pointerId) return;
      const {root, image, mode, revision, start, drafts, frame} = drag;
      drag = null;
      if (frame) cancelAnimationFrame(frame);
      drafts.forEach(draft => draft.remove());
      if (type === 'pointerup' && mode === root.dataset.polarDrag && revision === root.dataset.polarRevision) {
        emit(root, {kind: 'select', start, end: point(event, image), mode, revision});
      }
    });
  }
})();
