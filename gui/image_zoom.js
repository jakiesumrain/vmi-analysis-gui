(() => {
  const overlay = document.getElementById('vmi-zoom-overlay');
  const img = document.getElementById('vmi-zoom-img');
  const toolbar = document.getElementById('vmi-zoom-toolbar');
  const selection = document.getElementById('vmi-zoom-selection');
  const panButton = document.getElementById('vmi-zoom-pan');
  const selectButton = document.getElementById('vmi-zoom-select');
  const hint = document.getElementById('vmi-zoom-hint');
  let scale = 1, tx = 0, ty = 0;
  let selecting = false, gesture = null, suppressClick = false;
  let originalSrc = '', cropped = false;

  function apply() {
    img.style.transform = `translate(calc(-50% + ${tx}px), calc(-50% + ${ty}px)) scale(${scale})`;
  }

  function cursor() {
    overlay.style.cursor = selecting ? 'crosshair' : gesture ? 'grabbing' : 'grab';
  }

  function cancelGesture() {
    if (gesture && overlay.hasPointerCapture(gesture.id)) overlay.releasePointerCapture(gesture.id);
    gesture = null;
    selection.style.display = 'none';
    cursor();
  }

  function setMode(select) {
    cancelGesture();
    selecting = select;
    panButton.setAttribute('aria-pressed', String(!select));
    selectButton.setAttribute('aria-pressed', String(select));
    panButton.style.background = select ? '#efe9de' : '#cc785c';
    selectButton.style.background = select ? '#cc785c' : '#efe9de';
    hint.textContent = select ? 'Drag a rectangle to crop the inspection view · Esc cancels selection'
                      : cropped ? `Cropped view (${img.naturalWidth} × ${img.naturalHeight} pixels) · Reset restores full image`
                                : 'Drag to pan · Mouse wheel to zoom · Crop area to inspect only a region';
    cursor();
  }

  function fitImage() {
    if (!img.naturalWidth || !img.naturalHeight) return;
    setMode(false);
    scale = Math.min(cropped ? 30 : 1, overlay.clientWidth * .92 / img.naturalWidth,
                     Math.max(1, overlay.clientHeight - 100) / img.naturalHeight);
    tx = 0; ty = 0;
    apply();
  }

  window.vmiZoomOpen = src => {
    cancelGesture();
    setMode(false);
    suppressClick = false;
    originalSrc = src;
    cropped = false;
    overlay.style.display = 'block';
    img.src = src;
    // Opening the same cached image need not generate another load event.
    if (img.complete && img.naturalWidth) fitImage();
  };
  window.vmiZoomClose = () => {
    setMode(false);
    overlay.style.display = 'none';
    img.src = '';
    originalSrc = '';
    cropped = false;
  };
  window.vmiZoomReset = () => {
    if (cropped) {
      cropped = false;
      img.src = originalSrc;
    }
    fitImage();
  };
  window.vmiZoomSelect = () => setMode(true);
  window.vmiZoomPan = () => setMode(false);
  window.vmiZoomAt = (event, factor) => {
    if (gesture || !img.naturalWidth) return;
    const bounds = overlay.getBoundingClientRect();
    const cx = bounds.left + bounds.width / 2, cy = bounds.top + bounds.height / 2;
    const px = (event ? event.clientX : cx) - cx - tx;
    const py = (event ? event.clientY : cy) - cy - ty;
    const next = Math.min(30, Math.max(.01, scale * factor));
    tx -= px * (next / scale - 1);
    ty -= py * (next / scale - 1);
    scale = next;
    apply();
  };

  function point(event, bounds) {
    return {x: Math.max(bounds.left, Math.min(bounds.right, event.clientX)),
            y: Math.max(bounds.top, Math.min(bounds.bottom, event.clientY))};
  }

  function rectangle(event) {
    const end = point(event, gesture.bounds);
    return {left: Math.min(gesture.x, end.x), top: Math.min(gesture.y, end.y),
            width: Math.abs(end.x - gesture.x), height: Math.abs(end.y - gesture.y)};
  }

  function cropRectangle(box) {
    const bounds = img.getBoundingClientRect();
    // Convert the selected screen rectangle to pixels of the currently
    // displayed bitmap, accounting for any prior zoom, pan, or crop.
    const x = Math.max(0, Math.floor((box.left - bounds.left) * img.naturalWidth / bounds.width));
    const y = Math.max(0, Math.floor((box.top - bounds.top) * img.naturalHeight / bounds.height));
    const right = Math.min(img.naturalWidth, Math.ceil((box.left + box.width - bounds.left) * img.naturalWidth / bounds.width));
    const bottom = Math.min(img.naturalHeight, Math.ceil((box.top + box.height - bounds.top) * img.naturalHeight / bounds.height));
    const canvas = document.createElement('canvas');
    canvas.width = right - x;
    canvas.height = bottom - y;
    if (canvas.width < 1 || canvas.height < 1) return;
    try {
      canvas.getContext('2d').drawImage(img, x, y, canvas.width, canvas.height,
                                      0, 0, canvas.width, canvas.height);
      const source = canvas.toDataURL('image/png');
      cropped = true;
      img.src = source;
      if (img.complete && img.naturalWidth) fitImage();
    } catch (error) {
      // Keep the current view if an external image disallows canvas readback.
      hint.textContent = 'This image cannot be cropped in the browser. Pan and zoom are still available.';
    }
  }

  img.addEventListener('load', fitImage);
  img.addEventListener('dragstart', event => event.preventDefault());
  overlay.addEventListener('wheel', event => {
    event.preventDefault();
    if (!toolbar.contains(event.target)) window.vmiZoomAt(event, event.deltaY < 0 ? 1.2 : 1 / 1.2);
  }, {passive: false});

  overlay.addEventListener('pointerdown', event => {
    if (event.button !== 0 || gesture || toolbar.contains(event.target) || !img.naturalWidth) return;
    suppressClick = false;
    const viewport = overlay.getBoundingClientRect();
    const image = img.getBoundingClientRect();
    const bounds = {left: Math.max(viewport.left, image.left), top: Math.max(viewport.top, image.top),
                    right: Math.min(viewport.right, image.right), bottom: Math.min(viewport.bottom, image.bottom)};
    if (selecting && (event.clientX < bounds.left || event.clientX > bounds.right ||
                      event.clientY < bounds.top || event.clientY > bounds.bottom)) return;
    gesture = {id: event.pointerId, x: event.clientX, y: event.clientY,
               tx, ty, bounds, moved: false, backdrop: event.target === overlay};
    overlay.setPointerCapture(event.pointerId);
    event.preventDefault();
    cursor();
  });

  overlay.addEventListener('pointermove', event => {
    if (!gesture || event.pointerId !== gesture.id) return;
    gesture.moved ||= Math.hypot(event.clientX - gesture.x, event.clientY - gesture.y) > 3;
    if (selecting) {
      const box = rectangle(event), bounds = overlay.getBoundingClientRect();
      Object.assign(selection.style, {display: 'block', left: `${box.left - bounds.left}px`,
        top: `${box.top - bounds.top}px`, width: `${box.width}px`, height: `${box.height}px`});
    } else {
      tx = gesture.tx + event.clientX - gesture.x;
      ty = gesture.ty + event.clientY - gesture.y;
      apply();
    }
  });

  overlay.addEventListener('pointerup', event => {
    if (!gesture || event.pointerId !== gesture.id) return;
    const box = selecting ? rectangle(event) : null;
    suppressClick = selecting || gesture.moved || !gesture.backdrop;
    cancelGesture();
    if (box && box.width >= 8 && box.height >= 8) {
      setMode(false);
      cropRectangle(box);
    }
  });
  overlay.addEventListener('pointercancel', () => { suppressClick = true; cancelGesture(); });
  overlay.addEventListener('lostpointercapture', () => {
    if (gesture) { suppressClick = true; cancelGesture(); }
  });
  overlay.addEventListener('click', event => {
    if (suppressClick) { suppressClick = false; return; }
    if (event.target === overlay && !selecting) window.vmiZoomClose();
  });
  overlay.addEventListener('dblclick', event => {
    if (!toolbar.contains(event.target) && !selecting) window.vmiZoomReset();
  });
  window.addEventListener('blur', () => { if (gesture) suppressClick = true; cancelGesture(); });
  window.addEventListener('keydown', event => {
    if (event.key !== 'Escape' || overlay.style.display === 'none') return;
    if (gesture || selecting) {
      if (gesture) suppressClick = true;
      setMode(false);
    }
    else window.vmiZoomClose();
  });
})();
