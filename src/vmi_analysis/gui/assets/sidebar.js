(() => {
  const sidebar = document.querySelector('.vmi-sidebar');
  if (!sidebar || sidebar.dataset.sidebarReady) return;
  sidebar.dataset.sidebarReady = 'true';
  const toggle = document.querySelector('[data-sidebar-toggle]');
  const rail = sidebar.closest('.vmi-sidebar-rail');
  const resizer = rail.querySelector('.vmi-sidebar-resizer');
  const settingsLabel = sidebar.querySelector('.vmi-sidebar-settings .block');
  const groups = [...sidebar.querySelectorAll('.vmi-sidebar-disclosure')];
  const states = new Map();
  const desktop = matchMedia('(min-width: 901px)');
  const read = key => { try { return localStorage.getItem(key); } catch { return null; } };
  const write = (key, value) => { try { localStorage.setItem(key, String(value)); } catch {} };
  const collapseKey = 'vmi-workspace.sidebar.collapsed';
  const widthKey = 'vmi-workspace.sidebar.width';
  let preference = read(collapseKey) === 'true';
  let preferredWidth = Number(read(widthKey)) || null;
  let collapsed = false;
  let activeBeforeCollapse = null;

  for (const group of groups) {
    group.dataset.workspaceReady = 'true';
    const key = `vmi-workspace.${group.dataset.workspaceGroup}.open`;
    const saved = read(key);
    if (saved !== null) group.open = saved === 'true';
    states.set(group, group.open);
    group.addEventListener('toggle', () => {
      if (collapsed) {
        group.open = true;
        return;
      }
      states.set(group, group.open);
      write(key, group.open);
    });
  }

  function apply() {
    const next = preference && desktop.matches;
    if (next !== collapsed) {
      const active = sidebar.querySelector('.q-tab--active, .vmi-sidebar-primary[aria-current="page"]');
      const activeName = active?.getAttribute('aria-label') ?? active?.textContent.trim();
      if (next) {
        groups.forEach(group => states.set(group, group.open));
        activeBeforeCollapse = activeName;
      }
      collapsed = next;
      groups.forEach(group => group.open = collapsed || states.get(group));
      // Reveal a tool selected while only icons were shown.
      if (!collapsed && activeName !== activeBeforeCollapse) {
        for (let group = active?.closest('.vmi-sidebar-disclosure'); group;
             group = group.parentElement.closest('.vmi-sidebar-disclosure')) {
          group.open = true;
        }
      }
    }
    sidebar.classList.toggle('vmi-sidebar--collapsed', collapsed);
    // Quasar's .block utility uses display: block !important; match its priority.
    if (settingsLabel) {
      if (collapsed) settingsLabel.style.setProperty('display', 'none', 'important');
      else settingsLabel.style.removeProperty('display');
    }
    const label = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
    toggle.setAttribute('aria-label', label);
    toggle.setAttribute('aria-expanded', String(!collapsed));
    toggle.setAttribute('aria-controls', sidebar.id);
    toggle.title = label;
    applyWidth();
  }

  const minWidth = 192;
  const maxWidth = () => Math.min(512, Math.floor(innerWidth * 0.45));
  const clampWidth = width => Math.max(minWidth, Math.min(maxWidth(), Math.round(width)));
  function applyWidth() {
    if (collapsed || !desktop.matches) return;
    if (preferredWidth !== null) {
      rail.style.setProperty('--vmi-sidebar-w', `${clampWidth(preferredWidth)}px`);
    }
    resizer.setAttribute('aria-valuemin', String(minWidth));
    resizer.setAttribute('aria-valuemax', String(maxWidth()));
    resizer.setAttribute('aria-valuenow', String(Math.round(rail.getBoundingClientRect().width)));
  }

  resizer.setAttribute('aria-controls', sidebar.id);
  let drag = null;
  function resize(width) {
    preferredWidth = clampWidth(width);
    applyWidth();
  }
  resizer.addEventListener('pointerdown', event => {
    if (event.button !== 0 || drag) return;
    drag = {id: event.pointerId, x: event.clientX, width: rail.getBoundingClientRect().width};
    resizer.setPointerCapture(drag.id);
    document.body.classList.add('vmi-sidebar-resizing');
  });
  resizer.addEventListener('pointermove', event => {
    if (drag?.id === event.pointerId) resize(drag.width + event.clientX - drag.x);
  });
  function endDrag(event) {
    if (drag?.id !== event.pointerId) return;
    drag = null;
    document.body.classList.remove('vmi-sidebar-resizing');
    if (preferredWidth !== null) write(widthKey, preferredWidth);
  }
  for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    resizer.addEventListener(name, endDrag);
  }
  resizer.addEventListener('keydown', event => {
    const width = rail.getBoundingClientRect().width;
    const targets = {ArrowLeft: width - 16, ArrowRight: width + 16, Home: minWidth, End: maxWidth()};
    if (!Object.hasOwn(targets, event.key)) return;
    event.preventDefault();
    resize(targets[event.key]);
    write(widthKey, preferredWidth);
  });
  window.addEventListener('resize', applyWidth);
  toggle.addEventListener('click', () => {
    preference = !preference;
    write(collapseKey, preference);
    apply();
  });
  desktop.addEventListener('change', apply);
  apply();
})();
