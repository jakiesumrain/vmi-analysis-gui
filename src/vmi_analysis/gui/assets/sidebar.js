(() => {
  const sidebar = document.querySelector('.vmi-sidebar');
  if (!sidebar || sidebar.dataset.sidebarReady) return;
  sidebar.dataset.sidebarReady = 'true';
  const toggle = sidebar.querySelector('[data-sidebar-toggle]');
  const settingsLabel = sidebar.querySelector('.vmi-sidebar-settings .block');
  const groups = [...sidebar.querySelectorAll('.vmi-sidebar-disclosure')];
  const states = new Map();
  const desktop = matchMedia('(min-width: 901px)');
  const read = (key, fallback = false) => {
    try {
      const value = localStorage.getItem(key);
      return value === null ? fallback : value === 'true';
    } catch { return fallback; }
  };
  const write = (key, value) => { try { localStorage.setItem(key, String(value)); } catch {} };
  const collapseKey = 'vmi-workspace.sidebar.collapsed';
  let preference = read(collapseKey);
  let collapsed = false;
  let activeBeforeCollapse = null;

  for (const group of groups) {
    group.dataset.workspaceReady = 'true';
    const key = `vmi-workspace.${group.dataset.workspaceGroup}.open`;
    group.open = read(key, group.open);
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
      const active = sidebar.querySelector('.q-tab--active');
      if (next) {
        groups.forEach(group => states.set(group, group.open));
        activeBeforeCollapse = active?.getAttribute('aria-label');
      }
      collapsed = next;
      groups.forEach(group => group.open = collapsed || states.get(group));
      // Reveal a tool selected while only icons were shown.
      if (!collapsed && active?.getAttribute('aria-label') !== activeBeforeCollapse) {
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
  }
  toggle.addEventListener('click', () => {
    preference = !preference;
    write(collapseKey, preference);
    apply();
  });
  desktop.addEventListener('change', apply);
  apply();
})();
