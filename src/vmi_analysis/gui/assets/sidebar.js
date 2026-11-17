for (const group of document.querySelectorAll('.vmi-sidebar-disclosure')) {
  if (group.dataset.workspaceReady) continue;
  group.dataset.workspaceReady = 'true';
  const key = `vmi-workspace.${group.dataset.workspaceGroup}.open`;
  try {
    group.open = localStorage.getItem(key) === 'true';
    group.addEventListener('toggle', () => {
      localStorage.setItem(key, String(group.open));
    });
  } catch {
    // The disclosures still work when browser storage is disabled.
  }
}
