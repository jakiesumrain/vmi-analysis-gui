window.vmiAgentScroller = id => {
  const root = getHtmlElement(id);
  return root ? root.querySelector('.q-scrollarea__container') : null;
};

window.vmiAgentNearBottom = scroller =>
  scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 96;

window.vmiAgentUpdateLatest = (scroller, button) => {
  const visible = scroller.scrollHeight > scroller.clientHeight && !window.vmiAgentNearBottom(scroller);
  button.classList.toggle('vmi-agent-latest-visible', visible);
  button.setAttribute('aria-hidden', String(!visible));
  button.tabIndex = visible ? 0 : -1;
};

window.vmiAgentSyncLatest = (scroller, latestId) => {
  const button = getHtmlElement(latestId);
  if (button) window.vmiAgentUpdateLatest(scroller, button);
};

document.addEventListener('scroll', event => {
  const scroller = event.target;
  if (!(scroller instanceof HTMLElement) || !scroller.classList.contains('q-scrollarea__container')) return;
  const timeline = scroller.closest('.vmi-agent-timeline');
  const button = timeline ? timeline.querySelector('.vmi-agent-latest') : null;
  if (button) window.vmiAgentUpdateLatest(scroller, button);
}, {capture: true, passive: true});

window.vmiAgentChatChanged = (transcriptId, latestId) => {
  const scroller = window.vmiAgentScroller(transcriptId);
  if (!scroller) return;
  if (window.vmiAgentNearBottom(scroller)) {
    scroller.scrollTo({
      top: scroller.scrollHeight,
      behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
    });
  }
  requestAnimationFrame(() => window.vmiAgentSyncLatest(scroller, latestId));
};

window.vmiAgentChatLatest = (transcriptId, latestId) => {
  const scroller = window.vmiAgentScroller(transcriptId);
  if (!scroller) return;
  scroller.scrollTo({
    top: scroller.scrollHeight,
    behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
  });
  window.vmiAgentSyncLatest(scroller, latestId);
};
