
(() => {
  const tabs = document.querySelector('.vmi-tab-panels');
  if (!tabs || tabs.dataset.agentChatReady) return;
  tabs.dataset.agentChatReady = 'true';
  let page;
  let disconnect;

  let serial = 0;
  let completionResults;
  window.vmiAgentCompletionResults = (...args) => completionResults?.(...args);

  function setupCompletion(prompt, menu) {
    if (!menu) return () => {};
    const eventName = menu.dataset.agentEvent;
    const promptId = Number(menu.dataset.agentPrompt);
    let request;
    let timer;
    let items = [];
    let index = 0;

    prompt.setAttribute('aria-autocomplete', 'list');
    prompt.setAttribute('aria-controls', menu.id);

    function hide() {
      clearTimeout(timer);
      request = undefined;
      items = [];
      menu.hidden = true;
      menu.replaceChildren();
      prompt.removeAttribute('aria-activedescendant');
    }

    function token() {
      if (prompt.selectionStart !== prompt.selectionEnd) return;
      const end = prompt.selectionStart;
      const before = prompt.value.slice(0, end);
      const command = /^\/([\w:.-]*)$/.exec(before);
      if (command) return {kind: '/', query: command[1], start: 0, end};
      const path = /(?:^|\s)@(?:"([^"]*)|([^\s@"]*))$/.exec(before);
      if (path) {
        return {kind: '@', query: path[1] ?? path[2], start: path.index + (path[0][0] === '@' ? 0 : 1), end};
      }
    }

    function schedule() {
      const next = document.activeElement === prompt ? token() : undefined;
      if (next && request && next.kind === request.kind && next.query === request.query &&
          next.start === request.start && next.end === request.end) return;
      hide();
      if (!next) return;
      request = {...next, id: ++serial};
      timer = setTimeout(() => emitEvent(eventName, request), 115);
    }

    function select(indexToSelect) {
      if (!request || !items[indexToSelect]) return;
      const {kind, start, end} = request;
      const choice = items[indexToSelect].value;
      hide();
      if (kind === '@') emitEvent(eventName, {kind: 'attach', query: choice});
      prompt.setRangeText(kind === '/' ? choice + ' ' : '', start, end, 'end');
      prompt.dispatchEvent(new Event('input', {bubbles:true}));
      prompt.focus({preventScroll:true});
    }

    function highlight(nextIndex) {
      index = (nextIndex + items.length) % items.length;
      for (const [i, option] of Array.from(menu.children).entries()) {
        option.setAttribute('aria-selected', String(i === index));
      }
      const current = menu.children[index];
      prompt.setAttribute('aria-activedescendant', current.id);
      current.scrollIntoView({block:'nearest'});
    }

    completionResults = (id, requestId, choices) => {
      if (id !== promptId || request?.id !== requestId || document.activeElement !== prompt) return;
      if (!choices.length) return hide();
      items = choices;
      menu.replaceChildren();
      choices.forEach((choice, i) => {
        const option = document.createElement('div');
        option.id = menu.id + '-' + i;
        option.setAttribute('role', 'option');
        option.className = 'vmi-agent-completion-option';
        const label = document.createElement('span');
        label.textContent = choice.label;
        option.append(label);
        if (choice.detail) {
          const detail = document.createElement('span');
          detail.className = 'vmi-agent-completion-detail';
          detail.textContent = choice.detail;
          option.append(detail);
        }
        option.addEventListener('pointerdown', event => event.preventDefault());
        option.addEventListener('click', () => select(i));
        menu.append(option);
      });
      menu.hidden = false;
      highlight(0);
    };

    function keydown(event) {
      if (event.isComposing) return;
      const key = event.key;
      if (menu.hidden) {
        if (request && key === 'Enter' && !event.shiftKey) {
          event.preventDefault();
          event.stopImmediatePropagation();
        }
        return;
      }
      if (!['ArrowDown','ArrowUp','Enter','Tab','Escape'].includes(key) || event.shiftKey) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      if (key === 'ArrowDown') highlight(index + 1);
      if (key === 'ArrowUp') highlight(index - 1);
      if (key === 'Escape') hide();
      if (key === 'Enter' || key === 'Tab') select(index);
    }

    function onCursor(event) {
      if (!event.ctrlKey && !event.metaKey && !event.altKey &&
          (event.key.length === 1 || ['ArrowLeft','ArrowRight','Home','End','Backspace','Delete'].includes(event.key))) {
        schedule();
      }
    }

    const afterInput = () => setTimeout(schedule, 0);
    prompt.addEventListener('input', afterInput, true);
    prompt.addEventListener('click', schedule);
    prompt.addEventListener('focus', schedule);
    prompt.addEventListener('blur', hide);
    prompt.addEventListener('keydown', keydown, true);
    prompt.addEventListener('keyup', onCursor);
    return () => {
      completionResults = undefined;
      hide();
      prompt.removeEventListener('input', afterInput, true);
      prompt.removeEventListener('click', schedule);
      prompt.removeEventListener('focus', schedule);
      prompt.removeEventListener('blur', hide);
      prompt.removeEventListener('keydown', keydown, true);
      prompt.removeEventListener('keyup', onCursor);
    };
  }

  function watchConversation() {
    const next = tabs.querySelector('.vmi-agent-page');
    if (next === page) return;
    disconnect?.();
    page = next;
    if (!page) return;

    const entries = page.querySelector('.vmi-agent-entries');
    const latest = page.querySelector('.vmi-agent-latest');
    const prompt = page.querySelector('.vmi-agent-prompt textarea');
    const stopCompletion = setupCompletion(prompt, page.querySelector('.vmi-agent-completions'));
    const scroller = document.scrollingElement;
    let follow = true;

    function updateLatest() {
      follow = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 96;
      latest.hidden = follow || !entries.querySelector('.vmi-agent-entry');
    }

    function jumpToLatest() {
      if (document.activeElement === latest) prompt.focus({preventScroll: true});
      scroller.scrollTop = scroller.scrollHeight;
      updateLatest();
    }

    const observer = new ResizeObserver(() => {
      if (follow && entries.querySelector('.vmi-agent-entry')) jumpToLatest();
      else updateLatest();
    });
    observer.observe(entries);
    window.addEventListener('scroll', updateLatest, {passive: true});
    latest.addEventListener('click', jumpToLatest);
    disconnect = () => {
      stopCompletion();
      observer.disconnect();
      window.removeEventListener('scroll', updateLatest);
      latest.removeEventListener('click', jumpToLatest);
    };
  }

  new MutationObserver(watchConversation).observe(tabs, {childList: true, subtree: true});
  watchConversation();
})();
