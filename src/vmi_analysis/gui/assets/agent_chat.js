
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
      const quoted = prompt.value[start + 1] === '"' || /[\s"\\]/.test(choice);
      const value = kind === '/' ? choice : '@' + (quoted ? JSON.stringify(choice) : choice);
      const separator = /\s/.test(prompt.value[end] ?? '') ? '' : ' ';
      hide();
      prompt.setRangeText(value + separator, start, end, 'end');
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
        label.className = 'vmi-agent-completion-label';
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

  async function uploadFiles(files, signal) {
    const form = new FormData();
    files.forEach(file => form.append('files', file, file.name));
    const response = await fetch('/agent-attachments', {
      method: 'POST', body: form, signal, credentials: 'same-origin',
    });
    let payload;
    try { payload = await response.json(); } catch { /* The server may return plain text. */ }
    if (!response.ok) {
      const detail = typeof payload?.detail === 'string' ? payload.detail : `Upload failed (${response.status}).`;
      throw new Error(detail);
    }
    if (!Array.isArray(payload?.paths)) throw new Error('The upload returned an invalid response.');
    return payload;
  }

  window.vmiAgentCapturePlot = async (element, filename) => {
    try {
      const plot = element?.$el;
      if (!plot?.getClientRects().length || !plot.classList.contains('js-plotly-plot')) {
        throw new Error('Wait for the plot to display before capturing it.');
      }
      const {Plotly} = await import('nicegui-plotly');
      const {width, height} = plot.getBoundingClientRect();
      const dataUrl = await Plotly.toImage(plot, {
        format: 'png', width: Math.round(width), height: Math.round(height), scale: 1.5,
      });
      const blob = await (await fetch(dataUrl)).blob();
      return await uploadFiles([new File([blob], filename, {type: 'image/png'})]);
    } catch (cause) {
      return {error: cause.message || 'Could not capture this plot.'};
    }
  };

  function setupAttachments(prompt) {
    const composer = prompt?.closest('.vmi-agent-composer');
    const eventName = composer?.dataset.agentUploadEvent;
    if (!composer || !eventName) return () => {};
    const maxFiles = Number(composer.dataset.agentMaxFiles);
    const maxFileBytes = Number(composer.dataset.agentMaxFileBytes);
    const maxBatchBytes = Number(composer.dataset.agentMaxBatchBytes);
    const controllers = new Set();
    let dragDepth = 0;
    let pending = 0;

    const report = payload => emitEvent(eventName, payload);
    const hasFiles = event => Array.from(event.dataTransfer?.types || []).includes('Files');

    function screenshot(file) {
      const extension = {'image/jpeg': 'jpg', 'image/gif': 'gif', 'image/webp': 'webp'}[file.type] || 'png';
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      return new File([file], `screenshot-${stamp}.${extension}`, {type: file.type || 'image/png'});
    }

    async function receive(fileList, {paste = false} = {}) {
      let files = Array.from(fileList || []);
      if (!files.length) return;
      if (paste) files = files.map(file =>
        file.type.startsWith('image/') && (!file.name || file.name === 'image.png') ? screenshot(file) : file);
      const total = files.reduce((sum, file) => sum + file.size, 0);
      let error = '';
      if (files.length > maxFiles) error = `Attach up to ${maxFiles} files at a time.`;
      else if (files.some(file => file.size > maxFileBytes)) error = 'Each attachment must be 25 MiB or smaller.';
      else if (total > maxBatchBytes) error = 'Keep each attachment batch below 100 MiB.';
      if (error) return report({action: 'error', message: error});

      const controller = new AbortController();
      controllers.add(controller);
      pending += 1;
      composer.setAttribute('aria-busy', 'true');
      report({action: 'start', count: files.length});
      try {
        const {paths} = await uploadFiles(files, controller.signal);
        report({action: 'add', paths});
        prompt.focus({preventScroll: true});
      } catch (cause) {
        if (cause.name !== 'AbortError') report({action: 'error', message: cause.message});
      } finally {
        controllers.delete(controller);
        pending -= 1;
        if (!pending) composer.setAttribute('aria-busy', 'false');
        report({action: 'finish'});
      }
    }

    function onPaste(event) {
      const files = Array.from(event.clipboardData?.files || []);
      if (!files.length) return;
      event.preventDefault();
      receive(files, {paste: true});
    }

    function onDragEnter(event) {
      if (!hasFiles(event)) return;
      event.preventDefault();
      dragDepth += 1;
      composer.classList.add('vmi-agent-composer-receiving');
    }

    function onDragOver(event) {
      if (!hasFiles(event)) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = 'copy';
      composer.classList.add('vmi-agent-composer-receiving');
    }

    function onDragLeave(event) {
      if (!hasFiles(event)) return;
      dragDepth = Math.max(0, dragDepth - 1);
      if (!dragDepth || !composer.contains(event.relatedTarget)) {
        dragDepth = 0;
        composer.classList.remove('vmi-agent-composer-receiving');
      }
    }

    function onDrop(event) {
      if (!hasFiles(event)) return;
      event.preventDefault();
      dragDepth = 0;
      composer.classList.remove('vmi-agent-composer-receiving');
      receive(event.dataTransfer.files);
    }

    prompt.addEventListener('paste', onPaste);
    composer.addEventListener('dragenter', onDragEnter);
    composer.addEventListener('dragover', onDragOver);
    composer.addEventListener('dragleave', onDragLeave);
    composer.addEventListener('drop', onDrop);
    return () => {
      controllers.forEach(controller => controller.abort());
      prompt.removeEventListener('paste', onPaste);
      composer.removeEventListener('dragenter', onDragEnter);
      composer.removeEventListener('dragover', onDragOver);
      composer.removeEventListener('dragleave', onDragLeave);
      composer.removeEventListener('drop', onDrop);
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
    const stopAttachments = setupAttachments(prompt);
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
      stopAttachments();
      observer.disconnect();
      window.removeEventListener('scroll', updateLatest);
      latest.removeEventListener('click', jumpToLatest);
    };
  }

  new MutationObserver(watchConversation).observe(tabs, {childList: true, subtree: true});
  watchConversation();
})();
