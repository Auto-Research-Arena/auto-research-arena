(() => {
  'use strict';
  // Keep expandable tables compact for linear reading, but open either one
  // when a reader follows an explicit link to it.
  const expandableTables = [...document.querySelectorAll('.metric-table-details[id]')];
  const openFromHash = () => {
    const target = expandableTables.find((table) => window.location.hash === `#${table.id}`);
    if (target) target.open = true;
  };
  expandableTables.forEach((table) => {
    document.querySelectorAll(`a[href="#${table.id}"]`).forEach((link) => {
      link.addEventListener('click', () => { table.open = true; });
    });
  });
  window.addEventListener('hashchange', openFromHash);
  openFromHash();

  // Print the full specification without changing the reader's saved state.
  let printClosed = [];
  window.addEventListener('beforeprint', () => {
    printClosed = [...document.querySelectorAll('.metric-table-details:not([open])')];
    printClosed.forEach((details) => { details.open = true; });
  });
  window.addEventListener('afterprint', () => {
    printClosed.forEach((details) => { details.open = false; });
    printClosed = [];
  });

  document.querySelectorAll('[data-metric-explorer]').forEach((widget) => {
    const picker = widget.querySelector('.metric-picker');
    const tabs = [...picker.querySelectorAll('[role="tab"]')];
    const panels = tabs.map((tab) => widget.querySelector(`#${tab.getAttribute('aria-controls')}`));
    // Keep the complete static reading view if a panel is missing.
    if (panels.some((panel) => !panel)) return;

    const alignedRows = [
      ['--metric-title-row', 'h4'],
      ['--metric-purpose-row', '.metric-purpose'],
      ['--metric-meaning-row', '.metric-meaning'],
      ['--metric-optimize-row', '.metric-contract > div:nth-child(1)'],
      ['--metric-quality-row', '.metric-contract > div:nth-child(2)'],
      ['--metric-limits-row', '.metric-contract > div:nth-child(3)'],
    ];
    let alignmentFrame = 0;
    const alignPanelRows = () => {
      window.cancelAnimationFrame(alignmentFrame);
      alignmentFrame = window.requestAnimationFrame(() => {
        const container = widget.closest('details');
        if (container && !container.open) return;
        alignedRows.forEach(([property]) => widget.style.removeProperty(property));
        alignedRows.forEach(([property, selector]) => {
          const height = Math.max(...panels.map((panel) => panel.querySelector(selector).getBoundingClientRect().height));
          widget.style.setProperty(property, `${Math.ceil(height)}px`);
        });
      });
    };

    const select = (tab, { focus = false, scroll = false } = {}) => {
      tabs.forEach((candidate, index) => {
        const active = candidate === tab;
        candidate.setAttribute('aria-selected', String(active));
        candidate.tabIndex = active ? 0 : -1;
        panels[index].hidden = false;
        panels[index].classList.toggle('is-active', active);
        panels[index].setAttribute('aria-hidden', String(!active));
        panels[index].inert = !active;
      });
      if (focus) tab.focus();
      if (scroll && window.matchMedia('(max-width: 780px)').matches) {
        const behavior = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
        panels[tabs.indexOf(tab)].scrollIntoView({ behavior, block: 'start' });
      }
    };

    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => {
        select(tab, { scroll: true });
      });
      tab.addEventListener('keydown', (event) => {
        const keys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End'];
        if (!keys.includes(event.key)) return;
        event.preventDefault();
        let next = index;
        if (['ArrowUp', 'ArrowLeft'].includes(event.key)) next = (index - 1 + tabs.length) % tabs.length;
        if (['ArrowDown', 'ArrowRight'].includes(event.key)) next = (index + 1) % tabs.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = tabs.length - 1;
        select(tabs[next], { focus: true });
      });
    });
    picker.hidden = false;
    widget.classList.add('is-enhanced');
    select(tabs.find((tab) => tab.getAttribute('aria-selected') === 'true') || tabs[0]);
    const container = widget.closest('details');
    if (container) container.addEventListener('toggle', alignPanelRows);
    window.addEventListener('resize', alignPanelRows, { passive: true });
    if (document.fonts?.ready) document.fonts.ready.then(alignPanelRows);
    alignPanelRows();
  });
})();
