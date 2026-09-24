(() => {
  'use strict';
  document.querySelectorAll('[data-research-comparison]').forEach(widget => {
    const switcher = widget.querySelector('[data-research-switch]');
    if (!switcher) return;
    const tabs = [...switcher.querySelectorAll('[role="tab"]')];
    const panels = tabs.map(tab => document.getElementById(tab.getAttribute('aria-controls')));
    // Preserve both static tables if the widget is incomplete or JS is disabled.
    if (tabs.length !== 2 || panels.some(panel => !panel || !widget.contains(panel))) return;
    const select = (selected, focus = false) => {
      tabs.forEach((tab, index) => {
        const active = tab === selected;
        tab.setAttribute('aria-selected', String(active));
        tab.tabIndex = active ? 0 : -1;
        panels[index].hidden = !active;
      });
      if (focus) selected.focus();
    };
    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => select(tab));
      tab.addEventListener('keydown', event => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 :
          (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
        select(tabs[next], true);
      });
    });
    select(tabs.find(tab => tab.getAttribute('aria-selected') === 'true') || tabs[0]);
    switcher.hidden = false;
    widget.classList.add('is-enhanced');
  });
  // Print the explanations in both tables, then restore the reader's state.
  const connections = [...document.querySelectorAll('[data-connection-details]')];
  let printState;
  addEventListener('beforeprint', () => {
    if (printState) return;
    printState = connections.map(item => item.open);
    connections.forEach(item => { item.open = true; });
  });
  addEventListener('afterprint', () => {
    if (!printState) return;
    connections.forEach((item, index) => { item.open = printState[index]; });
    printState = undefined;
  });
})();
