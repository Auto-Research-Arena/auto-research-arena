/* The shared blog script handles clicks and arrow keys. This adds deep links
   and prints all targets without changing the reader's selected target. */
(() => {
  'use strict';
  const widget = document.querySelector('[data-performance-explorer]');
  if (!widget) return;
  const tabs = [...widget.querySelectorAll('[role="tab"]')];
  const panels = [...widget.querySelectorAll('[role="tabpanel"]')];
  const revealHash = () => {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    const target = document.getElementById(id);
    if (!target || !widget.contains(target)) return;
    const panel = target.closest('[role="tabpanel"]');
    const tab = panel ? tabs.find(t => t.getAttribute('aria-controls') === panel.id) : tabs.find(t => t === target);
    if (tab) tab.click();
  };
  revealHash();
  addEventListener('hashchange', revealHash);

  const curveKeys = [...widget.querySelectorAll('.harness-curve-key[data-method]')];
  const curveRows = [...widget.querySelectorAll('.benchmark-target-results tbody tr[data-method]')];
  const focusCurve = source => {
    const panel = source.closest('.native-progress-panel');
    if (!panel) return;
    const method = source.dataset.method;
    panel.classList.add('has-curve-focus');
    panel.querySelectorAll('.native-curve').forEach(curve => {
      curve.classList.toggle('is-highlighted', curve.dataset.method === method);
    });
  };
  const clearCurve = source => {
    const panel = source.closest('.native-progress-panel');
    if (!panel) return;
    panel.classList.remove('has-curve-focus');
    panel.querySelectorAll('.native-curve.is-highlighted').forEach(curve => curve.classList.remove('is-highlighted'));
  };
  curveKeys.forEach(key => {
    key.addEventListener('focus', () => focusCurve(key));
    key.addEventListener('blur', () => clearCurve(key));
  });
  curveRows.forEach(row => {
    row.addEventListener('mouseenter', () => focusCurve(row));
    row.addEventListener('mouseleave', () => clearCurve(row));
  });

  let printState;
  addEventListener('beforeprint', () => {
    if (printState) return;
    printState = panels.map(panel => panel.hidden);
    panels.forEach(panel => { panel.hidden = false; });
  });
  addEventListener('afterprint', () => {
    if (!printState) return;
    panels.forEach((panel, i) => { panel.hidden = printState[i]; });
    printState = undefined;
  });

  const overall = document.querySelector('[data-overall-ranking]');
  if (overall) {
    const rows = [...overall.querySelectorAll('tbody tr[data-method]')];
    const focusRadar = row => {
      const method = row.dataset.method;
      overall.classList.add('has-radar-focus');
      rows.forEach(candidate => candidate.classList.toggle('is-highlighted', candidate === row));
      overall.querySelectorAll('.overall-radar-series[data-method]').forEach(series => {
        series.classList.toggle('is-highlighted', series.dataset.method === method);
      });
    };
    const clearRadar = () => {
      overall.classList.remove('has-radar-focus');
      rows.forEach(row => row.classList.remove('is-highlighted'));
      overall.querySelectorAll('.overall-radar-series.is-highlighted').forEach(series => series.classList.remove('is-highlighted'));
    };
    rows.forEach(row => {
      row.addEventListener('mouseenter', () => focusRadar(row));
      row.addEventListener('mouseleave', clearRadar);
      row.addEventListener('focus', () => focusRadar(row));
      row.addEventListener('blur', clearRadar);
    });
  }
})();
