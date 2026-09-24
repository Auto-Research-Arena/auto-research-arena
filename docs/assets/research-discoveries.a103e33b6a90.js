(() => {
  'use strict';
  const findings = [...document.querySelectorAll('[data-research-finding]')];
  const toolbar = document.querySelector('[data-research-tools]');
  if (toolbar && findings.length) {
    toolbar.hidden = false;
    toolbar.querySelector('[data-expand]').addEventListener('click', () => findings.forEach(item => { item.open = true; }));
    toolbar.querySelector('[data-collapse]').addEventListener('click', () => findings.forEach(item => { item.open = false; }));
  }
  const revealHash = (hash = location.hash) => {
    let id;
    try { id = decodeURIComponent(hash.slice(1)); } catch { return; }
    const target = id && document.getElementById(id);
    if (!target) return;
    let node = target;
    while (node) {
      if (node.tagName === 'DETAILS') node.open = true;
      node = node.parentElement;
    }
    requestAnimationFrame(() => target.scrollIntoView({block: 'start'}));
  };
  addEventListener('hashchange', () => revealHash());
  document.addEventListener('click', event => {
    const anchor = event.target.closest('a[href^="#"]');
    if (anchor && !event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey && event.button === 0) revealHash(anchor.hash);
  });
  revealHash();
  const printItems = [...findings, ...document.querySelectorAll('[data-literature-entry]')];
  let printState;
  addEventListener('beforeprint', () => {
    if (printState) return;
    printState = printItems.map(item => item.open);
    printItems.forEach(item => { item.open = true; });
  });
  addEventListener('afterprint', () => {
    if (!printState) return;
    printItems.forEach((item, index) => { item.open = printState[index]; });
    printState = undefined;
  });
})();
