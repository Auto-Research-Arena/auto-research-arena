/* Native links/details work without JS. Enhance with active-section tracking. */
(() => {
  'use strict';
  const page = document.querySelector('.arena-page');
  if (!page) return;
  const links = [...page.querySelectorAll('[data-toc-link]')];
  const headings = [...new Set(links.map(a => a.hash.slice(1)))].map(id => document.getElementById(id)).filter(Boolean);
  const mobile = page.querySelector('.mobile-toc');
  const tldr = page.querySelector('.article-tldr');
  let selected;
  let scheduled = false;
  let printing = false;
  let printState;
  const mark = id => {
    if (selected === id) return;
    selected = id;
    for (const link of links) {
      if (link.hash === `#${id}`) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    }
  };
  const update = () => {
    scheduled = false;
    if (printing) return;
    let current = null;
    const boundary = Math.min(140, window.innerHeight * .2);
    for (const heading of headings) {
      if (heading.getBoundingClientRect().top <= boundary) current = heading.id;
      else break;
    }
    mark(current);
  };
  const schedule = () => {
    if (!scheduled) { scheduled = true; requestAnimationFrame(update); }
  };
  const fromHash = () => {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    if (headings.some(h => h.id === id)) mark(id);
    else schedule();
  };
  for (const link of links) link.addEventListener('click', event => {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    mark(link.hash.slice(1));
    if (mobile?.contains(link)) mobile.open = false;
    // Leave anchor navigation and browser history to the browser.
  });
  addEventListener('scroll', schedule, {passive: true});
  addEventListener('resize', schedule);
  addEventListener('load', schedule);
  addEventListener('hashchange', fromHash);
  page.addEventListener('toggle', schedule, true);
  if ('ResizeObserver' in window) new ResizeObserver(schedule).observe(page.querySelector('.article'));
  update();
  fromHash();
  addEventListener('beforeprint', () => {
    if (printing) return;
    printing = true;
    printState = tldr?.open;
    if (tldr) tldr.open = true;
  });
  addEventListener('afterprint', () => {
    if (!printing) return;
    if (tldr) tldr.open = printState;
    printing = false;
    schedule();
  });
})();
