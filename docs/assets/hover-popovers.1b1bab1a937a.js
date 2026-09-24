(() => {
  'use strict';

  const triggers = [...document.querySelectorAll('[data-tooltip-id]')];
  let activeTrigger = null;
  let activePopover = null;
  let hideTimer = null;

  const cancelHide = () => {
    if (hideTimer !== null) window.clearTimeout(hideTimer);
    hideTimer = null;
  };

  const hide = () => {
    cancelHide();
    if (activePopover) {
      activePopover.classList.remove('is-visible');
      activePopover.setAttribute('aria-hidden', 'true');
      activePopover.style.removeProperty('left');
      activePopover.style.removeProperty('top');
    }
    activeTrigger = null;
    activePopover = null;
  };

  const scheduleHide = () => {
    cancelHide();
    hideTimer = window.setTimeout(hide, 120);
  };

  const place = () => {
    if (!activeTrigger || !activePopover) return;
    const triggerBox = activeTrigger.getBoundingClientRect();
    const popoverBox = activePopover.getBoundingClientRect();
    const margin = 12;
    const gap = 7;

    if (activeTrigger.dataset.tooltipPlacement === 'right-margin') {
      const paragraphBox = activeTrigger.closest('p')?.getBoundingClientRect();
      if (paragraphBox) {
        const rightLeft = paragraphBox.right + 12;
        if (rightLeft + popoverBox.width <= window.innerWidth - margin) {
          const top = Math.max(margin, Math.min(paragraphBox.top, window.innerHeight - popoverBox.height - margin));
          activePopover.style.left = `${rightLeft}px`;
          activePopover.style.top = `${top}px`;
          return;
        }
      }
    }

    let left = triggerBox.left + triggerBox.width / 2 - popoverBox.width / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - popoverBox.width - margin));
    let top = triggerBox.bottom + gap;
    if (top + popoverBox.height > window.innerHeight - margin) {
      top = Math.max(margin, triggerBox.top - popoverBox.height - gap);
    }
    activePopover.style.left = `${left}px`;
    activePopover.style.top = `${top}px`;
  };

  const show = trigger => {
    cancelHide();
    const popover = document.getElementById(trigger.dataset.tooltipId);
    if (!popover) return;
    if (activePopover && activePopover !== popover) hide();
    activeTrigger = trigger;
    activePopover = popover;
    popover.classList.add('is-visible');
    popover.setAttribute('aria-hidden', 'false');
    place();
  };

  for (const trigger of triggers) {
    trigger.addEventListener('mouseenter', () => show(trigger));
    trigger.addEventListener('mouseleave', scheduleHide);
    trigger.addEventListener('focus', () => show(trigger));
    trigger.addEventListener('blur', scheduleHide);
  }

  for (const popover of document.querySelectorAll('.hover-popover')) {
    popover.addEventListener('mouseenter', cancelHide);
    popover.addEventListener('mouseleave', scheduleHide);
  }

  window.addEventListener('resize', place, { passive: true });
  window.addEventListener('scroll', place, { passive: true });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') hide();
  });
})();
