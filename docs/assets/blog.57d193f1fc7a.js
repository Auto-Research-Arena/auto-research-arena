(() => {
  "use strict";

  const progress = document.querySelector(".reading-progress span");
  const updateProgress = () => {
    const available = document.documentElement.scrollHeight - window.innerHeight;
    const ratio = available > 0 ? window.scrollY / available : 0;
    progress.style.width = `${Math.min(1, Math.max(0, ratio)) * 100}%`;
  };
  updateProgress();
  addEventListener("scroll", updateProgress, { passive: true });
  addEventListener("resize", updateProgress);

  document.querySelectorAll("[data-progress-tabs], [data-discovery-tabs]").forEach((widget) => {
    const tabs = [...widget.querySelectorAll('[role="tab"]')];
    const select = (tab, moveFocus = false) => {
      tabs.forEach((candidate) => {
        const selected = candidate === tab;
        candidate.setAttribute("aria-selected", String(selected));
        candidate.tabIndex = selected ? 0 : -1;
        const panel = widget.querySelector(`#${candidate.getAttribute("aria-controls")}`);
        if (panel) panel.hidden = !selected;
      });
      if (moveFocus) tab.focus();
    };
    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => select(tab));
      tab.addEventListener("keydown", (event) => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        let next = index;
        if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
        if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = tabs.length - 1;
        select(tabs[next], true);
      });
    });
  });

  const dialog = document.querySelector(".image-dialog");
  const expanded = dialog?.querySelector("img");
  const close = dialog?.querySelector("button");
  document.querySelectorAll(".article img").forEach((image) => {
    image.dataset.expandable = "true";
    image.tabIndex = 0;
    image.setAttribute("role", "button");
    image.setAttribute("aria-haspopup", "dialog");
    image.setAttribute("aria-label", `${image.alt}. Expand figure.`);
    image.addEventListener("click", () => {
      expanded.src = image.currentSrc || image.src;
      expanded.alt = image.alt;
      dialog.showModal();
    });
    image.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        image.click();
      }
    });
  });
  close?.addEventListener("click", () => dialog.close());
  dialog?.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
})();
