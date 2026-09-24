"use strict";

const data = JSON.parse(document.getElementById("leaderboard-data").textContent);
const byId = id => document.getElementById(id);
const targetLabels = {
  params: {label: "Parameters", short: "Params", unit: "M parameters", divisor: 1e6},
  flops: {label: "FLOPs", short: "FLOPs", unit: "M / token", divisor: 1e6},
  memory: {label: "Memory", short: "Memory", unit: "MiB", divisor: 1048576},
  data: {label: "Training data", short: "Data", unit: "M tokens", divisor: 1e6},
  steps: {label: "Training steps", short: "Steps", unit: "updates", divisor: 1},
  traintime: {label: "Training time", short: "Train time", unit: "seconds", divisor: 1},
  decode: {label: "Decode latency", short: "Decode", unit: "ms", divisor: 1},
  request: {label: "Request latency", short: "Request", unit: "ms", divisor: 1},
  kvcache: {label: "KV cache", short: "KV cache", unit: "MiB", divisor: 1048576},
};
const methodInfo = method => data.method_metadata?.[method] || {};
const order = Object.keys(targetLabels);
const tasks = [...data.tasks].sort((a, b) => {
  const ai = order.indexOf(a.id), bi = order.indexOf(b.id);
  return (ai < 0 ? order.length : ai) - (bi < 0 ? order.length : bi) || a.id.localeCompare(b.id);
});
const methods = [...new Set(tasks.flatMap(task => task.runs.map(run => run.method_id)))].sort();
const palette = ["#3972c6", "#de8740", "#249478", "#9873b5", "#31a4b1", "#ce6971", "#7d8998"];
const colorFor = method => palette[methods.indexOf(method) % palette.length];
const methodName = method => methodInfo(method).name || method;
const labelFor = task => targetLabels[task.id] || {label: task.id, short: task.id, unit: task.unit || task.metric, divisor: 1};
let selected = tasks.find(task => task.id === location.hash.slice(1)) || tasks[0];
let hidden = new Set();
let focusedRun = null;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function link(text, url) {
  const element = node("a", text);
  element.href = url;
  element.rel = "noopener noreferrer";
  return element;
}
function format(value, task = selected, axis = false) {
  if (value === null || value === undefined) return "—";
  const scaled = value / labelFor(task).divisor;
  return new Intl.NumberFormat("en-US", axis ? {maximumFractionDigits: 1, maximumSignificantDigits: 4} : {maximumFractionDigits: 3}).format(scaled);
}
function dot(method) {
  const element = node("span", "", "method-dot");
  element.style.backgroundColor = colorFor(method);
  return element;
}
function selectTask(task, runId = null, scroll = false) {
  selected = task;
  focusedRun = runId;
  hidden = new Set();
  if (runId) for (const run of task.runs) if (run.run_id !== runId) hidden.add(run.run_id);
  history.replaceState(null, "", `#${task.id}`);
  renderTarget();
  if (scroll) byId("target-details").scrollIntoView({behavior: "smooth", block: "start"});
}

byId("method-count").textContent = methods.length;
byId("task-count").textContent = tasks.length;
byId("run-count").textContent = tasks.reduce((sum, task) => sum + task.runs.length, 0);
for (const task of tasks) {
  const label = labelFor(task);
  const th = node("th", undefined, "target-column");
  th.scope = "col";
  const button = node("button", label.short);
  button.type = "button";
  button.dataset.task = task.id;
  button.append(node("span", label.unit, "column-unit"));
  button.addEventListener("click", () => selectTask(task, null, true));
  th.append(button);
  byId("matrix-head").insertBefore(th, byId("matrix-head").querySelector(".source-column"));
  const tab = node("button", label.label);
  tab.type = "button";
  tab.dataset.task = task.id;
  tab.addEventListener("click", () => selectTask(task));
  byId("target-tabs").append(tab);
}
for (const method of data.overall.methods) {
  const row = node("tr");
  row.dataset.method = method.method_id;
  row.append(node("td", method.rank ?? "—", "overall-rank"));
  const name = node("td", undefined, "method-column");
  const title = node("strong", methodName(method.method_id), "method-name");
  title.title = method.method_id;
  title.prepend(dot(method.method_id));
  name.append(title, node("span", `${method.coverage} / ${tasks.length} targets`, "coverage"));
  const source = node("td", undefined, "source-column");
  const info = methodInfo(method.method_id);
  if (info.source_url && info.license) {
    const badge = link("Yes", info.source_url);
    badge.className = "source-badge source-open";
    source.append(badge);
  } else {
    source.append(node("span", "No", "source-no"));
  }
  row.append(name, node("td", method.elo === null ? "—" : method.elo.toFixed(1), "elo-column elo-value"));
  for (const task of tasks) {
    const cell = node("td", undefined, "result-cell");
    const result = method.cells[task.id];
    if (result) {
      if (result.rank === 1) cell.classList.add("best-cell");
      const button = node("button", undefined, "cell-button");
      button.type = "button";
      button.title = `${result.run_id}: ${result.value} ${task.unit || task.metric}`;
      button.setAttribute("aria-label", `${methodName(method.method_id)}, ${labelFor(task).label}: rank ${result.rank}, ${result.value} ${task.unit || ""}. Inspect run.`);
      button.append(node("span", format(result.value, task), "cell-value"), node("span", `#${result.rank}`, "cell-rank"));
      button.addEventListener("click", () => selectTask(task, result.run_id, true));
      cell.append(button);
    } else {
      cell.append(node("span", "—", "missing"));
      cell.title = "No accepted result on this target";
    }
    row.append(cell);
  }
  row.append(source);
  byId("matrix-body").append(row);
}
if (!data.overall.methods.length) {
  const row = node("tr"), cell = node("td", "No accepted methods yet. Submit the first result.", "empty-table");
  cell.colSpan = tasks.length + 4;
  row.append(cell);
  byId("matrix-body").append(row);
}
const statusText = {
  ready: `${data.overall.comparison_count} shared-target comparisons across the accepted methods. Elo is descriptive; individual task results remain the benchmark measurements.`,
  no_comparisons: "Overall Elo requires results from at least two methods on a shared target.",
  disconnected: "Overall Elo requires connected comparisons across methods; separate comparison groups have no global ordering.",
  nonpositive_values: "Overall Elo uses logarithmic ratios and requires positive target values. Per-target results remain available.",
};
byId("elo-status").textContent = statusText[data.overall.status];
if (data.overall.status !== "ready") byId("coverage-note").textContent = statusText[data.overall.status];
document.querySelector('a[href="#elo-method"]').addEventListener("click", () => {byId("elo-method").open = true;});

function renderTarget() {
  for (const tab of byId("target-tabs").children) tab.setAttribute("aria-pressed", String(tab.dataset.task === selected.id));
  byId("target-description").textContent = selected.title;
  if (selected.comparison_step) byId("target-description").textContent +=
    ` Values are rounded to checkpoints every ${selected.comparison_step} ${selected.unit} for comparison. Downloaded JSON retains raw measurements.`;
  byId("chart-title-text").textContent = labelFor(selected).label;
  byId("chart-unit").textContent = `${labelFor(selected).unit} · ${selected.direction === "minimize" ? "lower" : "higher"} is better`;
  byId("report-count").textContent = `(${selected.runs.length})`;
  byId("reset-comparison").disabled = hidden.size === 0;
  for (const button of document.querySelectorAll("#matrix-head button")) button.classList.toggle("selected", button.dataset.task === selected.id);
  const legend = byId("legend");
  legend.replaceChildren(node("p", "Method / best value", "legend-heading"));
  for (const run of selected.runs) {
    const button = node("button", undefined, "legend-row");
    button.type = "button";
    button.dataset.run = run.run_id;
    button.title = `${run.run_id}: click to show or hide`;
    button.setAttribute("aria-pressed", String(!hidden.has(run.run_id)));
    const title = node("span", methodName(run.method_id), "legend-name");
    title.prepend(dot(run.method_id));
    if (selected.runs.filter(other => other.method_id === run.method_id).length > 1) title.append(node("small", run.run_id));
    button.append(title, node("span", format(run.comparison_value), "legend-value"));
    button.addEventListener("click", () => {
      if (hidden.has(run.run_id)) hidden.delete(run.run_id); else hidden.add(run.run_id);
      focusedRun = null;
      button.setAttribute("aria-pressed", String(!hidden.has(run.run_id)));
      byId("reset-comparison").disabled = hidden.size === 0;
      drawChart();
    });
    legend.append(button);
  }
  if (focusedRun) {
    const run = selected.runs.find(run => run.run_id === focusedRun);
    if (run) legend.append(link(run.report_url ? "Open this run’s report ↗" : "View run JSON ↗", run.report_url || run.submission_url));
  }
  const body = byId("run-records");
  body.replaceChildren();
  for (const run of selected.runs) {
    const row = node("tr"), method = node("td");
    method.append(node("strong", methodName(run.method_id)), node("span", run.run_id, "subtext"));
    const evidence = node("td", undefined, "evidence");
    if (run.report_url) evidence.append(link("Report ↗", run.report_url));
    if (run.code_url) evidence.append(link("Code ↗", run.code_url));
    if (run.logs_url) evidence.append(link("Logs ↗", run.logs_url));
    evidence.append(link("JSON ↗", run.submission_url));
    row.append(method, node("td", `${format(run.comparison_value)} ${labelFor(selected).unit}`, "numeric"),
      node("td", `${run.budget.spent} / ${run.budget.max_launches}`, "numeric"),
      node("td", run.research_llm.used === false ? "No LLM" : run.research_llm.model || "Not recorded", "model-cell"), evidence);
    body.append(row);
  }
  drawChart();
}

function svgNode(tag, attributes = {}, text) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  if (text !== undefined) element.textContent = text;
  return element;
}
function drawChart() {
  const container = byId("chart"), tooltip = byId("chart-tooltip");
  container.replaceChildren();
  tooltip.hidden = true;
  const runs = selected.runs.filter(run => !hidden.has(run.run_id));
  const showPoints = byId("show-points").checked;
  const values = runs.flatMap(run => [...run.curve, ...(showPoints ? run.points : [])].map(point => point.value)).filter(value => value !== null && Number.isFinite(value));
  if (!values.length) {
    const empty = node("div", undefined, "empty-chart");
    empty.append(node("strong", selected.runs.length ? "Choose a method to compare" : "No accepted results yet"),
      node("p", selected.runs.length ? "Use the method list to show its progress." : "This target is ready for its first submission."));
    if (!selected.runs.length) empty.append(link("Submit a result →", "submit.html"));
    container.append(empty);
    return;
  }
  const width = Math.max(280, container.clientWidth), height = width < 500 ? 280 : 325;
  const left = width < 500 ? 50 : 64, right = 18, top = 15, bottom = 49;
  let low = Math.min(...values), high = Math.max(...values);
  const padding = (high - low || Math.abs(high) || 1) * 0.08;
  low = Math.max(0, low - padding); high += padding;
  const xmax = Math.max(1, ...selected.runs.map(run => run.budget.spent));
  const x = step => left + step / xmax * (width - left - right);
  const y = value => top + (high - value) / (high - low) * (height - top - bottom);
  const svg = svgNode("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-labelledby": "plot-title plot-description"});
  svg.append(svgNode("title", {id: "plot-title"}, `${labelFor(selected).label}: best qualifying result by experiment step`),
    svgNode("desc", {id: "plot-description"}, "Step lines show the best qualifying result through each charged experiment. Each line stops at the run’s actual budget usage. Enable individual experiments to see every qualifying measurement."));
  const divisor = labelFor(selected).divisor;
  const roughStep = (high - low) / divisor / 5;
  const power = 10 ** Math.floor(Math.log10(roughStep));
  const tickStep = [1, 2, 2.5, 5, 10].find(factor => factor * power >= roughStep) * power;
  const firstTick = Math.ceil(low / divisor / tickStep) * tickStep;
  for (let tick = firstTick; tick <= high / divisor + tickStep * 1e-9; tick += tickStep) {
    const value = tick * divisor;
    svg.append(svgNode("line", {x1: left, x2: width - right, y1: y(value), y2: y(value), class: "grid-line"}),
      svgNode("text", {x: left - 12, y: y(value) + 4, "text-anchor": "end", class: "axis-label"}, format(value, selected, true)));
  }
  for (const step of [...new Set(Array.from({length: 6}, (_, i) => Math.round(xmax * i / 5)))]) {
    svg.append(svgNode("text", {x: x(step), y: height - bottom + 23, "text-anchor": "middle", class: "axis-label"}, step));
  }
  svg.append(svgNode("text", {x: (left + width - right) / 2, y: height - 5, "text-anchor": "middle", class: "axis-title"}, "Experiment step"));
  for (const run of runs) {
    const color = colorFor(run.method_id);
    let path = "", started = false;
    for (const point of run.curve) {
      if (point.value === null) continue;
      path += started ? ` H${x(point.step)} V${y(point.value)}` : `M${x(point.step)},${y(point.value)}`;
      started = true;
    }
    if (path) svg.append(svgNode("path", {d: path, stroke: color, fill: "none", "stroke-width": 2, "stroke-linejoin": "round", class: "progress-line", "data-run": run.run_id}));
    const last = run.curve.filter(point => point.value !== null).at(-1);
    if (last) svg.append(svgNode("circle", {cx: x(last.step), cy: y(last.value), r: 3.5, fill: color, stroke: "white", "stroke-width": 1.5, class: "final-point"}));
    if (showPoints) for (const point of run.points) {
      const circle = svgNode("circle", {cx: x(point.step), cy: y(point.value), r: 2.5, fill: color, "fill-opacity": .35, class: "measurement"});
      circle.append(svgNode("title", {}, `${methodName(run.method_id)} · ${point.candidate_id} · step ${point.step} · ${point.value} ${selected.unit || ""}`));
      svg.append(circle);
    }
  }
  const cursor = svgNode("line", {y1: top, y2: height - bottom, class: "cursor-line", visibility: "hidden"});
  svg.append(cursor);
  const inspect = event => {
    const rect = svg.getBoundingClientRect();
    const position = (event.clientX - rect.left) / rect.width * width;
    if (position < left || position > width - right) {tooltip.hidden = true; cursor.setAttribute("visibility", "hidden"); return;}
    const step = Math.max(1, Math.min(xmax, Math.round((position - left) / (width - left - right) * xmax)));
    cursor.setAttribute("x1", x(step)); cursor.setAttribute("x2", x(step)); cursor.setAttribute("visibility", "visible");
    tooltip.replaceChildren(node("strong", `Experiment ${step}`));
    for (const run of runs) {
      if (step > run.budget.spent) continue;
      const point = run.curve.filter(point => point.step <= step && point.value !== null).at(-1);
      if (!point) continue;
      const line = node("div", undefined, "tooltip-row"), name = node("span", methodName(run.method_id));
      name.prepend(dot(run.method_id));
      line.append(name, node("span", format(point.value), "numeric")); tooltip.append(line);
    }
    tooltip.hidden = false;
    tooltip.style.left = `${Math.min(Math.max(8, position + 14), Math.max(8, width - tooltip.offsetWidth - 8))}px`;
  };
  svg.addEventListener("pointermove", inspect);
  svg.addEventListener("click", inspect);
  svg.addEventListener("pointerleave", () => {tooltip.hidden = true; cursor.setAttribute("visibility", "hidden");});
  container.append(svg);
}

byId("show-points").addEventListener("change", drawChart);
byId("reset-comparison").addEventListener("click", () => {hidden = new Set(); focusedRun = null; renderTarget();});
window.addEventListener("resize", drawChart);
window.addEventListener("hashchange", () => {
  const task = tasks.find(task => task.id === location.hash.slice(1));
  if (task) {selected = task; hidden = new Set(); focusedRun = null; renderTarget();}
});
renderTarget();
