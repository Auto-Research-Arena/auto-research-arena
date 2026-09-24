"""Compare collected runs by task in offline HTML with inline CSS and SVG.

Rank eligible runs by canonical rank keys. Show excluded runs, integrity findings
and each run's measured reference alongside its results.
"""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

# Deliberately few colours, each legible on white and distinguishable in greyscale, since
# these pages get printed and pasted into documents.
PALETTE = (
    "#1f4e79",
    "#b45309",
    "#146b3a",
    "#7c2d8f",
    "#9b1c1c",
    "#0e7490",
    "#4b5563",
    "#a16207",
    "#3730a3",
)

CSS = """
:root { --ink:#111; --dim:#5b6470; --line:#d8dde3; --bg:#fff; --accent:#1f4e79;
        --warn:#8a4b00; --bad:#9b1c1c; --good:#146b3a; }
* { box-sizing:border-box; }
body { margin:0; padding:2rem 1.5rem 4rem; background:var(--bg); color:var(--ink);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
main { max-width:1180px; margin:0 auto; }
h1 { font-size:1.6rem; margin:0 0 .25rem; letter-spacing:-.01em; }
h2 { font-size:1.15rem; margin:2.5rem 0 .5rem; padding-bottom:.3rem;
     border-bottom:2px solid var(--line); }
h3 { font-size:.98rem; margin:1.6rem 0 .4rem; }
p, li { max-width:74ch; }
.sub { color:var(--dim); margin:0 0 1.5rem; font-size:.9rem; }
.mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.86em; }
table { border-collapse:collapse; width:100%; margin:.6rem 0 1rem; font-size:.88rem; }
th, td { text-align:right; padding:.36rem .55rem; border-bottom:1px solid var(--line);
         white-space:nowrap; }
th { font-weight:600; font-size:.78rem; text-transform:uppercase; letter-spacing:.04em;
     color:var(--dim); border-bottom:2px solid var(--line); }
th:first-child, td:first-child, th.l, td.l { text-align:left; }
tbody tr:hover { background:#f6f8fa; }
tr.blocked td { color:var(--dim); background:#fdf6f6; }
.rank { color:var(--dim); font-variant-numeric:tabular-nums; }
.best { font-weight:600; }
.tag { display:inline-block; padding:.03rem .38rem; border-radius:.6rem; font-size:.7rem;
       font-weight:600; text-transform:uppercase; letter-spacing:.04em; vertical-align:.08em; }
.tag.blocked { background:#fdecec; color:var(--bad); border:1px solid #f0c0c0; }
.card { border:1px solid var(--line); border-radius:6px; padding:.9rem 1.1rem; margin:.8rem 0;
        background:#fbfcfd; }
.card h3 { margin-top:0; }
.finding { margin:.35rem 0; padding-left:.7rem; border-left:3px solid var(--line);
           font-size:.86rem; color:#333; }
.finding.blocking, .finding.fail, .finding.high { border-left-color:var(--bad); }
.finding.warn { border-left-color:var(--warn); }
.finding code { font-weight:600; }
.kv { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.5rem 1.2rem;
      margin:.6rem 0; font-size:.86rem; }
.kv div span { display:block; color:var(--dim); font-size:.74rem; text-transform:uppercase;
               letter-spacing:.04em; }
figure { margin:1rem 0 1.6rem; }
figcaption { color:var(--dim); font-size:.82rem; margin-top:.35rem; max-width:74ch; }
.legend { font-size:.8rem; color:var(--dim); margin:.3rem 0 0; }
.legend b { font-weight:600; }
.swatch { display:inline-block; width:.7rem; height:.7rem; border-radius:2px; margin-right:.3rem;
          vertical-align:-.05em; }
footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line);
         color:var(--dim); font-size:.82rem; }
.empty { color:var(--dim); font-style:italic; }
"""


def build(summaries: Sequence[Dict[str, Any]], *, title: str = "AutoArena results") -> str:
    """Render every summary into one page, grouped by task."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for summary in summaries:
        by_task.setdefault(summary["task"]["task_id"], []).append(summary)

    parts = [
        _head(title),
        "<main>",
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="sub">{len(summaries)} run(s) across {len(by_task)} task(s). '
        f"Generated {generated} from each run's <span class='mono'>launches.jsonl</span>. "
        "Measurements, charging and ranking come from the collected ledger records.</p>",
    ]
    if not summaries:
        parts.append('<p class="empty">No runs collected.</p>')
    for task_id in sorted(by_task):
        parts.append(_task_section(task_id, by_task[task_id]))
    parts.append(_footer(summaries))
    parts.append("</main></body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def _task_section(task_id: str, summaries: List[Dict[str, Any]]) -> str:
    task = summaries[0]["task"]
    ranked, blocked = _partition(summaries)

    out = [f'<h2 id="{html.escape(task_id)}">{html.escape(task_id)}</h2>']
    out.append(f'<p class="sub">{html.escape(task["title"])}</p>')
    out.append(_objective_card(task, summaries[0]))

    out.append("<h3>Ranking</h3>")
    if ranked:
        out.append(_leaderboard(task, ranked))
    else:
        out.append(
            '<p class="empty">No benchmark-rankable run has an eligible candidate on this task.</p>'
        )

    if blocked:
        out.append("<h3>Excluded from the ranking</h3>")
        out.append(
            '<p class="sub">These runs have no eligible candidate or carry a blocking '
            "finding. See the per-run detail for the recorded evidence.</p>"
        )
        out.append(_leaderboard(task, blocked, blocked_rows=True))

    out.append(_trajectory_figure(task, ranked + blocked))
    out.append("<h3>Per-run detail</h3>")
    for summary in sorted(summaries, key=lambda item: item["run"]["run_id"]):
        out.append(_run_card(summary))
    return "\n".join(out)


def _partition(
    summaries: List[Dict[str, Any]],
) -> Tuple[List[Dict], List[Dict]]:
    """Split runs into ranked and excluded groups."""
    ranked, blocked = [], []
    for summary in summaries:
        (ranked if is_rankable(summary) else blocked).append(summary)
    ranked.sort(key=_sort_key)
    blocked.sort(key=_sort_key)
    return ranked, blocked


def is_rankable(summary: Dict[str, Any]) -> bool:
    """Return the renderer's single definition of a row that enters a ranking."""
    best = summary.get("best")
    if not best or not best.get("rank_key"):
        return False
    has_blocking = any(
        finding["level"] == "blocking" for finding in summary["integrity"]
    )
    return not has_blocking


def _sort_key(summary: Dict[str, Any]) -> tuple:
    """Best first. Runs with no admissible candidate sort last, not best.

    Explicitly: an empty rank key would compare as smaller than every real one, so a run
    that measured nothing would top the table.
    """
    best = summary.get("best")
    if not best or not best.get("rank_key"):
        return (1, ())
    return (0, tuple(best["rank_key"]))


def _objective_card(task: Dict[str, Any], summary: Dict[str, Any]) -> str:
    gate = task.get("gate")
    if gate:
        gate_text = (
            f"{html.escape(str(gate.get('metric', task.get('objective_metric', ''))))} "
            f"{html.escape(gate['operator'])} {gate['value']}"
        ).strip()
    else:
        gate_text = "none"
    rows = [
        ("Comparison mode", html.escape(task["comparison_mode"])),
        (
            "Ranking key",
            html.escape(", ".join(task["ranking_metrics"]) or "—")
            + f" ({html.escape(task['headline_direction'])})",
        ),
        ("Gate", gate_text),
        ("Launch cap", str(summary["budget"]["max_launches"])),
        (
            "Charging",
            html.escape(str(summary["budget"]["charging_rule"]))
            + (" · failures count" if summary["budget"]["includes_failures"] else ""),
        ),
    ]
    ceilings = task.get("admissibility") or []
    if ceilings:
        rows.append(
            (
                "Ceilings",
                "; ".join(
                    f"{html.escape(item['metric'])} {html.escape(item['operator'])} "
                    f"{item['value']}"
                    for item in ceilings
                ),
            )
        )
    cells = "".join(
        f"<div><span>{label}</span>{value}</div>" for label, value in rows
    )
    note = ""
    if task["comparison_mode"] == "gated":
        note = (
            "<p class='sub' style='margin:.4rem 0 0'>Under <span class='mono'>gated</span> "
            "the gate is a threshold, not a target: a candidate inside it is ranked on the "
            "ranking key alone, and the gate is not compared again.</p>"
        )
    return f'<div class="card"><div class="kv">{cells}</div>{note}</div>'


def _leaderboard(
    task: Dict[str, Any], summaries: List[Dict[str, Any]], blocked_rows: bool = False
) -> str:
    columns = task["columns"]
    head = (
        "<tr><th class='l'>#</th><th class='l'>Method</th><th class='l'>Run</th>"
        + "".join(f"<th>{html.escape(name)}</th>" for name in columns)
        + "<th>Spent</th><th>OK</th><th>Margin</th></tr>"
    )
    rows = []
    for position, summary in enumerate(summaries, start=1):
        best = summary.get("best") or {}
        metrics = best.get("metrics") or {}
        method = summary["method"]
        tags = ""
        if blocked_rows:
            tags += ' <span class="tag blocked">not ranked</span>'
            if not best.get("rank_key"):
                tags += ' <span class="tag blocked">no eligible candidate</span>'
        headline = task["headline"]
        cells = "".join(
            f'<td class="{"best" if name == headline else ""}">{_fmt(metrics.get(name))}</td>'
            for name in columns
        )
        margin = best.get("gate_margin")
        rows.append(
            f'<tr class="{"blocked" if blocked_rows else ""}">'
            f'<td class="l rank">{"—" if blocked_rows else position}</td>'
            f'<td class="l">{html.escape(method["method_id"])}{tags}</td>'
            f'<td class="l mono">{html.escape(summary["run"]["run_id"])}</td>'
            f"{cells}"
            f'<td>{summary["budget"]["spent"]}/{summary["budget"]["max_launches"]}</td>'
            f'<td>{summary["totals"]["measured"]}</td>'
            f"<td>{'—' if margin is None else f'{margin:+.6f}'}</td></tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _run_card(summary: Dict[str, Any]) -> str:
    budget = summary["budget"]
    method = summary["method"]
    totals = summary["totals"]
    kv = [
        ("Method", html.escape(method["method_id"])),
        ("Charged", f"{budget['spent']}/{budget['max_launches']}"),
        ("Refunded", str(budget["refunded"])),
        ("Unresolved", str(budget["unresolved"])),
        ("Measured ok", str(totals["measured"])),
        ("Distinct candidate IDs", str(totals["distinct_candidates"])),
        ("GPU seconds", f"{totals['gpu_seconds']:,.0f}"),
        ("Lanes", str(summary["run"].get("lanes") or "—")),
        ("Research model", html.escape(summary["run"].get("research_model") or "—")),
    ]
    cells = "".join(f"<div><span>{label}</span>{value}</div>" for label, value in kv)

    statuses = summary["status_counts"]
    status_line = (
        "<p class='sub' style='margin:.2rem 0'>Statuses: "
        + html.escape(", ".join(f"{name} {count}" for name, count in statuses.items()))
        + "</p>"
        if statuses
        else ""
    )
    refunds = budget.get("refund_reasons") or {}
    refund_line = (
        "<p class='sub' style='margin:.2rem 0'>Refunds: "
        + html.escape(", ".join(f"{name} {count}" for name, count in refunds.items()))
        + ". Charging follows the task rules and recorded adjudications.</p>"
        if refunds
        else ""
    )

    findings = "".join(
        f'<p class="finding {html.escape(item["level"])}"><code>'
        f'{html.escape(item["code"])}</code> — {html.escape(item["detail"])}</p>'
        for item in summary["integrity"]
    )
    reference = summary["reference"]
    baseline = "<p class='sub'>No reference launch recorded.</p>"
    if reference["present"]:
        metrics = reference.get("recorded") or {}
        baseline = (
            f"<p class='sub'>Reference launch {_fmt(reference.get('launch_seq'))}; "
            f"asserted values reproduced: {_fmt(reference.get('reproduced'))}.</p>"
            + '<div class="kv">'
            + "".join(f"<div><span>{html.escape(name)}</span>{_fmt(value)}</div>"
                      for name, value in metrics.items())
            + "</div>"
        )
    return (
        f'<div class="card"><h3>{html.escape(summary["run"]["run_id"])}</h3>'
        f'<div class="kv">{cells}</div>{status_line}{refund_line}'
        + baseline
        + (f"<p class='sub' style='margin:.6rem 0 .2rem'>Findings</p>{findings}"
           if findings else "")
        + "</div>"
    )


# ---------------------------------------------------------------------------
# the chart
# ---------------------------------------------------------------------------


def _trajectory_figure(task: Dict[str, Any], summaries: List[Dict[str, Any]]) -> str:
    series = []
    for index, summary in enumerate(summaries):
        points = [
            (point["spent"], point["best_so_far"])
            for point in summary["trajectory"]
            if point["best_so_far"] is not None
        ]
        if points:
            series.append(
                (summary["method"]["method_id"], PALETTE[index % len(PALETTE)], points)
            )
    if not series:
        return ""
    svg = _line_chart(
        series,
        x_label="charged launches",
        y_label=f"{task['headline']} (best so far)",
        y_invert=task["headline_direction"] == "minimize",
    )
    legend = " ".join(
        f'<span><span class="swatch" style="background:{colour}"></span>'
        f"<b>{html.escape(name)}</b></span>"
        for name, colour, _ in series
    )
    return (
        f"<figure>{svg}<p class='legend'>{legend}</p>"
        "<figcaption>Best-so-far against <b>charged</b> launches, not wall-clock and not "
        "row index. Charged launches are the budget, so this is the axis on which two "
        "methods are actually comparable; plotting against rows would reward an arm whose "
        "refunded substrate failures padded the count. A flat line is a method that spent "
        "budget without improving.</figcaption></figure>"
    )


def _line_chart(
    series: Sequence[Tuple[str, str, List[Tuple[float, float]]]],
    *,
    x_label: str,
    y_label: str,
    y_invert: bool,
    width: int = 880,
    height: int = 340,
) -> str:
    """A step-line chart as inline SVG. Steps, because best-so-far is a step function.

    Drawing it smooth would imply the metric improved between launches, which is a claim
    about measurements that were never taken.
    """
    pad_l, pad_r, pad_t, pad_b = 78, 18, 18, 46
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    xs = [x for _, _, points in series for x, _ in points]
    ys = [y for _, _, points in series for _, y in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        span = abs(y_min) * 0.05 or 1.0
        y_min, y_max = y_min - span, y_max + span
    else:
        margin = (y_max - y_min) * 0.08
        y_min, y_max = y_min - margin, y_max + margin

    def sx(value: float) -> float:
        return pad_l + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        fraction = (value - y_min) / (y_max - y_min)
        # A minimized metric improves downward on the value axis, so better must be *up*
        # on the page. Getting this backwards makes every chart read as a regression.
        return pad_t + (fraction if y_invert else 1 - fraction) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="{html.escape(y_label)} against {html.escape(x_label)}" '
        'style="max-width:100%;height:auto;font:12px sans-serif">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>',
    ]
    for step in range(5):
        value = y_min + (y_max - y_min) * step / 4
        y = sy(value)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            'stroke="#e8ecf0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#5b6470">'
            f"{_axis_number(value)}</text>"
        )
    for step in range(5):
        value = x_min + (x_max - x_min) * step / 4
        x = sx(value)
        parts.append(
            f'<text x="{x:.1f}" y="{height - pad_b + 18}" text-anchor="middle" '
            f'fill="#5b6470">{value:.0f}</text>'
        )
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" '
        'stroke="#9aa4b0"/>'
    )
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" '
        f'y2="{pad_t + plot_h}" stroke="#9aa4b0"/>'
    )
    parts.append(
        f'<text x="{pad_l + plot_w / 2:.0f}" y="{height - 6}" text-anchor="middle" '
        f'fill="#5b6470">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="14" y="{pad_t + plot_h / 2:.0f}" text-anchor="middle" fill="#5b6470" '
        f'transform="rotate(-90 14 {pad_t + plot_h / 2:.0f})">{html.escape(y_label)}</text>'
    )

    for name, colour, points in series:
        path = []
        previous_y = None
        for x, y in points:
            px, py = sx(x), sy(y)
            if previous_y is None:
                path.append(f"M{px:.1f},{py:.1f}")
            else:
                path.append(f"L{px:.1f},{previous_y:.1f} L{px:.1f},{py:.1f}")
            previous_y = py
        last_x, last_y = points[-1]
        # The line stops where the method stopped. Extending it to the right edge would
        # draw an arm that ended at its own stagnation rule after 60 launches as though it
        # had spent all 100, which is the one difference between those two rows.
        parts.append(
            f'<path d="{" ".join(path)}" fill="none" stroke="{colour}" stroke-width="2" '
            'stroke-linejoin="round"/>'
        )
        parts.append(
            f'<circle cx="{sx(last_x):.1f}" cy="{sy(last_y):.1f}" r="3.2" fill="{colour}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _axis_number(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1e8:
        return f"{value / 1e6:.0f}M"
    if magnitude >= 1e6:
        return f"{value / 1e6:.1f}M"
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


# ---------------------------------------------------------------------------
# chrome
# ---------------------------------------------------------------------------


def _head(title: str) -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
    )


def _footer(summaries: Sequence[Dict[str, Any]]) -> str:
    runs = ", ".join(
        f"<span class='mono'>{html.escape(summary['run']['run_id'])}</span>"
        for summary in summaries
    )
    return (
        "<footer><p>This page renders canonical collected records with "
        "<span class='mono'>engine.submission.comparison.build</span>; "
        "audit a run with "
        "<span class='mono'>python3 -m engine verify --run &lt;dir&gt;</span>. "
        "Verification re-reads launch stdout and captured sources to check the "
        "recorded measurements.</p>"
        f"<p>Runs: {runs or 'none'}.</p></footer>"
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return html.escape(str(value)) + " (non-finite)"
        if value == int(value) and abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return html.escape(str(value))
