"""Offline submission pages. Reads records, never makes benchmark decisions."""
from html import escape
import json

CSS = '''
:root{--ink:#172b3a;--muted:#62717c;--line:#dce5e8;--paper:#fff;--bg:#f4f7f8;--blue:#1565a6;--green:#147d59;--red:#b13c31;--series-1:#1565a6;--grid:#e7edef;--ink-3:#62717c;--ink-2:#394e5c;--panel:#fff;--good:#147d59;--critical:#b13c31;--warning:#c68916;--serious:#c47835}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}a:focus-visible,summary:focus-visible{outline:3px solid #8cbfe7;outline-offset:4px}header{background:var(--paper);border-bottom:1px solid var(--line)}.top{max-width:1120px;margin:auto;padding:19px 32px;display:flex;justify-content:space-between;align-items:center;gap:20px}.brand{font-size:13px;font-weight:750;letter-spacing:.1em}.eyebrow{font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}.topnav{display:flex;gap:22px;font-size:13px}.wrap{max-width:1120px;padding:40px 32px 60px;margin:auto}.title{display:flex;justify-content:space-between;gap:20px;align-items:center;margin-bottom:23px}h1{font-size:32px;letter-spacing:-.035em;line-height:1.2;margin:7px 0 10px}h2{font-size:19px;margin:0 0 5px;letter-spacing:-.02em}h3{font-size:14px;margin:0 0 8px}p{margin:0 0 12px}.muted{color:var(--muted)}.small{font-size:12px}.badge{display:inline-flex;align-items:center;gap:7px;background:#eaf5ef;color:var(--green);padding:5px 12px;border-radius:20px;font-size:12px;font-weight:650;white-space:nowrap}.badge.fail{background:#fbece9;color:var(--red)}.dot{width:6px;height:6px;border-radius:50%;background:currentColor}.panel{background:var(--paper);border:1px solid var(--line);border-radius:12px;margin-bottom:20px;overflow:hidden}.section{padding:24px}.summary{display:grid;grid-template-columns:1.3fr 1fr 1fr 1fr;padding:25px 0}.stat{padding:0 25px;border-right:1px solid var(--line)}.stat:last-child{border:0}.label{font-size:12px;color:var(--muted);display:block;margin-bottom:8px}.value{font-size:29px;letter-spacing:-.045em;font-weight:650;line-height:1.3;font-variant-numeric:tabular-nums}.value.primary{color:var(--blue)}.value.good{color:var(--green)}.unit{font-size:11px;color:var(--muted);display:block;margin-top:5px}.strip{display:flex;flex-wrap:wrap;gap:8px 25px;border-top:1px solid var(--line);padding:12px 24px;font-size:12px;color:var(--muted);background:#fafcfc}.sectionhead{display:flex;justify-content:space-between;align-items:start;gap:20px;margin-bottom:17px}.setup{display:grid;grid-template-columns:1fr 1fr;gap:8px 35px}.setup dl{margin:0}.kv{display:grid;grid-template-columns:145px minmax(0,1fr);gap:8px 14px;font-size:13px}.kv dt{color:var(--muted)}.kv dd{margin:0;overflow-wrap:anywhere}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;overflow-wrap:anywhere}.legend{display:flex;flex-wrap:wrap;gap:18px;font-size:12px;color:var(--muted);margin-top:8px}.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}.chart{margin:12px 0 3px}.chart svg{display:block}.note{font-size:12px;color:var(--muted)}.buttons{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}.button{display:inline-block;padding:8px 13px;border:1px solid var(--line);border-radius:6px;font-size:13px;background:#fff}.button.primary{background:var(--blue);border-color:var(--blue);color:white}.callout{border-left:3px solid var(--blue);background:#f3f8fc;padding:12px 15px;font-size:13px;margin:17px 0}.warn{border-left-color:var(--red);background:#fdf5f3}.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12px}th{text-align:left;font-size:10px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);font-weight:650;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}td{padding:11px 12px;border-bottom:1px solid #edf1f3;vertical-align:top}tbody tr:last-child td{border-bottom:0}.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.pass{color:var(--green)}.failtext{color:var(--red)}.winner{background:#edf6fd}details{border-top:1px solid var(--line)}summary{cursor:pointer;padding:16px 24px;display:list-item;list-style-position:inside;font-size:13px;font-weight:600}details .inside{padding:0 24px 22px}details .inside .section{padding:0}ul{padding-left:20px;margin:6px 0}li{margin-bottom:6px}.footer{display:flex;justify-content:space-between;gap:20px;font-size:11px;color:var(--muted);margin-top:26px}.indexgrid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.indexgrid .panel{margin:0}.indexgrid .value{margin:15px 0 2px}.nowrap{white-space:nowrap}
@media(max-width:720px){.top{padding:15px 20px}.topnav{gap:12px;font-size:12px}.wrap{padding:25px 16px 40px}.title{align-items:start;flex-direction:column;gap:6px}h1{font-size:27px}.summary{grid-template-columns:1fr 1fr;gap:24px 0;padding:22px 0}.stat{padding:0 18px}.stat:nth-child(2){border:0}.value{font-size:25px}.section{padding:20px}.setup{grid-template-columns:1fr;gap:8px}.kv{grid-template-columns:130px minmax(0,1fr);font-size:12px}.sectionhead{flex-direction:column;gap:7px}.strip{padding:12px 18px}.footer{flex-direction:column;gap:4px}.indexgrid{grid-template-columns:1fr}details .inside{padding:0 18px 18px}summary{padding:15px 18px}}
'''


def e(value):
    return escape('' if value is None else str(value), quote=True)


def n(value, precision=3):
    if value is None:
        return '—'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (int, float)):
        return f'{value:,.{precision}f}'.rstrip('0').rstrip('.') if value % 1 else f'{value:,.0f}'
    return e(value)


def link(path, label, cls=''):
    return f'<a href="{e(path)}" class="{cls}">{e(label)}</a>'


def percent(value):
    if value is None:
        return '—'
    digits = 2
    while value < 1 and float(f'{value * 100:.{digits}f}') == 100 and digits < 12:
        digits += 2
    return f'{value * 100:+.{digits}f}%'


def kv(rows):
    return '<dl class="kv">' + ''.join(f'<dt>{e(label)}</dt><dd>{value}</dd>' for label, value in rows) + '</dl>'


def details(title, content, section_id=None):
    identity = f' id="{e(section_id)}"' if section_id else ''
    return f'<details{identity}><summary>{e(title)}</summary><div class="inside">{content}</div></details>'


def idea_summary(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ('summary', 'change', 'description', 'hypothesis', 'claim', 'proposal', 'mechanism', 'text', 'note'):
            if isinstance(value.get(key), str):
                return value[key]
        for text in value.values():
            if isinstance(text, str) and len(text) >= 60:
                return text
        if isinstance(value.get('id'), str):
            return value['id']
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return ''


def recorded_ideas(row):
    account = row.get('research_log') or {}
    return ' '.join(filter(None, (idea_summary(x) for x in account.get('ideas', [])))) or idea_summary(row.get('idea'))


def account_value(value):
    if isinstance(value, dict):
        return '<dl class="account">' + ''.join(f'<dt>{e(key.replace("_", " ").replace("-", " "))}</dt><dd>{account_value(item)}</dd>' for key, item in value.items()) + '</dl>'
    if isinstance(value, list):
        return '<ul>' + ''.join(f'<li>{account_value(item)}</li>' for item in value) + '</ul>'
    return '<span class="accounttext">' + e(value) + '</span>'


def render_account(record, row):
    account = row.get('research_log')
    if isinstance(account, dict):
        account = {key: account[key] for key in ['ideas', 'status', *[k for k in account if k not in ('ideas', 'status')]] if key in account}
    body = '<style>.account{margin:12px 0}.account dt{font-weight:650;font-size:12px;margin-top:15px;overflow-wrap:anywhere}.account dd{margin:3px 0 12px;padding-left:14px;border-left:2px solid var(--line);overflow-wrap:anywhere}.accounttext{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
    body += link('../../submission.html#history', '← Back to evaluation history')
    body += f'<div class="title"><div><div class="eyebrow" style="margin-top:24px">Recorded research account · {e(record["headline"]["task_id"])}</div><h1>Launch {n(row["launch_seq"])} · {e(row["candidate_id"])}</h1><p class="muted">The method’s own recorded ideas and status.</p></div></div>'
    body += '<section class="panel section">' + kv([('Candidate', e(row['candidate_id'])), ('Parent', e(row.get('parent_id') or 'Not supplied')), ('Evaluation status', e(row['status']))]) + '</section>'
    body += '<section class="panel section"><h2>Research account</h2>' + (account_value(account) if account else '<p class="muted">Reference evaluation; no research account was required.</p>' if row.get('is_reference') else '<p class="muted">No research account recorded.</p>') + '</section>'
    if row.get('idea'):
        body += '<section class="panel section"><h2>Additional idea metadata</h2>' + account_value(row['idea']) + '</section>'
    body += '<p class="note">This account describes the method’s intent. It does not establish the cause of a measured change.</p>'
    return shell(f'Launch {row["launch_seq"]} research account', body, nav=False)


def table(headers, rows):
    return '<div class="tablewrap"><table><thead><tr>' + ''.join(f'<th>{e(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def badge(verdict):
    ok = verdict.get('publishable')
    return f'<span class="badge {"" if ok else "fail"}"><span class="dot"></span>{"Eligible for submission" if ok else "Not eligible for submission"}</span>'


def shell(title, body, nav=True):
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' + f'<title>{e(title)} · AutoArena</title><style>{CSS}</style></head><body><header><div class="top"><div class="brand">AUTOARENA <span class="muted">/ SUBMISSION</span></div><nav class="topnav">' + (link('#progress', 'Progress') + link('submission.json', 'Submission JSON') if nav else '') + '</nav></div></header><main class="wrap">' + body + '</main></body></html>'


def chart(record):
    """Display the engine's recorded step trajectory at two readable viewport sizes."""
    points = [(r['spent'], r['best_so_far']) for r in record.get('trajectory', [])
              if isinstance(r.get('spent'), (int, float)) and isinstance(r.get('best_so_far'), (int, float))]
    if len(points) < 2:
        return '<p class="note">Too few measurements for a progress chart.</p>'
    metric = record['headline']['metric']
    unit = record['headline'].get('unit')
    factor = 1048576 if unit == 'bytes' else 1000000 if max(y for _, y in points) > 1000000 else 1
    axis_label = ('MiB' if unit == 'bytes' else 'millions' if factor == 1000000 else unit or metric)
    low, high = min(y for _, y in points), max(y for _, y in points)
    span = high - low or max(abs(high) * .02, 1)
    low, high = low - span * .07, high + span * .07
    start, finish = min(x for x, _ in points), max(x for x, _ in points)
    charts = []
    for width, cls in ((900, 'widechart'), (340, 'narrowchart')):
        height = 265 if width == 900 else 220
        left, right, top, bottom = 58, 15, 30, 40
        px = lambda x: left + (x - start) / (finish - start or 1) * (width - left - right)
        py = lambda y: top + (high - y) / (high - low) * (height - top - bottom)
        chunks = [f'<svg class="{cls}" viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Best qualifying {e(metric)} against charged evaluations"><text x="{left}" y="13" font-size="11" fill="var(--muted)">{e(axis_label)} · {"lower" if record["headline"].get("direction") == "minimize" else "higher"} is better</text>']
        for i in range(5):
            val = low + (high - low) * i / 4
            yy = py(val)
            chunks.append(f'<line x1="{left}" x2="{width-right}" y1="{yy}" y2="{yy}" stroke="var(--line)"/><text x="{left-9}" y="{yy+4}" text-anchor="end" font-size="11" fill="var(--muted)">{n(val/factor, 1)}</text>')
        steps = [f'M {px(points[0][0]):.2f} {py(points[0][1]):.2f}']
        for previous, current in zip(points, points[1:]):
            steps.append(f'L {px(current[0]):.2f} {py(previous[1]):.2f} L {px(current[0]):.2f} {py(current[1]):.2f}')
        chunks.append(f'<path d="{" ".join(steps)}" stroke="var(--blue)" stroke-width="2" fill="none"/>')
        chunks.append(f'<circle cx="{px(points[-1][0])}" cy="{py(points[-1][1])}" r="4" fill="var(--blue)"/>')
        for i in range(5):
            xx = start + (finish - start) * i / 4
            chunks.append(f'<text x="{px(xx)}" y="{height-22}" text-anchor="middle" font-size="11" fill="var(--muted)">{n(round(xx))}</text>')
        chunks.append(f'<text x="{width-right}" y="{height-3}" text-anchor="end" font-size="11" fill="var(--muted)">Charged evaluations</text></svg>')
        charts.append(''.join(chunks))
    return '<style>.chart .narrowchart{display:none}@media(max-width:720px){.chart .widechart{display:none}.chart .narrowchart{display:block}}</style>' + ''.join(charts)


def build(record):
    h = record['headline']
    cost = record.get('cost') or {}; budget = cost.get('budget') or {}
    setup = record.get('setup') or {}; gpu = setup.get('compute') or {}; llm = setup.get('research_llm') or {}
    impl = record.get('winning_implementation') or {}
    title = f"{h.get('method_id')} · {h.get('task_title') or h.get('task_id')}"
    body = f'<div class="title"><div><div class="eyebrow">One method · One task · One run</div><h1>{e(title)}</h1><p class="muted">{e(h.get("metric"))}</p></div>{badge(record["verdict"])}</div>'
    reasons = record['verdict'].get('reasons') or []
    if reasons:
        body += '<div class="callout warn"><strong>Submission findings</strong><ul>' + ''.join(f'<li>{e(x)}</li>' for x in reasons) + '</ul></div>'
    stats = [('Best qualifying result', n(h.get('best_value')), h.get('unit') or h.get('metric')),
             ('Reference', n(h.get('reference_value')), h.get('unit') or h.get('metric')),
             ('Improvement', percent(h.get('relative_improvement')), 'Lower is better' if h.get('direction') == 'minimize' else 'Higher is better'),
             ('Evaluation budget', f'{n(budget.get("spent"))} / {n(budget.get("max_launches"))}', 'charged launches')]
    body += '<section class="panel"><div class="summary">' + ''.join(f'<div class="stat"><span class="label">{label}</span><span class="value primary">{value}</span><span class="unit">{e(unit)}</span></div>' for label, value, unit in stats) + '</div></section>'
    setup_rows = [('Research model', llm.get('model')), ('LLM provider', llm.get('provider')),
                  ('Compute backend', gpu.get('backend')), ('GPU instance', gpu.get('gpu_instance_type')),
                  ('GPU model', ', '.join(sorted(set(gpu['gpu_model']))) if isinstance(gpu.get('gpu_model'), list) else gpu.get('gpu_model'))]
    if llm.get('used') is False:
        setup_rows.insert(0, ('Research LLM', 'Not used'))
    body += '<section class="panel section"><h2>Experiment setup</h2>' + kv([(label, e(value)) for label, value in setup_rows if value is not None])
    if gpu.get('workers') and gpu.get('gpus_per_evaluation'):
        body += kv([('Configured capacity', f'{n(gpu["workers"])} workers × {n(gpu["gpus_per_evaluation"])} GPUs per evaluation'),
                    ('Evaluation GPUs', n(gpu.get('evaluation_capacity_gpus')))])
    if llm.get('generation_settings') or llm.get('usage'):
        body += details('LLM settings and usage', account_value({k: llm[k] for k in ('generation_settings', 'usage') if llm.get(k)}))
    body += '</section><section class="panel section" id="progress"><h2>Search progress</h2><p class="muted small">Best qualifying result versus charged evaluations.</p><div class="chart">' + chart(record) + '</div></section>'
    body += '<section class="panel section"><h2>Winning implementation</h2>'
    winner = next((r for r in record.get('history', []) if r.get('candidate_id') == h.get('best_candidate_id')), {})
    idea = recorded_ideas(winner)
    if idea:
        body += '<div class="callout"><strong>Recorded proposal</strong><br>' + e(idea[:500]) + ('…' if len(idea) > 500 else '') + '</div>'
    if impl.get('source_verified'):
        body += '<p class="small pass">Exact source verified against the measurement-time identity.</p>'
        body += kv([('Candidate', e(h.get('best_candidate_id'))), ('Files changed', e(', '.join(impl.get('changed_files', [])) or 'None'))])
        body += '<div class="buttons">' + ''.join(link(impl[key], label, 'button') for key, label in (('download', 'Download winning code'), ('diff', 'View changes'), ('reproduce', 'Reproduce evaluation')) if impl.get(key)) + '</div>'
        body += details('Source files', table(['File', 'Size', 'Reference'], [(link(f['path'], f['name']), n(f['bytes']) + ' B', link('code/reference/' + f['name'], 'reference') if impl.get('reference') else '—') for f in impl.get('files', [])]))
    else:
        body += '<p class="muted">' + ('No qualifying candidate.' if not h.get('best_candidate_id') else 'Exact source was not found for this recorded candidate.') + '</p>'
    body += '</section><section class="panel"><div class="section"><h2>Evidence and details</h2><p>' + link('evidence/history.json', 'Complete research history') + ' · ' + link('reproduce/task.json', 'Frozen task definition') + '</p>'
    body += '<div class="buttons">' + ''.join(link(item[ch], f'{item["role"].title()} {ch}', 'button') for item in record.get('evidence', []) for ch in ('stdout', 'stderr') if item.get(ch)) + '</div></div>'
    clauses = record.get('eligibility', {}).get('clauses', [])
    body += details('Task constraint checks', table(['Metric', 'Required', 'Measured', 'Result'], [(e(c['metric']), e(c['operator']) + ' ' + n(c['bound'], 12), n(c['measured'], 12), 'Pass' if c['satisfied'] is True else 'Failed' if c['satisfied'] is False else 'Not measured') for c in clauses if c['kind'] in ('quality_gate', 'constraint')]) + account_value(record.get('obligations') or {}))
    rows = []
    for row in record.get('history', []):
        idea = recorded_ideas(row)
        summary = e(idea[:240]) + ('…' if len(idea) > 240 else '') if idea else 'Reference implementation' if row.get('is_reference') else 'No idea recorded'
        if row.get('research_account'):
            summary += '<p>' + link(row['research_account'], 'Read full account') + '</p>'
        logs = ' · '.join(link(row[ch], ch) for ch in ('stdout', 'stderr') if row.get(ch))
        rows.append([n(row['launch_seq']), e(row['candidate_id']), '<div style="min-width:260px;max-width:380px">' + summary + '</div>', e(row['status']), n((row.get('metrics') or {}).get(h.get('metric'))), 'Yes' if row.get('admissible') else 'No', n(row.get('wall_time_seconds'), 1), logs])
    body += details(f'Evaluation history · {len(rows)} launches', table(['Launch', 'Candidate', 'Recorded idea', 'Status', h.get('metric'), 'Qualifies', 'Seconds', 'Logs'], rows), 'history')
    outcome = record.get('outcome') or {}
    body += details('Budget and completion', kv([('Spent / budget', f'{n(budget.get("spent"))} / {n(budget.get("max_launches"))}'), ('Refunded', n(budget.get('refunded'))), ('Recorded GPU-hours', n(cost.get('evaluation_gpu_hours'), 3)), ('GPU-time coverage', e(cost.get('gpu_time_coverage'))), ('Outcome', e(outcome.get('category'))), ('Completion reason', e(outcome.get('stop_reason')))]))
    findings = record.get('limits', []) + record['verdict'].get('caveats', [])
    body += details('Audit and provenance', '<p>' + link('evidence/audit.json', 'Audit findings') + ' · ' + link('evidence/reconciliation.json', 'Ledger reconciliation') + ' · ' + link('evidence/source-manifest.json', 'Source hashes') + ' · ' + link('evidence/log-manifest.json', 'Log export hashes') + '</p>' + '<ul>' + ''.join(f'<li>{e(x)}</li>' for x in findings) + '</ul>' + account_value(record.get('identity') or {})) + '</section>'
    return shell(title, body)


def render_index(records):
    body = '<div class="title"><h1>Submission reports</h1></div><div class="indexgrid">'
    for row in records:
        h = row['headline']
        body += '<section class="panel section"><h2>' + e(h.get('method_id')) + ' · ' + e(h.get('task_title') or h.get('task_id')) + '</h2><p class="value primary">' + n(h.get('best_value')) + '</p><p>' + e(h.get('unit') or h.get('metric')) + ' · ' + percent(h.get('relative_improvement')) + ' improvement</p>' + badge(row['verdict']) + '<p>' + link(row['name'] + '/submission.html', 'Open submission') + '</p></section>'
    return shell('Submissions', body + '</div><p class="note">Each task is evaluated independently.</p>', nav=False)
