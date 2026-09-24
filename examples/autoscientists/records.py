"""Export native scientific records while the private service is still available."""
import json
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


def export_records(focus, destination, endpoint, token, workshop):
    focus, destination = Path(focus), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    secrets = [token]
    for path in focus.glob('agents/*/credentials.json'):
        key = json.loads(path.read_text()).get('api_key')
        if key:
            secrets.append(key)

    def write(path, text):
        for secret in secrets:
            text = text.replace(secret, '[REDACTED]')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    suffixes = {'.md', '.json', '.jsonl', '.py', '.patch', '.diff', '.csv', '.tsv', '.yaml', '.yml'}
    roots = ('agents', 'champion', 'teams', 'logs', 'system', 'task', 'diagnostics', 'reports', 'knowledge')
    excluded = {'credentials.json', 'agent_tokens.json'}
    for path in focus.rglob('*'):
        relative = path.relative_to(focus)
        if (not path.is_file() or path.is_symlink()
                or any(part.startswith('.') for part in relative.parts)
                or path.name in excluded or 'sessions' in relative.parts
                or not path.resolve().is_relative_to(focus.resolve())):
            continue
        if ((len(relative.parts) == 1 or relative.parts[0] in roots)
                and (path.suffix in suffixes or path.name in {'SOURCE', 'WORKSPACE_ID', 'WORKSHOP_NAME'})):
            write(destination / relative, path.read_text())

    def get(path):
        request = Request(endpoint + path, headers={'Authorization': 'Bearer ' + token})
        with urlopen(request, timeout=15) as response:
            return json.load(response)

    def pages(path, key):
        rows, offset = [], 0
        while True:
            separator = '&' if '?' in path else '?'
            batch = get(path + separator + urlencode({'limit': 200, 'offset': offset}))[key]
            rows.extend(batch)
            if len(batch) < 200:
                return rows
            offset += len(batch)

    query = '?' + urlencode({'workshop': workshop})
    for workspace in pages('/workspaces' + query, 'workspaces'):
        base = '/workspaces/' + quote(workspace['id'], safe='')
        files = []
        for entry in get(base + '/files')['files']:
            path = entry['path']
            if Path(path).name in excluded or any(part.startswith('.') for part in Path(path).parts):
                continue
            url = base + '/files/' + quote(path, safe='/')
            files.append({'file': get(url), 'revisions': pages(url + '/history', 'history')})
        document = {'workspace': workspace, 'files': files,
                    'comments': get(base + '/comments')['comments'],
                    'workspace_comment_limit': 100}
        write(destination / 'service' / ('workspace-' + workspace['id'] + '.json'),
              json.dumps(document, indent=2) + '\n')
    posts = pages('/posts' + query, 'posts')
    for post in posts:
        post['comments'] = pages('/posts/' + quote(post['id'], safe='') + '/comments', 'comments')
    write(destination / 'service/posts.json', json.dumps(posts, indent=2) + '\n')
