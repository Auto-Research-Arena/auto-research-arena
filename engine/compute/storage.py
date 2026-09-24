"""Repository-local runtime storage and process-local dataset mounts."""
from pathlib import Path
import shutil

from engine import ROOT


def cache_environment():
    root = ROOT / '.local'
    cache = root / 'cache'
    paths = {key: str(path) for key, path in {
        'UV_CACHE_DIR': cache / 'uv', 'UV_PYTHON_INSTALL_DIR': root / 'python',
        'PIP_CACHE_DIR': cache / 'pip', 'XDG_CACHE_HOME': cache,
        'HF_HOME': cache / 'huggingface', 'TORCH_HOME': cache / 'torch',
        'TORCH_EXTENSIONS_DIR': cache / 'torch_extensions',
        'TRITON_CACHE_DIR': cache / 'triton', 'CUDA_CACHE_PATH': cache / 'cuda',
        'TMPDIR': root / 'tmp',
    }.items()}
    return {**paths, "UV_LINK_MODE": "copy"}


def mount_command(command, mounts):
    """Expose repo data at a frozen task's cache path only inside its process."""
    if not mounts:
        return command
    executable = shutil.which('bwrap')
    if not executable:
        raise RuntimeError('dataset mounts require bubblewrap (bwrap)')
    prefix = [executable, '--die-with-parent', '--bind', '/', '/', '--dev-bind', '/dev', '/dev', '--proc', '/proc']
    for alias, source in mounts.items():
        target = Path(source).expanduser().resolve()
        if not target.is_dir():
            raise ValueError(f'dataset directory is missing: {target}')
        prefix += ['--bind', str(target), str(Path(alias).expanduser().resolve())]
    return prefix + ['--', *command]
