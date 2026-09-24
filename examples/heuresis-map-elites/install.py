"""Install the pinned native implementation plus explicit support overlays."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PIN = "51c737575090271075d5751f26ef6183c8b282b0"


def patch_native_hooks(upstream: Path) -> None:
    """Apply reviewed API lifecycle and installed-package hooks."""
    loop = upstream / "src/heuresis/loops/map_elites.py"
    text = loop.read_text()
    old = "        while loop.should_continue():\n"
    if text.count(old) != 1:
        raise RuntimeError("Native iteration boundary changed")
    text = text.replace(old, old + "            if adapter.stop_requested():\n                return\n")
    old = "classifier.classify_metrics(obj_verdict.metrics)"
    if text.count(old) != 1:
        raise RuntimeError("Native objective classification boundary changed")
    text = text.replace(old, "classifier.classify(idea, result.workspace)")
    loop.write_text(text)
    # Preserve native LLM calls, key rotation and fallback; identify the path
    # actually used for each executor in the native log.
    features = upstream / "src/heuresis/qd/core/features.py"
    text = features.read_text()
    old = "                return self._try_llm(idea, workspace, api_key)"
    if text.count(old) != 1:
        raise RuntimeError("Native LLM classifier changed")
    text = text.replace(old,
        '                result = self._try_llm(idea, workspace, api_key)\n'
        '                print(f"Classifier: model={self.model} workspace={workspace}", flush=True)\n'
        '                return result')
    old = "        return self.fallback.classify(idea, workspace)"
    if text.count(old) != 1:
        raise RuntimeError("Native classifier fallback changed")
    text = text.replace(old,
        '        print(f"Classifier: keyword_fallback model={self.model} workspace={workspace}", flush=True)\n'
        + old)
    features.write_text(text)
    # Suspicious evidence is regraded from the original canonical execution.
    # Re-running unchanged code would purchase the same benchmark candidate twice.
    experiment = upstream / "src/heuresis/experiment.py"
    text = experiment.read_text()
    old = "        if regenerate(task_dir, exec_workspace, gpu_ids=gpu_ids):"
    if text.count(old) != 1:
        raise RuntimeError("Native judge verification boundary changed")
    text = text.replace(old,
        "        from heuresis.tasks.nanogpt.verification import restore_evidence\n"
        "        if restore_evidence(exec_workspace):")
    experiment.write_text(text)
    # Wheel installs retain package templates but not the upstream checkout root.
    workspace = upstream / "src/heuresis/workspace.py"
    text = workspace.read_text()
    old = '        search_paths = [_PROJECT_ROOT, _PROJECT_ROOT / "src"]\n'
    if text.count(old) != 1:
        raise RuntimeError("Native template lookup changed")
    text = text.replace(old, '        search_paths = [Path(__file__).resolve().parents[1], _PROJECT_ROOT, _PROJECT_ROOT / "src"]\n')
    # The judge installs its sandbox dependencies from the supplied checkout.
    old = '_PROJECT_ROOT = Path(__file__).resolve().parents[2]\n'
    if text.count(old) != 1:
        raise RuntimeError("Native project root changed")
    text = text.replace(old, old + f"_CHECKOUT_ROOT = Path({str(upstream)!r})\n")
    old = '        install_source = f"{_PROJECT_ROOT}[{project_extra}]"\n'
    if text.count(old) != 1:
        raise RuntimeError("Native sandbox venv install source changed")
    text = text.replace(old, '        install_source = f"{_CHECKOUT_ROOT}[{project_extra}]"\n')
    workspace.write_text(text)
    # A fresh lane must never mount ~/.claude/projects or foreign session history.
    agent = upstream / "src/heuresis/agent.py"
    text = agent.read_text()
    old = '        DataMount("~/.claude", "/workspace/.claude"),'
    if text.count(old) != 1:
        raise RuntimeError("Native Claude profile changed")
    agent.write_text(text.replace(old,
        '        DataMount("~/.claude/.credentials.json", "/workspace/.claude/.credentials.json"),'))


def main():
    upstream = ROOT / "upstream"
    provenance = json.loads((ROOT / "support/provenance.json").read_text())
    for name, hashes in provenance["files"].items():
        if "upstream_sha256" in hashes:
            digest = hashlib.sha256((upstream / name).read_bytes()).hexdigest()
            if digest != hashes["upstream_sha256"]:
                raise RuntimeError(f"Pinned upstream file differs: {name}")
    subprocess.run(["git", "apply", "--unsafe-paths", str(ROOT / "support/runtime.patch")], cwd=upstream, check=True)
    shutil.copytree(ROOT / "support/nanogpt", upstream / "src/heuresis/tasks/nanogpt", dirs_exist_ok=True)
    patch_native_hooks(upstream)
    subprocess.run([sys.executable, "-m", "pip", "install", str(upstream)], check=True)


if __name__ == "__main__":
    main()
