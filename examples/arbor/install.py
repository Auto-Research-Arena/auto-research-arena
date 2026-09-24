"""Install pinned Arbor with benchmark evaluation and prompt adaptations."""
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def patch_native_hooks(source):
    path = Path(source) / "src/coordinator/tools/git_ops.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != "8a182c96884a08384d94e3d37dac68e639831828b66bb88a48ab6abe3513039e":
        raise RuntimeError("Pinned Arbor merge implementation differs")
    text = path.read_text()
    start = text.index("        # ── Auto-run B_test evaluation")
    end = text.index("        merge_threshold = self._config.merge_threshold", start)
    replacement = '''        # AutoArena supplies one canonical evaluation, with no B_test rerun.
        from arbor.arena_bridge import recorded_branch
        try:
            candidate = recorded_branch(self.cwd, source_branch)
            incumbent = recorded_branch(self.cwd, target_branch)
        except (ValueError, OSError, RuntimeError) as error:
            return f"Merge rejected: canonical evidence unavailable: {error}"
        if not candidate["eligible"]:
            return f"Merge rejected: task constraints failed: {candidate['reason']}"
        test_score = candidate["score"]
        test_trunk_score = incumbent["score"]
        medal_info = None
        log.info("Using canonical candidate/trunk results; no independent B_test measurement")

'''
    if text.count("        # ── Auto-run B_test evaluation") != 1:
        raise RuntimeError("Pinned Arbor merge boundary changed")
    text = text[:start] + replacement + text[end:]
    old = '        verified_tag = " (independently verified)" if eval_cmd_test else " (LLM-reported, NOT verified)"'
    if text.count(old) != 1:
        raise RuntimeError("Pinned Arbor merge evidence label changed")
    text = text.replace(old, '        verified_tag = " (canonical measurement; no independent confirmation)"')
    path.write_text(text)


def patch_executor_timeout_guidance(source):
    """State the evaluation command's walltime where RunTraining timeouts are set."""
    path = Path(source) / "src/executor/prompts.py"
    text = path.read_text()
    old = "Estimate: epochs * time_per_epoch * folds * 1.5.\n"
    if text.count(old) != 1:
        raise RuntimeError("Pinned Arbor RunTraining timeout guidance changed")
    path.write_text(text.replace(
        old, "Estimate: epochs * time_per_epoch * folds * 1.5. \\\nUse `timeout=7200` for the evaluation command.\n"))


def patch_measurement_guidance(source):
    """Include measurement rules whenever either native system prompt is built."""
    guidance = (ROOT / "measurement.md").read_text().strip()
    anchor = "        _system_section(),\n"
    for role in ("coordinator", "executor"):
        path = Path(source) / "src" / role / "prompts.py"
        text = path.read_text()
        if text.count(anchor) != 1:
            raise RuntimeError(f"Pinned Arbor {role} system prompt changed")
        path.write_text(text.replace(anchor, anchor + f"        {guidance!r},\n"))


def main():
    source = ROOT / "upstream"
    patch_native_hooks(source)
    patch_measurement_guidance(source)
    patch_executor_timeout_guidance(source)
    shutil.copyfile(ROOT / "bridge.py", source / "src/arena_bridge.py")
    subprocess.run([sys.executable, "-m", "pip", "install", str(source), "boto3"], check=True)


if __name__ == "__main__":
    main()
