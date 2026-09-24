"""Extract a task's declared metrics from one launch's own output.

Every method, and the audit, reads the same number from the same bytes by calling
this module. That is the point: a metric is an extraction spec, not a name, and a
method that parses the log its own way is producing a number that cannot be
compared with anyone else's.

Three failure modes this module exists to prevent, all of which have happened:

- **Reading a rounded twin.** The substrate prints `num_params_M` at `%.1f`, a
  100,000-parameter quantum on a metric that is a ranking key. A metric declares
  `never_use` so the trap stays visible, and ranking metrics are extracted from a
  source that has exact precision.
- **Believing one of two disagreeing sources.** `total_tokens` is a product of a
  printed step count and a constant in the *mutable* file, cross-checked against
  the printed `total_tokens_M`. A mismatch is an extraction failure, recorded as
  such, rather than a silent choice between them.
- **Reading a number the candidate computed about itself.** `metrics_json` reads
  the single `METRICS_JSON:` line emitted by an immutable instrument, and it is
  the only extractor that can be trusted on a substrate whose mutable file could
  otherwise print its own score. On a substrate where the editable file computed
  the metrics, a search on arithmetic cost reached exactly zero by routing the
  work through operations the counter did not register.

A metric that is absent is `None`, not zero and not omitted. A launch that
crashed before printing has null metrics and is still a ledger row.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TRUE_WORDS = ("true", "yes", "1")
FALSE_WORDS = ("false", "no", "0")

#: The one accessor on an instrumented substrate. One line, one JSON object, full
#: precision, printed by a file the candidate may not edit.
METRICS_JSON_PREFIX = "METRICS_JSON: "
_METRICS_JSON_LINE = re.compile(r"^" + METRICS_JSON_PREFIX + r"(\{.*)$", re.MULTILINE)


class ExtractionError(Exception):
    """A declared metric could not be read, and guessing is not an option."""


class _MetricsJson:
    """The launch's `METRICS_JSON` object, parsed at most once per extraction.

    Absent is distinguished from unparseable on purpose. An absent line means the
    instrument never reported, which is a forfeit; a malformed one means the line
    was corrupted or truncated, which is a different fault with a different fix.
    """

    def __init__(self, stdout: str) -> None:
        self._stdout = stdout
        self._parsed: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None
        self._done = False

    def get(self, name: str, field: str) -> Any:
        if not self._done:
            self._done = True
            # Carriage returns: the substrate's progress line uses `\r`, so the
            # instrument's line may not start at a `\n`. Normalise before matching.
            matches = _METRICS_JSON_LINE.findall(self._stdout.replace("\r", "\n"))
            if matches:
                try:
                    # The last line wins, as with every other extractor here: a
                    # restarted or resumed process can print more than one.
                    loaded = json.loads(matches[-1])
                    if isinstance(loaded, dict):
                        self._parsed = loaded
                    else:
                        self._error = "is not a JSON object"
                except json.JSONDecodeError as error:
                    self._error = f"does not parse as JSON ({error})"
        if self._parsed is None:
            raise ExtractionError(
                f"{name}: the {METRICS_JSON_PREFIX.strip()} line "
                + (self._error or "is absent from this launch's stdout")
                + ". It is the only accessor for this metric and there is no fallback: a "
                "human-readable line is rounded, and on a mutable substrate it is "
                "written by the candidate. Record the launch as a forfeit instead"
            )
        if field not in self._parsed:
            return None
        return self._parsed[field]


def extract_metrics(
    task: Any,
    stdout: str,
    source_root: Optional[Path] = None,
) -> Tuple[Dict[str, Optional[float]], List[str]]:
    """Return (metrics, errors) for every metric the task declares.

    `stdout` is the launch's own captured stdout. `source_root` is the candidate's
    run copy, needed only for `source_constant` metrics -- and it must be the
    candidate's copy, never the task's pristine substrate, because a constant in a
    mutable file is part of what the candidate changed.
    """
    specs: Dict[str, Any] = dict(task.metrics)
    values: Dict[str, Optional[float]] = {}
    errors: List[str] = []
    instrument = _MetricsJson(stdout)

    # Products depend on other metrics, so resolve them last. Two passes suffice
    # because a product's factors may not themselves be products (task validation
    # allows it structurally, so guard by iterating until stable).
    pending = list(specs.items())
    for _ in range(len(pending) + 1):
        deferred = []
        for name, spec in pending:
            kind = spec["extract"]["kind"]
            if kind == "product" and any(
                factor not in values for factor in spec["extract"]["factors"]
            ):
                deferred.append((name, spec))
                continue
            try:
                values[name] = _extract_one(
                    name, spec, stdout, source_root, values, instrument
                )
            except ExtractionError as error:
                values[name] = None
                errors.append(str(error))
        if not deferred:
            break
        if len(deferred) == len(pending):
            for name, _ in deferred:
                values[name] = None
                errors.append(f"{name}: product factors form a cycle or are unreadable")
            break
        pending = deferred

    errors.extend(_cross_check(specs, values))
    return values, errors


def _extract_one(
    name: str,
    spec: Dict[str, Any],
    stdout: str,
    source_root: Optional[Path],
    resolved: Dict[str, Optional[float]],
    instrument: "_MetricsJson",
) -> Optional[float]:
    extract = spec["extract"]
    kind = extract["kind"]
    if kind == "metrics_json":
        value = instrument.get(name, extract["field"])
        # JSON null is a reported absence, not a failure: the cache-latency probe
        # reports null on a model that has no cache protocol, and that is the
        # honest answer for it rather than a zero.
        return None if value is None else _coerce(name, spec, value)
    if kind == "summary_field":
        raw = _summary_field(stdout, extract["field"])
    elif kind == "block_field":
        raw = _block_field(stdout, extract["block"], extract["field"])
    elif kind == "source_constant":
        raw = _source_constant(source_root, extract["file"], extract["name"])
    elif kind == "source_regex":
        return _source_regex(name, source_root, extract["file"], extract["pattern"])
    elif kind == "product":
        factors = [resolved.get(factor) for factor in extract["factors"]]
        if any(factor is None for factor in factors):
            return None
        product: float = 1
        for factor in factors:
            product *= factor  # type: ignore[operator]
        return _coerce(name, spec, product)
    else:
        raise ExtractionError(f"{name}: unsupported extractor {kind!r}")

    if raw is None:
        return None
    for token in extract.get("strip", ()):
        raw = raw.replace(token, "")
    return _coerce(name, spec, raw.strip())


def _summary_field(stdout: str, field: str) -> Optional[str]:
    pattern = re.compile(rf"^{re.escape(field)}\s*:\s*(.+?)\s*$", re.MULTILINE)
    matches = pattern.findall(stdout)
    if not matches:
        return None
    # The last occurrence wins: the substrate's final summary block prints after
    # any progress line that happens to share the field name.
    return matches[-1]


def _block_field(stdout: str, block: str, field: str) -> Optional[str]:
    lines = stdout.splitlines()
    try:
        start = next(
            index for index, line in enumerate(lines) if line.strip() == block.strip()
        )
    except StopIteration:
        return None
    pattern = re.compile(rf"^\s*{re.escape(field)}\s*:\s*(.+?)\s*$")
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        match = pattern.match(line)
        if match:
            return match.group(1)
    return None


def _source_constant(source_root: Optional[Path], file: str, name: str) -> Optional[str]:
    if source_root is None:
        raise ExtractionError(
            f"{name}: declared as a source constant but no candidate source root was given; "
            "it must be read from the candidate's own copy, since the file is mutable"
        )
    path = Path(source_root) / file
    if not path.is_file():
        raise ExtractionError(f"{name}: {path} does not exist in the candidate source")
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as error:
        raise ExtractionError(f"{name}: {path} does not parse: {error}") from error
    # Walk module-level assignments in order, keeping every constant we can fold. A
    # candidate writes `TOTAL_BATCH_SIZE = 2**19`, and it may equally write
    # `TOTAL_BATCH_SIZE = 4 * DEVICE_BATCH_SIZE * MAX_SEQ_LEN`, so a literal-only read
    # fails on ordinary code. Folding is restricted to arithmetic over constants and
    # earlier module-level names: enough for a knob, and it never executes candidate code.
    known: Dict[str, Any] = {}
    found: Any = None
    for node in tree.body:  # module level only; a nested constant is not the knob
        targets: List[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        folded = _fold(value, known)
        for target in targets:
            if isinstance(target, ast.Name):
                if folded is not None:
                    known[target.id] = folded
                if target.id == name:
                    if folded is None:
                        raise ExtractionError(
                            f"{name}: {file} assigns it an expression this cannot fold "
                            f"({ast.dump(value)[:120]}); reading it would mean executing "
                            "candidate code, so it is refused rather than guessed"
                        )
                    # Last assignment wins: a later rebinding is what actually ran.
                    found = folded
    if found is None:
        return None
    return str(found)


_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}
MAX_EXPONENT = 64  # 2**64 is already absurd for a batch size; refuse a fork bomb


def _fold(node: ast.expr, known: Dict[str, Any]) -> Any:
    """Constant-fold an expression without executing anything. None means 'cannot'."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, (int, float, bool, str)) else None
    if isinstance(node, ast.Name):
        return known.get(node.id)
    if isinstance(node, ast.UnaryOp):
        operand = _fold(node.operand, known)
        if not isinstance(operand, (int, float)):
            return None
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        return None
    if isinstance(node, ast.BinOp):
        left, right = _fold(node.left, known), _fold(node.right, known)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return None
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                return None
            return left**right
        handler = _BINOPS.get(type(node.op))
        if handler is None or (right == 0 and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod))):
            return None
        return handler(left, right)
    return None


def _source_regex(
    name: str, source_root: Optional[Path], file: str, pattern: str
) -> Optional[bool]:
    """Whether a pattern appears in the candidate's source. A heuristic, on purpose.

    Some facts about a candidate are not printed and are not a constant -- whether it
    ties its embedding to its output head, for instance. This reads them from source
    by pattern, which is fragile in the way pattern-matching is always fragile, so a
    `source_regex` metric may not be an objective or a ceiling. It exists to make a
    cross-method comparison auditable: if one method ties weights and another does
    not, the reported parameter count means different things in the two rows while
    each method remains internally sound.
    """
    if source_root is None:
        raise ExtractionError(
            f"{name}: declared as a source pattern but no candidate source root was given"
        )
    path = Path(source_root) / file
    if not path.is_file():
        raise ExtractionError(f"{name}: {path} does not exist in the candidate source")
    return re.search(pattern, path.read_text()) is not None


def _coerce(name: str, spec: Dict[str, Any], raw: Any) -> Optional[float]:
    declared = spec["type"]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if declared == "boolean":
            lowered = text.lower()
            if lowered in TRUE_WORDS:
                return True  # type: ignore[return-value]
            if lowered in FALSE_WORDS:
                return False  # type: ignore[return-value]
            raise ExtractionError(f"{name}: {text!r} is not a boolean")
        try:
            number: float = float(text)
        except ValueError:
            raise ExtractionError(
                f"{name}: {text!r} is not numeric. A numeric field arriving as a string has "
                "produced NaN comparisons that silently rejected every result with a "
                "plausible-looking reason, so this refuses instead of coercing"
            ) from None
    elif isinstance(raw, bool):
        return raw  # type: ignore[return-value]
    elif isinstance(raw, (int, float)):
        number = float(raw)
    else:
        raise ExtractionError(f"{name}: cannot coerce {type(raw).__name__}")

    if declared == "integer":
        if abs(number - round(number)) > 1e-9:
            raise ExtractionError(
                f"{name}: declared integer but read {number!r}; a ranking key read at reduced "
                "precision equates candidates that differ"
            )
        return int(round(number))
    return number


def _cross_check(specs: Dict[str, Any], values: Dict[str, Optional[float]]) -> List[str]:
    errors: List[str] = []
    for name, spec in specs.items():
        cross = spec.get("cross_check")
        if not isinstance(cross, dict):
            continue
        mine, theirs = values.get(name), values.get(cross["against"])
        if mine is None or theirs is None:
            continue
        expected = float(theirs) * float(cross["scale"])
        if abs(float(mine) - expected) > float(cross["tolerance"]):
            values[name] = None
            errors.append(
                f"{name}: {mine} disagrees with {cross['against']} ({theirs} x "
                f"{cross['scale']} = {expected}) beyond {cross['tolerance']}. Two sources for "
                "one quantity disagree, so neither is recorded as the value: fix the "
                "extraction rather than choosing one"
            )
    return errors
