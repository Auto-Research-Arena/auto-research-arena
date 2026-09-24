"""Keep a Claude Code session alive for its protocol-owned background work.

This is a site-selected process transport, not a search loop or completion judge.
Native result envelopes retain session IDs; verbose tool payloads are not stored.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, field


class TransportError(Exception):
    """The native protocol cannot establish a settled process handoff."""


class Interrupted(Exception):
    def __init__(self, number):
        self.number = number


@dataclass
class Protocol:
    pending: set[tuple[str, str | None]] = field(default_factory=set)
    terminal: set[tuple[str, str | None]] = field(default_factory=set)
    task_tools: dict[str, set[str]] = field(default_factory=dict)
    awaiting_start: set[str] = field(default_factory=set)
    associated_tools: set[str] = field(default_factory=set)
    result: dict | None = None
    idle: bool = False
    capability_seen: bool = False

    @staticmethod
    def identifier(value):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
            raise TransportError("background event has no task identity")
        return value

    def invocation(self, identity, tool, *, starting=False):
        """Task IDs survive native resume; tool IDs distinguish invocations.

        A tool-less event can identify a task's sole invocation, but cannot
        distinguish a stale completion from a resumed invocation's completion.
        Refuse that ambiguity rather than closing live work.
        """
        known = self.task_tools.setdefault(identity, set())
        if tool is None:
            if starting and (known or any(task == identity for task, _ in self.pending | self.terminal)):
                raise TransportError("repeated task start lacks invocation identity")
            if len(known) > 1:
                raise TransportError("resumed task event lacks invocation identity")
            tool = next(iter(known), None)
        else:
            if not known:
                if (identity, None) in self.terminal:
                    raise TransportError("terminal task lacks invocation identity for later event")
                if (identity, None) in self.pending:
                    if starting:
                        raise TransportError("prior task start lacks invocation identity for later start")
                    self.pending.remove((identity, None))
                    self.pending.add((identity, tool))
            known.add(tool)
        return identity, tool

    def consume(self, event):
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise TransportError("invalid native JSON event")
        kind, subtype = event["type"], event.get("subtype")
        if kind in {"assistant", "user"}:
            self.result = None
            self.idle = False
            message = event.get("message", {})
            content = message.get("content", []) if isinstance(message, dict) else []
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    arguments = block.get("input", {})
                    if isinstance(arguments, dict) and arguments.get("run_in_background") is True:
                        identity = self.identifier(block.get("id"))
                        if identity not in self.associated_tools:
                            self.awaiting_start.add(identity)
                elif block.get("type") == "tool_result" and block.get("is_error") is True:
                    # A rejected tool invocation never created a background task.
                    self.awaiting_start.discard(block.get("tool_use_id"))
        elif kind == "system" and subtype in {"task_started", "task_notification"}:
            identity = self.identifier(event.get("task_id"))
            tool = event.get("tool_use_id")
            if tool is not None:
                tool = self.identifier(tool)
                self.associated_tools.add(tool)
                self.awaiting_start.discard(tool)
            invocation = self.invocation(identity, tool, starting=subtype == "task_started")
            if subtype == "task_started":
                if invocation not in self.terminal:
                    self.result = None
                    self.idle = False
                    self.pending.add(invocation)
            else:
                self.result = None
                self.idle = False
                if event.get("status") not in {"completed", "failed", "stopped"}:
                    raise TransportError("unknown background terminal status")
                self.pending.discard(invocation)
                self.terminal.add(invocation)
                # A notification may trigger another native model turn. Do not
                # close on the earlier result that preceded this notification.
        elif kind == "system" and subtype == "init":
            # Several completions can queue several native turns. An init after
            # one result announces the next turn even with no task left running.
            self.result = None
            self.idle = False
        elif kind == "system" and subtype == "session_state_changed":
            value = event.get("state")
            if value not in {"idle", "running", "requires_action"}:
                raise TransportError("unknown native session state")
            self.capability_seen = True
            # Native authoritative turn-over event, emitted after its queued
            # background-notification turns and held result have been flushed.
            # An earlier idle (before the result) cannot acknowledge that result.
            self.idle = value == "idle" and self.result is not None
            if value != "idle":
                self.result = None
        elif kind == "result":
            if type(event.get("is_error")) is not bool:
                raise TransportError("native result lacks a typed error status")
            queued = event.get("queued_turn_count", 0)
            if type(queued) is not int or queued < 0:
                raise TransportError("native result has invalid queued-turn count")
            self.result = event
            self.idle = False

    @property
    def settled(self):
        return (self.idle and self.result is not None and self.result.get("queued_turn_count", 0) == 0
                and not self.pending and not self.awaiting_start)


def native_command(arguments):
    """Change framing only; retain model, permissions, resume and all other argv."""
    if not arguments or not all(isinstance(x, str) and "\0" not in x for x in arguments):
        raise TransportError("a native CLI argv is required")
    result, formats = [arguments[0]], set()
    index = 1
    while index < len(arguments):
        item = arguments[index]
        name, separator, inline = item.partition("=")
        if name in {"--input-format", "--output-format"}:
            if name in formats:
                raise TransportError("duplicate native framing option")
            formats.add(name)
            if separator:
                value = inline
            else:
                index += 1
                if index == len(arguments):
                    raise TransportError("missing native framing value")
                value = arguments[index]
            allowed = {"text", "stream-json"} if name == "--input-format" else {"text", "json", "stream-json"}
            if value not in allowed:
                raise TransportError("invalid native framing value")
        elif item in {"--bg", "--background"}:
            raise TransportError("whole-session background mode conflicts with print transport")
        else:
            result.append(item)
        index += 1
    if not any(x in result for x in ("--print", "-p")):
        result.append("--print")
    if "--verbose" not in result:
        result.append("--verbose")
    return result + ["--input-format", "stream-json", "--output-format", "stream-json"]


def _positive(value, name, *, zero=False):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise TransportError(f"invalid {name}")


def visible_event(event):
    """Retain the old final-result contract, not newly verbose private payloads."""
    if event.get("type") == "result":
        return event
    if event.get("type") != "system" or event.get("subtype") not in {
            "init", "task_started", "task_notification", "session_state_changed"}:
        return None
    result = {"type": "system", "subtype": "autoarena_transport",
              "event": event["subtype"]}
    for key in ("session_id", "task_id", "tool_use_id"):
        if key in event:
            result[key] = Protocol.identifier(event[key])
    if event["subtype"] == "task_notification":
        result["status"] = event["status"]
    if event["subtype"] == "session_state_changed":
        result["state"] = event["state"]
    return result


@contextmanager
def _defer_signals():
    """Assign ownership before raising; never pass a blocked mask to a child."""
    requested, previous = [], {}
    try:
        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGTERM, signal.SIGINT):
                previous[number] = signal.signal(number, lambda number, frame: requested.append(number))
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    if requested:
        raise Interrupted(requested[0])


def _exited(process):
    # Keep the direct child unreaped until exceptional group cleanup has finished.
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _cleanup(process, timeout, failed):
    if process.stdin and not process.stdin.closed:
        try:
            process.stdin.close()
        except (OSError, BrokenPipeError):
            pass
    if failed:
        # The unreaped direct child reserves the original group identity even
        # after abnormal EOF. Different-group acknowledged workers are untouched.
        for number in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, number)
            except ProcessLookupError:
                pass
            if number == signal.SIGTERM:
                deadline = time.monotonic() + timeout
                while not _exited(process) and time.monotonic() < deadline:
                    time.sleep(.01)
        process.wait()


def run(arguments, prompt, *, stdout=None, stderr=None, settle_seconds=0.25,
        shutdown_timeout=10.0, timeout_seconds=None):
    _positive(settle_seconds, "settle interval", zero=True)
    _positive(shutdown_timeout, "shutdown timeout")
    if timeout_seconds is not None:
        _positive(timeout_seconds, "transport timeout")
    stdout = sys.stdout.buffer if stdout is None else stdout
    stderr = sys.stderr.buffer if stderr is None else stderr
    command = native_command(arguments)
    payload = json.dumps({"type": "user", "message": {"role": "user", "content": prompt},
                          "parent_tool_use_id": None, "session_id": "default"}) + "\n"
    process = None
    selector = selectors.DefaultSelector()
    state = Protocol()
    buffer = b""
    started = last_event = time.monotonic()
    closing = None
    failed = True
    exit_observed = None
    try:
        with _defer_signals():
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=True,
                                       env={**os.environ, "CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS": "1"})
        outgoing = payload.encode()
        os.set_blocking(process.stdin.fileno(), False)
        selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while True:
            now = time.monotonic()
            if not state.capability_seen and now - started > 30:
                raise TransportError("native session-state startup handshake timed out")
            if timeout_seconds is not None and now - started > timeout_seconds:
                raise TransportError("transport wall timeout; no completion inferred")
            for key, _ in selector.select(0.1):
                if key.data == "stdin":
                    try:
                        outgoing = outgoing[os.write(key.fileobj.fileno(), outgoing):]
                    except BlockingIOError:
                        continue
                    if not outgoing:
                        selector.unregister(key.fileobj)
                    continue
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                if key.data == "stderr":
                    # Native stderr was already part of the previous CLI contract.
                    stderr.write(chunk)
                    stderr.flush()
                    continue
                last_event = time.monotonic()
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeError) as error:
                        raise TransportError("malformed native JSON; no completion inferred") from error
                    if isinstance(event, dict) and event.get("type") == "result" and not state.capability_seen:
                        raise TransportError("native CLI lacks session-state handshake; supported protocol required")
                    state.consume(event)
                    visible = visible_event(event)
                    if visible is not None:
                        stdout.write((json.dumps(visible) + "\n").encode())
                        stdout.flush()
                    if closing is not None and not state.settled:
                        raise TransportError("native activity appeared after input closure")
                if len(buffer) > 16 * 1024 * 1024:
                    raise TransportError("oversized incomplete native JSON event")
            now = time.monotonic()
            if closing is None and state.settled and not buffer and not outgoing and now-last_event >= settle_seconds:
                process.stdin.close()
                closing = now
            if closing is not None and now-closing > shutdown_timeout:
                raise TransportError("native CLI did not close after a settled result")
            if _exited(process):
                if exit_observed is None:
                    exit_observed = now
                if selector.get_map() and now - exit_observed > shutdown_timeout:
                    raise TransportError("native CLI exited with unclosed protocol pipes")
            if exit_observed is not None and not selector.get_map():
                if buffer.strip():
                    raise TransportError("truncated native JSON; no completion inferred")
                if not state.settled:
                    raise TransportError("native EOF with unfinished protocol work")
                if closing is None:
                    raise TransportError("native CLI exited before transport settlement")
                with _defer_signals():
                    code = process.wait()
                    failed = False
                return code or (1 if state.result["is_error"] else 0)
    finally:
        with _defer_signals():
            try:
                if process is not None:
                    _cleanup(process, shutdown_timeout, failed)
            finally:
                selector.close()
                if process is not None:
                    process.stdout.close()
                    process.stderr.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--settle-seconds", type=float, default=0.25)
    parser.add_argument("--shutdown-timeout", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    previous = {}
    def interrupted(number, _frame):
        raise Interrupted(number)
    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, interrupted)
        return run(command, sys.stdin.read(), settle_seconds=args.settle_seconds,
                   shutdown_timeout=args.shutdown_timeout, timeout_seconds=args.timeout_seconds)
    except Interrupted as error:
        return 128 + error.number
    except (TransportError, OSError) as error:
        # Do not print argv, prompt, credentials or arbitrary native content here.
        message = str(error) if isinstance(error, TransportError) else type(error).__name__
        print(f"Claude transport error: {message}", file=sys.stderr)
        return 1
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
