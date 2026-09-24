"""Local API delivery to the engine outside a method's device sandbox."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import socket
import socketserver
import threading

from engine.runtime import lifecycle as runtime


def address(run_dir):
    # Linux abstract sockets avoid filesystem path limits and sandbox mount aliases.
    digest = hashlib.sha256(str(Path(run_dir).resolve()).encode()).hexdigest()[:32]
    return "\0autoarena-" + digest


def _execute(run_dir, operation, arguments):
    operations = {
        "evaluate": runtime.evaluate,
        "experiment-status": runtime.experiment_status,
        "status": runtime.status,
        "finish": runtime.finish,
    }
    if operation not in operations:
        raise runtime.EngineError("unknown experiment API operation")
    return operations[operation](run_dir, **arguments)


def invoke(run_dir, operation, **arguments):
    """Use the active engine; allow direct operator calls while it is offline."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        try:
            connection.connect(address(run_dir))
        except ConnectionRefusedError:
            # An active controller without its service must not launch evaluation
            # in the caller (which may be inside the method's device sandbox).
            try:
                with runtime.lock(Path(run_dir) / "engine/controller.lock", nonblocking=True):
                    pass
            except runtime.EngineError as error:
                raise runtime.EngineError("the active engine's API service is unavailable; "
                                          "inspect the original request before recovery") from error
            return _execute(run_dir, operation, arguments)
        connection.sendall((json.dumps({"operation": operation, "arguments": arguments},
                                       allow_nan=False) + "\n").encode())
        with connection.makefile("rb") as incoming:
            response = incoming.readline()
    if not response:
        raise runtime.EngineError("API connection closed; inspect the original request before recovery")
    response = json.loads(response)
    if "error" in response:
        raise runtime.EngineError(response["error"])
    return response["result"]


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            request = json.loads(self.rfile.readline())
            result = _execute(self.server.run_dir, request["operation"], request["arguments"])
            response = {"result": result}
        except Exception as error:
            response = {"error": f"{type(error).__name__}: {error}"}
        try:
            # Raw measurement responses may contain nonfinite failure metrics.
            self.wfile.write((json.dumps(response) + "\n").encode())
        except OSError:
            pass  # A disconnected client does not cancel an accepted evaluation.


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


@contextmanager
def serve(run_dir):
    """Own the listener with the controller; detached evaluation workers survive it."""
    with _Server(address(run_dir), _Handler) as server:
        server.run_dir = Path(run_dir).resolve()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield
        finally:
            server.shutdown()
            thread.join()
