import json
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
sys.path[:0] = [str(EXAMPLES), str(EXAMPLES / "distributed_http")]

import categorical  # noqa: E402
import client  # noqa: E402
import quickstart  # noqa: E402
import server  # noqa: E402

# --- stub objective server ----------------------------------------------------


class Stub:
    """A stdlib HTTP server that answers each request with the next (status, body) in ``replies``
    (the last one repeats), or computes the objective when ``replies`` is None."""

    def __init__(self, replies=None):
        stub = self
        self.replies, self.requests = replies, []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.path).query))
                stub.requests.append(query)
                if stub.replies is None:
                    status, body = (
                        200,
                        {"score": server.objective(**{k: json.loads(v) for k, v in query.items()})},
                    )
                else:
                    status, body = stub.replies[min(len(stub.requests), len(stub.replies)) - 1]
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(client.time, "sleep", sleeps.append)
    return sleeps


# --- client (spec 008) --------------------------------------------------------


def test_X_1_zero_score_is_returned_on_first_request(no_sleep):
    stub = Stub([(200, {"score": 0.0})])
    try:
        assert client.remote_score(stub.port, {"x1": 0.0, "x2": 0.0}) == 0.0
        assert len(stub.requests) == 1 and no_sleep == []
    finally:
        stub.close()


def test_client_error_is_raised_without_retry(no_sleep):
    stub = Stub([(400, {"error": "missing=['x2']"})])
    try:
        with pytest.raises(client.ObjectiveError, match="400"):
            client.remote_score(stub.port, {"x1": 1.0})
        assert len(stub.requests) == 1
    finally:
        stub.close()


def test_server_errors_are_retried_then_succeed(no_sleep):
    stub = Stub([(503, {"error": "busy"}), (503, {"error": "busy"}), (200, {"score": 2.5})])
    try:
        assert client.remote_score(stub.port, {"x1": 1.0}) == 2.5
        assert len(stub.requests) == 3 and no_sleep == [1, 2]
    finally:
        stub.close()


def test_unreachable_server_gives_up_with_bounded_backoff(no_sleep):
    stub = Stub([(200, {"score": 1.0})])
    port = stub.port
    stub.close()  # nothing listens on this port any more
    with pytest.raises(client.ObjectiveError, match="4 attempts"):
        client.remote_score(port, {"x1": 1.0}, retries=3)
    assert no_sleep == [1, 2, 4]


def test_values_are_json_encoded_in_the_query():
    stub = Stub([(200, {"score": 1.0})])
    try:
        import numpy as np

        client.remote_score(stub.port, {"act": "relu", "n": np.int64(3), "lr": np.float64(0.01)})
        assert stub.requests[0] == {"act": '"relu"', "n": "3", "lr": "0.01"}
    finally:
        stub.close()


def test_distributed_run_end_to_end_against_stub_servers():
    stubs = [Stub(), Stub()]
    try:
        result = client.main(ports=[s.port for s in stubs], generations=3, population_size=8)
        assert result.n_evaluations == sum(len(s.requests) for s in stubs) > 0
        assert all(s.requests for s in stubs)  # both servers did work
    finally:
        for s in stubs:
            s.close()


# --- server (spec 009) --------------------------------------------------------


def test_server_parses_json_values():
    assert server.parse_params({"x1": "0.5", "x2": "-3"}) == {"x1": 0.5, "x2": -3}


@pytest.mark.parametrize(
    "args, match",
    [
        ({"x1": "0.5"}, "missing"),
        ({"x1": "0.5", "x2": "1", "evil": "1"}, "unknown"),
        ({"x1": "0.5", "x2": "not json"}, "x2"),
    ],
)
def test_server_rejects_bad_parameters(args, match):
    with pytest.raises(ValueError, match=match):
        server.parse_params(args)


def test_X_2_no_example_enables_debug_mode():
    for path in EXAMPLES.rglob("*.py"):
        assert "debug=True" not in path.read_text(), path


# --- runnable examples --------------------------------------------------------


@pytest.mark.plots  # the quickstart writes plots
def test_quickstart_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = quickstart.main(generations=3, population_size=10)
    assert result.score < 0
    assert (tmp_path / "cache" / "holder_table-plots" / "best_score.png").exists()


def test_categorical_runs():
    result = categorical.main(generations=20, population_size=20)
    assert set(result.params) == {"activation", "batch_size", "layers", "learning_rate", "optimizer"}
