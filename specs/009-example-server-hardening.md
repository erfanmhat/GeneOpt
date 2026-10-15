# 009 — Example Server Hardening

**Status:** proposed · **Severity:** 🟠 high · **Fixes:** X-2, X-3 · **Pairs with:** 008

## Problem

`server.py` (also copied into the README) is the template people will copy for their own
objective servers. It has these problems:

| # | Code | Problem |
|---|---|---|
| 9a | `app.run(debug=True, …)` | Turns on the Werkzeug interactive debugger and the auto-reloader. Flask's docs say debug mode must never be used in deployment: any unhandled exception gives an in-browser Python console (PIN-protected) and full source tracebacks. It's bound to `127.0.0.1` today, but the README pitches the setup for **scaling across multiple servers**. The first thing a user does is set `host="0.0.0.0"`, and then the debugger is on the network |
| 9b | Reloader | `debug=True` also starts a file-watching reloader, so each server is **two** processes. `RunNServers.py` launches N servers and actually runs 2N Python processes |
| 9c | `{key: float(value) for key, value in request.args.items()}` | Any query parameter is passed as `**kwargs` to the objective. An unknown name → `TypeError` → 500 → the 1.0.10 client retries forever (008 §8a). Every parameter is forced to `float`, so categorical environments (GC-6) can't be used at all |
| 9d | Errors in `data` | `{"data": "Bad request: …"}` shares a field with real scores (008 §8e) |
| 9e | `threaded=False` | Correct (one evaluation at a time per port), but nothing says why. Someone will "fix" it |
| 9f | `RunNServers.py` | Hard-codes port 5000 and `"python"` (not `sys.executable`, so a venv gets the wrong interpreter). If one child dies, the rest keep running and nothing is reported. The README calls it `RunNServer.py` (X-3) |

## Design

### `examples/distributed_http/server.py`

```python
"""Serve one objective evaluation per request: GET /run_objective?x1=0.5&x2=-3"""
import inspect, json, sys

from objective import objective          # the user's function, in its own module

PARAMS = set(inspect.signature(objective).parameters)


def parse_params(args) -> dict:
    """Validate query args against the objective's signature and JSON-decode them."""
    unknown, missing = set(args) - PARAMS, PARAMS - set(args)
    if unknown or missing:
        raise ValueError(f"unknown={sorted(unknown)} missing={sorted(missing)}")
    params = {}
    for name, raw in args.items():
        try:
            params[name] = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{name}: value {raw!r} is not JSON ({e})") from None
    return params


def create_app():
    from flask import Flask, jsonify, request    # imported here so parse_params is testable without flask

    app = Flask(__name__)

    @app.get("/run_objective")
    def run_objective():
        try:
            params = parse_params(request.args.to_dict())
        except ValueError as e:
            return jsonify(error=str(e)), 400
        return jsonify(score=float(objective(**params)))

    return app


if __name__ == "__main__":
    # threaded=False: one evaluation at a time per process. Parallelism comes from running
    # several processes (run_servers.py), not threads, so CPU-bound objectives aren't GIL-bound.
    create_app().run(host="127.0.0.1", port=int(sys.argv[1]), debug=False, threaded=False)
```

- **`debug=False`, `host="127.0.0.1"` written out explicitly** (9a, 9b). The README section on
  going multi-machine says to put the servers behind a production WSGI server
  (`gunicorn -w 1 -b 0.0.0.0:5000 server:app`) on a private network, and never to expose them
  publicly. The endpoint runs arbitrary computation and has no authentication.
- **Parameters are checked against the objective signature** (9c). A mismatch is a 400 that
  the 008 client raises at once instead of retrying.
- **Values are JSON-encoded** in the query string, so `"relu"`, `3`, `0.5` and `true` all
  round-trip. The client sends `urlencode({k: json.dumps(v)})`. This is what makes categorical
  environments work over HTTP.
- **Separate `score` / `error` fields** (9d).
- An objective exception is left as Flask's default 500, without a debug page. That's correct
  and is retried by the client.

### `examples/distributed_http/run_servers.py` (renamed, fixes X-3)

```python
import subprocess, sys
from pathlib import Path
n, base = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 5000
here = Path(__file__).resolve().parent
procs = [subprocess.Popen([sys.executable, "server.py", str(base + i)], cwd=here) for i in range(n)]
codes = []                      # defined before try: Ctrl-C during wait() must not hit a NameError
try:
    codes = [p.wait() for p in procs]
finally:
    for p in procs:
        p.terminate()
sys.exit(max(codes, default=0))
```

- `sys.executable` (the venv-correct interpreter) and a configurable base port. `cwd` is the
  script's own directory, so it can be started from anywhere.
- Ctrl-C or a child exit terminates the whole group, so no orphaned servers keep holding ports.

## Acceptance criteria

- [ ] `grep -rn "debug=True" examples/ README.md` returns nothing.
- [ ] Starting `run_servers.py 4` gives exactly 4 Python child processes.
- [ ] `GET /run_objective?x1=0.5` (missing `x2`) → 400 `{"error": …}`, objective not called.
- [ ] `GET /run_objective?x1=0.5&x2=0.5&evil=1` → 400.
- [ ] `GET /run_objective?act=%22relu%22&lr=0.01` reaches the objective as `("relu", 0.01)`.
- [ ] Ctrl-C on `run_servers.py` leaves no server processes behind.
- [ ] Every README reference uses `run_servers.py`.
