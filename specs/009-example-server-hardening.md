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
"""One objective evaluation per request. Run one process per CPU core / GPU."""
import inspect, json, sys
from flask import Flask, jsonify, request

from objective import objective          # the user's function, in its own module

app = Flask(__name__)
PARAMS = inspect.signature(objective).parameters


@app.get("/run_objective")
def run_objective():
    unknown = request.args.keys() - PARAMS.keys()
    missing = PARAMS.keys() - request.args.keys()
    if unknown or missing:
        return jsonify(error=f"unknown={sorted(unknown)} missing={sorted(missing)}"), 400
    try:
        params = {k: json.loads(v) for k, v in request.args.items()}   # numbers, strings, bools
    except json.JSONDecodeError as e:
        return jsonify(error=f"bad value: {e}"), 400
    return jsonify(score=float(objective(**params)))


if __name__ == "__main__":
    # threaded=False: one evaluation at a time per process. Parallelism comes from running
    # several processes (run_servers.py), not threads, so CPU-bound objectives aren't GIL-bound.
    app.run(host="127.0.0.1", port=int(sys.argv[1]), debug=False, threaded=False)
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
n, base = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 5000
procs = [subprocess.Popen([sys.executable, "server.py", str(base + i)]) for i in range(n)]
try:
    codes = [p.wait() for p in procs]
finally:
    for p in procs:
        p.terminate()
sys.exit(max(codes, default=0))
```

- `sys.executable` (the venv-correct interpreter) and a configurable base port.
- Ctrl-C or a child exit terminates the whole group, so no orphaned servers keep holding ports.

## Acceptance criteria

- [ ] `grep -rn "debug=True" examples/ README.md` returns nothing.
- [ ] Starting `run_servers.py 4` gives exactly 4 Python child processes.
- [ ] `GET /run_objective?x1=0.5` (missing `x2`) → 400 `{"error": …}`, objective not called.
- [ ] `GET /run_objective?x1=0.5&x2=0.5&evil=1` → 400.
- [ ] `GET /run_objective?act=%22relu%22&lr=0.01` reaches the objective as `("relu", 0.01)`.
- [ ] Ctrl-C on `run_servers.py` leaves no server processes behind.
- [ ] Every README reference uses `run_servers.py`.
