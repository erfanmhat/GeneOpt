# 008 — HTTP Example Client Hangs on a Score of 0

**Status:** proposed · **Severity:** 🔴 critical (for distributed users) · **Fixes:** X-1 · **Depends on:** 004

## Problem

The distributed example client (`main2.py:15-27`, copied into the README) only accepts a
response if the score is truthy:

```python
while True:
    try:
        ...
                res_json = await response.json()
                if res_json and res_json.get("data"):     # 0.0 is falsy
                    return float(res_json["data"])
    except Exception as e:
        print(f"Error occurred: {e}, retrying...")
        await asyncio.sleep(5)
```

A score of exactly `0.0` takes neither branch, so the loop goes round again **with no log and no
sleep**. The client sends requests to the server back to back, forever.

`0.0` is a normal score: the global optimum of many benchmark functions, a zero loss, a zero
profit. With the Holder table, any `x1 = 0` gives `sin(0) = 0` and so a score of `0.0`.

### Reproduction (verified)

A stdlib stand-in server always answers `{"data": 0.0}`, and `main2.objective_func` is called as is:

```
asyncio.wait_for(objective_func(5099, x1=0.0, x2=0.0), 12)  →  HUNG: still retrying after 12s
```

Nothing is printed, because the `except` branch never runs.

### Related defects in the same client

| # | Defect | Effect |
|---|---|---|
| 8a | `except Exception` retries **every** error forever, including HTTP 400 ("parameters must be numeric") and 500 (objective raised) | A deterministic failure hangs the run, with a log line every 5 s and no end |
| 8b | `except KeyboardInterrupt: raise` | Dead code: `KeyboardInterrupt` isn't an `Exception`, so it was never caught anyway |
| 8c | A new `aiohttp.ClientSession` per call and a new event loop per call (`asyncio.run` in `run_async_objective_func`) | A TCP + loop setup for every evaluation, and no connection reuse |
| 8d | `timeout=60*30` passed as a bare int | aiohttp expects `ClientTimeout`. A bare number is deprecated |
| 8e | A score of `NaN` or a non-number string in `data` | `float("Bad request…")` raises `ValueError` → retried forever (same as 8a) |

## Design

The client lives in `examples/distributed_http/client.py` (006) and is called from threads in
an executor (004), so it doesn't need asyncio at all. The stdlib is enough:

```python
import json, time, urllib.error, urllib.parse, urllib.request

class ObjectiveError(RuntimeError):
    """The server answered, but the evaluation itself failed. Not retried."""

def remote_score(port: int, params: dict, retries: int = 3, timeout: float = 1800) -> float:
    query = urllib.parse.urlencode({k: json.dumps(v.item() if isinstance(v, np.generic) else v)
                                    for k, v in params.items()})   # JSON values, see 009
    url = f"http://127.0.0.1:{port}/run_objective?{query}"
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                body = json.load(r)
            return float(body["score"])                  # 0.0 is a valid score
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise ObjectiveError(f"{e.code} for {params}: {e.read()[:200]!r}") from e
            err = e                                      # 5xx: the server may recover
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e                                      # network: retry
        if attempt < retries:                            # no pointless sleep before giving up
            time.sleep(min(2 ** attempt, 30))
    raise ObjectiveError(f"gave up after {retries + 1} attempts for {params}") from err
```

Rules:
1. **Presence, not truthiness.** Read `body["score"]` directly. A missing key is a `KeyError`
   (a server bug), not a retry.
2. **Bounded retries**, and only for transient failures (network errors, 5xx). 4xx is raised at
   once.
3. **Every retry sleeps** with exponential backoff, capped at 30 s. There's no path through the
   loop without a sleep.
4. The engine's `on_error` policy (004 §4) decides what an `ObjectiveError` does to the run:
   it either aborts or scores the genome as worst. The client doesn't decide that.
5. The response field is renamed from `data` to `score`, and errors use `{"error": "…"}`. The
   current server puts error messages in `data`, which is how 8e happens.

`aiohttp` is dropped from the example entirely. That's one less dependency to install.

## Acceptance criteria

- [ ] With a stub server that returns `{"score": 0.0}`, `remote_score` returns `0.0` on the
      first request (the stub's request counter is 1).
- [ ] Stub returns 400 → `ObjectiveError` raised after exactly 1 request.
- [ ] Stub returns 503 twice, then 200 → the score is returned after 3 requests.
- [ ] Stub unreachable → `ObjectiveError` after `retries + 1` attempts, with total sleep ≤ 1 + 2 + 4 s
      for `retries=3`.
- [ ] `tests/test_examples.py` runs these against a `http.server` stub, with no network and no Flask.
- [ ] The README embeds `client.py` verbatim (006 §3), so the README can't drift from the file.
