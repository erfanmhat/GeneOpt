"""Serve one objective evaluation per request: GET /run_objective?x1=0.5&x2=-3

Query values are JSON ("relu" needs its quotes), so numbers, strings and booleans
all round-trip. Responses are {"score": float} or {"error": str} with status 400.

Usage: python server.py PORT   (needs flask; run_servers.py starts several)
"""
import inspect
import json
import sys

from objective import objective

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
    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.get("/run_objective")
    def run_objective():
        try:
            params = parse_params(request.args.to_dict())
        except ValueError as e:
            return jsonify(error=str(e)), 400
        return jsonify(score=float(objective(**params)))  # an objective exception -> plain 500, retried

    return app


if __name__ == "__main__":
    # debug stays off: the Werkzeug debugger must never be reachable. For several machines, run
    # behind a WSGI server on a private network, e.g. gunicorn -w 1 -b 0.0.0.0:5000 server:create_app()
    # threaded=False: one evaluation per process; parallelism comes from running several processes.
    create_app().run(host="127.0.0.1", port=int(sys.argv[1]), debug=False, threaded=False)
