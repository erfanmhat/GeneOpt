"""Exact genome -> score memo, optionally persisted as append-only JSONL."""
import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from types import CodeType

import numpy as np

log = logging.getLogger("geneopt")

FORMAT_VERSION = 2


class CacheMismatchError(Exception):
    """A cache file's header fingerprint differs from the run's."""


def _py(value):
    """numpy scalars -> Python scalars; anything else unchanged."""
    return value.item() if isinstance(value, np.generic) else value


def _jsonable(value):
    value = _py(value)
    return value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)


def _code_digest(code: CodeType, h) -> None:
    # repr() of a code object embeds its memory address, so hash nested code structurally
    h.update(code.co_code)
    h.update(repr(code.co_names).encode())
    for const in code.co_consts:
        if isinstance(const, CodeType):
            _code_digest(const, h)
        else:
            h.update(repr(const).encode())


def fingerprint(environment, objective, objective_version=None) -> str:
    """Identify a search space + objective; any change gives a different cache file."""
    code = getattr(objective, "__code__", None)
    code_hash = None
    if code is not None:
        h = hashlib.sha256()
        _code_digest(code, h)
        code_hash = h.hexdigest()
    payload = {
        "env": {name: [repr(_py(v)) for v in values] for name, values in sorted(environment.items())},
        "objective": f"{getattr(objective, '__module__', '')}."
                     f"{getattr(objective, '__qualname__', type(objective).__qualname__)}",
        "objective_code": code_hash,
        "objective_version": objective_version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class GenomeCache:
    """Scores keyed by gene-index tuples. ``path=None`` keeps the cache in memory only."""

    def __init__(self, path: Path | str | None = None, fingerprint: str = "", params=()):
        self._path = Path(path) if path is not None else None
        self._fingerprint = fingerprint
        self._params = list(params)
        self._scores: dict[tuple[int, ...], float] = {}
        if self._path is not None:
            if self._path.exists() and self._path.stat().st_size:
                self._load()
            else:  # new file, or empty after a crash while writing the header
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text("")
                self._append({"geneopt_cache": FORMAT_VERSION, "fingerprint": fingerprint,
                              "params": self._params,
                              "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    def _load(self) -> None:
        with open(self._path, "rb") as f:
            lines = f.readlines()
        header = json.loads(lines[0])
        if header.get("fingerprint") != self._fingerprint:
            raise CacheMismatchError(
                f"{self._path} has fingerprint {header.get('fingerprint')!r}, expected {self._fingerprint!r}")
        offset = len(lines[0])
        for i, line in enumerate(lines[1:], start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise ValueError(f"{self._path}: corrupt record on line {i + 1}") from None
                log.warning("%s: skipping truncated last record (interrupted write)", self._path)
                with open(self._path, "r+b") as f:  # drop the partial line so appends stay valid
                    f.truncate(offset)
                break
            self._scores[tuple(record["k"])] = record["s"]
            offset += len(line)

    def _append(self, record: dict) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_jsonable) + "\n")

    def get(self, key):
        return self._scores.get(tuple(key))

    def put(self, key, score, values=None) -> None:
        key = tuple(int(i) for i in key)
        score = float(score)
        if math.isnan(score):
            raise ValueError(f"objective returned NaN for {values if values is not None else key}")
        self._scores[key] = score
        if self._path is not None:
            record = {"k": key, "s": score}
            if values is not None:
                record["v"] = [_jsonable(v) for v in values]
            self._append(record)

    def __contains__(self, key) -> bool:
        return tuple(key) in self._scores

    def __len__(self) -> int:
        return len(self._scores)

    def items(self):
        return self._scores.items()
