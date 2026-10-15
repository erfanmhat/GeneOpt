# 006 — Packaging & Repo Hygiene

**Status:** proposed · **Severity:** 🟡 medium · **Fixes:** A-4 · **Related:** 009 (X-2, X-3), 011 (removed files)

## Problem

| Finding | Evidence |
|---|---|
| Build output is committed | `build/` (76 K), `dist/` (68 K, wheels for 1.0.0 *and* 1.0.10), `GeneOpt.egg-info/`, 14 `__pycache__/*.pyc` files |
| `build/lib` has already drifted | `build/lib/GeneOpt/GeneticAlgorithm.py:584` still uses `np.NaN`, which was removed in NumPy 2 |
| Two build configs | `setup.py` holds everything, `pyproject.toml` has only `[build-system]` |
| Heavy hard deps | `catboost` (~100 MB wheel) + `scikit-learn` are required just to *import* the library, even though the README says CatBoost is optional (A-4) |
| Version claims | `python_requires=">=3.6"`. 3.6–3.8 are EOL, and the code isn't tested on any version |
| README drift | It says `RunNServer.py` but the file is `RunNServers.py` (X-3). `Logger` is part of every example. Plots are said to go to `./cache/` |
| Example server | `app.run(debug=True)`. The Werkzeug debugger allows arbitrary code execution for anyone who can reach the port (X-2) |
| Loose scripts in root | `main.py` (objective silently changed to `x1 * x2`, with the docstring still saying "holder table"), `main2.py`, `server.py`, `RunNServers.py` |
| No tests, no CI, no changelog, no version tags | — |

## Design

### 1. `pyproject.toml` only (PEP 621)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "GeneOpt"
version = "2.0.0"
description = "Genetic-algorithm optimisation over discrete search spaces, with caching and plots"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [{ name = "Erfan Mahdavi Athar" }]
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
plots    = ["matplotlib>=3.7", "pandas>=2.0"]
analysis = ["GeneOpt[plots]", "catboost>=1.2"]
dev      = ["pytest>=8", "pytest-cov>=5", "hypothesis>=6", "ruff>=0.6", "mypy>=1.10"]

[project.urls]
Source = "https://github.com/erfanmhat/GeneOpt"

[tool.ruff]
line-length = 110

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- **The core needs only numpy.** pandas becomes a plots-only dependency, because 001 removes it
  from the cache.
- `scikit-learn` is dropped (its one use was `r2_score`, see 002).
- Delete `setup.py`, `build/`, `dist/`, `GeneOpt.egg-info/`, and all `__pycache__/` from git.

### 2. `.gitignore`

Add the standard Python entries the current one is missing:

```gitignore
__pycache__/
*.py[cod]
build/
dist/
*.egg-info/
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
```

### 3. Repo layout

```
geneopt/                 # package (005)
tests/                   # 007
examples/
  quickstart.py          # holder table, local, the README snippet verbatim
  categorical.py         # new: proves GC-6 is fixed, a real-world-shaped search
  distributed_http/
    objective.py         # the user's function, imported by server.py
    server.py            # 009: debug=False, signature-checked params
    run_servers.py       # 009: renamed from RunNServers.py (X-3)
    client.py            # 008: bounded retries, 0.0-safe, + port pool (004 §3)
docs/images/             # moved from image/
specs/                   # these files
CHANGELOG.md
README.md
```

- The README embeds the example files exactly as they are, and a test runs
  `examples/quickstart.py`, so the README can't drift again.
- README image URLs point at `raw.githubusercontent.com`, because `blob/` URLs don't render
  on PyPI.

### 4. Release process

- `CHANGELOG.md` in Keep-a-Changelog format. The `2.0.0` entry lists every ledger ID it closes
  and states clearly that **1.x CSV caches are not reused**.
- Tag `v2.0.0`. Publish from CI with PyPI trusted publishing when a tag is pushed, not from a
  laptop. That way `dist/` never needs to exist in the repo.
- `__version__ = "2.0.0"` lives in `geneopt/__init__.py`, and hatchling reads it from there
  (`[tool.hatch.version] path`). It's a single source of truth that also works uninstalled.
  `importlib.metadata` would need an install, so tests running from a checkout would break.

### 5. Case-only rename hazard

`GeneOpt/` → `geneopt/` differs only in case. This repo had `core.ignorecase=true` on a
case-sensitive filesystem, so `git add geneopt/x.py` was recorded under the existing `GeneOpt/`
path and `geneopt/__init__.py` was skipped. Six commits went in without an importable package
before it was caught.
- Run `git config core.ignorecase false` on Linux checkouts, or use `git -c core.ignorecase=false`
  for the rename commits.
- Verify the result from a clean checkout (`git worktree add`), not the working tree. The working
  tree was correct the whole time, so the tests passed.

## Acceptance criteria

- [ ] `git ls-files | grep -E '(^build/|^dist/|egg-info|__pycache__|\.pyc$)'` returns nothing.
- [ ] `pip install GeneOpt` in a clean venv pulls only numpy. `python -c "import geneopt"` works.
- [ ] `pip install GeneOpt[analysis]` enables feature plots.
- [ ] `python -m build` gives an sdist and wheel that `twine check` accepts.
- [ ] The README renders on PyPI with images.
- [ ] `examples/distributed_http/server.py` doesn't enable debug mode.
