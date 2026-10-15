# 013 — Releasing 2.0.0

**Status:** in progress · **Severity:** 🟡 process · **Depends on:** 006, 007, 011 (012 optional)

The 2.0.0 code is on the `v2.0.0` branch. This is the checklist from there to a published
release on [PyPI](https://pypi.org/project/GeneOpt/), whose current versions are 1.0.0 and 1.0.10.

## How publishing works

`ci.yml` has a `publish` job. It runs only when a tag starting with `v` is pushed, and only after
lint, all tests and the build pass. It uploads to PyPI with **trusted publishing**: PyPI accepts
the upload because it comes from this repo's workflow, so no PyPI password or API token is stored
anywhere. PyPI has to be told once which workflow to trust. That's the only step that needs a
PyPI login.

## Checklist

### On GitHub (done by Claude)
- [x] Push the `v2.0.0` branch and open a pull request into `master`
      ([PR #1](https://github.com/erfanmhat/GeneOpt/pull/1), draft). The PR runs `ci.yml` on
      the full matrix (3 OSes × Python 3.10–3.13).
- [x] CI green on the full matrix in < 5 min: 1 min 40 s. The one Windows/macOS failure (a
      case-insensitive path check in a test) was fixed.

### On PyPI (owner only, one time, needs a login)
- [ ] pypi.org → *Your projects* → **GeneOpt** → *Manage* → **Publishing** → *Add a new
      publisher* → **GitHub**, with:

      | Field | Value |
      |---|---|
      | Owner | `erfanmhat` |
      | Repository name | `GeneOpt` |
      | Workflow name | `ci.yml` |
      | Environment name | `pypi` |

      (The `pypi` environment appears on GitHub automatically the first time the publish job
      runs. Nothing to create there.)

### Release (owner decides when)
- [ ] Decide whether 012 goes into 2.0.0 or waits for 2.1.0.
- [ ] Merge the PR into `master`.
- [ ] Tag and push: `git tag v2.0.0 && git push origin v2.0.0`. CI runs, then `publish` uploads.
- [ ] Check https://pypi.org/project/GeneOpt/2.0.0/ shows the README with images, and that
      `pip install GeneOpt==2.0.0` in a clean venv imports `geneopt` with numpy as the only
      dependency.

> **A PyPI version number can never be reused**, even after deleting a release. A broken 2.0.0
> can only be fixed with a 2.0.1, so the tag is pushed only after CI is green on the PR.

## Notes for after the release

- **1.x users:** `pip install -U GeneOpt` moves them to 2.0.0, which is a hard break (011). People
  who must stay on 1.x pin `GeneOpt<2`. The CHANGELOG and README "Upgrading from 1.x" section say so.
- **Local clones on Linux:** run `git config core.ignorecase false` (see 006 §5). Otherwise a
  case-only rename such as `GeneOpt/` → `geneopt/` can be committed under the wrong path.
