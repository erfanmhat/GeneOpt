"""Suite problems for the fair benchmark (spec 015). Needs the bench requirements (cocoex, ioh)."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cocoex")
pytest.importorskip("ioh")
sys.path.insert(0, str(Path(__file__).resolve().parent / "bench"))
import suites  # noqa: E402


@pytest.mark.parametrize(
    "dim, sizes",
    [
        (10, (2, 2, 4, 4, 8, 8, 16, 16, 256, 256)),
        (20, (2,) * 4 + (4,) * 4 + (8,) * 4 + (16,) * 4 + (256,) * 4),
    ],
)
def test_mixint_cardinalities_match_spec_015(tmp_path, dim, sizes):
    p = suites.mixint(1, dim, 1, tmp_path)
    assert p.sizes == sizes
    assert p.finish() is None  # nothing evaluated, so COCO wrote no log to read f_opt from


def test_mixint_f_opt_comes_from_cocos_own_log(tmp_path):
    p = suites.mixint(15, 10, 1, tmp_path)
    rng = np.random.default_rng(0)
    values = [p.objective(tuple(int(rng.integers(s)) for s in p.sizes)) for _ in range(20)]
    f_opt = p.finish()
    assert values == p.scored  # scored in evaluation order, as the optimiser saw them
    assert all(v - f_opt >= 0 for v in values)
    dat = next(tmp_path.rglob("*.dat")).read_text().splitlines()
    first = dat[1].split()  # evaluations, g-evals, f - f_opt, measured f, ...
    assert float(first[3]) - float(first[2]) == pytest.approx(f_opt)


def test_mixint_runs_log_to_separate_folders(tmp_path):
    for inst in (1, 2):
        p = suites.mixint(1, 10, inst, tmp_path)
        p.objective((0,) * 10)
        p.finish()
    assert len(list(tmp_path.rglob("*.info"))) == 2
    assert not Path("exdata").exists() or not any(Path("exdata").iterdir())


def test_pbo_scores_untransformed_values_and_minimises_transformed_ones(tmp_path):
    import ioh

    p = suites.pbo(19, 64, 2, tmp_path)
    x = tuple(int(b) for b in np.random.default_rng(1).integers(0, 2, 64))
    seen = p.objective(x)
    q = ioh.get_problem(19, instance=2, dimension=64, problem_class=ioh.ProblemClass.PBO)
    assert seen == -q(list(x))  # the optimiser minimises the transformed value
    assert p.scored == [q.state.current_internal.y]
    assert p.maximize and p.sizes == (2,) * 64
    assert p.finish() == 64.0  # Ising ring: stated optimum, from instance 1


@pytest.mark.parametrize("fid", [18, 25])
def test_pbo_unknown_optimum_is_none(tmp_path, fid):
    assert suites.pbo(fid, 64, 1, tmp_path).finish() is None


def test_targets():
    assert len(suites.MIXINT_TARGETS) == 51
    assert suites.MIXINT_TARGETS[0] == pytest.approx(100) and suites.MIXINT_TARGETS[-1] == pytest.approx(1e-8)
    assert 1e-2 in suites.PBO_TARGETS and suites.PBO_TARGETS[-1] == 0
    assert any(t == pytest.approx(1e-2) for t in suites.MIXINT_TARGETS)


# --- run records (spec 015 task 2) -----------------------------------------------

import run as bench  # noqa: E402


@pytest.mark.parametrize("suite, fid, dim", [("mixint", 1, 10), ("pbo", 19, 64)])
@pytest.mark.parametrize("label", ["random", "geneopt[mr=1/D]"])
def test_run_fair_records_an_anytime_trace(tmp_path, suite, fid, dim, label):
    r = bench.run_fair(suite, fid, dim, 1, label, tmp_path, budget_scale=0.2)
    assert r["error"] is None and 0 < r["evaluations"] <= r["budget"] == 20 * dim
    evals = [e for e, _ in r["trace"]]
    best = [b for _, b in r["trace"]]
    assert evals == sorted(evals) and evals[0] == 1
    assert best == sorted(best, reverse=not r["maximize"])  # improvements only
    assert best[-1] == r["best"]
    assert (r["early_best"] <= r["best"]) if r["maximize"] else (r["early_best"] >= r["best"])
    assert r["reference"] is not None
    assert len(r["sizes"]) == dim
    if label.startswith("geneopt"):
        assert r["config"]["mutation_rate"] == 1 / dim


def test_nevergrad_is_fast_on_binary_problems():
    pytest.importorskip("nevergrad")
    import time

    p = bench.Problem("pbo", (2,) * 64, 500, categorical=True)
    f = bench.Budget(lambda k: float(sum(k)), 500, bench.CALL_CAP * 500)
    started = time.perf_counter()
    with pytest.raises(bench.Exhausted):
        bench.run_nevergrad(p, f, 1)
    assert time.perf_counter() - started < 30


# --- CLI (spec 015 task 4) ---------------------------------------------------------

SMOKE = ["--functions", "1", "--instances", "1-2", "--budget-scale", "0.1", "--jobs", "2"]


@pytest.mark.bench
@pytest.mark.parametrize("suite, dims, function", [("mixint", "10", "1"), ("pbo", "64", "19")])
def test_fair_suite_runs_end_to_end(tmp_path, capsys, suite, dims, function):
    args = ["--suite", suite, "--dims", dims, "--only", "geneopt,random", "--out-dir", str(tmp_path)]
    args += [*SMOKE, "--functions", function, "--out", str(tmp_path / "r.md")]
    code = bench.main(args)
    text = (tmp_path / "r.md").read_text()
    assert code in (0, 1) and "## Configuration" in text and "Head-start check" in text
    # (three GeneOpt rows + random) x 2 instances: one file each, so CI shards never overwrite each other
    assert len(list((tmp_path / suite / "records").glob("*.jsonl"))) == 8
    assert "crashed" not in capsys.readouterr().out.lower()


@pytest.mark.bench
def test_report_only_on_shards_equals_one_combined_run(tmp_path):
    common = ["--suite", "mixint", "--dims", "10", *SMOKE]
    bench.main(
        [
            *common,
            "--only",
            "geneopt,random",
            "--out-dir",
            str(tmp_path / "all"),
            "--out",
            str(tmp_path / "all.md"),
        ]
    )
    for only in ("geneopt", "random"):
        bench.main([*common, "--only", only, "--out-dir", str(tmp_path / "shards" / only)])
    bench.main(["--report-only", str(tmp_path / "shards"), "--out", str(tmp_path / "merged.md")])
    assert (tmp_path / "merged.md").read_text() == (tmp_path / "all.md").read_text()
