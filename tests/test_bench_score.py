"""Scoring of fair-suite run records (spec 015). Pure Python: runs without the suites installed."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "bench"))
import score  # noqa: E402


def rec(optimiser, best, trace=None, *, suite="mixint", function=1, dim=10, instance=1, reference=0.0,
        early=None, budget=1000, error=None, maximize=None):  # fmt: skip
    maximize = suite == "pbo" if maximize is None else maximize
    return {
        "suite": suite, "function": function, "dim": dim, "instance": instance, "optimiser": optimiser,
        "budget": budget, "evaluations": budget, "seconds": 1.0, "error": error, "maximize": maximize,
        "reference": reference, "best": best, "early_best": best if early is None else early,
        "trace": trace or [(1, best)], "config": None, "sizes": [2] * dim,
    }  # fmt: skip


def test_evaluations_to_target_reads_the_trace():
    r = rec("a", 0.005, [(1, 10.0), (5, 1.0), (20, 0.005)])
    hits = score.evaluations_to_targets(r, 0.0)
    assert hits[100.0] == 1 and hits[10.0 ** (2 - 0.2 * 5)] == 1  # 10.0 reaches the 1e1 target
    assert hits[10 ** (2 - 0.2 * 10)] == 5  # 1e0
    assert hits[10 ** (2 - 0.2 * 20)] == 20  # 1e-2: 0.005 <= 0.01
    assert hits[10 ** (2 - 0.2 * 25)] is None  # 1e-3 never reached


def test_pbo_precision_is_the_relative_gap_to_the_reference():
    assert score.precision("pbo", 9.0, 10.0) == pytest.approx(0.1)
    assert score.precision("pbo", 10.0, 10.0) == 0
    assert score.precision("pbo", 10.0 - 1e-15, 10.0) == 0  # rounding noise is a hit
    assert score.precision("mixint", 3.5, 1.5) == 2.0


def test_pbo_reference_is_the_best_of_stated_and_observed():
    rs = [
        rec("a", 10.35, suite="pbo", function=24, reference=8.75),  # beats IOH's stated optimum
        rec("b", 8.0, suite="pbo", function=24, reference=8.75),
        rec("a", 5.0, suite="pbo", function=18, reference=None),  # no stated optimum
        rec("b", 6.0, suite="pbo", function=18, reference=None),
        rec("a", 4.0, suite="pbo", function=19, reference=64.0),
    ]
    refs = score.references(rs)
    assert refs[("pbo", 24, 10, 1)] == (10.35, True)
    assert refs[("pbo", 18, 10, 1)] == (6.0, True)
    assert refs[("pbo", 19, 10, 1)] == (64.0, False)


def test_ecdf_area_is_the_log_budget_share_after_each_hit():
    # budget 1000 at D = 10: the ECDF x-axis spans [10, 1000] evaluations, i.e. 2 decades
    hit_at_100 = {"evaluations_to_target": 100, "budget": 1000, "dimension": 10}
    hit_at_5 = {"evaluations_to_target": 5, "budget": 1000, "dimension": 10}  # before D: full area
    missed = {"evaluations_to_target": None, "budget": 1000, "dimension": 10}
    assert score.ecdf_area([hit_at_100]) == pytest.approx(0.5)
    assert score.ecdf_area([hit_at_5, missed]) == pytest.approx(0.5)


def test_ranks_average_ties():
    assert score.ranks({"a": 1.0, "b": 1.0, "c": 3.0}) == {"a": 1.5, "b": 1.5, "c": 3.0}
    assert score.ranks({"a": math.inf, "b": 0.0}) == {"a": 2.0, "b": 1.0}


def suite_records(wins, ties_at_zero=0, losses=0, main_trace_hits=True):
    """One (function, D) pair per outcome, instance 1, minimisation with f_opt = 0."""
    rs, fid = [], 0
    for outcome in ["win"] * wins + ["zero"] * ties_at_zero + ["loss"] * losses:
        fid += 1
        g, r = {"win": (1.0, 2.0), "zero": (0.0, 0.0), "loss": (3.0, 2.0)}[outcome]
        rs.append(rec("geneopt[mr=1/D]", g, [(10 if main_trace_hits else 999, g)], function=fid))
        rs.append(rec("random", r, [(500, r)], function=fid))
    return rs


def test_gate_needs_90_percent_of_pairs_and_a_higher_ecdf_area():
    assert score.gate(score.score(suite_records(9, losses=1))) == []
    assert score.gate(score.score(suite_records(8, ties_at_zero=1, losses=1))) == []  # both solved = win
    assert score.gate(score.score(suite_records(8, losses=2))) != []
    assert score.gate(score.score(suite_records(10, main_trace_hits=False))) != []  # lower ECDF area


def test_missing_or_crashed_runs_count_as_unsolved():
    rs = [rec("geneopt[mr=1/D]", 1.0), rec("random", 2.0, error="boom", trace=[])]
    rs[1]["best"] = None
    s = score.score(rs)
    assert s["final"]["mixint"][(1, 10)]["random"] == math.inf
    assert s["crashes"] == [rs[1]]


def test_head_start_flags_runs_already_at_the_1e_minus_2_target():
    rs = [rec("a", 0.0, early=0.005), rec("b", 0.0, early=0.5)]
    assert [r["optimiser"] for r in score.score(rs)["head_starts"]] == ["a"]


def test_rows_have_the_016_long_format():
    rows = score.runs_rows([rec("a", 0.005, [(1, 10.0), (20, 0.005)])])
    assert len(rows) == 51
    assert set(rows[0]) >= {
        "algorithm", "function_id", "dimension", "instance", "target",
        "evaluations_to_target", "success_bool", "elapsed_time", "evaluations",
    }  # fmt: skip
    assert rows[0]["function_id"] == "bbob-mixint/f1"
    assert sum(r["success_bool"] for r in rows) == 22  # targets 1e2 .. 10^-2.2 (0.0063 >= 0.005)


def test_report_shows_configuration_gate_and_flags():
    rs = suite_records(10)
    rs[0]["config"] = {"mutation_rate": 0.1, "population_size": 50}
    text = score.report(rs)
    for part in ("## Configuration", "mutation_rate", "## mixint", "Gate", "PASS", "Head-start check"):
        assert part in text


def test_wilcoxon_reports_counts_and_a_p_value():
    pytest.importorskip("scipy")
    s = score.score(suite_records(9, losses=1))
    [(o, wins, ties, losses, p)] = score.wilcoxon(s, "mixint")
    assert (o, wins, ties, losses) == ("random", 9, 0, 1) and 0 < p < 0.05


def test_compact_sizes():
    assert score.compact([2, 2, 4, 4, 8, 8, 16, 16, 256, 256]) == "2 ×2, 4 ×2, 8 ×2, 16 ×2, 256 ×2"
    assert score.compact([2] * 64) == "2 ×64" and score.compact([5, 7]) == "5, 7"


def test_budget_used_shows_optimisers_that_stall():
    rs = [rec("a", 1.0, budget=1000), rec("b", 1.0, budget=1000)]
    rs[1]["evaluations"] = 250  # stopped by the duplicate-call cap
    s = score.score(rs)
    assert s["budget_used"]["mixint"] == {"a": 1.0, "b": 0.25}
    assert "budget used" in score.report(rs)
