"""Benchmark figures (spec 016): hand-computed values for every figure and every missing-data rule."""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.plots
pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent / "bench"))
import plots  # noqa: E402


def row(alg, fid, inst, target, hit, *, dim=10, time=1.0, evals=100, budget=100):
    return {
        "algorithm": alg, "function_id": f"bbob-mixint/f{fid}", "dimension": dim, "instance": inst,
        "target": target, "evaluations_to_target": np.nan if hit is None else float(hit),
        "elapsed_time": time, "evaluations": evals, "budget": budget,
    }  # fmt: skip


def frame(rows):
    return pd.DataFrame(rows)


# --- validation (spec 016 § Missing and incomplete data) ----------------------


@pytest.mark.parametrize("drop", ["algorithm", "function_id", "evaluations_to_target", "target"])
def test_missing_columns_are_named(drop):
    with pytest.raises(ValueError, match=drop):
        plots.validate(frame([row("a", 1, 1, 1.0, 10)]).drop(columns=drop))


def test_empty_frame_and_nan_labels_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        plots.validate(frame([row("a", 1, 1, 1.0, 10)]).iloc[0:0])
    bad = frame([row("a", 1, 1, 1.0, 10)])
    bad.loc[0, "algorithm"] = np.nan
    with pytest.raises(ValueError, match="algorithm"):
        plots.validate(bad)


def test_success_is_derived_and_checked():
    runs = plots.validate(frame([row("a", 1, 1, 1.0, 10), row("a", 1, 1, 0.1, None)]))
    assert runs["success_bool"].tolist() == [True, False]
    bad = frame([row("a", 1, 1, 1.0, None)]).assign(success_bool=True)
    with pytest.raises(ValueError, match="success_bool"):
        plots.validate(bad)


def test_duplicate_run_target_rows_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        plots.validate(frame([row("a", 1, 1, 1.0, 10), row("a", 1, 1, 1.0, 20)]))


def test_incomplete_grid_is_reported_and_warned(caplog):
    runs = frame([row("a", 1, 1, 1.0, 10), row("a", 2, 1, 1.0, 10), row("b", 1, 1, 1.0, 10)])
    with caplog.at_level(logging.WARNING):
        assert plots.completeness(plots.validate(runs)) == {"a": (2, 2), "b": (1, 2)}
    assert "b" in caplog.text and "f2" in caplog.text


# --- 1. ECDF ------------------------------------------------------------------


def ecdf_runs():
    # 2 problems x 2 targets = 4 (problem, target) pairs; b has no run on problem 2 at all
    return plots.validate(frame([
        row("a", 1, 1, 1.0, 10), row("a", 1, 1, 0.1, 50),
        row("a", 2, 1, 1.0, 20), row("a", 2, 1, 0.1, None),
        row("b", 1, 1, 1.0, 30), row("b", 1, 1, 0.1, None),
    ]))  # fmt: skip


def test_ecdf_counts_hits_over_all_pairs_including_missing_runs():
    x, y = plots.ecdf_curve(ecdf_runs(), "a")
    assert x.tolist() == [1.0, 2.0, 5.0] and y.tolist() == [0.25, 0.5, 0.75]  # evaluations / D
    x, y = plots.ecdf_curve(ecdf_runs(), "b")
    assert x.tolist() == [3.0] and y.tolist() == [0.25]  # its missing problem stays in the denominator


def test_ecdf_figure_has_one_panel_per_dimension_and_labels_gaps():
    runs = plots.validate(pd.concat([ecdf_runs(), frame([row("a", 1, 1, 1.0, 40, dim=20)])]))
    fig = plots.plot_ecdf(runs)
    assert len(fig.axes) == 2
    labels = [t.get_text() for t in fig.legends[0].get_texts()]  # one shared legend below the panels
    assert labels[0] == "a" and labels[1].startswith("b (1/2 runs")  # best final value first
    assert len(plots.plot_ecdf(runs, per_dimension=False).axes) == 1


# --- 2. performance profile --------------------------------------------------------


def profile_runs():
    t = {
        1: {"a": 10, "b": 20, "c": None},
        2: {"a": 40, "b": 20, "c": 20},
        3: {"a": None, "b": None, "c": None},
    }
    return plots.validate(
        frame([row(alg, fid, 1, 0.01, hit) for fid, by in t.items() for alg, hit in by.items()])
    )


def test_profile_ratios_and_excluded_problems():
    ratios, excluded = plots.profile_ratios(profile_runs(), 0.01)
    assert excluded == 1  # problem 3: unsolved by everyone
    assert ratios["a"].tolist() == [1.0, 2.0] and ratios["b"].tolist() == [1.0, 2.0]
    assert ratios["c"].tolist() == [1.0, np.inf]
    assert plots.rho(ratios["c"], 1.0) == 0.5 and plots.rho(ratios["c"], 1e9) == 0.5
    assert plots.rho(ratios["a"], 1.0) == 0.5 and plots.rho(ratios["a"], 2.0) == 1.0


def test_profile_caption_counts_the_excluded_problems():
    fig = plots.plot_performance_profile(profile_runs())
    assert "1 of 3 problems unsolved by all" in fig.axes[0].get_title()


def test_default_target_must_exist():
    with pytest.raises(ValueError, match="available: 1"):
        plots.plot_performance_profile(plots.validate(frame([row("a", 1, 1, 1.0, 10)])))


# --- 3. Pareto ----------------------------------------------------------------------


def pareto_runs(times):
    """One target (0.01), 10 problems; successes per algorithm as given."""
    successes = {"a": 9, "b": 10, "c": 5, "d": 2, "e": 9}
    return plots.validate(frame([
        row(alg, fid, 1, 0.01, 10 if fid < n else None, time=times[alg][fid % len(times[alg])])
        for alg, n in successes.items() for fid in range(10)
    ]))  # fmt: skip


def test_pareto_frontier_wilson_interval_and_ties():
    times = {"a": [1.0], "b": [2.0], "c": [3.0], "d": [0.5], "e": [1.0]}
    pts = plots.pareto_points(pareto_runs(times), 0.01, "elapsed_time").set_index("algorithm")
    assert pts.loc["a", "success"] == 0.9 and pts.loc["a", "cost"] == 1.0
    assert pts.loc["a", ["low", "high"]].round(4).tolist() == [0.5958, 0.9821]  # Wilson 95% for 9/10
    assert pts.index[pts.frontier].tolist() == ["d", "a", "e", "b"]  # x order; a and e tie, neither beaten
    assert not pts.loc["c", "frontier"]


def test_pareto_handles_missing_times(caplog):
    times = {"a": [1.0, np.nan], "b": [np.nan], "c": [3.0], "d": [0.5], "e": [1.0]}
    with caplog.at_level(logging.WARNING):
        pts = plots.pareto_points(pareto_runs(times), 0.01, "elapsed_time").set_index("algorithm")
    assert "b" not in pts.index and "b" in caplog.text  # all NaN: left out, with a warning
    assert pts.loc["a", "cost"] == 1.0 and pts.loc["a", "hollow"]  # mean of the remaining runs
    by_evals = plots.pareto_points(pareto_runs(times), 0.01, "evaluations")
    assert "b" in by_evals["algorithm"].tolist()


# --- shared style and output ----------------------------------------------------------


def test_palette_is_stable_distinct_and_emphasises_geneopt():
    labels = ["random", "pymoo", "geneopt[mr=1/D]", "geneopt[mr=2/D]", "deap", "optuna",
              "nevergrad", "pygad", "geneopt[mr=0.5/D]"]  # fmt: skip
    styles = plots.palette(labels)
    assert styles == plots.palette(reversed(labels))
    assert len({(s.color, s.linestyle, s.marker) for s in styles.values()}) == len(labels)
    assert len({styles[x].color for x in labels if x.startswith("geneopt")}) == 1  # hue = library
    assert styles["geneopt[mr=1/D]"].linestyle == "-"  # the main row is solid
    assert styles["geneopt[mr=1/D]"].linewidth > styles["deap"].linewidth
    assert styles["geneopt[mr=1/D]"].zorder > styles["deap"].zorder


def test_every_figure_saves_as_pdf_and_png(tmp_path):
    runs = pareto_runs({"a": [1.0], "b": [2.0], "c": [3.0], "d": [0.5], "e": [1.0]})
    for name, fig in plots.figures(runs).items():
        written = plots.save(fig, tmp_path / name)
        assert [p.suffix for p in written] == [".pdf", ".png"] and all(p.stat().st_size > 0 for p in written)


def test_profile_when_nobody_reaches_the_target():
    runs = plots.validate(frame([row("a", 1, 1, 0.01, None), row("b", 1, 1, 0.01, None)]))
    fig = plots.plot_performance_profile(runs)
    assert "1 of 1 problems unsolved by all" in fig.axes[0].get_title()
