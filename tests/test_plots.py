import sys

import numpy as np
import pytest

from geneopt import GeneticAlgorithm
from geneopt.plots import diversity, gene_frequency_matrix

pytestmark = pytest.mark.plots  # needs pandas + matplotlib (catboost optional)

ENV = {"x": np.linspace(-2, 2, 21), "y": np.linspace(-2, 2, 21), "act": ["relu", "tanh", "gelu"]}


def objective(x, y, act):
    return -(x * x + y * y) + {"relu": 0.0, "tanh": 0.5, "gelu": 1.0}[act]


def run(**kwargs):
    ga = GeneticAlgorithm(objective, ENV, population_size=20, generations=8, seed=3, **kwargs)
    ga.start()
    return ga


def test_diversity_is_exported_from_plots():
    assert diversity(np.zeros((5, 2), dtype=int)) == 0.0


def test_H_3_gene_frequency_rows_sum_to_one():
    ga = run()
    matrix = gene_frequency_matrix(ga.history, "act")
    assert matrix.shape == (len(ga.history), 3)
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_gene_frequency_bins_large_parameters():
    ga = GeneticAlgorithm(lambda n: n, {"n": list(range(1000))}, population_size=10, generations=2)
    ga.start()
    matrix = gene_frequency_matrix(ga.history, "n")
    assert matrix.shape == (2, 64) and np.allclose(matrix.sum(axis=1), 1.0)


def test_evaluations_returns_unique_evaluated_points():
    ga = run()
    frame = ga.evaluations()
    assert list(frame.columns) == ["act", "x", "y", "score"]
    assert len(frame) == ga.result.n_evaluations
    assert not frame.duplicated(["act", "x", "y"]).any()


def test_plot_writes_named_pngs(tmp_path):
    ga = run()
    out = ga.plot(output_dir=tmp_path, dpi=50)
    errors = {name: e for name, e in out.items() if isinstance(e, Exception)}
    assert not errors
    for name in (
        "best_score",
        "diversity",
        "cache_hit_rate",
        "param_x",
        "param_act",
        "gene_freq_act",
        "corr_with_score",
        "corr_matrix",
        "hist_x",
        "regression_y",
        "scatter_x_y",
        "predicted_vs_actual",
        "feature_importance",
    ):
        path = tmp_path / f"{name}.png"
        assert out[name] == path and path.stat().st_size > 0, name


def test_default_output_dir_is_per_optimizer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = GeneticAlgorithm(objective, ENV, optimizer_name="a", population_size=6, generations=2)
    b = GeneticAlgorithm(
        objective, ENV, optimizer_name="b", population_size=6, generations=2, cache_dir=tmp_path / "cache"
    )
    a.start(), b.start()
    assert a.plot(dpi=30)["best_score"] == tmp_path / "a-plots" / "best_score.png"
    assert b.plot(dpi=30)["best_score"] == tmp_path / "cache" / "b-plots" / "best_score.png"


def test_H_4_minimize_with_inf_scores_still_plots_features(tmp_path):
    def sometimes_inf(x, y, act):
        return np.inf if x > 1.5 else x * x + y * y

    ga = GeneticAlgorithm(sometimes_inf, ENV, maximize=False, population_size=20, generations=6, seed=1)
    ga.start()
    out = ga.plot(output_dir=tmp_path, dpi=30)
    assert not [n for n, v in out.items() if isinstance(v, Exception)]


def test_missing_catboost_only_drops_the_model_plots(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "catboost", None)  # makes `import catboost` raise ImportError
    out = run().plot(output_dir=tmp_path, dpi=30)
    assert isinstance(out["predicted_vs_actual"], ImportError)
    assert "GeneOpt[analysis]" in str(out["predicted_vs_actual"])
    assert out["best_score"].exists() and out["hist_x"].exists()


def test_plot_before_start_is_an_error():
    with pytest.raises(RuntimeError, match="start"):
        GeneticAlgorithm(objective, ENV).plot()


def test_plot_writes_no_stray_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run().plot(output_dir=tmp_path / "out", dpi=30)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]  # e.g. no catboost_info/


def test_H_5_one_failing_plot_does_not_stop_the_others(tmp_path, monkeypatch):
    # 1.x re-raised before logging, so the first failing plot aborted all the rest
    def broken(genomes):
        raise RuntimeError("boom")

    monkeypatch.setattr("geneopt.plots.diversity", broken)
    out = run().plot(output_dir=tmp_path, dpi=30)
    assert isinstance(out["diversity"], RuntimeError)
    assert out["best_score"].exists() and out["cache_hit_rate"].exists() and out["hist_x"].exists()
