"""Plots over a run's History and its evaluated points. Needs ``GeneOpt[plots]``
(feature-importance plots additionally need ``GeneOpt[analysis]``)."""

from itertools import combinations
from pathlib import Path

import numpy as np

from .core import diversity

__all__ = ["diversity", "gene_frequency_matrix", "plot_history", "plot_features"]

MAX_BINS = 64  # gene-frequency columns per parameter
MAX_PAIR_FEATURES = 8  # pairwise scatter plots grow O(p^2); keep the most score-correlated features


def gene_frequency_matrix(history, param: str) -> np.ndarray:
    """(generations, values) share of the population holding each value index (binned above MAX_BINS)."""
    j = list(history.environment).index(param)
    size = len(history.environment[param])
    bins = min(size, MAX_BINS)
    return np.array([np.bincount(g[:, j] * bins // size, minlength=bins) / len(g) for g in history.genomes])


def _figure():
    from matplotlib.figure import Figure  # OO API: no pyplot global state, no GUI backend needed

    fig = Figure(figsize=(7, 4.5), layout="constrained")
    return fig, fig.add_subplot()


def _save(fig, path: Path, dpi: int) -> Path:
    fig.savefig(path, dpi=dpi)
    return path


def _best_so_far(history):
    sign = 1.0 if history.maximize else -1.0
    best_index: tuple[int, int] | None = None
    best = -np.inf
    out: list[tuple[int, int]] = []
    for scores in history.scores:
        i = int(np.argmax(sign * scores))
        if best_index is None or sign * scores[i] > best:
            best, best_index = sign * scores[i], (len(out), i)
        out.append(best_index)
    return out  # per generation: (generation, individual) of the best-so-far genome


def plot_history(history, out_dir: Path, dpi: int) -> dict:
    gens = np.arange(1, len(history) + 1)
    best_at = _best_so_far(history)
    best_scores = [history.scores[g][i] for g, i in best_at]
    results: dict[str, Path | Exception] = {}

    def plot(name, draw):
        try:
            fig, ax = _figure()
            draw(ax)
            results[name] = _save(fig, out_dir / f"{name}.png", dpi)
        except Exception as e:  # one broken plot must not cost the others
            results[name] = e

    def best_score(ax):
        pick = np.nanmax if history.maximize else np.nanmin
        finite = [np.where(np.isfinite(s), s, np.nan) for s in history.scores]
        ax.plot(gens, best_scores, lw=2, label="best so far")
        ax.plot(
            gens, [pick(s) if np.isfinite(s).any() else np.nan for s in finite], lw=1, label="generation best"
        )
        ax.plot(
            gens,
            [np.nanmedian(s) if np.isfinite(s).any() else np.nan for s in finite],
            lw=1,
            label="generation median",
        )
        ax.set(title="Score over generations", xlabel="Generation", ylabel="Score")
        ax.legend()

    plot("best_score", best_score)
    plot(
        "diversity",
        lambda ax: (
            ax.plot(gens, [diversity(g) for g in history.genomes]),
            ax.set(
                title="Population diversity",
                xlabel="Generation",
                ylabel="Mean gene disagreement",
                ylim=(0, 1),
            ),
        ),
    )
    plot(
        "cache_hit_rate",
        lambda ax: (
            ax.plot(gens, [h.mean() for h in history.hits]),
            ax.set(
                title="Share of population served from cache",
                xlabel="Generation",
                ylabel="Cache-hit rate",
                ylim=(0, 1),
            ),
        ),
    )

    for j, param in enumerate(history.environment):
        values = history.environment[param]
        indices = [history.genomes[g][i, j] for g, i in best_at]

        def best_param(ax, values=values, indices=indices, param=param):
            numeric = all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in values)
            if numeric:
                ax.step(gens, [values[k] for k in indices], where="post")
            else:
                ax.step(gens, indices, where="post")
                ax.set_yticks(range(len(values)), [str(v) for v in values])
            ax.set(title=f"Best {param} over generations", xlabel="Generation", ylabel=param)

        def gene_freq(ax, param=param, values=values):
            matrix = gene_frequency_matrix(history, param)
            image = ax.imshow(
                matrix.T,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=0,
                vmax=1,
                extent=(0.5, len(history) + 0.5, -0.5, matrix.shape[1] - 0.5),
            )
            if len(values) <= MAX_BINS:
                ax.set_yticks(range(len(values)), [str(v) for v in values])
            ax.figure.colorbar(image, ax=ax, label="share of population")
            ax.set(
                title=f"Gene frequencies: {param}",
                xlabel="Generation",
                ylabel=param if len(values) <= MAX_BINS else f"{param} (binned)",
            )

        plot(f"param_{param}", best_param)
        plot(f"gene_freq_{param}", gene_freq)
    return results


def plot_features(evaluations, out_dir: Path, dpi: int) -> dict:
    """Landscape analysis over the unique evaluated points (numeric parameters, finite scores)."""
    data = evaluations[np.isfinite(evaluations["score"])]
    features = [c for c in data.columns[:-1] if data[c].dtype.kind in "iuf" and data[c].nunique() > 1]
    data = data[features + ["score"]].astype(float)
    results: dict[str, Path | Exception] = {}

    def plot(name, draw):
        try:
            fig, ax = _figure()
            draw(ax)
            results[name] = _save(fig, out_dir / f"{name}.png", dpi)
        except Exception as e:
            results[name] = e

    corr = data.corr(method="spearman")
    plot(
        "corr_with_score",
        lambda ax: (
            corr["score"].drop("score").sort_values().plot.barh(ax=ax),
            ax.set(title="Spearman correlation with score"),
        ),
    )

    def corr_matrix(ax):
        image = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr)), corr.columns)
        for (r, c), v in np.ndenumerate(corr.values):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8)
        ax.figure.colorbar(image, ax=ax)
        ax.set_title("Spearman correlation")

    plot("corr_matrix", corr_matrix)

    try:
        from catboost import CatBoostRegressor
    except ImportError:
        error = ImportError("feature-importance plots need catboost: pip install 'GeneOpt[analysis]'")
        results["predicted_vs_actual"] = results["feature_importance"] = error
    else:
        model = CatBoostRegressor(verbose=0, loss_function="RMSE", allow_writing_files=False)
        model.fit(data[features], data["score"])
        y, pred = data["score"].to_numpy(), model.predict(data[features])
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

        def predicted(ax):
            ax.scatter(y, pred, s=8)
            lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
            ax.plot([lo, hi], [lo, hi], "r--", label="y = x")
            ax.set(
                title=f"Predicted vs actual (in-sample R² = {r2:.3f})", xlabel="Actual", ylabel="Predicted"
            )
            ax.legend()

        def importance(ax):
            imp = np.asarray(model.feature_importances_)
            order = np.argsort(imp)
            ax.barh([features[i] for i in order], imp[order] / imp.sum() * 100)
            ax.set(title="Feature importance", xlabel="%")

        plot("predicted_vs_actual", predicted)
        plot("feature_importance", importance)

    for f in features:
        plot(
            f"hist_{f}", lambda ax, f=f: (ax.hist(data[f], bins=20), ax.set(title=f"Evaluated {f}", xlabel=f))
        )

        def regression(ax, f=f):
            ax.scatter(data[f], data["score"], s=8)
            slope, intercept = np.polyfit(data[f], data["score"], 1)
            xs = np.linspace(data[f].min(), data[f].max(), 100)
            ax.plot(xs, slope * xs + intercept, "r")
            ax.set(title=f"{f} vs score", xlabel=f, ylabel="score")

        plot(f"regression_{f}", regression)

    top = corr["score"].drop("score").abs().sort_values(ascending=False).index[:MAX_PAIR_FEATURES]
    for a, b in combinations(sorted(top), 2):

        def scatter(ax, a=a, b=b):
            points = ax.scatter(data[a], data[b], c=data["score"], s=10, cmap="viridis")
            ax.figure.colorbar(points, ax=ax, label="score")
            ax.set(title=f"{a} & {b}", xlabel=a, ylabel=b)

        plot(f"scatter_{a}_{b}", scatter)
    return results
