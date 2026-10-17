"""Benchmark figures (spec 016): ECDF, Dolan–Moré performance profile, speed vs success Pareto.

Input: the long ``runs`` frame, one row per (run, target), as ``score.runs_rows`` builds it.
Each figure has a pure function underneath (``ecdf_curve``, ``profile_ratios``,
``pareto_points``); the ``plot_*`` functions only draw what those return.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("geneopt.bench")

REQUIRED = ["algorithm", "function_id", "dimension", "instance", "target", "evaluations_to_target",
            "elapsed_time", "evaluations"]  # fmt: skip
PROBLEM = ["function_id", "dimension", "instance"]
DEFAULT_TARGET = 1e-2  # owner decision (spec 016): the exact optimum would put everyone near 0%
DPI = 200

# Hue = library, in a fixed order that never depends on which libraries are present (dataviz rule:
# colour follows the entity). Okabe–Ito plus Tol's wine instead of yellow, which failed the
# lightness band; validated with the dataviz palette checker (all pairs: CVD ΔE >= 7.6, so every
# library also gets its own marker). Random search is the neutral baseline.
LIBRARY_COLOURS = {"geneopt": "#0072B2", "deap": "#E69F00", "pymoo": "#009E73", "optuna": "#D55E00",
                   "nevergrad": "#CC79A7", "pygad": "#56B4E9"}  # fmt: skip
SPARE_COLOUR, BASELINE_COLOUR = "#882255", "#6E6E6E"
MARKERS = {
    "geneopt": "o",
    "deap": "s",
    "pymoo": "^",
    "optuna": "D",
    "nevergrad": "v",
    "pygad": "P",
    "random": "X",
}
VARIANT_STYLES = ["--", ":", "-."]  # a library's extra rows; its main row is solid


@dataclass(frozen=True)
class Style:
    color: str
    marker: str
    linestyle: str
    linewidth: float
    zorder: int


def palette(algorithms: Iterable[str]) -> dict[str, Style]:
    """Fixed style per label: hue from the library, line style from the variant (``lib[...]``)."""
    labels = sorted(set(algorithms))
    out = {}
    for label in labels:
        library = label.split("[")[0]
        variants = [x for x in labels if x.split("[")[0] == library]
        main = next((x for x in variants if "1/D" in x or "[" not in x), variants[0])
        extras = [x for x in variants if x != main]
        linestyle = "-" if label == main else VARIANT_STYLES[extras.index(label) % len(VARIANT_STYLES)]
        colour = BASELINE_COLOUR if library == "random" else LIBRARY_COLOURS.get(library, SPARE_COLOUR)
        emphasis = library == "geneopt"  # the subject of the report
        out[label] = Style(
            colour, MARKERS.get(library, "*"), linestyle, 2.4 if emphasis else 1.5, 3 if emphasis else 2
        )
    return out


# --- validation (spec 016 § Missing and incomplete data) ------------------------


def validate(runs: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in runs.columns]
    if missing:
        raise ValueError(f"runs frame is missing column(s): {', '.join(missing)}")
    if runs.empty:
        raise ValueError("runs frame is empty")
    for column in ("algorithm", "function_id"):
        if runs[column].isna().any():
            raise ValueError(f"runs frame has NaN in {column!r}")
    runs = runs.copy()
    reached = runs["evaluations_to_target"].notna()
    if "success_bool" not in runs.columns:
        runs["success_bool"] = reached
    elif (runs["success_bool"].astype(bool) != reached).any():
        bad = runs[runs["success_bool"].astype(bool) != reached].iloc[0]
        raise ValueError(f"success_bool disagrees with evaluations_to_target in row {bad.to_dict()}")
    duplicated = runs.duplicated(["algorithm", *PROBLEM, "target"])
    if duplicated.any():
        raise ValueError(f"duplicate (run, target) rows, a harness bug: {runs[duplicated].iloc[0].to_dict()}")
    return runs


def completeness(runs: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Algorithm -> (runs it has, problems in the frame). Gaps are logged."""
    problems = set(runs[PROBLEM].itertuples(index=False, name=None))
    out = {}
    for alg, own in runs.groupby("algorithm"):
        mine = set(own[PROBLEM].itertuples(index=False, name=None))
        out[alg] = (len(mine), len(problems))
        if gaps := sorted(problems - mine):
            log.warning("%s has no run on %d problem(s), counted as unsolved: %s", alg, len(gaps), gaps[:5])
    return out


def _label(alg: str, counts: dict[str, tuple[int, int]]) -> str:
    n, total = counts[alg]
    return alg if n == total else f"{alg} ({n}/{total} runs)"


def _pick_target(runs: pd.DataFrame, target: float | None) -> float:
    target = DEFAULT_TARGET if target is None else target
    available = np.sort(runs["target"].unique())
    match = available[np.isclose(available, target, rtol=1e-9, atol=0)]
    if not len(match):
        raise ValueError(
            f"target {target:g} not in the frame; available: {', '.join(f'{t:g}' for t in available)}"
        )
    return float(match[0])


def _figure(panels: int = 1, width: float = 6.5):
    """Figure, axes and font size. Fonts scale with width so a 3.5 in single column stays readable."""
    from matplotlib.figure import Figure  # OO API: no pyplot state, no GUI backend needed

    fs = 7 if width < 5 else 9
    fig = Figure(figsize=(width, width * 0.62 + 0.6), layout="constrained")  # + room for the legend below
    axes = fig.subplots(1, panels, sharey=True, squeeze=False)[0]
    for ax in axes:
        ax.grid(True, color="#E0E0E0", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=fs)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return fig, axes, fs


def _line(ax, x, y, s: Style, label: str, fs: int) -> None:
    """A step curve with sparse markers: the marker tells libraries apart in greyscale and for
    colour-blind readers, the line style tells a library's variants apart."""
    ax.step(x, y, where="post", color=s.color, linestyle=s.linestyle, linewidth=s.linewidth, zorder=s.zorder,
            marker=s.marker, markevery=0.1, markersize=fs * 0.55, label=label)  # fmt: skip


def _legend(fig, axes, fs: int, width: float) -> None:
    """One legend below the axes, so it never covers data; entries in plotting order (best first)."""
    seen = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            seen.setdefault(label, handle)
    fig.legend(seen.values(), seen.keys(), loc="outside lower center", ncol=3 if width < 5 else 4,
               frameon=False, fontsize=fs - 1)  # fmt: skip


# --- 1. ECDF ----------------------------------------------------------------------


def ecdf_curve(runs: pd.DataFrame, algorithm: str) -> tuple[np.ndarray, np.ndarray]:
    """(budgets per dimension, fraction of (problem, target) pairs solved) as a step function."""
    pairs = len(runs.drop_duplicates([*PROBLEM, "target"]))
    own = runs[runs["algorithm"] == algorithm]
    hits = np.sort(np.maximum((own["evaluations_to_target"] / own["dimension"]).dropna().to_numpy(), 1.0))
    return hits, np.arange(1, len(hits) + 1) / pairs


def plot_ecdf(runs: pd.DataFrame, *, per_dimension: bool = True, width: float = 6.5):
    runs = validate(runs)
    styles = palette(runs["algorithm"])
    dims = sorted(runs["dimension"].unique()) if per_dimension else [None]
    fig, axes, fs = _figure(len(dims), width)
    for ax, dim in zip(axes, dims, strict=True):
        part = runs if dim is None else runs[runs["dimension"] == dim]
        counts = completeness(part)
        budget = part["budget"] if "budget" in part.columns else part["evaluations"]
        end = float((budget / part["dimension"]).max())
        curves = {alg: ecdf_curve(part, alg) for alg in styles if alg in counts}
        for alg, (x, y) in sorted(curves.items(), key=lambda kv: -(kv[1][1][-1] if len(kv[1][1]) else 0)):
            s = styles[alg]
            xs, ys = np.r_[1.0, x, end], np.r_[0.0, y, y[-1] if len(y) else 0.0]
            _line(ax, xs, ys, s, _label(alg, counts), fs)
            # budget end; unclipped, or the axis edge cuts the × in half
            ax.plot([end], [ys[-1]], marker="x", color=s.color, markersize=7, zorder=s.zorder, clip_on=False)
        ax.set_xscale("log")
        ax.set_xlim(1, end)
        ax.set_ylim(0, 1)
        ax.set_xlabel("evaluations / D", fontsize=fs)
        ax.set_title("all dimensions" if dim is None else f"D = {dim}", fontsize=fs)
    axes[0].set_ylabel("share of (problem, target) pairs", fontsize=fs)
    _legend(fig, axes, fs, width)
    return fig


# --- 2. Dolan–Moré performance profile ------------------------------------------------


def profile_ratios(runs: pd.DataFrame, target: float) -> tuple[dict[str, np.ndarray], int]:
    """Algorithm -> sorted ratios t / best t per problem (inf = unsolved); problems solved by
    nobody are excluded and counted."""
    at = runs[np.isclose(runs["target"], target, rtol=1e-9, atol=0)]
    # pivot, not pivot_table: pivot_table drops problems nobody solved, which must be counted
    table = at.pivot(index=PROBLEM, columns="algorithm", values="evaluations_to_target")
    problems = pd.MultiIndex.from_frame(runs[PROBLEM].drop_duplicates())
    table = table.reindex(
        index=problems, columns=sorted(runs["algorithm"].unique())
    )  # missing runs: unsolved
    solved = table.notna().any(axis=1)
    table = table[solved]
    ratios = table.div(table.min(axis=1), axis=0).fillna(np.inf)
    return {alg: np.sort(ratios[alg].to_numpy()) for alg in ratios.columns}, int((~solved).sum())


def rho(ratios: np.ndarray, tau: float) -> float:
    return float((ratios <= tau).mean()) if len(ratios) else 0.0


def plot_performance_profile(runs: pd.DataFrame, *, target: float | None = None, tau_max: float | None = None,
                             width: float = 6.5):  # fmt: skip
    runs = validate(runs)
    target = _pick_target(runs, target)
    ratios, excluded = profile_ratios(runs, target)
    counts, styles = completeness(runs), palette(runs["algorithm"])
    finite = np.concatenate(
        [np.array([1.0]), *(r[np.isfinite(r)] for r in ratios.values())]
    )  # nobody solved: 1
    tau_max = tau_max or 2 * max(float(finite.max()), 1.0)
    fig, (ax,), fs = _figure(1, width)
    for alg, r in sorted(ratios.items(), key=lambda kv: -rho(kv[1], tau_max)):
        s = styles[alg]
        taus = np.r_[1.0, r[np.isfinite(r) & (r <= tau_max)], tau_max]
        _line(ax, taus, [rho(r, t) for t in taus], s, _label(alg, counts), fs)
    total = len(runs.drop_duplicates(PROBLEM))
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, tau_max)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"τ = evaluations to {target:g} ÷ fastest's", fontsize=fs)
    ax.set_ylabel("share of problems within τ", fontsize=fs)
    ax.set_title(f"{excluded} of {total} problems unsolved by all, excluded", fontsize=fs)
    _legend(fig, [ax], fs, width)
    return fig


# --- 3. speed vs success Pareto ---------------------------------------------------------


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def pareto_points(runs: pd.DataFrame, target: float, cost: str) -> pd.DataFrame:
    """One row per algorithm: cost, success rate with Wilson 95% bounds, hollow (some costs
    missing), frontier (no other point has both lower cost and higher success). Sorted by cost."""
    problems = len(runs.drop_duplicates(PROBLEM))
    at = runs[np.isclose(runs["target"], target, rtol=1e-9, atol=0)]
    rows = []
    for alg, own in at.groupby("algorithm"):
        costs = own[cost]
        if costs.isna().all():
            log.warning("%s has no %s at all; left out of the Pareto plot", alg, cost)
            continue
        wins = int(own["success_bool"].sum())  # missing runs count as failures: n is every problem
        low, high = _wilson(wins, problems)
        rows.append({"algorithm": alg, "cost": float(costs.mean()), "success": wins / problems,
                     "low": low, "high": high, "hollow": bool(costs.isna().any())})  # fmt: skip
    pts = pd.DataFrame(rows).sort_values(["cost", "algorithm"], ignore_index=True)
    pts["frontier"] = [
        not ((pts["cost"] < c) & (pts["success"] > s)).any()
        for c, s in zip(pts["cost"], pts["success"], strict=True)
    ]
    return pts


def plot_pareto(
    runs: pd.DataFrame, *, target: float | None = None, cost: str = "elapsed_time", width: float = 6.5
):
    runs = validate(runs)
    target = _pick_target(runs, target)
    pts, styles = pareto_points(runs, target, cost), palette(runs["algorithm"])
    fig, (ax,), fs = _figure(1, width)
    front = pts[pts["frontier"]]
    ax.plot(front["cost"], front["success"], linestyle="--", color="#9E9E9E", linewidth=1, zorder=1)
    for _, p in pts.iterrows():
        s = styles[p["algorithm"]]
        ax.errorbar(p["cost"], p["success"], yerr=[[p["success"] - p["low"]], [p["high"] - p["success"]]],
                    fmt=s.marker, color=s.color, markersize=8, mfc="white" if p["hollow"] else s.color,
                    elinewidth=1, capsize=2, zorder=s.zorder + 2)  # fmt: skip
    ax.set_xscale("log")
    ax.set_ylim(-0.02, min(1.02, float(pts["high"].max()) * 1.25 + 0.03))  # low success rates get room
    ax.set_xlabel(
        "mean seconds per run" if cost == "elapsed_time" else "mean evaluations per run", fontsize=fs
    )
    ax.set_ylabel(f"success rate (reached {target:g})", fontsize=fs)
    ax.set_title("dashed: frontier · bars: 95% CI · hollow: times missing", fontsize=fs)
    _place_labels(ax, pts)
    return fig


# label positions to try, in points from the marker: 4 rings x 8 directions, nearest ring first
LABEL_OFFSETS = [
    (round(r * np.cos(np.radians(a))), round(r * np.sin(np.radians(a))))
    for r in (10, 22, 36, 52)
    for a in (30, -30, 150, -150, 90, -90, 0, 180)
]


def _place_labels(ax, pts: pd.DataFrame) -> None:
    """Greedy, dependency-free label placement: each label takes the first candidate offset whose
    text overlaps no marker and no earlier label and stays inside the axes; if none is free, the one
    with the fewest collisions. Frontier labels go first. Labels away from their point get a leader."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.transforms import Bbox

    canvas = FigureCanvasAgg(ax.figure)
    canvas.draw()  # settle the constrained layout so pixel positions are final
    renderer = canvas.get_renderer()
    xy = [ax.transData.transform((c, s)) for c, s in zip(pts["cost"], pts["success"], strict=True)]
    taken = [Bbox.from_bounds(x - 7, y - 7, 14, 14) for x, y in xy]  # markers
    for c, low, high in zip(pts["cost"], pts["low"], pts["high"], strict=True):  # and error bars
        (x0, y0), (_, y1) = ax.transData.transform([(c, low), (c, high)])
        taken.append(Bbox.from_extents(x0 - 4, y0, x0 + 4, y1))
    inside = ax.get_window_extent(renderer)  # labels stay inside the axes, or the figure edge clips them

    def label(p, offset, leader=None):
        size, ink = (7, "#222222") if p["frontier"] else (6, "#8A8A8A")  # dominated points recede
        dx, dy = offset
        text = ax.annotate(
            p["algorithm"], (p["cost"], p["success"]), xytext=offset, textcoords="offset points", va="center",
            ha="center" if dx == 0 else "left" if dx > 0 else "right", fontsize=size, color=ink,
            arrowprops=dict(arrowstyle="-", color=ink, lw=0.5, shrinkA=0, shrinkB=5) if leader else None,
        )  # fmt: skip
        text.set_in_layout(False)  # else the layout shrinks the axes afterwards and labels collide again
        return text

    for i in sorted(range(len(pts)), key=lambda k: not pts["frontier"].iloc[k]):
        p = pts.iloc[i]
        tried = []
        for offset in LABEL_OFFSETS:
            text = label(p, offset)  # measured without a leader: Annotation boxes include the leader
            box = text.get_window_extent(renderer)
            text.remove()
            outside = not (
                inside.x0 <= box.x0 and box.x1 <= inside.x1 and inside.y0 <= box.y0 and box.y1 <= inside.y1
            )
            tried.append((sum(box.overlaps(t) for t in taken) + outside, len(tried), offset, box))
            if tried[-1][0] == 0:
                break
        _, _, offset, box = min(tried)  # the first free spot, else the fewest collisions
        label(p, offset, leader=offset not in LABEL_OFFSETS[:8])  # nearest ring: close enough to need none
        taken.append(box)


def figures(runs: pd.DataFrame) -> dict:
    return {"ecdf": plot_ecdf(runs), "profile": plot_performance_profile(runs), "pareto": plot_pareto(runs)}


def save(fig, path) -> list[Path]:
    """Write ``path``.pdf (vector) and ``path``.png (DPI)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = [path.with_suffix(".pdf"), path.with_suffix(".png")]
    fig.savefig(written[0])
    fig.savefig(written[1], dpi=DPI)
    return written
