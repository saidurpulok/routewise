"""Figures drawn from results.json, so plots never need the raw data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from routewise.evaluate import hull_accuracy, pareto  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
_SUITES = {"lite": "HELM Lite", "capabilities": "HELM Capabilities"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def cost_quality(summary: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor=SURFACE)
    _style(ax)
    models = summary["models"]
    suite = summary.get("suite", "lite")
    singles = np.array([[m["accuracy"], m["cost_per_1k"]] for m in models])
    ax.scatter(
        singles[:, 1],
        singles[:, 0],
        s=36,
        color=MUTED,
        zorder=3,
        edgecolors=SURFACE,
        linewidths=1.5,
        label="Single model",
    )
    best_acc = summary["best_single"]["accuracy"]
    for m in models:
        # Labels sit just right of each point. Points on the flat stretch of the mixing
        # line go below it, and the upper point of a near-overlapping pair goes above.
        x, y = np.log10(m["cost_per_1k"]), m["accuracy"]
        crowded = any(
            abs(np.log10(o["cost_per_1k"]) - x) < 0.12 and 0 < y - o["accuracy"] < 0.03
            for o in models
        )
        dy = 5 if crowded else -11 if y >= best_acc - 0.005 else -3
        ax.annotate(
            m["name"],
            (m["cost_per_1k"], y),
            xytext=(4, dy),
            textcoords="offset points",
            fontsize=8,
            color=MUTED,
        )
    # Randomly mixing two models traces a straight line in linear cost, curved on a log axis.
    grid = np.geomspace(singles[:, 1].min(), singles[:, 1].max(), 200)
    hull = [hull_accuracy(singles[:, :2], c) for c in grid]
    ax.plot(
        grid,
        hull,
        color=MUTED,
        linestyle="--",
        linewidth=1.2,
        label="Best mix of two single models",
    )
    for color, (name, router) in zip(SERIES, summary["routers"].items(), strict=False):
        pts = np.array([[p["accuracy"], p["cost_per_1k"]] for p in router["curve"]])
        front = pareto(pts)
        ax.plot(front[:, 1], front[:, 0], color=color, linewidth=2, label=name, zorder=4)
    oracle = summary["oracle"]
    ax.scatter(
        [oracle["cost_per_1k"]],
        [oracle["accuracy"]],
        marker="*",
        s=140,
        color=INK,
        zorder=5,
        label="Oracle (cheapest correct model)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1,000 prompts (USD, log scale)", color=INK)
    ax.set_ylabel("Accuracy", color=INK)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("$%.2g"))
    ax.set_title(
        f"Routing across {len(models)} models on {summary['prompts']:,} {_SUITES.get(suite, suite)}"
        " prompts"
        " (5-fold CV)",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def mix(summary: dict[str, Any], router: str, path: Path) -> None:
    shares = summary["routers"][router].get("mix_at_match")
    if not shares:
        return
    items = sorted((v, k) for k, v in shares.items() if v >= 0.005)
    rest = sum(v for v in shares.values() if v < 0.005)
    if rest > 0:
        items.insert(0, (rest, "Others (<0.5% each)"))
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(items) + 1.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    values = [v for v, _ in items]
    ax.barh([k for _, k in items], values, color=SERIES[0], height=0.6)
    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f"{v:.0%}", va="center", fontsize=8, color=INK)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel("Share of prompts", color=INK)
    best = summary["best_single"]["name"]
    ax.set_title(
        f"{router}: where prompts go when matching {best}", color=INK, fontsize=10, loc="left"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def reliability(summary: dict[str, Any], path: Path) -> None:
    """Predicted probability of a correct answer against how often it was correct."""
    routers = {n: r["calibration"] for n, r in summary["routers"].items() if "calibration" in r}
    if not routers:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.2), facecolor=SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Perfect calibration")
    for color, (name, cal) in zip(SERIES, routers.items(), strict=False):
        table = [b for b in cal["reliability"] if b["share"] >= 0.005]  # skip near-empty bins
        ax.plot(
            [b["predicted"] for b in table],
            [b["observed"] for b in table],
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"{name} (Brier {cal['brier']:.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted chance the model is correct", color=INK)
    ax.set_ylabel("Share actually correct", color=INK)
    ax.set_title(
        "Calibration (out-of-fold, all prompt-model pairs)", color=INK, fontsize=10, loc="left"
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def draw(summary: dict[str, Any], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cost_quality(summary, outdir / "cost_quality.png")
    best_router = max(summary["routers"], key=lambda r: summary["routers"][r]["max_accuracy"])
    mix(summary, best_router, outdir / "routing_mix.png")
    reliability(summary, outdir / "calibration.png")
