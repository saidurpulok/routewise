"""Cross-validated cost-quality evaluation of routers against single models."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from routewise.config import Model
from routewise.routers import FloatArray, IntArray, expected_cost, select

LAMBDAS = np.concatenate([[0.0], np.logspace(0, 5, 120)])


class Router(Protocol):
    name: str

    def predict(self, data: Any, train: IntArray, test: IntArray) -> FloatArray: ...


@dataclass
class RoutingData:
    uids: list[str]
    texts: list[str]
    scenarios: NDArray[np.str_]
    correct: FloatArray
    cost: FloatArray
    prompt_tokens: FloatArray
    output_tokens: FloatArray
    input_price: FloatArray
    output_price: FloatArray
    models: list[Model] = field(default_factory=list)
    imputed: dict[str, list[str]] = field(default_factory=dict)


def impute_output_tokens(
    output_tokens: FloatArray, scenarios: NDArray[np.str_], models: list[Model]
) -> tuple[FloatArray, dict[str, list[str]]]:
    """Fill in output lengths HELM didn't record.

    For some providers HELM stores zero output tokens on every prompt, which would
    make those models look almost free. Where a model's output count is zero for a
    whole task, use the median output length of the other models on each prompt.
    """
    out = output_tokens.copy()
    imputed: dict[str, list[str]] = {}
    for scenario in np.unique(scenarios):
        rows = scenarios == scenario
        missing = (output_tokens[rows] == 0).all(axis=0)
        if missing.all() or not missing.any():
            continue
        fill = np.median(output_tokens[rows][:, ~missing], axis=1)
        block = out[rows]
        block[:, missing] = fill[:, None]
        out[rows] = block
        for m in np.flatnonzero(missing):
            imputed.setdefault(models[m].name, []).append(str(scenario))
    return out, imputed


def load_data(processed: Path, models: list[Model]) -> RoutingData:
    index = {m.id: i for i, m in enumerate(models)}
    prompts: dict[str, dict[str, str]] = {}
    with gzip.open(processed / "prompts.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            prompts[row["uid"]] = row
    values: dict[str, dict[int, tuple[int, int, int]]] = {}
    with gzip.open(processed / "records.csv.gz", "rt") as handle:
        for rec in csv.DictReader(handle):
            if rec["model"] in index:
                values.setdefault(rec["uid"], {})[index[rec["model"]]] = (
                    int(rec["correct"]),
                    int(rec["prompt_tokens"]),
                    int(rec["output_tokens"]),
                )
    uids = sorted(u for u, v in values.items() if len(v) == len(models) and u in prompts)
    arr = np.array([[values[u][m] for m in range(len(models))] for u in uids], dtype=np.float64)
    input_price = np.array([m.input for m in models]) / 1e6
    output_price = np.array([m.output for m in models]) / 1e6
    correct, prompt_tokens = arr[..., 0], arr[..., 1]
    scenarios = np.array([prompts[u]["scenario"] for u in uids])
    output_tokens, imputed = impute_output_tokens(arr[..., 2], scenarios, models)
    return RoutingData(
        uids=uids,
        texts=[prompts[u]["text"] for u in uids],
        scenarios=scenarios,
        correct=correct,
        cost=prompt_tokens * input_price + output_tokens * output_price,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        input_price=input_price,
        output_price=output_price,
        models=models,
        imputed=imputed,
    )


def folds(scenarios: NDArray[np.str_], k: int, seed: int) -> list[IntArray]:
    """Stratified by task so every fold has the same task mix."""
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[] for _ in range(k)]
    for scenario in np.unique(scenarios):
        idx = rng.permutation(np.flatnonzero(scenarios == scenario))
        for i, chunk in enumerate(np.array_split(idx, k)):
            parts[i].extend(chunk.tolist())
    return [np.sort(np.array(p, dtype=np.int64)) for p in parts]


def route_cv(data: RoutingData, router: Router, k: int = 5, seed: int = 0) -> NDArray[np.int64]:
    """Out-of-fold model choice for every prompt at every lambda: (lambdas, prompts)."""
    choices = np.zeros((len(LAMBDAS), len(data.uids)), dtype=np.int64)
    parts = folds(data.scenarios, k, seed)
    everything = np.arange(len(data.uids))
    for test in parts:
        train = np.setdiff1d(everything, test)
        p = router.predict(data, train, test)
        c = expected_cost(data, train, test)
        for i, lam in enumerate(LAMBDAS):
            choices[i, test] = select(p, c, float(lam))
    return choices


def curve(data: RoutingData, choices: NDArray[np.int64], rows: IntArray) -> FloatArray:
    """(lambdas, 2) array of mean accuracy and mean cost on the given prompts."""
    pick = choices[:, rows]
    acc = data.correct[rows[None, :], pick].mean(axis=1)
    cost = data.cost[rows[None, :], pick].mean(axis=1)
    return np.stack([acc, cost], axis=1)


def pareto(points: FloatArray) -> FloatArray:
    """Points (acc, cost) not dominated by a cheaper, at-least-as-accurate point, sorted by cost."""
    order = np.lexsort((-points[:, 0], points[:, 1]))
    out: list[NDArray[np.float64]] = []
    best = -np.inf
    for p in points[order]:
        if p[0] > best + 1e-12:
            out.append(p)
            best = p[0]
    return np.array(out)


def hull_accuracy(singles: FloatArray, cost: float) -> float:
    """Best accuracy reachable at a given cost by randomly mixing two single models."""
    pts = pareto(singles)
    if cost <= pts[0, 1]:
        return float(pts[0, 0])
    best = float(pts[0, 0])
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            a, b = pts[i], pts[j]
            if a[1] <= cost <= b[1]:
                t = (cost - a[1]) / (b[1] - a[1]) if b[1] > a[1] else 1.0
                best = max(best, float(a[0] + t * (b[0] - a[0])))
        if pts[i, 1] <= cost:
            best = max(best, float(pts[i, 0]))
    return best


def cost_to_reach(points: FloatArray, target: float) -> float:
    """Lowest cost on a curve reaching at least ``target`` accuracy (inf if never)."""
    ok = points[:, 0] >= target - 1e-12
    return float(points[ok, 1].min()) if ok.any() else float("inf")


def single_points(data: RoutingData, rows: IntArray) -> FloatArray:
    return np.stack([data.correct[rows].mean(axis=0), data.cost[rows].mean(axis=0)], axis=1)


def oracle_point(data: RoutingData, rows: IntArray) -> tuple[float, float]:
    """Cheapest correct model when any is correct, otherwise the cheapest model."""
    c = data.cost[rows]
    masked = np.where(data.correct[rows] > 0, c, np.inf)
    pick = np.where(np.isfinite(masked.min(axis=1)), masked.argmin(axis=1), c.argmin(axis=1))
    r = np.arange(len(rows))
    return float(data.correct[rows][r, pick].mean()), float(c[r, pick].mean())


def _per_1k(cost: float) -> float | None:
    return float(1000 * cost) if np.isfinite(cost) else None


def _best_within(points: FloatArray, budget: float) -> float | None:
    """Highest accuracy on a curve at or below a cost budget (None if nothing fits)."""
    ok = points[:, 1] <= budget + 1e-15
    return float(points[ok, 0].max()) if ok.any() else None


def summarize(
    data: RoutingData, results: dict[str, NDArray[np.int64]], n_boot: int = 1000, seed: int = 0
) -> dict[str, Any]:
    rows = np.arange(len(data.uids))
    singles = single_points(data, rows)
    best = int(np.argmax(singles[:, 0]))
    best_acc, best_cost = singles[best]
    oracle_acc, oracle_cost = oracle_point(data, rows)
    summary: dict[str, Any] = {
        "prompts": len(rows),
        "tasks": {s: int((data.scenarios == s).sum()) for s in np.unique(data.scenarios)},
        "models": [
            {
                "name": m.name,
                "accuracy": float(singles[i, 0]),
                "cost_per_1k": float(1000 * singles[i, 1]),
                "per_task_accuracy": {
                    s: float(data.correct[data.scenarios == s, i].mean())
                    for s in np.unique(data.scenarios)
                },
            }
            for i, m in enumerate(data.models)
        ],
        "best_single": {
            "name": data.models[best].name,
            "accuracy": float(best_acc),
            "cost_per_1k": float(1000 * best_cost),
        },
        "oracle": {"accuracy": oracle_acc, "cost_per_1k": 1000 * oracle_cost},
        "imputed_output_tokens": data.imputed,
        "routers": {},
    }
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(rows), len(rows)) for _ in range(n_boot)]
    for name, choices in results.items():
        pts = curve(data, choices, rows)
        front = pareto(pts)
        reach = cost_to_reach(pts, best_acc)
        # Cost saving vs the best single model at matched accuracy, with a bootstrap interval.
        savings = []
        for b in boots:
            s = single_points(data, b)
            target_acc, target_cost = s[best]
            c = cost_to_reach(curve(data, choices, b), target_acc)
            savings.append(1 - c / target_cost if np.isfinite(c) else np.nan)
        sv = np.array(savings)
        gains = [p[0] - hull_accuracy(singles, p[1]) for p in front]
        summary["routers"][name] = {
            "max_accuracy": float(np.max(pts[:, 0])),
            "cost_to_match_best_per_1k": float(1000 * reach) if np.isfinite(reach) else None,
            "saving_vs_best": float(1 - reach / best_cost) if np.isfinite(reach) else None,
            "saving_vs_best_ci95": (
                [float(np.nanpercentile(sv, 2.5)), float(np.nanpercentile(sv, 97.5))]
                if np.isfinite(sv).mean() > 0.95
                else None
            ),
            "share_bootstrap_reaching_best": float(np.isfinite(sv).mean()),
            "max_gain_over_single_model_hull": float(max(gains)),
            "curve": [
                {"lambda": float(lam), "accuracy": float(a), "cost_per_1k": float(1000 * c)}
                for lam, (a, c) in zip(LAMBDAS, pts, strict=True)
            ],
        }
        summary["routers"][name]["vs_single"] = {
            m.name: {
                "cost_to_match_per_1k": _per_1k(cost_to_reach(pts, singles[j, 0])),
                "accuracy_at_same_cost": _best_within(pts, singles[j, 1]),
            }
            for j, m in enumerate(data.models)
        }
        if np.isfinite(reach):
            i = int(np.flatnonzero(np.isclose(pts[:, 1], reach) & (pts[:, 0] >= best_acc))[0])
            picks = choices[i]
            summary["routers"][name]["mix_at_match"] = {
                m.name: float((picks == j).mean()) for j, m in enumerate(data.models)
            }
    return summary
