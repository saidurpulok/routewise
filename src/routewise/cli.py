"""Command-line entry point: fetch data, run the benchmark, draw plots."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from routewise.config import load_models, load_scenarios
from routewise.data import build_table
from routewise.evaluate import Router, load_data, route_cv, summarize
from routewise.routers import KNNRouter, LogisticRouter, TaskRouter


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="routewise")
    parser.add_argument("--models", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--scenarios", type=Path, default=Path("configs/scenarios.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download HELM Lite results and build the routing table")
    run = sub.add_parser("run", help="cross-validate routers and write results")
    run.add_argument("--output", type=Path, default=Path("results"))
    run.add_argument("--folds", type=int, default=5)
    run.add_argument("--seed", type=int, default=0)
    plot = sub.add_parser("plot", help="draw figures from a results file")
    plot.add_argument("--results", type=Path, default=Path("results/results.json"))
    args = parser.parse_args(argv)

    models = load_models(args.models)
    if args.command == "fetch":
        rows, prompts = build_table(
            [m.id for m in models],
            load_scenarios(args.scenarios),
            args.data / "cache",
            args.data / "processed",
        )
        print(f"wrote {rows} records for {prompts} prompts")
    elif args.command == "run":
        data = load_data(args.data / "processed", models)
        routers: list[Router] = [TaskRouter(), KNNRouter(), LogisticRouter()]
        results: dict[str, Any] = {}
        for router in routers:
            start = time.time()
            results[router.name] = route_cv(data, router, k=args.folds, seed=args.seed)
            print(f"{router.name}: {time.time() - start:.1f}s")
        summary = summarize(data, results)
        summary["config"] = {
            "folds": args.folds,
            "seed": args.seed,
            "prices_usd_per_1m": {m.name: [m.input, m.output] for m in models},
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
        np.savez_compressed(args.output / "choices.npz", **results)
        print(json.dumps({k: v for k, v in summary.items() if k != "routers"}, indent=2)[:2000])
        for name, r in summary["routers"].items():
            print(name, {k: v for k, v in r.items() if k not in ("curve",)})
    elif args.command == "plot":
        from routewise.plot import draw

        draw(json.loads(args.results.read_text()), args.results.parent / "plots")


if __name__ == "__main__":
    main()
