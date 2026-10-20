"""Command-line entry point: fetch data, run the benchmark, draw plots."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from routewise.config import load_models, load_suite
from routewise.data import build_table
from routewise.evaluate import Router, load_data, route_cv, summarize
from routewise.routers import KNNRouter, LogisticRouter, TaskRouter


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="routewise")
    parser.add_argument(
        "--suite",
        default="capabilities",
        help="HELM leaderboard to use; reads configs/<suite>/, writes data/<suite>/ and "
        "results/<suite>/ (default: capabilities)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download HELM results and build the routing table")
    run = sub.add_parser("run", help="cross-validate routers and write results")
    run.add_argument("--folds", type=int, default=5)
    run.add_argument("--seed", type=int, default=0)
    sub.add_parser("plot", help="draw figures from the results file")
    args = parser.parse_args(argv)

    configs, data_dir = Path("configs") / args.suite, Path("data") / args.suite
    output = Path("results") / args.suite
    models = load_models(configs / "models.yaml")
    suite = load_suite(configs / "scenarios.yaml")
    if args.command == "fetch":
        rows, prompts = build_table(
            suite, [m.id for m in models], data_dir / "cache", data_dir / "processed"
        )
        print(f"wrote {rows} records for {prompts} prompts")
    elif args.command == "run":
        data = load_data(data_dir / "processed", models)
        routers: list[Router] = [TaskRouter(), KNNRouter(), LogisticRouter()]
        results: dict[str, Any] = {}
        for router in routers:
            start = time.time()
            results[router.name] = route_cv(data, router, k=args.folds, seed=args.seed)
            print(f"{router.name}: {time.time() - start:.1f}s")
        summary = {"suite": suite.name, **summarize(data, results)}
        summary["config"] = {
            "folds": args.folds,
            "seed": args.seed,
            "prices_usd_per_1m": {m.name: [m.input, m.output] for m in models},
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
        np.savez_compressed(output / "choices.npz", **results)
        print(json.dumps({k: v for k, v in summary.items() if k != "routers"}, indent=2)[:2000])
        for name, r in summary["routers"].items():
            print(name, {k: v for k, v in r.items() if k not in ("curve",)})
    elif args.command == "plot":
        from routewise.plot import draw

        draw(json.loads((output / "results.json").read_text()), output / "plots")


if __name__ == "__main__":
    main()
