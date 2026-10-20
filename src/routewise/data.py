"""Download HELM Lite per-instance results and build the routing table.

HELM publishes, for every model and scenario, each question it asked, the model's
answer, the token counts and whether the answer was scored correct. That is
exactly what an offline router study needs: for every prompt, which models get
it right and what each one costs.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import urllib.parse
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

LIST_URL = "https://storage.googleapis.com/storage/v1/b/crfm-helm-public/o"
FILE_URL = "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/runs"
RUN_PREFIX = "lite/benchmark_output/runs/"
RELEASES = [f"v1.{i}.0" for i in range(14)]


@dataclass(frozen=True)
class Run:
    release: str
    name: str

    @property
    def key(self) -> str:
        """Run name with the model removed, used to match runs across models."""
        return re.sub(r",?model=[^,]+", "", self.name)

    @property
    def model(self) -> str:
        match = re.search(r"model=([^,]+)", self.name)
        return match.group(1) if match else ""


def _get_json(url: str, retries: int = 4) -> object:
    last: Exception | None = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.load(response)
        except Exception as error:  # network hiccups through proxies are common
            last = error
    raise RuntimeError(f"failed to fetch {url}") from last


def list_runs() -> list[Run]:
    runs: list[Run] = []
    for release in RELEASES:
        token = None
        while True:
            query = {"prefix": f"{RUN_PREFIX}{release}/", "delimiter": "/", "maxResults": "1000"}
            if token:
                query["pageToken"] = token
            page = _get_json(f"{LIST_URL}?{urllib.parse.urlencode(query)}")
            assert isinstance(page, dict)
            for prefix in page.get("prefixes", []):
                runs.append(Run(release, prefix.rstrip("/").split("/")[-1]))
            token = page.get("nextPageToken")
            if not token:
                break
    return runs


def select_runs(
    runs: Iterable[Run], models: list[str], scenarios: dict[str, dict[str, str]]
) -> dict[tuple[str, str], Run]:
    """Pick one run per (model, run key) for the configured scenarios.

    The first model defines the set of run keys. A model whose run carries an
    extra ``stop=none`` suffix is matched to the same key.
    """
    by_model: dict[str, dict[str, Run]] = {}
    for run in runs:
        by_model.setdefault(run.model, {})[run.key] = run
    prefixes = tuple(spec["prefix"] for spec in scenarios.values())
    reference = by_model[models[0]]
    keys = sorted(key for key in reference if key.startswith(prefixes))
    selected: dict[tuple[str, str], Run] = {}
    for model in models:
        available = by_model.get(model, {})
        for key in keys:
            found = available.get(key)
            if found is None:
                alt = key.replace(",stop=none", "") if "stop=none" in key else f"{key},stop=none"
                found = available.get(alt)
            if found is None:
                raise KeyError(f"{model} has no run for {key}")
            selected[(model, key)] = found
    return selected


def _download(run: Run, filename: str, cache: Path) -> Path:
    path = cache / run.release / run.name / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{FILE_URL}/{run.release}/{urllib.parse.quote(run.name)}/{filename}"
    data = _get_json(url)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)
    return path


def scenario_of(key: str, scenarios: dict[str, dict[str, str]]) -> str:
    for name, spec in scenarios.items():
        if key.startswith(spec["prefix"]):
            return name
    raise KeyError(key)


def build_table(
    models: list[str], scenarios: dict[str, dict[str, str]], cache: Path, out: Path
) -> tuple[int, int]:
    """Download what is needed and write records.csv.gz and prompts.jsonl.gz."""
    selected = select_runs(list_runs(), models, scenarios)
    jobs = [(run, "display_predictions.json") for run in selected.values()]
    keys = sorted({key for _, key in selected})
    jobs += [(selected[(models[0], key)], "instances.json") for key in keys]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda job: _download(job[0], job[1], cache), jobs))

    out.mkdir(parents=True, exist_ok=True)
    prompts: dict[str, dict[str, str]] = {}
    for key in keys:
        scenario = scenario_of(key, scenarios)
        run = selected[(models[0], key)]
        instances = json.loads((cache / run.release / run.name / "instances.json").read_text())
        for instance in instances:
            uid = f"{key}|{instance['id']}"
            prompts[uid] = {"uid": uid, "scenario": scenario, "text": instance["input"]["text"]}

    rows = 0
    with gzip.open(out / "records.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uid", "scenario", "model", "correct", "prompt_tokens", "output_tokens"])
        for (model, key), run in sorted(selected.items()):
            scenario = scenario_of(key, scenarios)
            metric = scenarios[scenario]["metric"]
            path = cache / run.release / run.name / "display_predictions.json"
            for prediction in json.loads(path.read_text()):
                if prediction.get("train_trial_index", 0) != 0:
                    continue
                uid = f"{key}|{prediction['instance_id']}"
                stats = prediction["stats"]
                if uid not in prompts or metric not in stats:
                    continue
                writer.writerow(
                    [
                        uid,
                        scenario,
                        model,
                        int(stats[metric] >= 0.5),
                        int(stats.get("num_prompt_tokens", 0)),
                        int(stats.get("num_output_tokens", 0)),
                    ]
                )
                rows += 1
    with gzip.open(out / "prompts.jsonl.gz", "wt") as handle:
        for prompt in prompts.values():
            handle.write(json.dumps(prompt) + "\n")
    return rows, len(prompts)
