"""Model pool, prices and scenario configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from routewise.data import Suite


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    input: float  # USD per 1M input tokens
    output: float  # USD per 1M output tokens

    def cost(self, prompt_tokens: float, output_tokens: float) -> float:
        return (prompt_tokens * self.input + output_tokens * self.output) / 1e6


def load_models(path: Path) -> list[Model]:
    data = yaml.safe_load(path.read_text())
    return [Model(**{k: m[k] for k in ("id", "name", "input", "output")}) for m in data["models"]]


def load_suite(path: Path) -> Suite:
    data = yaml.safe_load(path.read_text())
    return Suite(data["suite"], int(data["releases"]), dict(data["scenarios"]))
