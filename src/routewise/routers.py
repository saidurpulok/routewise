"""Routers that pick one model per prompt, trading predicted quality against cost.

Every router estimates, for each prompt and model, the probability that the
model answers correctly. Selection is then shared: pick the model maximizing
``p_correct - lam * expected_cost`` and sweep ``lam`` to trace a cost-quality
curve. Routers only see information available before a request is sent: the
prompt text, its task, and costs observed on training prompts.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class Dataset(Protocol):
    texts: list[str]
    scenarios: NDArray[np.str_]
    correct: FloatArray  # (prompts, models) in {0, 1}
    cost: FloatArray  # (prompts, models) USD
    prompt_tokens: FloatArray  # (prompts, models)
    output_tokens: FloatArray  # (prompts, models)
    input_price: FloatArray  # (models,) USD per token
    output_price: FloatArray  # (models,) USD per token


def expected_cost(
    data: Dataset, train: IntArray, test: IntArray, by_task: bool = True
) -> FloatArray:
    """Cost known before sending: real prompt tokens plus a mean output length.

    The output length is each model's mean over training prompts of the same task,
    or over all training prompts when the router may not use the task label.
    """
    out = np.zeros((len(test), data.correct.shape[1]))
    groups = np.unique(data.scenarios[test]) if by_task else [None]
    for scenario in groups:
        tr = train if scenario is None else train[data.scenarios[train] == scenario]
        te = np.ones(len(test), bool) if scenario is None else data.scenarios[test] == scenario
        mean_output = data.output_tokens[tr].mean(axis=0)
        out[te] = data.prompt_tokens[test[te]] * data.input_price + mean_output * data.output_price
    return out


def select(p_correct: FloatArray, cost: FloatArray, lam: float) -> IntArray:
    """Index of the chosen model per prompt; ties go to the cheaper model."""
    score = p_correct - lam * cost
    best = score.max(axis=1, keepdims=True)
    tied = np.isclose(score, best, rtol=0, atol=1e-12)
    return np.where(tied, cost, np.inf).argmin(axis=1).astype(np.int64)


class TaskRouter:
    """Uses only the task label: each model's training accuracy on that task."""

    name = "Task-aware"
    uses_task = True

    def predict(self, data: Dataset, train: IntArray, test: IntArray) -> FloatArray:
        out = np.zeros((len(test), data.correct.shape[1]))
        for scenario in np.unique(data.scenarios[test]):
            tr = train[data.scenarios[train] == scenario]
            out[data.scenarios[test] == scenario] = data.correct[tr].mean(axis=0)
        return out


def _features(
    data: Dataset, train: IntArray, test: IntArray, use_task: bool = True
) -> tuple[csr_matrix, csr_matrix]:
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True)
    texts = np.array(data.texts, dtype=object)
    x_train, x_test = tfidf.fit_transform(texts[train]), tfidf.transform(texts[test])
    if use_task:
        onehot = OneHotEncoder(handle_unknown="ignore")
        x_train = hstack([x_train, onehot.fit_transform(data.scenarios[train, None])])
        x_test = hstack([x_test, onehot.transform(data.scenarios[test, None])])
    return csr_matrix(x_train), csr_matrix(x_test)


class KNNRouter:
    """Nearest training prompts by TF-IDF cosine similarity, smoothed toward the task prior."""

    name = "kNN (TF-IDF)"
    uses_task = True

    def __init__(self, k: int = 25, prior_weight: float = 5.0) -> None:
        self.k = k
        self.prior_weight = prior_weight

    def predict(self, data: Dataset, train: IntArray, test: IntArray) -> FloatArray:
        prior = TaskRouter().predict(data, train, test)
        x_train, x_test = _features(data, train, test)
        out = np.zeros_like(prior)
        for start in range(0, len(test), 512):
            block = slice(start, start + 512)
            sim = (x_test[block] @ x_train.T).toarray()
            top = np.argsort(-sim, axis=1)[:, : self.k]
            hits = data.correct[train[top]].sum(axis=1)
            out[block] = (hits + self.prior_weight * prior[block]) / (self.k + self.prior_weight)
        return out


class LogisticRouter:
    """One logistic regression per model on TF-IDF features, plus the task label if allowed."""

    def __init__(self, c: float = 1.0, use_task: bool = True) -> None:
        self.c = c
        self.uses_task = use_task
        self.name = "Logistic (TF-IDF)" if use_task else "Logistic (text only)"

    def predict(self, data: Dataset, train: IntArray, test: IntArray) -> FloatArray:
        x_train, x_test = _features(data, train, test, self.uses_task)
        out = np.zeros((len(test), data.correct.shape[1]))
        for m in range(data.correct.shape[1]):
            y = data.correct[train, m]
            if y.min() == y.max():
                out[:, m] = y[0]
                continue
            clf = LogisticRegression(C=self.c, max_iter=2000)
            clf.fit(x_train, y)
            out[:, m] = clf.predict_proba(x_test)[:, 1]
        return out
