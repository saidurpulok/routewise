import numpy as np

from routewise.config import Model
from routewise.evaluate import (
    RoutingData,
    cost_to_reach,
    folds,
    hull_accuracy,
    oracle_point,
    pareto,
    route_cv,
    summarize,
)
from routewise.routers import KNNRouter, LogisticRouter, TaskRouter, expected_cost, select


def synthetic(n: int = 300, seed: int = 0) -> RoutingData:
    """Two tasks; the cheap model is good on 'easy' and bad on 'hard', the big one good on both."""
    rng = np.random.default_rng(seed)
    scenarios = np.array(["easy"] * (n // 2) + ["hard"] * (n - n // 2))
    texts = [f"{s} question number {i % 7} about topic {i % 5}" for i, s in enumerate(scenarios)]
    easy = scenarios == "easy"
    correct = np.stack(
        [
            np.where(easy, rng.random(n) < 0.95, rng.random(n) < 0.2),
            rng.random(n) < 0.9,
        ],
        axis=1,
    ).astype(np.float64)
    prompt_tokens = np.full((n, 2), 100.0)
    output_tokens = np.full((n, 2), 10.0)
    input_price = np.array([0.1, 10.0]) / 1e6
    output_price = np.array([0.4, 40.0]) / 1e6
    return RoutingData(
        uids=[str(i) for i in range(n)],
        texts=texts,
        scenarios=scenarios,
        correct=correct,
        cost=prompt_tokens * input_price + output_tokens * output_price,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        input_price=input_price,
        output_price=output_price,
        models=[Model("small", "Small", 0.1, 0.4), Model("big", "Big", 10.0, 40.0)],
    )


def test_select_prefers_cheaper_model_on_ties() -> None:
    p = np.array([[0.8, 0.8], [0.2, 0.9]])
    cost = np.array([[1.0, 2.0], [1.0, 2.0]])
    assert select(p, cost, 0.0).tolist() == [0, 1]
    assert select(p, cost, 10.0).tolist() == [0, 0]


def test_pareto_drops_dominated_points() -> None:
    pts = np.array([[0.5, 1.0], [0.4, 2.0], [0.7, 3.0], [0.7, 4.0]])
    assert pareto(pts).tolist() == [[0.5, 1.0], [0.7, 3.0]]


def test_hull_interpolates_between_models() -> None:
    singles = np.array([[0.5, 1.0], [0.9, 3.0]])
    assert np.isclose(hull_accuracy(singles, 2.0), 0.7)
    assert hull_accuracy(singles, 0.5) == 0.5
    assert hull_accuracy(singles, 10.0) == 0.9


def test_cost_to_reach() -> None:
    curve = np.array([[0.5, 1.0], [0.8, 2.0], [0.9, 5.0]])
    assert cost_to_reach(curve, 0.8) == 2.0
    assert cost_to_reach(curve, 0.95) == float("inf")


def test_folds_are_a_stratified_partition() -> None:
    scenarios = np.array(["a"] * 50 + ["b"] * 30)
    parts = folds(scenarios, 5, seed=0)
    joined = np.sort(np.concatenate(parts))
    assert joined.tolist() == list(range(80))
    for part in parts:
        assert (scenarios[part] == "a").sum() == 10
        assert (scenarios[part] == "b").sum() == 6


def test_expected_cost_uses_only_training_output_lengths() -> None:
    data = synthetic()
    train, test = np.arange(0, 300, 2), np.arange(1, 300, 2)
    data.output_tokens[test] = 1e6  # unseen at routing time
    c = expected_cost(data, train, test)
    assert np.allclose(c, data.cost[train][: len(test)])


def test_routers_return_probabilities() -> None:
    data = synthetic()
    train, test = np.arange(0, 300, 2), np.arange(1, 300, 2)
    for router in (TaskRouter(), KNNRouter(k=5), LogisticRouter()):
        p = router.predict(data, train, test)
        assert p.shape == (len(test), 2)
        assert np.all((p >= 0) & (p <= 1))


def test_router_learns_the_task_split() -> None:
    data = synthetic()
    choices = route_cv(data, TaskRouter(), k=5, seed=0)
    # A moderate cost penalty: cheap model for easy prompts, big model for hard ones.
    mid = choices[len(choices) // 3]
    easy = data.scenarios == "easy"
    assert (mid[easy] == 0).mean() > 0.9
    assert (mid[~easy] == 1).mean() > 0.9


def test_oracle_and_summary() -> None:
    data = synthetic()
    acc, cost = oracle_point(data, np.arange(len(data.uids)))
    assert acc >= data.correct.mean(axis=0).max()
    assert cost <= data.cost[:, 1].mean()
    summary = summarize(data, {"Task-aware": route_cv(data, TaskRouter())}, n_boot=20)
    router = summary["routers"]["Task-aware"]
    assert summary["best_single"]["name"] == "Big"
    assert router["saving_vs_best"] > 0.3
    assert router["max_gain_over_single_model_hull"] > 0
