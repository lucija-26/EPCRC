"""Tests for the E1 physical-to-virtual compression frontier (claim C2)."""

from __future__ import annotations

import numpy as np
import pytest

from epcrc.judge import JudgeResponses
from experiments.experiment_e1_compression_frontier import (
    E1_GAMMAS,
    Panel,
    evaluate_subset,
    select_backward,
    select_exhaustive,
    select_forward,
    select_one_per_family,
    select_random,
    tolerance_frontier,
)


def _panel(n_judges: int = 4, n_contexts: int = 2, seed: int = 0) -> Panel:
    """A small random panel with the same shape contract as the real one."""
    rng = np.random.default_rng(seed)
    judge_ids = [f"J{i:02d}" for i in range(n_judges)]
    model_ids = [f"org{i % 2}/model{i}" for i in range(n_judges)]
    contexts = [f"I{c}" for c in range(n_contexts)]

    splits = {}
    n_items = {}
    for name, n in (("FIT", 12), ("CERT", 7), ("TEST", 5)):
        blocks = [rng.dirichlet(np.ones(3), size=(n, n_judges))
                  for _ in range(n_contexts)]
        splits[name] = JudgeResponses(blocks, contexts)
        n_items[name] = n
    return Panel(judge_ids, model_ids, contexts, splits, n_items)


def test_full_panel_reconstructs_itself_exactly():
    """Every judge is in the physical panel, so the held-out error must vanish."""
    panel = _panel()
    row = evaluate_subset(panel, range(panel.N), "TEST")

    assert row["worst_judge_worst_context_tv"] == pytest.approx(0.0, abs=1e-12)
    assert row["verdict_agreement"] == pytest.approx(1.0)
    assert row["calls_avoided_frac"] == pytest.approx(0.0)


def test_kept_judges_are_reconstructed_by_themselves():
    panel = _panel()
    row = evaluate_subset(panel, [0, 2], "TEST")

    for judge_id in ("J00", "J02"):
        assert row["per_judge"][judge_id]["in_physical_panel"] is True
        assert row["per_judge"][judge_id]["worst_context_mean_tv"] == pytest.approx(0.0)
    assert row["per_judge"]["J01"]["in_physical_panel"] is False


def test_worst_judge_error_dominates_the_mean():
    panel = _panel()
    row = evaluate_subset(panel, [0], "TEST")

    assert row["worst_judge_worst_context_tv"] >= row["mean_judge_worst_context_tv"]
    assert row["p95_item_tv"] >= row["median_item_tv"]


def test_greedy_chains_are_nested_and_cover_every_budget():
    """A nested chain is what makes the frontier a single sequence of decisions."""
    panel = _panel()

    for chain in (select_backward(panel), select_forward(panel)):
        assert sorted(chain) == list(range(1, panel.N + 1))
        for k in range(1, panel.N):
            assert set(chain[k]).issubset(set(chain[k + 1]))


def test_exhaustive_is_at_least_as_good_as_greedy_on_the_fitting_split():
    """Enumeration minimises FIT error by construction, so greedy cannot beat it."""
    panel = _panel()
    backward = select_backward(panel)
    best = select_exhaustive(panel, max_n=panel.N)

    for k in range(1, panel.N + 1):
        greedy_err = evaluate_subset(panel, backward[k], "FIT")[
            "worst_judge_worst_context_tv"]
        exact_err = evaluate_subset(panel, best[k], "FIT")[
            "worst_judge_worst_context_tv"]
        assert exact_err <= greedy_err + 1e-9


def test_exhaustive_is_skipped_for_large_panels():
    assert select_exhaustive(_panel(), max_n=2) == {}


def test_one_per_family_spreads_across_families_first():
    panel = _panel(n_judges=4)
    chain = select_one_per_family(panel)

    families = {panel.model_ids[j].split("/")[0] for j in chain[2]}
    assert len(families) == 2


def test_random_selection_is_reproducible():
    panel = _panel()
    assert select_random(panel, 7) == select_random(panel, 7)
    assert select_random(panel, 7) != select_random(panel, 8)


def test_tolerance_frontier_reports_the_smallest_feasible_budget():
    panel = _panel()
    chain = select_backward(panel)
    rows = {k: evaluate_subset(panel, kept, "TEST") for k, kept in chain.items()}
    frontier = tolerance_frontier(panel, chain, rows)

    assert sorted(frontier) == sorted(str(g) for g in E1_GAMMAS)
    # The full panel has zero error, so every tolerance is met by some budget.
    for gamma in E1_GAMMAS:
        entry = frontier[str(gamma)]
        assert entry["min_k"] is not None
        assert rows[entry["min_k"]]["worst_judge_worst_context_tv"] <= gamma
        for k in range(1, entry["min_k"]):
            assert rows[k]["worst_judge_worst_context_tv"] > gamma


def test_tolerance_frontier_reports_none_when_no_budget_qualifies():
    panel = _panel()
    chain = {1: [0], 2: [0, 1]}
    rows = {1: {"worst_judge_worst_context_tv": 0.9},
            2: {"worst_judge_worst_context_tv": 0.8}}

    frontier = tolerance_frontier(panel, chain, rows)
    assert frontier["0.02"] == {"min_k": None, "kept": None}
