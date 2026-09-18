"""Tests for C5's aggregators and downstream metrics.

The blocks here are built by hand so the right answer is known without running
the code being tested.  Where a metric has a standard definition the test pins
the exact value; where it is a comparison the test pins the direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from epcrc.downstream import (
    accuracy,
    apply_logistic_aggregate,
    brier_score,
    disagreement_rate,
    evaluate_aggregate,
    expected_calibration_error,
    fit_logistic_aggregate,
    macro_accuracy,
    majority_aggregate,
    mean_aggregate,
    negative_log_likelihood,
    preference_score,
    ranking_metrics,
    top_k_overlap,
    verdict_agreement,
)


def _block(rows) -> np.ndarray:
    """(n_items, n_judges, 3) from a nested list."""
    return np.asarray(rows, dtype=float)


# --------------------------------------------------------------------------
# aggregators
# --------------------------------------------------------------------------

def test_a_one_judge_panel_aggregates_to_that_judge():
    block = _block([[[0.7, 0.2, 0.1]], [[0.1, 0.8, 0.1]]])
    assert mean_aggregate(block) == pytest.approx(block[:, 0, :])


def test_the_mean_can_be_weighted_and_the_weights_are_renormalised():
    block = _block([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    out = mean_aggregate(block, weights=np.array([3.0, 1.0]))
    assert out[0] == pytest.approx([0.75, 0.25, 0.0])


def test_majority_reports_vote_shares_not_confidence():
    """Two judges say A, one says B, but one of the two is barely sure."""
    block = _block([[[0.34, 0.33, 0.33], [0.9, 0.05, 0.05], [0.1, 0.8, 0.1]]])
    assert majority_aggregate(block)[0] == pytest.approx([2 / 3, 1 / 3, 0.0])


def test_majority_and_mean_disagree_when_one_judge_is_very_confident():
    block = _block([[[0.4, 0.35, 0.25], [0.4, 0.35, 0.25], [0.0, 1.0, 0.0]]])
    assert majority_aggregate(block).argmax(axis=1) == 0
    assert mean_aggregate(block).argmax(axis=1) == 1


def test_the_learned_aggregator_never_sees_the_split_it_is_scored_on():
    """Fit on one block, apply to another, and still recover a separable rule."""
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 3, size=200)

    def make(labels):
        block = np.full((len(labels), 2, 3), 0.1)
        block[np.arange(len(labels)), 0, labels] = 0.8
        block[np.arange(len(labels)), 1, labels] = 0.8
        return block / block.sum(axis=2, keepdims=True)

    model = fit_logistic_aggregate(make(gold), gold)

    held_out = rng.integers(0, 3, size=200)
    out = apply_logistic_aggregate(model, make(held_out))
    assert out.shape == (200, 3)
    assert out.sum(axis=1) == pytest.approx(1.0)
    assert accuracy(out, held_out) > 0.95


def test_a_label_missing_from_fit_still_gets_a_column():
    """Otherwise the aggregate would have two columns and the metrics would lie."""
    gold = np.array([0, 1, 0, 1] * 10)
    block = np.full((len(gold), 1, 3), 0.1)
    block[np.arange(len(gold)), 0, gold] = 0.8
    block = block / block.sum(axis=2, keepdims=True)

    model = fit_logistic_aggregate(block, gold)
    out = apply_logistic_aggregate(model, block)
    assert out.shape[1] == 3
    assert out[:, 2] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# item-level metrics
# --------------------------------------------------------------------------

def test_a_perfect_aggregate_scores_perfectly():
    gold = np.array([0, 1, 2])
    aggregate = np.eye(3)[gold]
    assert accuracy(aggregate, gold) == 1.0
    assert macro_accuracy(aggregate, gold) == 1.0
    assert negative_log_likelihood(aggregate, gold) == pytest.approx(0.0)
    assert brier_score(aggregate, gold) == pytest.approx(0.0)


def test_macro_accuracy_does_not_let_a_rare_class_be_drowned_out():
    """Ninety-nine A items and one tie item, and the tie is missed."""
    gold = np.array([0] * 99 + [2])
    aggregate = np.tile([1.0, 0.0, 0.0], (100, 1))
    assert accuracy(aggregate, gold) == pytest.approx(0.99)
    assert macro_accuracy(aggregate, gold) == pytest.approx(0.5)


def test_macro_accuracy_ignores_classes_that_never_occur():
    gold = np.array([0, 0, 1])
    aggregate = np.eye(3)[gold]
    assert macro_accuracy(aggregate, gold) == pytest.approx(1.0)


def test_a_confident_mistake_gives_a_large_but_finite_nll():
    gold = np.array([1])
    aggregate = np.array([[1.0, 0.0, 0.0]])
    value = negative_log_likelihood(aggregate, gold)
    assert np.isfinite(value)
    assert value > 20


def test_brier_matches_its_definition_on_a_uniform_aggregate():
    gold = np.array([0])
    aggregate = np.full((1, 3), 1 / 3)
    expected = (1 - 1 / 3) ** 2 + 2 * (1 / 3) ** 2
    assert brier_score(aggregate, gold) == pytest.approx(expected)


def test_a_perfectly_calibrated_aggregate_has_no_calibration_error():
    aggregate = np.tile([1.0, 0.0, 0.0], (100, 1))
    gold = np.zeros(100, dtype=int)
    assert expected_calibration_error(aggregate, gold) == pytest.approx(0.0)


def test_calibration_error_catches_confident_and_wrong():
    aggregate = np.tile([1.0, 0.0, 0.0], (100, 1))
    gold = np.ones(100, dtype=int)
    assert expected_calibration_error(aggregate, gold) == pytest.approx(1.0)


def test_a_panel_that_agrees_on_everything_has_no_disagreement():
    block = _block([[[0.8, 0.1, 0.1], [0.6, 0.3, 0.1]]])
    assert disagreement_rate(block) == 0.0


def test_disagreement_counts_items_not_judge_pairs():
    block = _block([
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
        [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.3, 0.1]],
    ])
    assert disagreement_rate(block) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# agreement with the full panel
# --------------------------------------------------------------------------

def test_an_aggregate_agrees_with_itself():
    rng = np.random.default_rng(1)
    raw = rng.random((40, 3))
    aggregate = raw / raw.sum(axis=1, keepdims=True)

    assert verdict_agreement(aggregate, aggregate) == 1.0
    metrics = ranking_metrics(aggregate, aggregate)
    assert metrics["kendall_tau"] == pytest.approx(1.0)
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["max_rank_displacement"] == 0.0
    assert metrics["top1_agreement"] == 1.0


def test_ranking_is_driven_by_the_a_minus_b_margin_not_by_ties():
    """Two items with the same margin but different tie mass rank equally."""
    aggregate = np.array([[0.5, 0.2, 0.3], [0.6, 0.3, 0.1]])
    assert preference_score(aggregate) == pytest.approx([0.3, 0.3])


def test_reversing_every_preference_reverses_the_ranking():
    reference = np.array([[0.7, 0.1, 0.2], [0.4, 0.4, 0.2], [0.1, 0.7, 0.2]])
    flipped = reference[:, [1, 0, 2]]
    assert ranking_metrics(reference, flipped)["kendall_tau"] == pytest.approx(-1.0)


def test_grouping_ranks_domains_rather_than_items():
    reference = np.array([[0.9, 0.0, 0.1], [0.1, 0.8, 0.1], [0.5, 0.4, 0.1]])
    groups = np.array([0, 0, 1])
    assert ranking_metrics(reference, reference, groups)["n_units"] == 2


def test_top_k_overlap_is_a_fraction_of_k():
    reference = np.array([[0.9, 0.0, 0.1], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1]])
    candidate = reference[[1, 0, 2]]
    assert top_k_overlap(reference, candidate, 2) == pytest.approx(1.0)
    assert top_k_overlap(reference, candidate, 1) == pytest.approx(0.0)


def test_agreement_metrics_are_omitted_when_there_is_no_reference():
    """The `full` arm is the reference and has nothing to be compared against."""
    gold = np.array([0, 1, 2])
    aggregate = np.eye(3)[gold]
    metrics = evaluate_aggregate(aggregate, gold)
    assert "verdict_agreement_vs_full" not in metrics
    assert set(metrics) == {"accuracy", "macro_accuracy", "nll", "brier", "ece"}


def test_domain_metrics_appear_only_when_domains_are_supplied():
    gold = np.array([0, 1, 2, 0])
    aggregate = np.eye(3)[gold]
    without = evaluate_aggregate(aggregate, gold, reference=aggregate)
    assert any(k.startswith("item_rank_") for k in without)
    assert not any(k.startswith("domain_rank_") for k in without)

    with_domains = evaluate_aggregate(
        aggregate, gold, reference=aggregate, groups=np.array([0, 0, 1, 1])
    )
    assert with_domains["domain_rank_n_units"] == 2
