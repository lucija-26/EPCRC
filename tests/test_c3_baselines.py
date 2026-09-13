"""Properties the C3 comparison depends on.

C3 is an equal-k, equal-cost head-to-head, so most of what can go wrong is not
a crash but a silently unfair comparison: a selector returning the wrong number
of judges, a floor that is not actually a floor, or a reconstruction rule that
is scored on data it was fitted on.  These tests pin exactly those things.
"""

from __future__ import annotations

import numpy as np
import pytest

from epcrc.judge import JudgeResponses
from epcrc.reconstruction import RECONSTRUCTORS, fit_weights, score_weights
from epcrc.selection_baselines import (
    GEOMETRY_SELECTORS,
    judge_matrix,
    lowrank_floor,
    pairwise_tv,
)


def _responses(n_items: int = 40, n_judges: int = 6, n_contexts: int = 3,
               seed: int = 0) -> JudgeResponses:
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(n_contexts):
        raw = rng.dirichlet(np.ones(3), size=(n_items, n_judges))
        blocks.append(raw)
    return JudgeResponses(blocks, [f"c{c}" for c in range(n_contexts)])


# --------------------------------------------------------------------------
# representations
# --------------------------------------------------------------------------

def test_judge_matrix_round_trips_a_block():
    r = _responses(n_items=5, n_judges=4, n_contexts=2)
    X = judge_matrix(r)
    assert X.shape == (4, 5 * 3 * 2)
    # first context, judge 2, item 3 must survive the flattening unchanged
    np.testing.assert_allclose(X[2, 3 * 3:4 * 3], r.blocks[0][3, 2, :])


def test_pairwise_tv_is_a_metric_shaped_matrix():
    d = pairwise_tv(_responses())
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)
    assert d.min() >= 0.0
    assert d.max() <= 1.0


# --------------------------------------------------------------------------
# selectors
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(GEOMETRY_SELECTORS))
def test_selector_returns_exactly_k_distinct_judges(name):
    """The failure that would corrupt the equal-k comparison."""
    r = _responses(n_judges=7)
    chain = GEOMETRY_SELECTORS[name](r)

    assert sorted(chain) == list(range(1, 8))
    for k, kept in chain.items():
        assert len(set(kept)) == k, f"{name} gave {len(set(kept))} judges at k={k}"
        assert set(kept) <= set(range(7))
        assert all(isinstance(j, int) for j in kept), "indices must be JSON-safe"


@pytest.mark.parametrize("name", sorted(GEOMETRY_SELECTORS))
def test_selector_is_deterministic(name):
    r = _responses(n_judges=6)
    assert GEOMETRY_SELECTORS[name](r) == GEOMETRY_SELECTORS[name](r)


def test_duplicate_judges_are_not_both_selected_early():
    """A cloned judge carries no new information, so geometry should skip it."""
    rng = np.random.default_rng(7)
    base = rng.dirichlet(np.ones(3), size=(30, 4))
    twinned = np.concatenate([base, base[:, :1, :]], axis=1)   # judge 4 == judge 0
    r = JudgeResponses([twinned], ["c0"])

    for name in ("farthest_first", "kmedoids", "hierarchical"):
        kept = GEOMETRY_SELECTORS[name](r)[2]
        assert not {0, 4} <= set(kept), f"{name} picked a judge and its clone at k=2"


# --------------------------------------------------------------------------
# the low-rank floor
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["pca", "nmf"])
def test_lowrank_floor_is_finite_and_improves_with_rank(kind):
    fit = _responses(seed=1)
    table = lowrank_floor(fit, fit, kind)

    values = [table[k]["worst_judge_worst_context_tv"] for k in sorted(table)]
    assert all(np.isfinite(v) for v in values)
    assert values[-1] <= values[0] + 1e-9


def test_pca_floor_is_exact_at_full_rank():
    """Rank n must reconstruct n judges perfectly; anything else means the
    basis is not spanning judge space and the floor is not a floor."""
    fit = _responses(n_judges=5, seed=2)
    table = lowrank_floor(fit, fit, "pca")
    assert table[5]["worst_judge_worst_context_tv"] < 1e-8


def test_pca_floor_transfers_to_a_held_out_split():
    """The basis lives in judge space, so it must apply to unseen items."""
    fit = _responses(n_items=40, n_judges=5, seed=3)
    held = _responses(n_items=17, n_judges=5, seed=4)
    table = lowrank_floor(fit, held, "pca")
    assert all(np.isfinite(table[k]["worst_judge_worst_context_tv"]) for k in table)


# --------------------------------------------------------------------------
# reconstruction rules
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rule", RECONSTRUCTORS)
def test_retained_judge_reconstructs_itself_exactly(rule):
    r = _responses(n_judges=5)
    kept = [0, 2, 4]
    w = fit_weights(rule, r, 2, kept)
    row = score_weights(r, 2, kept, w)
    assert row["worst_context_mean_tv"] < 1e-9
    assert row["verdict_agreement"] == pytest.approx(1.0)


@pytest.mark.parametrize("rule", RECONSTRUCTORS)
def test_weights_have_the_right_shape(rule):
    r = _responses(n_judges=5)
    kept = [1, 3]
    assert fit_weights(rule, r, 0, kept).shape == (2,)


@pytest.mark.parametrize("rule", ["simplex", "uniform", "nearest"])
def test_simplex_rules_stay_on_the_simplex(rule):
    r = _responses(n_judges=5)
    w = fit_weights(rule, r, 0, [1, 2, 3])
    assert w.min() >= -1e-9
    assert w.sum() == pytest.approx(1.0)
    assert score_weights(r, 0, [1, 2, 3], w)["off_simplex_frac"] == 0.0


def test_simplex_beats_uniform_on_its_own_objective():
    """The minimax LP must be at least as good as any fixed simplex point."""
    r = _responses(n_judges=6, seed=5)
    kept = [1, 2, 3, 4]
    best = score_weights(r, 0, kept, fit_weights("simplex", r, 0, kept))
    flat = score_weights(r, 0, kept, fit_weights("uniform", r, 0, kept))
    assert best["worst_context_mean_tv"] <= flat["worst_context_mean_tv"] + 1e-9


def test_unconstrained_rules_are_flagged_when_they_leave_the_simplex():
    """The whole point of reporting off_simplex_frac: ridge may not return a
    probability vector, and the table has to say so."""
    r = _responses(n_judges=6, seed=6)
    kept = [1, 2, 3]
    w = fit_weights("ridge", r, 0, kept)
    row = score_weights(r, 0, kept, w)
    if abs(w.sum() - 1.0) > 1e-6 or w.min() < -1e-6:
        assert row["off_simplex_frac"] > 0.0


def test_empty_physical_panel_is_rejected():
    r = _responses()
    with pytest.raises(ValueError):
        fit_weights("simplex", r, 0, [])


def test_unknown_rule_is_rejected():
    r = _responses()
    with pytest.raises(ValueError):
        fit_weights("not_a_rule", r, 0, [1, 2])
