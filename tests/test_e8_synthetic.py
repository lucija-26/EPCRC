"""Tests for E8, the synthetic structure sweep.

E8 exists to check the pipeline against a known answer, so its own tests have
to check the *generator* against the properties its docstring claims.  If the
generator does not really plant a unique minimum backbone, then E8 recovering
that backbone would prove nothing about the selection code.

The assertions below are therefore the provable facts of the construction --
exact convex reconstruction, archetypes outside each other's hull, duplicates
that reconstruct each other -- not thresholds tuned to whatever the code
currently returns.
"""

from __future__ import annotations

import numpy as np
import pytest

from epcrc.judge import JudgeCoverageFunctional
from epcrc.synthetic_judges import archetype_panel
from experiments.experiment_e8_synthetic import (
    backward_to_size,
    calibration,
    composition,
    exhaustive_best,
    jaccard,
    recovery,
)


def _cov(panel) -> JudgeCoverageFunctional:
    return JudgeCoverageFunctional(
        panel.splits["FIT"], panel.splits["CERT"], panel.names
    )


# --------------------------------------------------------------------------
# the construction itself
# --------------------------------------------------------------------------

def test_blocks_are_valid_distributions():
    panel = archetype_panel(n_items=40, noise=0.1, seed=0)
    for split in panel.splits.values():
        for block in split.blocks:
            assert block.min() >= 0.0
            assert np.allclose(block.sum(axis=2), 1.0)


def test_noiseless_backbone_reconstructs_the_panel_exactly():
    """Every non-archetype is a convex combination of the archetypes, so the
    planted set covers the panel at numerical zero."""
    panel = archetype_panel(n_judges=10, n_archetypes=4, n_items=120,
                            noise=0.0, seed=0)
    error, _ = _cov(panel).compute_coverage(set(panel.backbone))
    assert error < 1e-9


def test_every_archetype_is_necessary():
    """An archetype lies outside the convex hull of the others, so dropping one
    cannot be repaired -- this is what makes the backbone the unique minimum."""
    panel = archetype_panel(n_judges=10, n_archetypes=4, n_items=120,
                            noise=0.0, seed=0)
    cov = _cov(panel)
    for drop in panel.backbone:
        error, _ = cov.compute_coverage(set(panel.backbone) - {drop})
        assert error > 0.05


def test_no_other_set_of_the_same_size_does_as_well():
    panel = archetype_panel(n_judges=9, n_archetypes=3, n_items=120,
                            noise=0.0, n_duplicates=0, seed=3)
    cov = _cov(panel)
    truth, _ = cov.compute_coverage(set(panel.backbone))
    best, best_error = exhaustive_best(cov, len(panel.backbone))
    assert best == list(panel.backbone)
    assert best_error <= truth + 1e-12


def test_duplicates_reconstruct_each_other():
    """The planted cycles: a copy and its source are exact substitutes."""
    panel = archetype_panel(n_judges=10, n_archetypes=3, n_items=80,
                            noise=0.0, n_duplicates=2, seed=1)
    assert panel.duplicate_of
    for target, source in panel.duplicate_of.items():
        assert np.allclose(panel.mixing[target], panel.mixing[source])
        for split in panel.splits.values():
            for block in split.blocks:
                assert np.allclose(block[:, target, :], block[:, source, :])


def test_fully_correlated_noise_preserves_the_backbone():
    """Shared noise is one affine map applied to every judge, so it moves the
    panel without breaking any convex combination."""
    panel = archetype_panel(n_judges=10, n_archetypes=4, n_items=120,
                            noise=0.4, noise_correlation=1.0, seed=0)
    error, _ = _cov(panel).compute_coverage(set(panel.backbone))
    assert error < 1e-9


def test_private_noise_does_break_reconstruction():
    """The contrast that makes the previous test meaningful."""
    panel = archetype_panel(n_judges=10, n_archetypes=4, n_items=120,
                            noise=0.4, noise_correlation=0.0, seed=0)
    error, _ = _cov(panel).compute_coverage(set(panel.backbone))
    assert error > 0.01


def test_mixing_rows_are_convex_weights():
    panel = archetype_panel(n_judges=12, n_archetypes=5, n_items=40, seed=2)
    assert np.allclose(panel.mixing.sum(axis=1), 1.0)
    assert panel.mixing.min() >= 0.0
    for slot, judge in enumerate(panel.backbone):
        assert panel.mixing[judge, slot] == pytest.approx(1.0)


def test_backbone_placement_is_not_the_leading_indices():
    """Recovery would be trivially passable if archetypes always sat first."""
    placements = {
        tuple(archetype_panel(n_judges=12, n_archetypes=4, n_items=20,
                              seed=seed).backbone)
        for seed in range(8)
    }
    assert len(placements) > 1
    assert (0, 1, 2, 3) not in placements or len(placements) > 2


def test_rejects_impossible_configurations():
    with pytest.raises(ValueError):
        archetype_panel(n_judges=3, n_archetypes=5)
    with pytest.raises(ValueError):
        archetype_panel(n_archetypes=1)
    with pytest.raises(ValueError):
        archetype_panel(noise=1.5)
    with pytest.raises(ValueError):
        archetype_panel(n_judges=6, n_archetypes=4, n_duplicates=3)


# --------------------------------------------------------------------------
# the experiment's own helpers
# --------------------------------------------------------------------------

def test_backward_to_size_returns_that_size():
    panel = archetype_panel(n_judges=9, n_archetypes=3, n_items=60, seed=0)
    kept, _ = backward_to_size(_cov(panel), 4)
    assert len(kept) == 4


def test_greedy_never_beats_exhaustive():
    """Exhaustive is the minimum over all size-k panels, so the gap is >= 0."""
    panel = archetype_panel(n_judges=9, n_archetypes=3, n_items=60,
                            noise=0.05, seed=4)
    cov = _cov(panel)
    _, greedy_error = backward_to_size(cov, 4)
    _, exact_error = exhaustive_best(cov, 4)
    assert greedy_error >= exact_error - 1e-12


def test_recovery_reports_a_nonnegative_gap():
    panel = archetype_panel(n_judges=9, n_archetypes=3, n_items=60, seed=0)
    out = recovery(panel, _cov(panel))
    assert out["greedy_gap"] >= -1e-12
    assert 0.0 <= out["greedy_jaccard"] <= 1.0


def test_jaccard_edges():
    assert jaccard([1, 2], [1, 2]) == 1.0
    assert jaccard([1, 2], [3, 4]) == 0.0
    assert jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)


def test_composition_detects_the_planted_cycle():
    panel = archetype_panel(n_judges=10, n_archetypes=3, n_items=80,
                            noise=0.0, n_duplicates=2, seed=1)
    rows = composition(panel, _cov(panel), [0.01, 0.2])
    assert any(row["n_cycles"] >= 1 for row in rows)
    # A tolerance loose enough to delete an archetype must break the panel.
    assert rows[-1]["naive_violates_gamma"]


def test_calibration_uses_a_held_out_split():
    """TEST must be a different draw from CERT, or the check is circular."""
    panel = archetype_panel(n_judges=9, n_archetypes=3, n_items=80, seed=0)
    assert not np.allclose(
        panel.splits["CERT"].blocks[0], panel.splits["TEST"].blocks[0]
    )
    rows = calibration(panel, [0.02, 0.08])
    for row in rows:
        assert row["cert_within_gamma"]
        assert row["selected_size"] >= len(panel.backbone)
