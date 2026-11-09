import numpy as np
import pytest

from supplementary.exp03_ecosystem_replication.run import validate_manifest, provenance_overlap, joint_gap_interval


def entry(name, models, relationship):
    return dict(name=name, model_ids=models, relationship=relationship, input="data.npz",
                assembly_protocol="Predeclared registry", benchmark="fixed benchmark")


def test_same_panel_cannot_be_rebranded_independent():
    original = entry("original", ["a", "b", "c"], "original_panel")
    for models in (["a", "b", "c"], ["a", "b"]):
        with pytest.raises(ValueError, match="subset"):
            validate_manifest([original, entry("new", models, "independent_ecosystem")])


def test_benchmark_shift_is_allowed_and_overlap_is_visible():
    entries = [entry("original", ["a", "b"], "original_panel"),
               entry("shift", ["a", "b"], "same_panel_benchmark_shift")]
    validate_manifest(entries)
    assert provenance_overlap(entries)[0]["same_model_set"]
    assert provenance_overlap(entries)[0]["jaccard"] == 1


def test_grouped_gap_bootstrap_uses_common_draws():
    losses = np.random.default_rng(0).uniform(0, .2, (2, 8, 2))
    result = joint_gap_interval(losses, losses + .1, np.repeat(np.arange(4), 2), 100, 2)
    assert result["gap"] == pytest.approx(.1)
    assert result["paired_cluster_percentile_interval_95"] == pytest.approx([.1, .1])
