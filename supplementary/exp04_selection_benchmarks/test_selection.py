import numpy as np
import pytest

from epcrc.judge import JudgeResponses
from supplementary.exp04_selection_benchmarks.run import exact_search, fitted_losses, paired_bootstrap


def tiny_panel():
    # Two duplicate A judges and one B judge: keeping A and B is exact.
    block = np.tile(np.array([[1., 0, 0], [1., 0, 0], [0, 1., 0]])[None], (6, 1, 1))
    return JudgeResponses([block, block.copy()])


def test_exact_and_budget_are_not_confused():
    panel = tiny_panel()
    limited = exact_search(panel, 2, max_evaluations=1)
    assert not limited["complete"] and limited["optimum"] is None
    assert limited["status"] == "budget_exhausted"
    full = exact_search(panel, 2, max_evaluations=3)
    assert full["complete"] and full["optimum"] == pytest.approx(0)
    assert 2 in full["kept"]


def test_reconstruction_keeps_physical_endpoints_exact():
    panel = tiny_panel()
    losses, weights, agreement = fitted_losses(panel, panel, [0, 2])
    assert losses.shape == (2, 6, 3)
    assert np.max(losses) == pytest.approx(0)
    assert np.all(agreement)
    assert np.allclose(weights.sum(1), 1)


def test_paired_bootstrap_constant_difference_and_group_alignment():
    reference = np.zeros((2, 6, 3))
    candidate = reference + .12
    result = paired_bootstrap(candidate, reference, [0, 0, 1, 1, 2, 2], 50, 17, 4)
    assert result["difference_method_minus_backward"] == pytest.approx(.12)
    assert result["bonferroni_percentile_interval"] == pytest.approx([.12, .12])
    with pytest.raises(ValueError, match="align"):
        paired_bootstrap(candidate, reference, [0, 1], 50)
