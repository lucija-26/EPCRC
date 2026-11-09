"""Equal-budget selection remains valid with exactly redundant endpoints."""
import numpy as np
import pytest

from epcrc.judge import JudgeCoverageFunctional, JudgeResponses
from supplementary.common import select_subset


@pytest.mark.parametrize("method", ["kmedoids", "qr", "random", "backward", "forward", "swap1", "swap2"])
def test_duplicate_responses_preserve_requested_cardinality(method):
    block = np.broadcast_to(np.array([0.2, 0.3, 0.5]), (8, 5, 3)).copy()
    fit = JudgeResponses([block])
    for k in [1, 3, 4, 5]:
        kept = select_subset(fit, k, method, seed=7)
        assert len(kept) == len(set(kept)) == k
        assert kept == sorted(kept)
        assert all(0 <= j < 5 for j in kept)


def test_kmedoids_is_deterministic_and_not_forced_to_duplicates():
    # Exactly duplicated extreme points require two distinct representatives,
    # but extra equal-budget slots must still be filled by distinct endpoints.
    block = np.tile(np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0.]]), (6, 1, 1))
    fit = JudgeResponses([block])
    first = select_subset(fit, 2, 'kmedoids', seed=5)
    second = select_subset(fit, 2, 'kmedoids', seed=5)
    assert first == second
    assert len(set(first) & {0, 1}) == len(set(first) & {2, 3}) == 1
    assert JudgeCoverageFunctional(fit, fit).compute_coverage(set(first))[0] == 0


def test_full_budget_still_rejects_unknown_method():
    fit = JudgeResponses([np.ones((4, 3, 3)) / 3])
    with pytest.raises(ValueError):
        select_subset(fit, 3, 'unknown')
