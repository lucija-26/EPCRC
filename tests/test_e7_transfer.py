"""Tests for the E7 cross-benchmark transfer experiment (plan section 29)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from epcrc.judge import JudgeResponses
from epcrc.panel import Panel
from experiments.experiment_e7_transfer import (
    backward_chain,
    fit_basis_weights,
    jaccard,
    reconstruct,
    run,
    score,
)


def _panel(n_judges: int = 5, n_contexts: int = 2, seed: int = 0,
           sizes=(30, 12, 12)) -> Panel:
    """A small random panel with the same shape contract as the real one."""
    rng = np.random.default_rng(seed)
    judge_ids = [f"J{i:02d}" for i in range(n_judges)]
    model_ids = [f"org{i % 2}/model{i}" for i in range(n_judges)]
    contexts = [f"I{c}" for c in range(n_contexts)]

    splits, n_items, gold, domains = {}, {}, {}, {}
    for name, n in zip(("FIT", "CERT", "TEST"), sizes):
        splits[name] = JudgeResponses(
            [rng.dirichlet(np.ones(3), size=(n, n_judges)) for _ in range(n_contexts)],
            contexts,
        )
        n_items[name] = n
        gold[name] = rng.integers(0, 3, size=n)
        domains[name] = np.array(["d0" if i % 2 else "d1" for i in range(n)])
    return Panel(judge_ids, model_ids, contexts, splits, n_items,
                 gold=gold, domains=domains)


def _fake_scores(panel: Panel, directory: str) -> str:
    """Just enough of a cached block for the top-accuracy baseline to rank on."""
    os.makedirs(directory, exist_ok=True)
    for rank, judge_id in enumerate(panel.judge_ids):
        for context in panel.context_names:
            path = os.path.join(directory, f"{judge_id}__{context}.json")
            with open(path, "w") as handle:
                json.dump({"accuracy": {"three_class_accuracy": 0.9 - 0.05 * rank}},
                          handle)
    return directory


def test_a_full_basis_reconstructs_every_judge_exactly():
    panel = _panel()
    kept = list(range(panel.N))
    weights = fit_basis_weights(panel.splits["FIT"], kept, panel.N)
    row = score(panel, "TEST", kept, weights)

    assert row["worst_judge_worst_context_tv"] == pytest.approx(0.0, abs=1e-12)
    assert row["verdict_agreement"] == pytest.approx(1.0)


def test_judges_inside_the_basis_are_copied_not_fitted():
    """A basis judge must survive any benchmark shift at zero error."""
    home, away = _panel(seed=0), _panel(seed=9)
    kept = [1, 3]
    weights = fit_basis_weights(home.splits["FIT"], kept, home.N)

    assert np.allclose(weights[1], [1.0, 0.0])
    assert np.allclose(weights[3], [0.0, 1.0])

    recon = reconstruct(away.splits["TEST"], kept, weights, away.N)
    for block, rec in zip(away.splits["TEST"].blocks, recon):
        assert np.allclose(block[:, kept, :], rec[:, kept, :])


def test_refitting_on_the_transfer_benchmark_never_hurts_on_that_benchmark():
    """Setting 2 exists because frozen weights are fitted for the wrong data.

    Weights refitted on the away benchmark's own FIT are a minimax fit there, so
    the frozen ones cannot beat them on that same split.
    """
    home, away = _panel(seed=0), _panel(seed=9)
    kept = [0, 2, 4]

    frozen = fit_basis_weights(home.splits["FIT"], kept, home.N)
    refit = fit_basis_weights(away.splits["FIT"], kept, away.N)

    frozen_error = score(away, "FIT", kept, frozen)["worst_judge_worst_context_tv"]
    refit_error = score(away, "FIT", kept, refit)["worst_judge_worst_context_tv"]

    assert refit_error <= frozen_error + 1e-9


def test_the_backward_chain_is_nested():
    panel = _panel()
    chain = backward_chain(panel.splits["FIT"], panel.judge_ids)

    for k in range(2, panel.N + 1):
        assert set(chain[k - 1]) < set(chain[k])


def test_jaccard_is_one_only_for_the_same_basis():
    assert jaccard([1, 2, 3], [1, 2, 3]) == 1.0
    assert jaccard([1, 2], [3, 4]) == 0.0
    assert jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)


def test_mismatched_judge_panels_are_refused():
    home = _panel(n_judges=5)
    away = _panel(n_judges=4)

    with pytest.raises(RuntimeError, match="different judges"):
        run(home, away, "", budgets=[2], calibration_sizes=[5], seed=0)


def test_run_reports_all_three_settings_at_every_budget(tmp_path):
    home, away = _panel(seed=0), _panel(seed=9)
    scores = _fake_scores(home, str(tmp_path / "scores"))
    payload = run(home, away, scores, budgets=[2, 3], calibration_sizes=[5, 10], seed=0)

    assert payload["away_has_ties"] is False
    for rows in payload["methods"].values():
        for k in ("2", "3"):
            row = rows[k]
            assert len(row["kept"]) == int(k)
            assert set(row["refit_weights_TEST"]) == {"5", "10", "30"}
            assert 0.0 <= row["oracle_reselection"]["basis_overlap_jaccard"] <= 1.0
            # The gap is a difference, so it may point either way; what must
            # hold is that both sides were actually measured.
            assert row["transfer_gap"] == pytest.approx(
                row["frozen_weights_TEST"]["worst_judge_worst_context_tv"]
                - row["in_domain_TEST"]["worst_judge_worst_context_tv"]
            )
            assert "accuracy" in row["frozen_weights_TEST"]["downstream"]
