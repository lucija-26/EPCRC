"""Tests for C4, multi-context robust selection and stress specialists.

The panel here is constructed, not random: one judge is *planted* to be exactly
redundant on the clean context and irreplaceable on the stress context.  That is
the situation claim C4 describes, so these tests check the mechanism rather than
merely checking that the code runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from epcrc.judge import JudgeResponses
from experiments.experiment_c4_stress_specialists import (
    ARMS,
    CLEAN,
    Panel,
    backward_chain,
    evaluate,
    find_specialists,
    loo_redundancy,
    only_contexts,
    per_context_loo,
    run_split_seed,
    summarise,
)

STRESS = "I1_stress"
SPECIALIST = "J04"


N_JUDGES = 5


def _planted_panel(seed: int = 0) -> Panel:
    """A panel whose last judge is redundant on clean and unique under stress.

    Clean: J00, J01, J02 are three independent anchors, and J03, J04 are exact
    midpoints of anchor pairs.  So ``{J00, J01, J02}`` reconstructs the whole
    panel on clean at *zero* error, and a clean-only selector sees both J03 and
    J04 as free to delete.

    Stress: J00..J03 are literally identical, hence perfect substitutes for each
    other, while J04 sits at the ``A`` corner that no convex combination of them
    can reach.  J04 is therefore the only judge whose deletion cost depends on
    being allowed to look past the clean context.
    """
    rng = np.random.default_rng(seed)
    judge_ids = [f"J{i:02d}" for i in range(N_JUDGES)]
    model_ids = [f"org{i % 2}/model{i}" for i in range(N_JUDGES)]
    contexts = [CLEAN, STRESS]

    splits, n_items = {}, {}
    for name, n in (("FIT", 14), ("CERT", 9), ("TEST", 11)):
        anchors = rng.dirichlet(np.ones(3), size=(n, 3))
        clean = np.stack([
            anchors[:, 0, :],
            anchors[:, 1, :],
            anchors[:, 2, :],
            0.5 * (anchors[:, 0, :] + anchors[:, 2, :]),
            0.5 * (anchors[:, 0, :] + anchors[:, 1, :]),
        ], axis=1)

        shared = rng.dirichlet([0.2, 4.0, 4.0], size=n)
        stress = np.repeat(shared[:, None, :], N_JUDGES, axis=1)
        stress[:, -1, :] = np.array([1.0, 0.0, 0.0])

        splits[name] = JudgeResponses([clean, stress], contexts)
        n_items[name] = n
    return Panel(judge_ids, model_ids, contexts, splits, n_items)


# --------------------------------------------------------------------------
# restricting the context axis
# --------------------------------------------------------------------------

def test_only_contexts_keeps_the_panel_order():
    panel = _planted_panel()
    clean = only_contexts(panel.splits["FIT"], [CLEAN])
    assert clean.context_names == [CLEAN]
    assert clean.n_contexts == 1
    assert clean.n_judges == panel.N
    np.testing.assert_array_equal(clean.blocks[0], panel.splits["FIT"].blocks[0])


def test_only_contexts_preserves_declaration_order_not_argument_order():
    panel = _planted_panel()
    both = only_contexts(panel.splits["FIT"], [STRESS, CLEAN])
    assert both.context_names == [CLEAN, STRESS]


def test_only_contexts_rejects_an_unknown_context():
    panel = _planted_panel()
    with pytest.raises(ValueError, match="unknown contexts"):
        only_contexts(panel.splits["FIT"], ["I9_does_not_exist"])


# --------------------------------------------------------------------------
# the specialist definition
# --------------------------------------------------------------------------

def test_the_planted_judge_is_redundant_on_clean_but_not_under_stress():
    panel = _planted_panel()
    clean = loo_redundancy(panel, [CLEAN])
    stress = loo_redundancy(panel, [STRESS])

    assert clean[SPECIALIST]["loo_worst_context_tv"] == pytest.approx(0.0, abs=1e-9)
    assert stress[SPECIALIST]["loo_worst_context_tv"] > 0.3
    assert stress[SPECIALIST]["binding_context"] == STRESS


def _found(panel: Panel) -> dict:
    return find_specialists(
        per_context_loo(panel), loo_redundancy(panel, panel.context_names)
    )


def test_only_the_planted_judge_is_flagged_a_specialist():
    found = _found(_planted_panel())
    flagged = [j for j, v in found.items() if v["is_specialist"]]
    assert flagged == [SPECIALIST]
    assert found[SPECIALIST]["binding_context"] == STRESS
    assert found[SPECIALIST]["stress_gap"] > 0.3


def test_the_anchors_are_not_specialists_despite_a_high_joint_loo():
    """The regression this definition exists to prevent.

    An anchor is expensive to delete under the *joint* fit, because clean and
    stress disagree about how to replace it.  That is contexts conflicting, not
    a judge being distinctive under perturbation, and C4 must not claim it.
    """
    found = _found(_planted_panel())
    for judge in ("J00", "J01", "J02"):
        assert found[judge]["loo_joint_worst_context_tv"] > 0.05
        assert found[judge]["loo_worst_stress_tv"] == pytest.approx(0.0, abs=1e-9)
        assert not found[judge]["is_specialist"]


def test_the_stress_gap_is_a_gap_not_a_level():
    """A judge irreplaceable on clean too is not what C4 is about."""
    found = _found(_planted_panel())
    for entry in found.values():
        expected = entry["loo_worst_stress_tv"] - entry["loo_clean_tv"]
        assert entry["stress_gap"] == pytest.approx(expected)


def test_per_context_loo_covers_every_judge_and_context():
    panel = _planted_panel()
    table = per_context_loo(panel)
    assert set(table) == set(panel.judge_ids)
    for row in table.values():
        assert set(row) == set(panel.context_names)
    # J00..J03 are identical under stress, so each substitutes for the others.
    for judge in ("J00", "J01", "J02", "J03"):
        assert table[judge][STRESS] == pytest.approx(0.0, abs=1e-9)
    # J03 and J04 are exact convex combinations of the anchors on clean.
    for judge in ("J03", "J04"):
        assert table[judge][CLEAN] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# selection depends on which contexts the selector may see
# --------------------------------------------------------------------------

def test_clean_only_selection_retires_the_specialist():
    """On clean the three anchors suffice, so the specialist is spent budget."""
    panel = _planted_panel()
    chain = backward_chain(panel, [CLEAN])
    target = panel.judge_ids.index(SPECIALIST)
    assert chain[3] == [0, 1, 2]
    assert target not in chain[3]


def test_robust_selection_keeps_the_specialist():
    panel = _planted_panel()
    chain = backward_chain(panel, panel.context_names)
    target = panel.judge_ids.index(SPECIALIST)
    for k in range(2, panel.N + 1):
        assert target in chain[k], f"specialist dropped at budget {k}"


def test_both_chains_are_nested():
    panel = _planted_panel()
    for contexts in ([CLEAN], panel.context_names):
        chain = backward_chain(panel, contexts)
        for k in range(2, panel.N + 1):
            assert set(chain[k - 1]) < set(chain[k])


# --------------------------------------------------------------------------
# the claim itself
# --------------------------------------------------------------------------

def _arms(panel: Panel, k: int) -> dict:
    found = _found(panel)
    chains = {
        "all": backward_chain(panel, panel.context_names),
        "clean": backward_chain(panel, [CLEAN]),
    }
    contexts_for = {"all": list(panel.context_names), "clean": [CLEAN]}
    return {
        arm: evaluate(panel, chains[sel][k], contexts_for[fit], found)
        for arm, (sel, fit) in ARMS.items()
    }


def test_robust_selection_lowers_worst_context_error():
    """The headline C4 prediction, on a panel built to exhibit it."""
    rows = _arms(_planted_panel(), k=3)
    assert (rows["robust"]["worst_judge_worst_context_tv"]
            < rows["clean_select"]["worst_judge_worst_context_tv"])
    assert (rows["robust"]["worst_judge_worst_context_tv"]
            < rows["clean_pipeline"]["worst_judge_worst_context_tv"])


def test_robust_selection_retains_the_specialist_the_baselines_drop():
    rows = _arms(_planted_panel(), k=3)
    assert rows["robust"]["specialists_kept"] == [SPECIALIST]
    assert rows["clean_select"]["specialists_kept"] == []
    assert rows["clean_pipeline"]["specialists_kept"] == []
    for row in rows.values():
        assert row["n_specialists_total"] == 1


def test_the_clean_only_failure_is_invisible_on_clean_items():
    """Why C4 matters: the clean numbers do not warn you about the stress loss."""
    rows = _arms(_planted_panel(), k=3)
    baseline = rows["clean_select"]
    # The three anchors reconstruct every judge exactly on clean items, so the
    # clean report is a flawless bill of health.
    assert baseline["worst_judge_clean_tv"] == pytest.approx(0.0, abs=1e-9)
    assert baseline["per_context_worst_judge_tv"][CLEAN] == pytest.approx(0.0, abs=1e-9)
    # And yet the panel it chose is blind under stress.
    assert baseline["per_context_worst_judge_tv"][STRESS] > 0.3
    assert baseline["worst_judge_worst_context_tv"] > 0.3


def test_every_arm_is_scored_on_every_context():
    panel = _planted_panel()
    rows = _arms(panel, k=3)
    for row in rows.values():
        assert set(row["per_context_worst_judge_tv"]) == set(panel.context_names)


def test_the_full_panel_reconstructs_itself_exactly_in_every_arm():
    panel = _planted_panel()
    for row in _arms(panel, k=panel.N).values():
        assert row["worst_judge_worst_context_tv"] == pytest.approx(0.0, abs=1e-9)
        assert row["verdict_agreement"] == pytest.approx(1.0)
        assert row["kept"] == panel.judge_ids


def test_evaluate_needs_the_clean_control_context():
    panel = _planted_panel()
    stripped = Panel(
        panel.judge_ids,
        panel.model_ids,
        [STRESS],
        {name: only_contexts(split, [STRESS])
         for name, split in panel.splits.items()},
        panel.n_items,
    )
    with pytest.raises(ValueError):
        evaluate(stripped, [0, 1, 2], [STRESS], {})


# --------------------------------------------------------------------------
# across split seeds
# --------------------------------------------------------------------------

def test_run_split_seed_reports_every_arm_and_budget():
    panel = _planted_panel()
    out = run_split_seed(panel, budgets=[2, 3])
    assert set(out["arms"]) == set(ARMS)
    for arm, block in out["arms"].items():
        assert set(block["budgets"]) == {"2", "3"}
    assert out["arms"]["clean_pipeline"]["fits_on"] == [CLEAN]
    assert out["arms"]["robust"]["fits_on"] == list(panel.context_names)


def test_summary_reports_mean_sd_min_and_max_across_seeds():
    """Plan section 34.4 asks for all four, and the sign convention must hold."""
    per_seed = {
        str(s): run_split_seed(_planted_panel(seed=s), budgets=[3])
        for s in (0, 1, 2)
    }
    summary = summarise(per_seed, [3])

    for arm in ("clean_select", "clean_pipeline"):
        row = summary[f"robust_vs_{arm}"]["3"]
        assert {"delta_worst_context_mean", "delta_worst_context_sd",
                "delta_worst_context_min", "delta_worst_context_max"} <= set(row)
        assert row["n_seeds"] == 3
        # Positive delta means the robust selector won.
        assert row["delta_worst_context_mean"] > 0
        assert row["n_seeds_robust_better"] == 3
        assert row["delta_worst_context_min"] <= row["delta_worst_context_mean"]
        assert row["delta_worst_context_mean"] <= row["delta_worst_context_max"]
        assert row["specialists_retained_advantage_mean"] == pytest.approx(1.0)


def test_specialist_stability_lists_only_judges_flagged_in_every_seed():
    per_seed = {
        str(s): run_split_seed(_planted_panel(seed=s), budgets=[3])
        for s in (0, 1)
    }
    stability = summarise(per_seed, [3])["specialist_stability"]
    assert stability["specialists_in_every_seed"] == [SPECIALIST]
    assert stability["binding_contexts"][SPECIALIST] == [STRESS]
    assert stability["count_per_seed"] == {"0": 1, "1": 1}
