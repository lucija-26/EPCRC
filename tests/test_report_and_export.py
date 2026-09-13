"""Tests for the reporting and packaging layer.

These are the numbers the professor reads, so what is guarded here is not the
maths -- that is tested elsewhere -- but the reporting contract:

  * a table never silently invents or drops a method or a budget;
  * a headline never quietly averages over the trivial budgets where no method
    can differ;
  * the package that gets zipped matches the tables it claims to contain.
"""

from __future__ import annotations

import json
import os
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc import report as R
from experiments import export_results as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

C3 = os.path.join(RESULTS, "c3_baselines.json")
E1 = os.path.join(RESULTS, "e1_frontier.json")
E0_REAL = os.path.join(RESULTS, "e0_real_core8.json")
E0_SYNTH = os.path.join(RESULTS, "e0_noncomposability.json")

needs_c3 = pytest.mark.skipif(not os.path.exists(C3), reason="C3 not run")
needs_e1 = pytest.mark.skipif(not os.path.exists(E1), reason="E1 not run")
needs_e0 = pytest.mark.skipif(not os.path.exists(E0_REAL), reason="E0 real not run")


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def test_fmt_ci_puts_the_interval_after_the_estimate():
    assert R.fmt_ci(0.5, 0.4, 0.6) == "0.500 [0.400, 0.600]"


def test_random_draws_collapse_to_one_baseline():
    assert R._family("random_17") == "random"
    assert R._family("random") == "random"
    assert R._family("kmedoids") == "kmedoids"


def test_every_reported_method_has_a_display_name():
    for method in R.COVERAGE_METHODS + R.BASELINE_ORDER:
        assert method in R.PRETTY, f"{method} would print as a raw key"


# --------------------------------------------------------------------------
# C1
# --------------------------------------------------------------------------

@needs_e0
def test_c1_marks_which_gammas_were_predeclared():
    df = R.c1_table(E0_REAL)
    assert set(df["gamma_source"]) <= {"declared", "loo_breakpoint"}
    assert (df["gamma_source"] == "declared").any()


@needs_e0
def test_c1_declared_grid_is_exactly_the_plan_grid():
    declared = R.c1_headline(E0_REAL, gamma_source="declared")
    payload = R.load(E0_REAL)
    assert sorted(declared["gamma"].unique()) == sorted(payload["gammas"])


@needs_e0
def test_c1_breakpoints_are_where_the_removable_set_changes():
    """A breakpoint must admit at least one removable judge.

    That is the whole reason the breakpoints exist: the predeclared grid can
    leave R empty everywhere, which says nothing about composition.
    """
    df = R.c1_table(E0_REAL)
    breaks = df[df["gamma_source"] == "loo_breakpoint"]
    assert len(breaks)
    assert (breaks["n_individually_removable"] >= 1).all()


@needs_e0
def test_c1_violation_is_consistent_with_the_reported_coverage():
    df = R.c1_table(E0_REAL)
    recomputed = df["naive_coverage"] > df["gamma"]
    assert (recomputed == df["naive_violates_gamma"]).all()


@needs_e0
def test_c1_pooling_both_grids_returns_more_rows_than_either():
    both = R.c1_headline(E0_REAL, gamma_source=None)
    declared = R.c1_headline(E0_REAL, gamma_source="declared")
    assert len(both) > len(declared)


# --------------------------------------------------------------------------
# C2
# --------------------------------------------------------------------------

@needs_e1
def test_c2_covers_every_budget_for_the_method_under_test():
    df = R.c2_table(E1, "TEST")
    cov = df[df["method"] == "coverage_backward"]
    assert sorted(cov["k"]) == list(range(1, int(cov["k"].max()) + 1))


@needs_e1
def test_c2_full_panel_reconstructs_itself_exactly():
    """At k = N every judge is in the panel, so the error must be zero."""
    df = R.c2_table(E1, "TEST")
    full = df[df["k"] == df["k"].max()]
    assert np.allclose(full["worst_judge_tv"], 0.0, atol=1e-9)
    assert np.allclose(full["verdict_agreement"], 1.0, atol=1e-9)


@needs_e1
def test_c2_calls_avoided_agrees_with_the_budget():
    df = R.c2_table(E1, "TEST")
    n = int(df["k"].max())
    assert np.allclose(df["calls_avoided_frac"], 1.0 - df["k"] / n)


@needs_e1
def test_c2_total_variation_stays_in_range():
    df = R.c2_table(E1, "TEST")
    for col in ("worst_judge_tv", "mean_judge_tv", "median_item_tv", "p95_item_tv"):
        assert df[col].between(0.0, 1.0).all(), col
    assert df["verdict_agreement"].between(0.0, 1.0).all()


@needs_e1
def test_c2_worst_judge_is_never_below_the_mean_judge():
    df = R.c2_table(E1, "TEST")
    assert (df["worst_judge_tv"] >= df["mean_judge_tv"] - 1e-12).all()


# --------------------------------------------------------------------------
# C3
# --------------------------------------------------------------------------

@needs_c3
def test_c3_headline_excludes_the_trivial_budgets():
    """k=1 admits no convex combination and k=N is exact for every method.

    Averaging either in would dilute the comparison with budgets at which no
    method can possibly differ.
    """
    df = R.c3_table(C3)
    head = R.c3_headline(C3)
    sizes = sorted(int(k) for k in df["k"].unique())
    assert head["k_values"].iloc[0] == [k for k in sizes if 1 < k < max(sizes)]


@needs_c3
def test_c3_headline_k_values_are_plain_ints():
    """numpy scalars render as `np.int64(2)` when interpolated into prose."""
    for k in R.c3_headline(C3)["k_values"].iloc[0]:
        assert type(k) is int


@needs_c3
def test_c3_headline_has_one_row_per_method():
    head = R.c3_headline(C3)
    assert head["method"].is_unique
    assert not head["method"].str.startswith("random_").any()


@needs_c3
def test_c3_marks_exactly_the_lowest_error_as_best():
    head = R.c3_headline(C3)
    assert head["best"].sum() >= 1
    assert head.loc[head["best"], "worst_judge_tv"].max() == head["worst_judge_tv"].min()


@needs_c3
def test_c3_bootstrap_interval_brackets_the_estimate():
    head = R.c3_headline(C3)
    assert (head["boot_lo"] <= head["worst_judge_tv"] + 1e-9).all()
    assert (head["boot_hi"] >= head["worst_judge_tv"] - 1e-9).all()


@needs_c3
def test_c3_averages_over_every_split_seed():
    payload = R.load(C3)
    df = R.c3_table(C3)
    cov = df[(df["method"] == "coverage_backward")]
    assert (cov["n_observations"] == len(payload["split_seeds"])).all()


@needs_c3
def test_c3_random_baseline_pools_all_draws():
    payload = R.load(C3)
    df = R.c3_table(C3)
    rnd = df[df["method"] == "random"]
    expected = payload["n_random_draws"] * len(payload["split_seeds"])
    assert (rnd["n_observations"] == expected).all()


# How far the FIT-selected optimum may fall behind the best method on TEST
# before the selection objective should be considered not to transfer.  This is
# a test tolerance, not a claim: ranks are meaningless at budgets where several
# methods pick the same subset, so the magnitude of the loss is what matters.
MAX_TRANSFER_DEFICIT = 0.05


@needs_c3
def test_c3_exhaustive_selection_transfers_to_held_out_items():
    """The enumerated optimum is optimal on FIT, and only scored on TEST.

    It can therefore lose on TEST, and at budgets close to the full panel it
    routinely does, because only a handful of subsets exist and the ordering
    between them is noise.  What would be damaging is losing by a *lot*: that
    would mean choosing the panel by coverage on FIT tells you little about
    held-out behaviour.
    """
    df = R.c3_table(C3)
    if "exhaustive" not in set(df["method"]):
        pytest.skip("panel too large for enumeration")

    non_trivial = R.c3_headline(C3)["k_values"].iloc[0]
    for k, group in df[df["k"].isin(non_trivial)].groupby("k"):
        ex = group[group["method"] == "exhaustive"]["worst_judge_tv"]
        if not len(ex):
            continue
        deficit = ex.iloc[0] - group["worst_judge_tv"].min()
        assert deficit <= MAX_TRANSFER_DEFICIT, (
            f"at k={k} the FIT-optimal subset scored {deficit:.4f} TV worse on "
            f"TEST than the best method, so selection is not transferring"
        )


# -- section 34.2, the paired comparison ------------------------------------

@needs_c3
def test_paired_table_covers_the_same_budgets_as_the_headline():
    """The paired test is only meaningful where methods can differ."""
    paired = R.c3_paired_table(C3)
    assert len(paired)
    assert sorted(paired["k"].unique()) == R.c3_headline(C3)["k_values"].iloc[0]


@needs_c3
def test_paired_table_never_compares_the_reference_with_itself():
    paired = R.c3_paired_table(C3)
    assert "coverage_backward" not in set(paired["method"])


@needs_c3
def test_paired_interval_brackets_the_point_estimate():
    paired = R.c3_paired_table(C3)
    assert (paired["lo"] <= paired["delta"] + 1e-9).all()
    assert (paired["hi"] >= paired["delta"] - 1e-9).all()


@needs_c3
def test_paired_outcome_labels_a_zero_difference_as_a_tie():
    """Two methods picking the identical subset is a tie, not a loss."""
    paired = R.c3_paired_table(C3)
    zero = paired[np.isclose(paired["delta"], 0.0)]
    assert (zero["outcome"] == "tie").all()
    assert set(paired["outcome"]) <= {"tie", "reference better", "baseline better"}


@needs_c3
def test_paired_is_more_sensitive_than_comparing_two_intervals():
    """This is the reason section 34.2 exists.

    Independently computed intervals share most of their sampling noise, so
    asking whether they overlap is a weak test.  The paired difference keeps
    that shared noise and must separate at least one baseline that the
    unpaired comparison cannot.
    """
    head = R.c3_headline(C3)
    cov = head[head["method"] == "coverage_backward"].iloc[0]
    unpaired = {
        row["method"] for _, row in head.iterrows()
        if row["method"] != "coverage_backward" and cov["boot_hi"] < row["boot_lo"]
    }

    paired = R.c3_paired_table(C3)
    cells = paired.groupby("method")["n_seeds"].sum()
    wins = paired.groupby("method")["n_seeds_significant"].sum()
    decided = set(wins[wins >= 0.5 * cells].index)

    assert decided - unpaired, (
        "the paired test resolved nothing the unpaired comparison did not, "
        "which would mean the pairing is not being applied"
    )


# -- section 34.4, split robustness ------------------------------------------

@needs_c3
def test_split_robustness_uses_every_predeclared_seed():
    payload = R.load(C3)
    rob = R.c3_split_robustness(C3)
    assert (rob["n_seeds"] == len(payload["split_seeds"])).all()


@needs_c3
def test_split_robustness_reports_min_sd_and_max_not_just_the_mean():
    """Section 34.4 asks for all four; a mean alone can hide one bad split."""
    rob = R.c3_split_robustness(C3)
    for col in ("mean", "sd", "min", "max"):
        assert col in rob.columns
    assert (rob["min"] <= rob["mean"] + 1e-9).all()
    assert (rob["max"] >= rob["mean"] - 1e-9).all()
    assert np.allclose(rob["range"], rob["max"] - rob["min"])


@needs_c3
def test_split_robustness_agrees_with_the_headline_mean():
    """Both average the same numbers over the same budgets, so they must match."""
    rob = R.c3_split_robustness(C3).set_index("method")["mean"]
    head = R.c3_headline(C3).set_index("method")["worst_judge_tv"]
    common = rob.index.intersection(head.index)
    assert len(common)
    assert np.allclose(rob[common], head[common])


@needs_c3
def test_c3_cost_speedup_is_consistent_with_seconds_saved():
    cost = R.c3_cost_table(C3)
    total = cost["panel_seconds"] + cost["seconds_saved"]
    assert np.allclose(total, total.iloc[0])
    finite = cost[np.isfinite(cost["speedup"])]
    assert np.allclose(finite["speedup"], total.iloc[0] / finite["panel_seconds"])


@needs_c3
def test_c3_cost_decreases_as_the_panel_shrinks():
    cost = R.c3_cost_table(C3).sort_values("k")
    assert cost["panel_seconds"].is_monotonic_increasing


@needs_c3
def test_lowrank_floor_is_exact_at_full_rank():
    floor = R.lowrank_floor_table(C3)
    for method, group in floor.groupby("method"):
        full = group[group["k"] == group["k"].max()]
        assert full["worst_judge_tv"].iloc[0] < 1e-2, method


@needs_c3
def test_lowrank_floor_improves_with_rank():
    floor = R.lowrank_floor_table(C3)
    for method, group in floor.groupby("method"):
        series = group.sort_values("k")["mean_judge_tv"].to_numpy()
        assert (np.diff(series) <= 1e-9).all(), method


@needs_c3
def test_simplex_reconstruction_never_leaves_the_simplex():
    """The point of the simplex rule: its output is always a distribution."""
    rec = R.reconstruction_table(C3)
    simplex = rec[rec["rule"] == "simplex"]
    assert (simplex["off_simplex_frac"] == 0.0).all()
    assert (simplex["min_weight"] >= -1e-9).all()


@needs_c3
def test_unconstrained_rules_are_flagged_when_they_leave_the_simplex():
    rec = R.reconstruction_table(C3)
    loose = rec[rec["rule"].isin(["nonneg", "ridge"])]
    assert (loose["off_simplex_frac"] > 0).any(), (
        "unconstrained rules that never leave the simplex would mean the "
        "off-simplex check is not actually measuring anything"
    )


@needs_c3
def test_every_reconstruction_rule_sees_the_same_kept_judges():
    """The ablation isolates the fitting rule, so the subset must be held fixed."""
    rec = R.reconstruction_table(C3)
    for k, group in rec.groupby("k"):
        assert group["kept"].nunique() == 1, k


# --------------------------------------------------------------------------
# packaging
# --------------------------------------------------------------------------

@needs_c3
def test_export_writes_a_table_for_every_claim(tmp_path):
    out = str(tmp_path / "pkg")
    E.export("core8", out, make_zip=False, with_notebooks=False)

    names = set(os.listdir(os.path.join(out, "tables")))
    for expected in ("c3_headline.csv", "c3_cost.csv", "c2_frontier_TEST.csv"):
        assert expected in names
    assert os.path.exists(os.path.join(out, "SUMMARY.md"))
    assert os.path.exists(os.path.join(out, "manifest.json"))


@needs_c3
def test_manifest_hashes_match_the_files_on_disk(tmp_path):
    """A manifest that does not match its own package is worse than none."""
    out = str(tmp_path / "pkg")
    E.export("core8", out, make_zip=False, with_notebooks=False)

    with open(os.path.join(out, "manifest.json")) as handle:
        manifest = json.load(handle)

    listed = {f["path"] for f in manifest["files"]}
    on_disk = {
        os.path.relpath(os.path.join(folder, name), out)
        for folder, _, names in os.walk(out)
        for name in names
    }
    assert listed == on_disk - {"manifest.json"}

    for entry in manifest["files"]:
        assert E.sha256(os.path.join(out, entry["path"])) == entry["sha256"]


@needs_c3
def test_manifest_records_the_commit_and_whether_the_tree_was_clean(tmp_path):
    out = str(tmp_path / "pkg")
    E.export("core8", out, make_zip=False, with_notebooks=False)
    with open(os.path.join(out, "manifest.json")) as handle:
        manifest = json.load(handle)

    assert manifest["git_commit"]
    assert isinstance(manifest["git_clean"], bool)
    assert manifest["panel"] == "core8"


@needs_c3
def test_summary_reports_the_headline_numbers_it_claims(tmp_path):
    """Guards against the summary prose drifting away from the tables."""
    out = str(tmp_path / "pkg")
    E.export("core8", out, make_zip=False, with_notebooks=False)
    text = open(os.path.join(out, "SUMMARY.md")).read()

    head = R.c3_headline(C3)
    cov = head[head["method"] == "coverage_backward"].iloc[0]
    assert f"{cov['worst_judge_tv']:.3f}" in text
    assert "C1" in text and "C2" in text and "C3" in text


@needs_c3
def test_zip_contains_the_whole_package_under_one_folder(tmp_path):
    out = str(tmp_path / "pkg")
    zip_path = E.export("core8", out, make_zip=True, with_notebooks=False)

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    assert names, "empty archive"
    assert all(n.startswith("epcrc_results_core8/") for n in names)
    assert "epcrc_results_core8/SUMMARY.md" in names
    os.remove(zip_path)


def test_input_paths_prefers_the_panel_specific_file():
    paths = E.input_paths("core20")
    assert paths["e0_real"].endswith("e0_real_core20.json")
