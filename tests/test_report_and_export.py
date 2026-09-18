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
C7 = os.path.join(RESULTS, "c7_certification_core20.json")
E6 = os.path.join(RESULTS, "e6_sparse_cost_core20.json")
BACKBONE = os.path.join(RESULTS, "backbone_core20.json")

needs_c3 = pytest.mark.skipif(not os.path.exists(C3), reason="C3 not run")
needs_e1 = pytest.mark.skipif(not os.path.exists(E1), reason="E1 not run")
needs_e0 = pytest.mark.skipif(not os.path.exists(E0_REAL), reason="E0 real not run")
needs_c7 = pytest.mark.skipif(not os.path.exists(C7), reason="C7 not run")
needs_e6 = pytest.mark.skipif(not os.path.exists(E6), reason="E6 not run")
needs_backbone = pytest.mark.skipif(not os.path.exists(BACKBONE),
                                    reason="backbone not run")


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


def _e0_payload(tmp_path, declared_rows):
    """A minimal E0 file. `declared_rows` are (gamma, removable, coverage) triples."""
    records = []
    for gamma, removable, coverage in declared_rows:
        records.append({
            "instance": "panel_real",
            "gamma": gamma,
            "gamma_source": "declared",
            "seed": 1000,
            "n_individually_removable": removable,
            "naive_retained_size": 19 - removable,
            "naive_coverage": coverage,
            "naive_violates_gamma": coverage > gamma,
            "min_feasible_size": 19 - removable,
            "min_feasible_is_exact": False,
            "min_feasible_lower_bound": 6,
            "composition_gap": 1,
            "dependency_graph": {"n_cycles": 1},
            "loo_errors": [0.13, 0.19, 0.32],
        })
    os.makedirs(str(tmp_path), exist_ok=True)
    path = str(tmp_path / "e0.json")
    with open(path, "w") as handle:
        json.dump({"experiment": "E0", "claim": "C1",
                   "gammas": [g for g, _, _ in declared_rows],
                   "records": records}, handle)
    return path


def _setup_only_c3(tmp_path, judges):
    """A C3 file carrying only what the summary's Setup block reads."""
    path = str(tmp_path / "c3.json")
    with open(path, "w") as handle:
        json.dump({
            "split_seeds": [1, 2],
            "n_bootstrap": 10,
            "per_split_seed": [{
                "judges": list(judges),
                "contexts": ["I0_clean"],
                "split_items": {"FIT": 10, "CERT": 5, "TEST": 5},
                "methods": {},
            }],
        }, handle)
    return path


def _summary_setup(tmp_path, judges, panel="core20"):
    return "\n".join(
        E._setup_section(panel, {"c3": _setup_only_c3(tmp_path, judges)}))


def test_summary_names_the_judges_the_panel_is_missing(tmp_path):
    """A short panel is a deviation from the plan, not a design choice.

    Every number in the package is over 19 of the 20 judges the plan's table
    names, because J07's weights are gated on Hugging Face.  A reader given only
    the count cannot tell a deliberate smaller panel from an unscored judge, so
    the absent ids and the reason are stated rather than left to be inferred
    from a gap in the numbering.
    """
    from epcrc.panel import PANELS

    short = [j for j in PANELS["core20"] if j != "J07"]
    text = _summary_setup(tmp_path, short)

    assert "Deviation from the plan's judge table" in text
    assert "J07" in text
    assert "19 of 20 judges" in text


def test_summary_calls_the_split_counts_pairs_not_items(tmp_path):
    """The recorded counts are rows, and a row is a comparison pair.

    One base item yields up to two pairs and both land in the same split, so
    labelling these counts as items overstates the number of distinct source
    tasks by about a factor of two.
    """
    from epcrc.panel import PANELS

    text = _summary_setup(tmp_path, PANELS["core20"])

    assert "Comparison pairs per split" in text
    assert "Items per split" not in text


def test_summary_claims_no_deviation_when_the_panel_is_complete(tmp_path):
    """The notice must not fire on a full panel, or it stops meaning anything."""
    from epcrc.panel import PANELS

    text = _summary_setup(tmp_path, PANELS["core20"])

    assert "Deviation from the plan's judge table" not in text


def test_c1_reports_a_violation_at_a_predeclared_tolerance(tmp_path):
    """The strongest form of C1 must not be dropped in favour of the weaker one.

    On Core-8 the predeclared grid left the removable set empty everywhere, so
    the section was written to fall back to leave-one-out breakpoints.  Core-19
    does admit two removable judges at the largest predeclared gamma, and the
    joint deletion breaks it -- the claim at a tolerance fixed before the data
    was seen, which is strictly better evidence than any breakpoint.  Reading
    only the breakpoint rows discarded it silently.
    """
    path = _e0_payload(tmp_path, [(0.15, 1, 0.134), (0.20, 2, 0.2158)])
    text = "\n".join(E._c1_section({"e0_real": path, "e0_synthetic": "absent"}))

    assert "**On the predeclared grid.**" in text
    assert "0.216" in text                   # the measured joint error
    assert "fixed before the data was seen" in text


def test_c1_claims_nothing_about_the_predeclared_grid_when_it_is_vacuous(tmp_path):
    """One removable judge is not a composition test, so it must not be claimed."""
    path = _e0_payload(tmp_path, [(0.15, 1, 0.134), (0.20, 1, 0.140)])
    text = "\n".join(E._c1_section({"e0_real": path, "e0_synthetic": "absent"}))

    assert "**On the predeclared grid.**" not in text


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


@needs_e1
def test_c2_section_quotes_the_one_judge_floor_that_makes_the_target_unreachable():
    """Missing the declared band and being unable to reach it are different.

    Dropping one judge is the smallest possible compression, so its error
    bounds every smaller panel from below.  Reporting that floor turns "we did
    not hit 0.08-0.10" into "no panel size on this panel can", which is a
    statement about judge redundancy rather than about our pruner -- and it
    stops a reader concluding the frontier was simply under-tuned.
    """
    df = R.c2_table(E1, "TEST")
    cov = df[df["method"] == "coverage_backward"].sort_values("k")
    full = int(cov["k"].max())
    floor = float(cov[cov["k"] == full - 1].iloc[0]["worst_judge_tv"])

    text = "\n".join(E._c2_section({"e1": E1}))
    assert f"{floor:.3f}" in text
    assert "0.08-0.10" in text
    assert (cov[cov["k"] < full]["worst_judge_tv"] >= floor - 1e-12).all()


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
def test_paired_win_and_loss_counts_are_split_by_direction():
    """A decided cell says the comparison resolved, not who won.

    Summing `excludes_zero` and calling the total a win count silently
    credits the reference with every cell a baseline took off it, which is
    the one direction of error a reader cannot detect from the table.
    """
    paired = R.c3_paired_table(C3)
    both = paired["n_seeds_reference_better"] + paired["n_seeds_baseline_better"]
    assert (both == paired["n_seeds_significant"]).all()
    # Losses exist in this data, so the split is not vacuous and any claim
    # phrased as a win count must be strictly smaller than the decided count.
    assert int(paired["n_seeds_baseline_better"].sum()) > 0


@needs_c3
def test_c3_section_states_where_coverage_loses():
    """The budgets where the method fails must be in the prose, not derivable.

    The losses are concentrated at the smallest budgets and are unanimous
    across seeds, so they are a property of greedy elimination rather than
    noise.  Reporting only the pooled delta would hide that.
    """
    paired = R.c3_paired_table(C3)
    lost = paired[paired["n_seeds_baseline_better"] > 0]
    text = "\n".join(E._c3_section({"c3": C3}))
    assert "Where coverage loses" in text
    assert str(int(lost["n_seeds_baseline_better"].sum())) in text
    assert "Verdict:" in text


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
    wins = paired.groupby("method")["n_seeds_reference_better"].sum()
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
    assert paths["c4"].endswith("c4_stress_specialists_core20.json")


# --------------------------------------------------------------------------
# C4
# --------------------------------------------------------------------------

def _c4_payload(tmp_path, robust_tv, baseline_tv, specialist_seeds,
                select_tv=None):
    """A C4 result file with dictated errors, summarised by the experiment itself.

    `summarise` is imported from the experiment rather than hand-rolled here, so
    the test pins the reader against the producer's own sign convention instead
    of against a second guess at it.  `robust_tv` and `baseline_tv` are per-seed
    lists of worst-context errors at the single budget k = 2.

    `baseline_tv` drives `clean_pipeline`; `select_tv` drives `clean_select` and
    defaults to it.  They are separable because the two arms carry different
    handicaps and the real panel separates them, so a fixture that forced them
    equal could not reproduce the shape the summary has to describe.
    """
    from experiments.experiment_c4_stress_specialists import summarise

    seeds = [str(1000 + i) for i in range(len(robust_tv))]
    judges = ["J01", "J02", "J03"]
    if select_tv is None:
        select_tv = baseline_tv

    def arm(tv, n_kept):
        return {
            "selects_on": ["I0_clean"],
            "fits_on": ["I0_clean"],
            "chain": {"2": judges[:2]},
            "budgets": {"2": {
                "kept": judges[:2],
                "k": 2,
                "worst_judge_worst_context_tv": tv,
                "worst_judge_clean_tv": tv / 2,
                "mean_judge_worst_context_tv": tv / 3,
                "verdict_agreement": 0.9,
                "n_specialists_kept": n_kept,
                "n_specialists_total": 2,
                "per_judge": {},
            }},
        }

    per_seed = {}
    for i, seed in enumerate(seeds):
        per_seed[seed] = {
            "split_items": {"FIT": 10, "CERT": 5, "TEST": 5},
            "specialists": {
                j: {
                    "is_specialist": seed in specialist_seeds.get(j, []),
                    "binding_context": "I1_swapped",
                    "stress_gap": 0.3,
                }
                for j in judges
            },
            "arms": {
                "robust": arm(robust_tv[i], 2),
                "clean_select": arm(select_tv[i], 0),
                "clean_pipeline": arm(baseline_tv[i], 0),
            },
        }

    payload = {
        "experiment": "C4",
        "claim": "C4",
        "seed": 20260817,
        "split_seeds": [int(s) for s in seeds],
        "judges": judges,
        "models": [f"org/{j}" for j in judges],
        "contexts": ["I0_clean", "I1_swapped"],
        "clean_context": "I0_clean",
        "specialist_margin": 0.05,
        "budgets": [2],
        "arms": {},
        "per_seed": per_seed,
        "summary": summarise(per_seed, [2]),
    }
    os.makedirs(str(tmp_path), exist_ok=True)
    path = str(tmp_path / "c4.json")
    with open(path, "w") as handle:
        json.dump(payload, handle)
    return path


def test_c4_delta_is_positive_when_robust_selection_is_better(tmp_path):
    """The sign convention is the whole claim, so it is pinned explicitly.

    `delta` is baseline minus robust.  If it ever flips, every C4 sentence in
    the summary inverts while still reading as fluent prose, which is the kind
    of error no amount of proofreading catches.
    """
    path = _c4_payload(tmp_path, robust_tv=[0.2, 0.2], baseline_tv=[0.5, 0.5],
                       specialist_seeds={})
    head = R.c4_headline(path)

    assert (head["delta_mean"] > 0).all()
    assert head["delta_mean"].iloc[0] == pytest.approx(0.3)


def test_c4_flags_a_split_result_rather_than_rounding_it_up(tmp_path):
    """A positive mean delta carried by one seed must not read as a clean win."""
    path = _c4_payload(tmp_path, robust_tv=[0.1, 0.6], baseline_tv=[0.9, 0.5],
                       specialist_seeds={})
    head = R.c4_headline(path)

    row = head.iloc[0]
    assert row["delta_mean"] > 0            # the mean says robust wins...
    assert row["n_seeds_better"] == 1       # ...but only one of two seeds does
    assert not row["robust_wins_every_seed"]
    assert row["delta_min"] < 0


def test_c4_table_covers_every_arm_seed_and_budget(tmp_path):
    path = _c4_payload(tmp_path, robust_tv=[0.2, 0.3], baseline_tv=[0.5, 0.5],
                       specialist_seeds={})
    df = R.c4_table(path)

    assert set(df["arm"]) == {"robust", "clean_select", "clean_pipeline"}
    assert len(df) == 3 * 2          # three arms, two seeds, one budget
    assert set(df["k"]) == {2}


def test_c4_specialists_separates_stable_labels_from_one_off_ones(tmp_path):
    """A judge flagged under one partition is a candidate, not a finding."""
    path = _c4_payload(
        tmp_path, robust_tv=[0.2, 0.2], baseline_tv=[0.5, 0.5],
        specialist_seeds={"J01": ["1000", "1001"], "J02": ["1000"]},
    )
    df = R.c4_specialists(path).set_index("judge")

    assert df.loc["J01", "in_every_seed"]
    assert df.loc["J01", "n_seeds_flagged"] == 2
    assert not df.loc["J02", "in_every_seed"]
    assert df.loc["J02", "n_seeds_flagged"] == 1
    assert "J03" not in df.index


def test_c4_summary_section_states_the_verdict_it_earned(tmp_path):
    """A split result has to say so in the prose, not just in the table."""
    split = _c4_payload(tmp_path / "a", robust_tv=[0.1, 0.6],
                        baseline_tv=[0.9, 0.5], specialist_seeds={})
    clean = _c4_payload(tmp_path / "b", robust_tv=[0.2, 0.2],
                        baseline_tv=[0.5, 0.5], specialist_seeds={})

    split_text = "\n".join(E._c4_section({"c4": split}))
    clean_text = "\n".join(E._c4_section({"c4": clean}))

    assert E.section_verdict(split_text.split("\n")) == "PARTIALLY SUPPORTED"
    assert E.section_verdict(clean_text.split("\n")) == "SUPPORTED"


def test_c4_does_not_claim_the_selection_result_the_pipeline_arm_earned(tmp_path):
    """The real Core-19 shape: unanimous vs the pipeline, not vs selection alone.

    `clean_pipeline` is handicapped at selection *and* fitting; `clean_select`
    only at selection.  C4 as stated is a claim about selection, so a verdict that
    pooled the two arms would report the pipeline arm's easy win as evidence for
    it.  The section has to separate them and say what share is selection.
    """
    path = _c4_payload(
        tmp_path, robust_tv=[0.20, 0.20], baseline_tv=[0.60, 0.60],
        select_tv=[0.25, 0.15], specialist_seeds={},
    )
    text = "\n".join(E._c4_section({"c4": path}))

    assert E.section_verdict(text.split("\n")) == "PARTIALLY SUPPORTED"
    assert "Strong for the deployed pipeline, weak for selection alone" in text
    assert "of the total effect is attributable to" in text


def test_c4_section_says_not_run_rather_than_failing(tmp_path):
    text = "\n".join(E._c4_section({"c4": str(tmp_path / "absent.json")}))
    assert "_Not run._" in text


def test_c4_figure_draws_from_the_same_tables_as_the_text(tmp_path):
    """The figure must survive the no-specialists case too, not just the happy one."""
    from epcrc import figures as Fg

    for specialists in ({}, {"J01": ["1000", "1001"]}):
        path = _c4_payload(tmp_path / f"f{len(specialists)}",
                           robust_tv=[0.2, 0.3], baseline_tv=[0.5, 0.5],
                           specialist_seeds=specialists)
        fig = Fg.fig_c4_stress(path)
        assert fig.axes
        Fg.plt.close(fig)


def test_save_all_writes_the_c4_figure_when_c4_exists(tmp_path):
    from epcrc import figures as Fg

    path = _c4_payload(tmp_path / "in", robust_tv=[0.2, 0.3],
                       baseline_tv=[0.5, 0.5], specialist_seeds={})
    written = Fg.save_all(str(tmp_path / "out"), c4=path)
    assert [os.path.basename(p) for p in written] == ["c4_stress.png"]


# --------------------------------------------------------------------------
# C7
# --------------------------------------------------------------------------


@needs_c7
def test_c7_verdict_does_not_report_violations_the_data_does_not_contain():
    """A wide interval and a broken bound are not the same failure.

    `within_nominal` folds two situations into one flag: the rule was violated
    too often, or the rule certified so few cases that the Wilson upper bound
    cannot fall below delta no matter how well it behaved.  On this panel the
    primary rule has zero violations and still fails the flag, purely because
    the tightest tolerance certifies a handful of cases.  Writing "violated
    more often than its nominal rate" there would state the opposite of what
    the run measured, so the wording has to follow `n_violated`.
    """
    head = R.c7_headline(C7, 0.05)
    primary = head[head["is_primary"]].iloc[0]

    text = "\n".join(E._c7_section({"c7": C7}))

    if int(primary["n_violated"]) == 0:
        assert "violated more often than its nominal rate" not in text
        assert f"not violated once in {int(primary['n_certified'])}" in text
    else:
        assert "not violated once" not in text


# --------------------------------------------------------------------------
# C8 and the backbone
# --------------------------------------------------------------------------


def test_sections_say_not_run_rather_than_inventing_numbers():
    for section in (E._c8_section, E._backbone_section):
        text = "\n".join(section({"e6": "", "backbone": ""}))
        assert "_Not run._" in text


@needs_e6
def test_c8_prose_reads_the_cap_column_and_not_its_rendering():
    """`support_cap` mixes ints with the string "none".

    Comparing it against "3" silently selects nothing and the ratios come out
    as NaN, which reads as a formatting glitch rather than as the bug it is.
    """
    text = "\n".join(E._c8_section({"e6": E6}))

    assert "nan" not in text.lower()
    sparse = R.c8_sparse_table(E6)
    worst = float(sparse[sparse["support_cap"] == 3]["ratio_to_uncapped"].max())
    assert f"{worst:.3f}" in text


@needs_e6
def test_c8_verdict_names_the_objectives_that_actually_disagreed():
    """The cost half is a negative result, so it must not read as a positive one."""
    cost = R.c8_cost_table(E6)
    differs = cost[~cost["same_panel_as_cardinality"]]
    text = "\n".join(E._c8_section({"e6": E6}))

    assert "not separable on this panel" in text
    for objective in set(differs["objective"]):
        assert objective in text
    # An objective that always returned the cardinality panel must not be
    # listed as having disagreed.
    for objective in set(cost["objective"]) - set(differs["objective"]):
        assert f"gamma 0.2 under `{objective}`" not in text


@needs_backbone
def test_backbone_section_only_claims_certification_it_has():
    per_seed = R.backbone_table(BACKBONE)
    text = "\n".join(E._backbone_section({"backbone": BACKBONE}))

    if bool(per_seed["certified"].all()):
        assert "certified on every seed" in text
    else:
        assert "is an upper bound" in text


@needs_backbone
def test_backbone_section_does_not_quote_an_unstable_judge_as_a_finding():
    """A judge whose category changes with the partition is not a result."""
    head = R.backbone_headline(BACKBONE)
    text = "\n".join(E._backbone_section({"backbone": BACKBONE}))

    loose = head[head["unstable"] != ""]
    for row in loose.itertuples():
        for judge in row.unstable.split(","):
            assert judge not in row.always_mandatory.split(",")
            assert judge not in row.never_in_any_optimum.split(",")
    assert "is not a finding" in text or len(loose) == 0


@needs_e6
@needs_backbone
def test_build_tables_ships_the_c8_and_backbone_csvs():
    tables = E.build_tables({
        key: "" for key in
        ("e0_synthetic", "e0_real", "e1", "c3", "c4", "c5", "e7", "c6", "c7")
    } | {"e6": E6, "backbone": BACKBONE})

    assert set(tables) == {
        "c8_sparse", "c8_support_stability", "c8_cost_panels",
        "backbone_headline", "backbone_per_seed", "backbone_per_judge",
    }
    assert all(len(df) for df in tables.values())


@needs_backbone
def test_the_backbone_figure_accounts_for_every_judge():
    """The four bands are a partition, so they must add up to the panel.

    The headline names only the two extreme categories. Stacking those and
    leaving the optional representatives out would draw a panel that is
    missing judges, which reads as the panel having shrunk.
    """
    from epcrc import figures as Fg

    per_judge = R.backbone_per_judge(BACKBONE)
    n_judges = per_judge["judge"].nunique()
    totals = per_judge.groupby("gamma").size()
    assert (totals == n_judges).all()

    fig = Fg.fig_backbone(BACKBONE)
    stacked = fig.axes[0].collections
    assert len(stacked) == 4
    Fg.plt.close(fig)


@needs_e6
def test_the_c8_figure_draws_a_line_per_budget():
    from epcrc import figures as Fg

    budgets = R.c8_sparse_table(E6)["k"].nunique()
    fig = Fg.fig_c8_sparse(E6)
    # One line per budget plus the dashed uncapped reference.
    assert len(fig.axes[0].lines) == budgets + 1
    Fg.plt.close(fig)


# --------------------------------------------------------------------------
# FINAL_REPORT.md
# --------------------------------------------------------------------------

_FAKE_PROV = {"generated_utc": "x", "git_commit": "0" * 40, "git_branch": "t"}


def test_the_final_report_has_the_section_66_headings_in_order():
    """Section 66 fixes the structure, so it is checked rather than trusted."""
    text = E.build_final_report("core20", E.input_paths("core20"), _FAKE_PROV)
    headings = [l for l in text.split("\n") if l.startswith("## ")]

    assert headings == [
        "## 1. Executive Findings",
        "## 2. Frozen Experimental Setting",
        "## 3. Data and Panel Completeness",
        "## 4. C1: Non-Composability",
        "## 5. C2: Physical-to-Virtual Compression",
        "## 6. C3: Baseline Comparison",
        "## 7. C4: Robust Contexts and Specialists",
        "## 8. C5: Downstream Preservation",
        "## 9. C6: Exact Optimality and Exchange Structure",
        "## 10. C7: Certification Reliability",
        "## 11. Optional C8 Results",
        "## 12. Negative and Null Results",
        "## 13. Deviations from the Plan",
        "## 14. Figure and Table Index",
        "## 15. Reproduction Commands",
        "## 16. Package Verification",
    ]


def test_every_claim_section_opens_with_one_of_the_four_allowed_words():
    text = E.build_final_report("core20", E.input_paths("core20"), _FAKE_PROV)
    allowed = {"SUPPORTED", "PARTIALLY SUPPORTED", "UNSUPPORTED",
               "CONTRADICTED", "NOT RUN"}

    for claim, heading, _ in E._FINAL_SECTIONS:
        block = text.split(f"## {heading}\n\n")[1]
        assert block.split("\n")[0].strip("* ") in allowed, claim


def test_the_headline_verdict_comes_from_the_section_that_argues_it():
    """A verdict stated twice can be changed in one place and not the other."""
    for token in ("SUPPORTED", "PARTIALLY SUPPORTED", "UNSUPPORTED",
                  "CONTRADICTED"):
        assert E.section_verdict(
            ["## X", "", f"**Verdict: {token}.** because"]) == token

    assert E.section_verdict(["## X", "", "_Not run._"]) == "NOT RUN"
    assert E.section_verdict(["## X", "", "no opinion"]) == "NO VERDICT"


def test_the_section_66_heading_list_is_not_duplicated_inside_a_claim():
    """A builder shipping its own `## C4 - ...` would add a heading section 66
    does not have, so claim bodies are demoted to subheads."""
    text = E.build_final_report("core20", E.input_paths("core20"), _FAKE_PROV)
    body = text.split("## 7. C4")[1].split("\n## ")[0]

    assert "### Backbone" in body


def test_every_script_the_report_tells_a_reader_to_run_exists():
    """A reproduction section naming a renamed script is worse than none."""
    for name, _ in E.REPRODUCTION_SCRIPTS:
        assert os.path.exists(os.path.join(E.ROOT, "experiments", name)), name


def test_the_deviations_section_points_at_a_file_the_package_carries():
    """Section 13 is the short list; DECISIONS.md is the record it defers to.

    Both halves are asserted together, because a pointer to a document the
    export does not copy sends the reader looking for a file that is not there.
    """
    text = "\n".join(E._deviations_section("core20", E.input_paths("core20")))

    assert "DECISIONS.md" in text
    assert os.path.exists(os.path.join(E.ROOT, "DECISIONS.md"))


def test_a_null_result_is_dropped_when_its_claim_comes_back_supported():
    verdicts = {claim: "SUPPORTED" for claim, _, _ in E._FINAL_SECTIONS}
    text = "\n".join(
        E._negative_results_section(E.input_paths("core20"), verdicts))

    assert "0 of 8 claims" in text
    assert "- " not in text


def test_the_result_index_agrees_with_the_report_it_indexes():
    """A script reading the index and a person reading the report must not
    come away with different verdicts."""
    inputs = E.input_paths("core20")
    index = E.build_result_index("core20", inputs, _FAKE_PROV)
    text = E.build_final_report("core20", inputs, _FAKE_PROV)

    for entry in index["claims"]:
        block = text.split(f"## {entry['section']}\n\n")[1]
        assert block.split("\n")[0].strip("* ") == entry["verdict"]
        assert entry["verdict"] in index["verdict_vocabulary"]


def test_every_table_in_the_index_is_claimed_by_exactly_one_claim():
    inputs = E.input_paths("core20")
    index = E.build_result_index("core20", inputs, _FAKE_PROV)

    listed = [t for entry in index["claims"] for t in entry["tables"]]
    assert len(listed) == len(set(listed))
    # The backbone tables are filed under C4, not left unclaimed.
    c4 = next(e for e in index["claims"] if e["claim"] == "C4")
    assert any("backbone" in t for t in c4["tables"])


def test_a_missing_experiment_is_named_rather_than_silently_dropped():
    inputs = dict(E.input_paths("core20"), e7="")
    index = E.build_result_index("core20", inputs, _FAKE_PROV)

    c5 = next(e for e in index["claims"] if e["claim"] == "C5")
    assert "e7" in c5["missing_inputs"]
    assert "e7" not in c5["inputs"]


# --------------------------------------------------------------------------
# E7 -- cross-benchmark transfer
# --------------------------------------------------------------------------

def _e7_result(tv, refit=None, oracle_tv=0.30, overlap=0.5):
    """One budget's worth of E7 output, in the shape the experiment writes."""
    def block(value):
        return {"worst_judge_worst_context_tv": value,
                "mean_judge_worst_context_tv": value * 0.6,
                "verdict_agreement": 0.9}

    return {
        "kept": ["J01", "J02"],
        "in_domain_TEST": block(0.20),
        "frozen_weights_TEST": block(tv),
        "transfer_gap": tv - 0.20,
        "refit_weights_TEST": {str(m): block(v)
                               for m, v in (refit or {10: tv, 310: tv - 0.05}).items()},
        "oracle_reselection": {"kept": ["J01", "J03"], "TEST": block(oracle_tv),
                               "basis_overlap_jaccard": overlap},
        "seconds": 1.0,
    }


def _e7_payload(tmp_path, coverage_tv=0.25, baseline_tv=0.40):
    payload = {
        "experiment": "E7",
        "claims": ["C2", "C4", "C5"],
        "seed": 1,
        "judges": [f"J{i:02d}" for i in range(1, 21)],
        "home_benchmark": "allenai/reward-bench-2",
        "away_benchmark": "ScalerLab/JudgeBench",
        "away_has_ties": False,
        "budgets": [4],
        "calibration_sizes": [10, 310],
        "n_random_draws": 1,
        "split_items": {"home": {"FIT": 1866, "CERT": 932, "TEST": 926},
                        "away": {"FIT": 310, "CERT": 153, "TEST": 157}},
        "methods": {
            "coverage_backward": {"4": _e7_result(coverage_tv)},
            "top_accuracy": {"4": _e7_result(baseline_tv)},
            "random_0": {"4": _e7_result(baseline_tv + 0.02)},
        },
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "e7_transfer_core20.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_e7_reports_the_refit_at_the_largest_calibration_sample(tmp_path):
    """The efficiency claim is about the most calibration data that was spent.

    Dict order in JSON is not numeric order, so picking the last key would
    report an arbitrary sample size as though it were the best available.
    """
    path = _e7_payload(tmp_path)
    row = R.e7_table(path)
    ours = row[row["raw_method"] == "coverage_backward"].iloc[0]

    assert ours["refit_tv"] == pytest.approx(0.20)   # 0.25 - 0.05, the m=310 row
    assert ours["refit_gain"] == pytest.approx(0.05)


def test_e7_headline_says_whether_the_basis_still_leads_off_benchmark(tmp_path):
    """E7's finding is the ordering, not the level, so the flag is the point."""
    ahead = R.e7_headline(_e7_payload(tmp_path / "a", coverage_tv=0.25,
                                      baseline_tv=0.40))
    behind = R.e7_headline(_e7_payload(tmp_path / "b", coverage_tv=0.45,
                                       baseline_tv=0.40))

    assert bool(ahead.iloc[0]["beats_every_baseline"]) is True
    assert ahead.iloc[0]["best_baseline_frozen_tv"] == pytest.approx(0.40)
    assert bool(behind.iloc[0]["beats_every_baseline"]) is False


def test_e7_does_not_declare_a_verdict_of_its_own(tmp_path):
    """C5 owns the verdict; E7 is filed under it as evidence.

    `section_verdict` scans the whole claim body, so a second token here could
    become the one printed at the top of C5 in the final report.
    """
    inputs = dict(E.input_paths("core20"), e7=_e7_payload(tmp_path))
    text = "\n".join(E._e7_section(inputs))

    assert "**Verdict:" not in text
    assert E.section_verdict(E._claim_body([E._c5_section, E._e7_section], inputs)) \
        == E.section_verdict(E._claim_body([E._c5_section], inputs))


def test_e7_describes_the_direction_the_error_actually_moved(tmp_path):
    """A benchmark with no ties can lower the error, and the prose must allow it.

    Asserting a rise unconditionally would print a false statement on exactly
    the data the reader would most want explained.
    """
    rose = dict(E.input_paths("core20"),
                e7=_e7_payload(tmp_path / "up", coverage_tv=0.35))
    fell = dict(E.input_paths("core20"),
                e7=_e7_payload(tmp_path / "down", coverage_tv=0.05,
                               baseline_tv=0.06))

    assert "rises on the new benchmark at every budget" in "\n".join(
        E._e7_section(rose))
    assert "does not rise on the new benchmark at any budget" in "\n".join(
        E._e7_section(fell))


def test_e7_tables_are_filed_under_c5(tmp_path):
    """Section 31 puts E7 in C5's required column, so its CSVs belong there."""
    inputs = dict(E.input_paths("core20"), e7=_e7_payload(tmp_path))
    index = E.build_result_index("core20", inputs, _FAKE_PROV)

    c5 = next(e for e in index["claims"] if e["claim"] == "C5")
    assert any("e7" in t for t in c5["tables"])
    assert "e7" not in c5["missing_inputs"]
