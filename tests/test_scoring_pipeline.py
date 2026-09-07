"""Tests for the RewardBench 2 converter, prompt protocols and label scoring.

Nothing here loads a model: the pieces that could silently corrupt a whole
response tensor -- pair construction, split grouping, orientation correction,
softmax normalisation -- are all pure functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from epcrc.prompts import (
    I0_CLEAN,
    I1_SWAP,
    I2_CORRECTNESS,
    I4_VERBOSITY,
    I5_SOURCE_TAGS,
    I6_REFERENCE_HIDDEN,
    LABELS,
    PROTOCOLS,
    REGISTERED_CONTEXTS,
    Context,
    canonical_gold_label,
    canonicalize,
    protocol_hashes,
    render,
)
from epcrc.rewardbench import (
    PRIMARY_SEED,
    JudgedPair,
    base_item_id,
    build_pairs,
    grouped_split,
    is_tie_row,
    split_manifest,
    stratified_subset,
)
from epcrc.scoring import accuracy_report, argmax_labels, normalize_label_scores


def _row(subset, item_id, chosen, rejected):
    return {
        "id": item_id,
        "subset": subset,
        "prompt": f"question for {subset}/{item_id}",
        "chosen": chosen,
        "rejected": rejected,
        "num_correct": len(chosen),
        "num_incorrect": len(rejected),
        "total_completions": len(chosen) + len(rejected),
        "models": [],
        "additional_metadata": None,
    }


def _corpus():
    """A miniature stand-in with the real dataset's shape, including the id clash."""
    rows = []
    for domain in ("Factuality", "Focus", "Math"):
        for k in range(8):
            rows.append(_row(domain, str(k), ["good"], ["bad0", "bad1", "bad2"]))
    # Precise IF reuses ids 0..7, which is why the grouping key must be qualified.
    for k in range(8):
        rows.append(_row("Precise IF", str(k), ["good"], ["bad0", "bad1", "bad2"]))
    for k in range(8):
        rows.append(_row("Ties", f"tied:{k}", ["ok0", "ok1", "ok2", "ok3"], ["no0"]))
        rows.append(_row("Ties", f"ref:{k}", ["good"], ["bad0", "bad1"]))
    return rows


# --------------------------------------------------------------------------
# dataset conversion
# --------------------------------------------------------------------------

def test_base_item_id_disambiguates_colliding_ids():
    a = _row("Factuality", "0", ["x"], ["y"])
    b = _row("Precise IF", "0", ["x"], ["y"])

    assert a["id"] == b["id"]
    assert base_item_id(a) != base_item_id(b)


def test_tie_status_comes_from_the_number_of_chosen_responses():
    """The Ties subset is not homogeneous; 'ref:*' rows have a single answer."""
    assert is_tie_row(_row("Ties", "tied:0", ["a", "b", "c"], ["d"]))
    assert not is_tie_row(_row("Ties", "ref:0", ["a"], ["b", "c"]))


def test_pairs_respect_the_two_per_task_cap_and_sum_to_unit_weight():
    pairs = build_pairs(_corpus(), seed=PRIMARY_SEED)

    by_task = {}
    for pair in pairs:
        by_task.setdefault(pair.base_item_id, []).append(pair)

    for task_pairs in by_task.values():
        assert len(task_pairs) <= 2
        assert sum(p.weight for p in task_pairs) == pytest.approx(1.0)


def test_tie_pairs_are_labelled_C_and_use_two_acceptable_responses():
    pairs = build_pairs(_corpus(), seed=PRIMARY_SEED)
    tie_pairs = [p for p in pairs if p.is_tie]

    assert tie_pairs
    for pair in tie_pairs:
        assert pair.gold_label == "C"
        assert pair.response_a.startswith("ok")
        assert pair.response_b.startswith("ok")
        assert pair.response_a != pair.response_b


def test_non_tie_pairs_put_the_preferred_response_on_both_sides():
    """A fixed gold position would make position bias unmeasurable."""
    pairs = [p for p in build_pairs(_corpus(), seed=PRIMARY_SEED) if not p.is_tie]
    labels = {p.gold_label for p in pairs}

    assert labels == {"A", "B"}
    for pair in pairs:
        preferred = pair.response_a if pair.gold_label == "A" else pair.response_b
        assert preferred == "good"


def test_pair_construction_is_deterministic_and_seed_dependent():
    corpus = _corpus()
    first = build_pairs(corpus, seed=PRIMARY_SEED)
    again = build_pairs(list(reversed(corpus)), seed=PRIMARY_SEED)
    other_seed = build_pairs(corpus, seed=PRIMARY_SEED + 1)

    assert {p.pair_id: p for p in first} == {p.pair_id: p for p in again}
    assert [p.response_b for p in first] != [p.response_b for p in other_seed]


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------

def test_split_is_a_partition_and_keeps_a_task_whole():
    pairs = build_pairs(_corpus(), seed=PRIMARY_SEED)
    assignment = grouped_split(pairs, seed=PRIMARY_SEED)

    members = [i for ids in assignment.values() for i in ids]
    assert len(members) == len(set(members))
    assert set(members) == {p.base_item_id for p in pairs}

    # Every pair of one base task lands in the same split, by construction.
    membership = {i: name for name, ids in assignment.items() for i in ids}
    for pair in pairs:
        assert pair.base_item_id in membership


def test_split_is_stratified_by_domain_and_tie_status():
    pairs = build_pairs(_corpus(), seed=PRIMARY_SEED)
    assignment = grouped_split(pairs, seed=PRIMARY_SEED)
    membership = {i: name for name, ids in assignment.items() for i in ids}

    tie_tasks = {p.base_item_id for p in pairs if p.is_tie}
    tie_splits = {membership[i] for i in tie_tasks}

    assert tie_splits == {"FIT", "CERT", "TEST"}
    assert len(assignment["FIT"]) > len(assignment["CERT"])


def test_split_is_reproducible_and_manifest_records_checksums():
    pairs = build_pairs(_corpus(), seed=PRIMARY_SEED)
    first = grouped_split(pairs, seed=PRIMARY_SEED)
    again = grouped_split(pairs, seed=PRIMARY_SEED)
    other = grouped_split(pairs, seed=PRIMARY_SEED + 1)

    assert first == again
    assert first != other

    manifest = split_manifest(pairs, first, PRIMARY_SEED)
    assert set(manifest["split_sha256"]) == {"FIT", "CERT", "TEST"}
    assert manifest["n_pairs"] == len(pairs)
    assert manifest["n_tie_pairs"] > 0


def test_stratified_subset_covers_every_domain():
    pairs = build_pairs(_corpus(), seed=PRIMARY_SEED)
    subset = stratified_subset(pairs, 20, seed=PRIMARY_SEED)

    assert len(subset) == 20
    assert len({p.domain for p in subset}) == len({p.domain for p in pairs})

    # A size that the per-domain quota cannot divide must still be filled
    # exactly, deterministically, and without repeating a pair.
    awkward = stratified_subset(pairs, 22, seed=PRIMARY_SEED)
    assert len(awkward) == 22
    assert len({p.pair_id for p in awkward}) == 22
    assert awkward == stratified_subset(pairs, 22, seed=PRIMARY_SEED)


# --------------------------------------------------------------------------
# prompts and interventions
# --------------------------------------------------------------------------

def test_every_protocol_offers_the_three_labels_and_ends_at_the_verdict():
    for name, template in PROTOCOLS.items():
        assert template.endswith("Verdict:"), name
        assert "Return one label only." in template, name
        for label in LABELS:
            assert f"\n{label} —" in template, (name, label)


def test_rendering_places_the_responses_in_order():
    text = render("why?", "ALPHA", "BETA", context=I0_CLEAN)

    assert text.index("ALPHA") < text.index("BETA")
    assert "why?" in text
    assert "None provided." in text


def test_swap_context_reverses_the_rendered_order():
    clean = render("why?", "ALPHA", "BETA", context=I0_CLEAN)
    swapped = render("why?", "ALPHA", "BETA", context=I1_SWAP)

    assert swapped.index("BETA") < swapped.index("ALPHA")
    assert clean != swapped


def test_orientation_correction_restores_canonical_identity():
    """The judge saw B first, so its 'A' mass belongs to canonical B."""
    seen = (0.7, 0.2, 0.1)

    assert canonicalize(seen, I0_CLEAN) == (0.7, 0.2, 0.1)
    assert canonicalize(seen, I1_SWAP) == (0.2, 0.7, 0.1)
    # Applying the swap twice is the identity, and tie mass never moves.
    assert canonicalize(canonicalize(seen, I1_SWAP), I1_SWAP) == seen


def test_gold_label_follows_the_swap_but_ties_do_not():
    assert canonical_gold_label("A", I1_SWAP) == "B"
    assert canonical_gold_label("B", I1_SWAP) == "A"
    assert canonical_gold_label("C", I1_SWAP) == "C"
    assert canonical_gold_label("A", I0_CLEAN) == "A"


def test_interventions_change_only_their_own_target():
    base = render("why?", "ALPHA", "BETA", context=I0_CLEAN, reference="REF")

    padded = render("why?", "ALPHA", "BETA", context=I4_VERBOSITY, reference="REF")
    assert len(padded) > len(base)
    assert "BETA" in padded

    tagged = render("why?", "ALPHA", "BETA", context=I5_SOURCE_TAGS, reference="REF")
    assert "System Alpha" in tagged and "System Beta" in tagged

    hidden = render("why?", "ALPHA", "BETA", context=I6_REFERENCE_HIDDEN, reference="REF")
    assert "REF" not in hidden
    assert "REF" in base

    rubric = render("why?", "ALPHA", "BETA", context=I2_CORRECTNESS, reference="REF")
    assert "strict evaluator" in rubric


def test_registered_contexts_are_uniquely_named_and_hashes_are_stable():
    names = [c.name for c in REGISTERED_CONTEXTS]
    assert len(names) == len(set(names))

    assert protocol_hashes() == protocol_hashes()
    assert len(set(protocol_hashes().values())) == 3


def test_render_rejects_an_unknown_protocol():
    with pytest.raises(ValueError, match="unknown protocol"):
        render("q", "a", "b", context=Context("bogus", protocol="P9"))


# --------------------------------------------------------------------------
# label scoring
# --------------------------------------------------------------------------

def test_normalization_is_a_softmax_over_the_three_labels():
    probabilities = normalize_label_scores([-1.0, -2.0, -3.0])

    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]
    # Shift invariance: only differences between label scores matter.
    assert np.allclose(probabilities, normalize_label_scores([9.0, 8.0, 7.0]))


def test_normalization_survives_the_large_negative_logprobs_it_will_see():
    probabilities = normalize_label_scores([-900.0, -901.0, -902.0])

    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)


def test_normalization_rejects_malformed_scores():
    with pytest.raises(ValueError, match="three label scores"):
        normalize_label_scores([-1.0, -2.0])
    with pytest.raises(ValueError, match="non-finite"):
        normalize_label_scores([-1.0, float("-inf"), -3.0])


def test_accuracy_report_separates_tie_and_binary_behaviour():
    pairs = [
        JudgedPair("p0", "d/0", "Math", False, "q", "a", "b", "A", 1.0, 0),
        JudgedPair("p1", "d/1", "Math", False, "q", "a", "b", "B", 1.0, 0),
        JudgedPair("p2", "d/2", "Ties", True, "q", "a", "b", "C", 1.0, 0),
    ]
    probabilities = np.array([
        [0.8, 0.1, 0.1],   # predicts A, correct
        [0.1, 0.8, 0.1],   # predicts B, correct
        [0.2, 0.2, 0.6],   # predicts tie, correct
    ])

    report = accuracy_report(probabilities, pairs)

    assert report["three_class_accuracy"] == pytest.approx(1.0)
    assert report["binary_accuracy_non_tie"] == pytest.approx(1.0)
    assert report["gold_tie_rate"] == pytest.approx(1 / 3)
    assert report["predicted_tie_rate"] == pytest.approx(1 / 3)
    assert argmax_labels(probabilities) == ["A", "B", "C"]
