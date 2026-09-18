"""Tests for the JudgeBench transfer set used by E7.

None of these touch the network: the loader is exercised on rows shaped like the
real ones, because what has to be guaranteed is the mapping onto `JudgedPair`
and the fact that the E7 cache cannot collide with the in-domain one.
"""

from __future__ import annotations

import pytest

from epcrc.judgebench import build_pairs, coverage_summary, source_family
from epcrc.panel import pairs_path, scores_dir, split_path
from epcrc.rewardbench import grouped_split


def _row(pair_id, source, label, question="q"):
    return {
        "pair_id": pair_id,
        "original_id": 1,
        "source": source,
        "question": question,
        "response_model": "gpt-4o",
        "response_A": "a",
        "response_B": "b",
        "label": label,
    }


ROWS = [
    ("gpt", _row("p1", "mmlu-pro-law", "A>B")),
    ("gpt", _row("p2", "livebench-math", "B>A")),
    ("claude", _row("p3", "livecodebench", "A>B")),
    ("claude", _row("p4", "mmlu-pro-physics", "B>A")),
]


def test_source_family_coarsens_to_the_three_benchmarks():
    assert source_family("mmlu-pro-computer science") == "mmlu-pro"
    assert source_family("livebench-reasoning") == "livebench"
    assert source_family("livecodebench") == "livecodebench"


def test_labels_map_onto_gold_and_nothing_is_a_tie():
    pairs = build_pairs(ROWS)

    assert [p.gold_label for p in pairs] == ["A", "B", "A", "B"]
    assert not any(p.is_tie for p in pairs)
    # The tie class is never correct here, so E7 says nothing about it.
    assert coverage_summary(pairs)["gold_label_counts"]["C"] == 0


def test_an_unknown_label_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unexpected JudgeBench label"):
        build_pairs([("gpt", _row("p1", "mmlu-pro-law", "tie"))])


def test_pair_ids_are_qualified_by_response_model():
    """Both splits could in principle reuse an id; the split name separates them."""
    pairs = build_pairs([
        ("gpt", _row("same", "livebench-math", "A>B")),
        ("claude", _row("same", "livebench-math", "A>B")),
    ])

    assert [p.pair_id for p in pairs] == ["gpt/same", "claude/same"]
    assert len({p.base_item_id for p in pairs}) == 2


def test_every_item_lands_in_exactly_one_split():
    pairs = build_pairs(ROWS)
    assignment = grouped_split(pairs, seed=20260818)

    placed = [i for ids in assignment.values() for i in ids]
    assert sorted(placed) == sorted(p.base_item_id for p in pairs)
    assert len(placed) == len(set(placed))


def test_judgebench_never_shares_a_cache_with_the_in_domain_run():
    """The block cache is keyed on (judge, context) only, so the paths must differ."""
    assert scores_dir("core20") != scores_dir("core20", "judgebench")
    assert pairs_path(20260818) != pairs_path(20260818, "judgebench")
    assert split_path(20260818) != split_path(20260818, "judgebench")


def test_unknown_dataset_names_are_refused():
    with pytest.raises(ValueError, match="unknown dataset"):
        scores_dir("core20", "arena-hard")
