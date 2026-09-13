"""The split seeds must re-partition one fixed item universe.

E1 needs confidence bands across split seeds (plan section 23).  Those bands are
only free of GPU cost if changing the split seed re-partitions the *same* scored
items rather than selecting different ones.  This pins that property.
"""

from __future__ import annotations

import json
import os

import pytest

from epcrc.panel import DATA, SCORES, SPLIT_SEEDS, load_panel
from epcrc.rewardbench import PRIMARY_SEED

SPLIT_FILES = {s: os.path.join(DATA, f"split_seed{s}.json") for s in SPLIT_SEEDS}

requires_splits = pytest.mark.skipif(
    not all(os.path.exists(p) for p in SPLIT_FILES.values()),
    reason="split files not built",
)
requires_scores = pytest.mark.skipif(
    not os.path.isdir(SCORES) or not os.listdir(SCORES),
    reason="Core-8 scores not present",
)


def _splits(seed: int) -> dict:
    with open(SPLIT_FILES[seed]) as handle:
        return json.load(handle)["splits"]


@requires_splits
def test_every_split_seed_covers_the_same_item_universe():
    universes = {s: set().union(*_splits(s).values()) for s in SPLIT_SEEDS}
    reference = universes[PRIMARY_SEED]
    for seed, universe in universes.items():
        assert universe == reference, f"split seed {seed} scores a different item set"


@requires_splits
def test_split_seeds_actually_disagree_about_membership():
    """Otherwise the 'bands' would be five copies of one number."""
    reference = set(_splits(PRIMARY_SEED)["FIT"])
    for seed in SPLIT_SEEDS:
        if seed == PRIMARY_SEED:
            continue
        assert set(_splits(seed)["FIT"]) != reference


@requires_splits
def test_splits_are_disjoint():
    for seed in SPLIT_SEEDS:
        parts = _splits(seed)
        fit, cert, test = (set(parts[k]) for k in ("FIT", "CERT", "TEST"))
        assert not (fit & cert) and not (fit & test) and not (cert & test)


@requires_scores
@requires_splits
def test_split_seed_defaults_to_the_pairs_seed():
    implicit = load_panel(PRIMARY_SEED)
    explicit = load_panel(PRIMARY_SEED, split_seed=PRIMARY_SEED)
    assert implicit.n_items == explicit.n_items


@requires_scores
@requires_splits
def test_resplitting_keeps_the_judges_and_the_item_total():
    base = load_panel(PRIMARY_SEED)
    total = sum(base.n_items.values())
    for seed in SPLIT_SEEDS:
        other = load_panel(PRIMARY_SEED, split_seed=seed)
        assert other.judge_ids == base.judge_ids
        assert other.context_names == base.context_names
        assert sum(other.n_items.values()) == total
