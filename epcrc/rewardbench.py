"""RewardBench 2 -> judged pairs, and the grouped stratified split.

Verified against `allenai/reward-bench-2` (test split, 1865 base tasks):

    subset        rows   shape
    Focus          495   1 chosen, 3 rejected
    Factuality     475   1 chosen, 3 rejected
    Safety         450   1 chosen, 3 rejected
    Math           183   1 chosen, 3 rejected
    Precise IF     160   1 chosen, 3 rejected
    Ties           102   51 "tied:*" with 2-26 equally acceptable chosen,
                         51 "ref:*" with a single chosen

Two schema facts drive the code.  First, the `id` field is *not* unique across
subsets -- Factuality and Precise IF both start at "0" -- so the grouping key is
``subset/id``.  Second, the Ties subset is not homogeneous: only the "tied:*"
rows carry several equally acceptable answers, so tie status is decided by
``len(chosen) >= 2`` rather than by the subset name.  The "ref:*" rows have one
correct answer and are ordinary preference items.

`additional_metadata` is None for every Ties row, so nothing may depend on it.

Every choice below is a deterministic function of (base item id, seed), never of
iteration order, so the same manifest is reproduced on any machine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DATASET_ID = "allenai/reward-bench-2"

# Predeclared split seeds from the execution plan; 20260818 is the primary.
PRIMARY_SEED = 20260818
ROBUSTNESS_SEEDS = (20260819, 20260820, 20260821, 20260822)
ALL_SEEDS = (PRIMARY_SEED,) + ROBUSTNESS_SEEDS

SPLIT_FRACTIONS = {"FIT": 0.50, "CERT": 0.25, "TEST": 0.25}
MAX_PAIRS_PER_TASK = 2


@dataclass(frozen=True)
class JudgedPair:
    """One A-versus-B comparison put to a judge.

    `gold_label` is "A", "B" or "C"; `weight` is 1/m_b so that a base task with
    several retained pairs does not count more than one with a single pair.
    `gold_is_first` records whether the preferred response sits in slot A before
    any order-swap intervention is applied.
    """

    pair_id: str
    base_item_id: str
    domain: str
    is_tie: bool
    prompt: str
    response_a: str
    response_b: str
    gold_label: str
    weight: float
    pair_index: int


def _digest(*parts: object) -> int:
    """Stable 64-bit integer from the string form of `parts`."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def base_item_id(row: Dict[str, object]) -> str:
    """Grouping key. Plain `id` collides across subsets, so qualify it."""
    return f"{row['subset']}/{row['id']}"


def is_tie_row(row: Dict[str, object]) -> bool:
    return len(row["chosen"]) >= 2


def load_reward_bench_2(revision: Optional[str] = None, split: str = "test"):
    """Load the raw dataset. Pin `revision` before formal inference."""
    from datasets import load_dataset

    return load_dataset(DATASET_ID, split=split, revision=revision)


def _select_deterministically(
    candidates: Sequence[str], k: int, *, key: str, seed: int
) -> List[int]:
    """Indices of the `k` candidates whose (seed, key, content) hash is smallest."""
    if len(candidates) <= k:
        return list(range(len(candidates)))
    ranked = sorted(
        range(len(candidates)),
        key=lambda i: _digest(seed, key, i, candidates[i]),
    )
    return sorted(ranked[:k])


def build_pairs(
    rows: Iterable[Dict[str, object]],
    seed: int = PRIMARY_SEED,
    max_pairs: int = MAX_PAIRS_PER_TASK,
) -> List[JudgedPair]:
    """Turn base tasks into at most `max_pairs` judged pairs each.

    Non-tie tasks pair the preferred response against deterministically chosen
    negatives.  Tie tasks pair two equally acceptable responses and carry gold
    label C, so the tie probability is a real target rather than a leftover.

    Which response occupies slot A is itself decided by hash, so the clean
    context does not put the preferred response first every time -- otherwise
    position bias would be indistinguishable from judge quality.
    """
    pairs: List[JudgedPair] = []

    for row in rows:
        key = base_item_id(row)
        domain = str(row["subset"])
        chosen = list(row["chosen"])
        rejected = list(row["rejected"])
        tie = is_tie_row(row)

        if tie:
            picks = _select_deterministically(chosen, 2 * max_pairs, key=key, seed=seed)
            couples = [
                (chosen[picks[i]], chosen[picks[i + 1]])
                for i in range(0, len(picks) - 1, 2)
            ][:max_pairs]
        else:
            if not chosen or not rejected:
                continue
            picks = _select_deterministically(rejected, max_pairs, key=key, seed=seed)
            couples = [(chosen[0], rejected[i]) for i in picks]

        if not couples:
            continue

        weight = 1.0 / len(couples)
        for index, (preferred, other) in enumerate(couples):
            first_is_preferred = _digest(seed, key, "orientation", index) % 2 == 0
            if first_is_preferred:
                response_a, response_b = preferred, other
                gold = "C" if tie else "A"
            else:
                response_a, response_b = other, preferred
                gold = "C" if tie else "B"

            pairs.append(JudgedPair(
                pair_id=f"{key}#{index}",
                base_item_id=key,
                domain=domain,
                is_tie=tie,
                prompt=str(row["prompt"]),
                response_a=response_a,
                response_b=response_b,
                gold_label=gold,
                weight=weight,
                pair_index=index,
            ))

    return pairs


def grouped_split(
    pairs: Sequence[JudgedPair], seed: int = PRIMARY_SEED
) -> Dict[str, List[str]]:
    """Assign base tasks to FIT / CERT / TEST, stratified and grouped.

    Splitting happens at the base-task level, never at the rendered-prompt
    level: every pair variant, orientation and intervention derived from one
    base task must land in the same split or the held-out error is optimistic.
    Strata are (domain, tie status) so both stay proportionally represented.
    """
    strata: Dict[Tuple[str, bool], List[str]] = {}
    for pair in pairs:
        strata.setdefault((pair.domain, pair.is_tie), [])
        if pair.base_item_id not in strata[(pair.domain, pair.is_tie)]:
            strata[(pair.domain, pair.is_tie)].append(pair.base_item_id)

    assignment: Dict[str, List[str]] = {name: [] for name in SPLIT_FRACTIONS}

    for stratum, item_ids in sorted(strata.items()):
        ordered = sorted(item_ids, key=lambda i: _digest(seed, "split", i))
        n = len(ordered)
        n_fit = int(round(n * SPLIT_FRACTIONS["FIT"]))
        n_cert = int(round(n * SPLIT_FRACTIONS["CERT"]))
        # Give any rounding remainder to TEST rather than silently dropping it.
        assignment["FIT"].extend(ordered[:n_fit])
        assignment["CERT"].extend(ordered[n_fit:n_fit + n_cert])
        assignment["TEST"].extend(ordered[n_fit + n_cert:])

    return {name: sorted(ids) for name, ids in assignment.items()}


def split_manifest(
    pairs: Sequence[JudgedPair],
    assignment: Dict[str, List[str]],
    seed: int,
) -> Dict[str, object]:
    """Checksummed record of one split, so results can be traced to their data."""
    per_split = {
        name: hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        for name, ids in assignment.items()
    }
    counts = {name: len(ids) for name, ids in assignment.items()}

    domains: Dict[str, Dict[str, int]] = {}
    membership = {i: name for name, ids in assignment.items() for i in ids}
    for pair in pairs:
        bucket = domains.setdefault(pair.domain, {name: 0 for name in assignment})
        bucket[membership[pair.base_item_id]] += 1

    return {
        "dataset": DATASET_ID,
        "seed": seed,
        "n_pairs": len(pairs),
        "n_base_items": len(membership),
        "n_tie_pairs": sum(1 for p in pairs if p.is_tie),
        "split_sizes": counts,
        "split_sha256": per_split,
        "pairs_per_domain_and_split": domains,
    }


def write_pairs(pairs: Sequence[JudgedPair], path: str) -> None:
    """One JSON object per line, so the scorer can stream it."""
    with open(path, "w") as handle:
        for pair in pairs:
            handle.write(json.dumps(asdict(pair)) + "\n")


def read_pairs(path: str) -> List[JudgedPair]:
    with open(path) as handle:
        return [JudgedPair(**json.loads(line)) for line in handle if line.strip()]


def stratified_subset(
    pairs: Sequence[JudgedPair], n: int, seed: int = PRIMARY_SEED
) -> List[JudgedPair]:
    """A deterministic domain-balanced subset, for smoke runs and audits."""
    by_domain: Dict[str, List[JudgedPair]] = {}
    for pair in pairs:
        by_domain.setdefault(pair.domain, []).append(pair)

    quota = max(1, n // max(1, len(by_domain)))
    ranked = {
        domain: sorted(members, key=lambda p: _digest(seed, "subset", p.pair_id))
        for domain, members in by_domain.items()
    }

    picked: List[JudgedPair] = []
    for domain in sorted(ranked):
        picked.extend(ranked[domain][:quota])

    # The floor divide leaves a remainder; fill it from the same per-domain
    # orderings so the subset has exactly n items and stays deterministic.
    leftovers = [p for domain in sorted(ranked) for p in ranked[domain][quota:]]
    leftovers.sort(key=lambda p: _digest(seed, "subset_fill", p.pair_id))
    picked.extend(leftovers[:max(0, n - len(picked))])

    return sorted(picked, key=lambda p: _digest(seed, "subset_order", p.pair_id))[:n]
