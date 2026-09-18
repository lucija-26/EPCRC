"""JudgeBench -> judged pairs, for the cross-benchmark transfer experiment (E7).

`ScalerLab/JudgeBench` ships two splits named after the model that produced the
responses being compared:

    split    rows   sources
    gpt       350   mmlu-pro (154), livebench (154), livecodebench (42)
    claude    270   mmlu-pro (154), livebench  (85), livecodebench (31)

Every row is already a finished A-versus-B comparison, so there is no negative
sampling and no orientation choice to make here: the dataset fixes which
response sits in slot A and `label` is either "A>B" or "B>A".  That is the whole
reason this benchmark was chosen for E7 -- the transfer set must not be built by
the same construction the panel was fitted under, or a transfer gap could be an
artefact of the pair builder rather than of the domain.

Two consequences to state wherever E7 is reported.

* **JudgeBench has no ties.**  The tie class is still predicted by every judge,
  but it is never the correct answer, so transfer numbers say nothing about how
  well the tie corner survives compression.
* `pair_id` is unique across both splits and one row is one base item, so the
  grouped split degenerates to an ordinary stratified split.  Nothing is lost:
  grouping only mattered on RewardBench 2 because one task produced several
  pairs.

Strata are (response model, source family), which keeps both the generator and
the subject area proportionally represented in FIT / CERT / TEST.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from epcrc.rewardbench import JudgedPair

DATASET_ID = "ScalerLab/JudgeBench"

SPLITS = ("gpt", "claude")

LABELS_SEEN = {"A>B": "A", "B>A": "B"}


def load_judge_bench(revision: Optional[str] = None) -> List[Tuple[str, Dict[str, object]]]:
    """Load both splits as (split name, row) pairs. Pin `revision` before inference."""
    from datasets import load_dataset

    data = load_dataset(DATASET_ID, revision=revision)
    return [(split, dict(row)) for split in SPLITS for row in data[split]]


def source_family(source: str) -> str:
    """Coarsen the 17 raw sources to the three benchmarks they come from.

    Seventeen strata over 620 rows would leave several of them too thin to split
    three ways, and the subject label inside MMLU-Pro is not the axis E7 is
    about.
    """
    return "mmlu-pro" if source.startswith("mmlu-pro") else source.split("-")[0]


def build_pairs(rows: Iterable[Tuple[str, Dict[str, object]]]) -> List[JudgedPair]:
    """Turn JudgeBench rows into `JudgedPair`s, one per row.

    No seed is taken because nothing here is random: the dataset already decided
    the pairing, the slot order and the label.
    """
    pairs: List[JudgedPair] = []

    for split, row in rows:
        label = str(row["label"])
        if label not in LABELS_SEEN:
            raise ValueError(f"unexpected JudgeBench label {label!r} in split {split}")

        pair_id = f"{split}/{row['pair_id']}"
        pairs.append(JudgedPair(
            pair_id=pair_id,
            base_item_id=pair_id,
            domain=f"{split}/{source_family(str(row['source']))}",
            is_tie=False,
            prompt=str(row["question"]),
            response_a=str(row["response_A"]),
            response_b=str(row["response_B"]),
            gold_label=LABELS_SEEN[label],
            weight=1.0,
            pair_index=0,
        ))

    return pairs


def coverage_summary(pairs: Sequence[JudgedPair]) -> Dict[str, object]:
    """Counts a reader needs to see before trusting an E7 transfer number."""
    per_domain: Dict[str, int] = {}
    for pair in pairs:
        per_domain[pair.domain] = per_domain.get(pair.domain, 0) + 1
    return {
        "dataset": DATASET_ID,
        "n_pairs": len(pairs),
        "n_tie_pairs": sum(1 for p in pairs if p.is_tie),
        "pairs_per_domain": dict(sorted(per_domain.items())),
        "gold_label_counts": {
            label: sum(1 for p in pairs if p.gold_label == label)
            for label in ("A", "B", "C")
        },
    }
