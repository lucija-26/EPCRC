"""Rebuild the scored judge panel from the cached (judge, context) blocks.

Two seeds are involved and they are deliberately separate:

``pairs_seed``
    fixes *which* 100 items were scored and *what text* each judge saw, so it
    cannot be varied without re-running inference on the GPU;

``split_seed``
    fixes only the grouped FIT / CERT / TEST partition of those same items, so
    it can be varied for free.

That is what makes across-seed confidence bands a CPU-only exercise.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from epcrc.judge import JudgeResponses
from epcrc.prompts import REGISTERED_CONTEXTS
from epcrc.rewardbench import PRIMARY_SEED, read_pairs, stratified_subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SCORES = os.path.join(ROOT, "results", "smoke_core8", "scores")

SPLIT_SEEDS = [20260818, 20260819, 20260820, 20260821, 20260822]

# --------------------------------------------------------------------------
# the formal panels (plan sections 10 and 10.3)
# --------------------------------------------------------------------------

# Section 10.  Judge ids are fixed by the plan and must not be renumbered:
# every result file keys on them.  The families matter as much as the sizes,
# because C1 needs within-family pairs to have any redundancy to detect and C3
# needs the one-per-family baseline to be meaningful.
CORE20: Dict[str, str] = {
    "J01": "Qwen/Qwen2.5-3B-Instruct",
    "J02": "Qwen/Qwen2.5-7B-Instruct",
    "J03": "Qwen/Qwen2.5-14B-Instruct",
    "J04": "Qwen/Qwen3-4B",
    "J05": "Qwen/Qwen3-8B",
    "J06": "Qwen/Qwen3-14B",
    "J07": "meta-llama/Llama-3.2-3B-Instruct",
    "J08": "meta-llama/Llama-3.1-8B-Instruct",
    "J09": "google/gemma-3-4b-it",
    "J10": "google/gemma-3-12b-it",
    "J11": "microsoft/Phi-4-mini-instruct",
    "J12": "microsoft/phi-4",
    "J13": "microsoft/Phi-3.5-mini-instruct",
    "J14": "mistralai/Mistral-7B-Instruct-v0.3",
    "J15": "mistralai/Mistral-Nemo-Instruct-2407",
    "J16": "ibm-granite/granite-3.3-8b-instruct",
    "J17": "tiiuae/Falcon3-7B-Instruct",
    "J18": "tiiuae/Falcon3-10B-Instruct",
    "J19": "CohereLabs/aya-expanse-8b",
    "J20": "allenai/OLMo-2-1124-7B-Instruct",
}

# Section 10.3.  The smoke panel: one representative per family, spanning the
# size range.  Core-20 formal inference starts only after these eight pass
# every quality gate.
CORE8_IDS = ["J02", "J05", "J08", "J10", "J12", "J14", "J16", "J20"]
CORE8: Dict[str, str] = {j: CORE20[j] for j in CORE8_IDS}

PANELS: Dict[str, Dict[str, str]] = {"core8": CORE8, "core20": CORE20}

# Declared bf16 parameter counts, used for the disk and memory budget.  These
# are the sizes the plan's table states, not values read off the hub.
PARAMS_B: Dict[str, float] = {
    "J01": 3, "J02": 7, "J03": 14, "J04": 4, "J05": 8,
    "J06": 14, "J07": 3, "J08": 8, "J09": 4, "J10": 12,
    "J11": 4, "J12": 14, "J13": 4, "J14": 7, "J15": 12,
    "J16": 8, "J17": 7, "J18": 10, "J19": 8, "J20": 7,
}


def panel_weight_gb(judge_ids: Sequence[str]) -> float:
    """Approximate bf16 download size of a set of judges, in GB."""
    return 2.0 * sum(PARAMS_B[j] for j in judge_ids)


def scores_dir(panel_name: str) -> str:
    """Where a panel's cached (judge, context) blocks live."""
    if panel_name not in PANELS:
        raise ValueError(f"unknown panel {panel_name!r}; expected one of {sorted(PANELS)}")
    folder = "smoke_core8" if panel_name == "core8" else panel_name
    return os.path.join(ROOT, "results", folder, "scores")


class Panel:
    """The scored panel, split into FIT / CERT / TEST along the item axis.

    ``splits[name]`` is a `JudgeResponses` over the same judges and the same
    contexts, restricted to the items belonging to that split.
    """

    def __init__(
        self,
        judge_ids: List[str],
        model_ids: List[str],
        context_names: List[str],
        splits: Dict[str, JudgeResponses],
        n_items: Dict[str, int],
        item_groups: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.judge_ids = judge_ids
        self.model_ids = model_ids
        self.context_names = context_names
        self.splits = splits
        self.n_items = n_items
        # Per split, the base item each row came from, as contiguous integers.
        # One base item can yield several comparison pairs, and those pairs are
        # not independent, so the bootstrap has to resample base items and take
        # all of their pairs together (plan section 34.1).
        self.item_groups = item_groups or {}

    @property
    def N(self) -> int:
        return len(self.judge_ids)


def load_panel(
    seed: int = PRIMARY_SEED,
    scores_dir: str = SCORES,
    split_seed: Optional[int] = None,
) -> Panel:
    """Rebuild the response tensor, optionally re-splitting it under another seed.

    `seed` must be the seed the cached blocks were scored under; `split_seed`
    only re-partitions those cached items and defaults to `seed`.
    """
    split_seed = seed if split_seed is None else split_seed
    contexts = [c.name for c in REGISTERED_CONTEXTS]

    judge_ids = sorted({
        os.path.basename(p).split("__")[0]
        for p in os.listdir(scores_dir)
        if p.endswith(".json")
    })
    judge_ids = [
        j for j in judge_ids
        if all(os.path.exists(os.path.join(scores_dir, f"{j}__{c}.json"))
               for c in contexts)
    ]
    if len(judge_ids) < 2:
        raise RuntimeError(f"need >=2 judges with complete blocks, found {judge_ids}")

    blocks: Dict[Tuple[str, str], dict] = {}
    for j in judge_ids:
        for c in contexts:
            with open(os.path.join(scores_dir, f"{j}__{c}.json")) as handle:
                blocks[(j, c)] = json.load(handle)

    # Every block must cover the same pairs in the same order, otherwise the
    # judge axis would not line up item for item.
    reference = blocks[(judge_ids[0], contexts[0])]["pair_ids"]
    for key, block in blocks.items():
        if block["pair_ids"] != reference:
            raise RuntimeError(f"block {key} does not match the reference pair order")

    all_pairs = read_pairs(os.path.join(DATA, f"pairs_seed{seed}.jsonl"))
    pairs = stratified_subset(all_pairs, len(reference), seed=seed)
    if [p.pair_id for p in pairs] != reference:
        raise RuntimeError("cached blocks do not match the stratified subset for this seed")

    with open(os.path.join(DATA, f"split_seed{split_seed}.json")) as handle:
        membership = {
            i: name
            for name, ids in json.load(handle)["splits"].items()
            for i in ids
        }
    where = np.array([membership[p.base_item_id] for p in pairs])

    base_items = np.array([p.base_item_id for p in pairs])

    splits: Dict[str, JudgeResponses] = {}
    n_items: Dict[str, int] = {}
    item_groups: Dict[str, np.ndarray] = {}
    for name in ("FIT", "CERT", "TEST"):
        rows = np.flatnonzero(where == name)
        if rows.size == 0:
            raise RuntimeError(f"split {name} has no scored items")
        tensor = [
            np.stack(
                [np.asarray(blocks[(j, c)]["probabilities"])[rows] for j in judge_ids],
                axis=1,
            )
            for c in contexts
        ]
        splits[name] = JudgeResponses(tensor, contexts)
        n_items[name] = int(rows.size)
        item_groups[name] = np.unique(base_items[rows], return_inverse=True)[1]

    model_ids = [blocks[(j, contexts[0])]["model_id"] for j in judge_ids]
    return Panel(judge_ids, model_ids, contexts, splits, n_items, item_groups)
