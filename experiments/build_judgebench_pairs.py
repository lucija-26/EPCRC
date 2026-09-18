"""Convert JudgeBench into judged pairs and write the split manifests (E7).

Produces, under data/:

    judgebench_pairs_seed<seed>.jsonl   the judged pairs
    judgebench_split_seed<seed>.json    FIT / CERT / TEST ids plus checksums

The pairs themselves do not depend on the seed -- JudgeBench fixes the pairing,
the slot order and the label -- so the file is written once per seed only so
that the scorer and the panel loader can key on a seed exactly as they do for
RewardBench 2.  Only the FIT / CERT / TEST partition actually varies.

On the E7 side of the transfer experiment FIT is the calibration pool the
refitted weights may look at and TEST is the held-out transfer set.  Nothing
here calls a model, so it is cheap to re-run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc.judgebench import DATASET_ID, build_pairs, coverage_summary, load_judge_bench
from epcrc.prompts import protocol_hashes
from epcrc.rewardbench import (
    ALL_SEEDS,
    PRIMARY_SEED,
    grouped_split,
    split_manifest,
    write_pairs,
)

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=[PRIMARY_SEED])
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--revision", default=None, help="pin a dataset revision")
    args = parser.parse_args()

    seeds = list(ALL_SEEDS) if args.all_seeds else args.seeds
    os.makedirs(DATA, exist_ok=True)

    rows = load_judge_bench(revision=args.revision)
    pairs = build_pairs(rows)
    summary = coverage_summary(pairs)
    print(f"loaded {len(rows)} rows from {DATASET_ID}")
    print("response models:", dict(Counter(split for split, _ in rows)))
    print("pairs per domain:", summary["pairs_per_domain"])
    print("gold labels:", summary["gold_label_counts"])

    for seed in seeds:
        assignment = grouped_split(pairs, seed=seed)
        manifest = split_manifest(pairs, assignment, seed, dataset=DATASET_ID)
        manifest["protocol_sha256"] = protocol_hashes()
        manifest["splits"] = assignment
        manifest["coverage"] = summary

        write_pairs(pairs, os.path.join(DATA, f"judgebench_pairs_seed{seed}.jsonl"))
        with open(os.path.join(DATA, f"judgebench_split_seed{seed}.json"), "w") as handle:
            json.dump(manifest, handle, indent=2)

        sizes = manifest["split_sizes"]
        print(
            f"seed {seed}: {len(pairs)} pairs over {manifest['n_base_items']} items "
            f"-> FIT {sizes['FIT']} / CERT {sizes['CERT']} / TEST {sizes['TEST']}"
        )

    print(f"\nwrote {len(seeds)} seed(s) to {DATA}")


if __name__ == "__main__":
    main()
