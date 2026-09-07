"""Convert RewardBench 2 into judged pairs and write the split manifests.

Produces, under data/:

    pairs_seed<seed>.jsonl     the judged pairs for one seed
    split_seed<seed>.json      FIT / CERT / TEST base-item ids plus checksums

Run once per seed; the primary seed is the one the paper tables use.  Nothing
here calls a model, so it is cheap to re-run and the manifests are the record
that ties every later result back to a specific slice of data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc.prompts import protocol_hashes
from epcrc.rewardbench import (
    ALL_SEEDS,
    PRIMARY_SEED,
    build_pairs,
    grouped_split,
    load_reward_bench_2,
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

    rows = list(load_reward_bench_2(revision=args.revision))
    print(f"loaded {len(rows)} base tasks from RewardBench 2")
    print("subsets:", dict(Counter(r["subset"] for r in rows)))

    for seed in seeds:
        pairs = build_pairs(rows, seed=seed)
        assignment = grouped_split(pairs, seed=seed)
        manifest = split_manifest(pairs, assignment, seed)
        manifest["protocol_sha256"] = protocol_hashes()
        manifest["splits"] = assignment

        pairs_path = os.path.join(DATA, f"pairs_seed{seed}.jsonl")
        split_path = os.path.join(DATA, f"split_seed{seed}.json")
        write_pairs(pairs, pairs_path)
        with open(split_path, "w") as handle:
            json.dump(manifest, handle, indent=2)

        sizes = manifest["split_sizes"]
        print(
            f"seed {seed}: {len(pairs)} pairs "
            f"({manifest['n_tie_pairs']} tie) over {manifest['n_base_items']} tasks "
            f"-> FIT {sizes['FIT']} / CERT {sizes['CERT']} / TEST {sizes['TEST']}"
        )

    print(f"\nwrote {len(seeds)} seed(s) to {DATA}")


if __name__ == "__main__":
    main()
