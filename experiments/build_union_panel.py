"""Assemble the union panel's block cache from the two panels it is made of.

The union is never scored directly.  This links every ``{judge}__{context}.json``
of Core-20 and of the second panel into ``results/core40/scores`` after
checking that all blocks cover the same pairs in the same order.  The two
source directories are only read.

Usage:  python experiments/build_union_panel.py [--copy]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from epcrc.panel import PANELS, scores_dir  # noqa: E402
from epcrc.prompts import REGISTERED_CONTEXTS  # noqa: E402


def collect(sources):
    """{file name: source path} for every complete judge, with the pair check."""
    contexts = [c.name for c in REGISTERED_CONTEXTS]
    reference = None
    files, incomplete = {}, []
    for name in sources:
        folder = scores_dir(name)
        for judge in PANELS[name]:
            paths = {c: os.path.join(folder, f"{judge}__{c}.json") for c in contexts}
            if not all(os.path.exists(p) for p in paths.values()):
                incomplete.append(judge)
                continue
            for c, path in paths.items():
                with open(path) as handle:
                    block = json.load(handle)
                if block["model_id"] != PANELS[name][judge]:
                    raise RuntimeError(f"{path}: scored with {block['model_id']}, "
                                       f"registry says {PANELS[name][judge]}")
                if reference is None:
                    reference = block["pair_ids"]
                elif block["pair_ids"] != reference:
                    raise RuntimeError(f"{path} does not cover the reference pairs in order")
                files[os.path.basename(path)] = path
    return files, incomplete, len(reference or [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", default=["core20", "ext20"])
    ap.add_argument("--target", default="core40")
    ap.add_argument("--copy", action="store_true", help="copy instead of symlinking")
    a = ap.parse_args()
    if a.target in a.sources:
        ap.error("the target must not be one of the sources")

    files, incomplete, n_pairs = collect(a.sources)
    out = scores_dir(a.target)
    os.makedirs(out, exist_ok=True)
    for name, src in sorted(files.items()):
        dst = os.path.join(out, name)
        if os.path.lexists(dst):
            os.remove(dst)
        if a.copy:
            shutil.copyfile(src, dst)
        else:
            os.symlink(os.path.relpath(src, out), dst)
    judges = sorted({n.split("__")[0] for n in files})
    print(f"{len(judges)} judges, {len(files)} blocks, {n_pairs} pairs -> {out}")
    if incomplete:
        print(f"skipped (blocks missing): {sorted(incomplete)}")


if __name__ == "__main__":
    main()
