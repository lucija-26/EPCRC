"""Re-run backward and k-swap on separator instances found by
experiments/experiment_forward_beats_backward.py.

Usage:
    python experiments/experiment_compare_separators.py [which]

`which` can be:
    smallest   - re-run on the single smallest separator (default)
    all        - re-run on every separator in results/forward_beats_backward

Output:
  results/forward_beats_backward/comparisons.json
  results/forward_beachd_backward/comparison_<idx>.png
"""
from __future__ import annotations

import json
import os
import sys
from typing import List

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from epcrc.coverage import CoverageFunctional
from epcrc.pruning import BackwardEliminationPruner, PriorityQueuePruner, ForwardSelectionPruner

IN_PATH = os.path.join(_root, "results", "forward_beats_backward", "separators.json")
OUT_DIR = os.path.join(_root, "results", "forward_beats_backward")

KSWAP_MAX_K = 2


def plot_instance(pts: np.ndarray, kept_sets: dict, path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, d = pts.shape[0], pts.shape[1]
    if d != 2:
        print(f"[plot] skipping non-2D instance (d={d}) -> {path}")
        return

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(pts[:, 0], pts[:, 1], s=40, c="lightgray", zorder=1, label="dropped")
    for i, (x, y) in enumerate(pts):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=9)

    colors = {"forward": "blue", "backward": "red", "kswap": "green"}
    markers = {"forward": "o", "backward": "s", "kswap": "D"}

    for alg, idxs in kept_sets.items():
        if len(idxs) == 0:
            continue
        arr = np.array(sorted(idxs))
        ax.scatter(
            pts[arr, 0], pts[arr, 1], s=120, c=colors.get(alg, "k"), marker=markers.get(alg, "o"),
            label=f"{alg} kept ({len(idxs)})", zorder=3, edgecolors="k"
        )

    ax.set_title(title)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"[plot] wrote {path}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "smallest"
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(IN_PATH, "r") as fh:
        data = json.load(fh)

    if which == "smallest":
        items = [data["smallest"]]
    elif which == "all":
        items = data.get("separators", [])
    else:
        raise ValueError("which must be 'smallest' or 'all'")

    results: List[dict] = []
    for idx, inst in enumerate(items):
        pts = np.array(inst.get("points"))
        N = pts.shape[0]
        Y = pts.T.copy()
        model_names = [f"m{i}" for i in range(N)]

        cov = CoverageFunctional(Y_fit=Y, Y_eval=Y, model_names=model_names, metric="mean_abs")

        gamma = float(inst["gamma"])

        fwd = ForwardSelectionPruner(cov, gamma).run()
        bwd = BackwardEliminationPruner(cov, gamma).run()
        pq = PriorityQueuePruner(cov, gamma, max_swap_k=KSWAP_MAX_K).run()

        rec = {
            "index": idx,
            "seed": inst.get("seed"),
            "N": inst.get("N"),
            "dim": inst.get("dim"),
            "gamma": gamma,
            "forward": sorted(int(x) for x in fwd.kept_set),
            "backward": sorted(int(x) for x in bwd.kept_set),
            "kswap": sorted(int(x) for x in pq.kept_set),
            "E_forward": float(fwd.coverage),
            "E_backward": float(bwd.coverage),
            "E_kswap": float(pq.coverage),
            "n_forward": int(len(fwd.kept_set)),
            "n_backward": int(len(bwd.kept_set)),
            "n_kswap": int(len(pq.kept_set)),
        }
        results.append(rec)

        title = f"seed={rec['seed']} N={rec['N']} gamma={gamma:.3f}"
        png = os.path.join(OUT_DIR, f"comparison_{idx}.png")
        plot_instance(pts, {"forward": rec["forward"], "backward": rec["backward"], "kswap": rec["kswap"]}, png, title)

    out_path = os.path.join(OUT_DIR, "comparisons.json")
    with open(out_path, "w") as fh:
        json.dump({"config": data.get("config", {}), "results": results}, fh, indent=2)

    print(f"[json] wrote {out_path}")


if __name__ == "__main__":
    main()
