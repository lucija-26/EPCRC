"""Task 2: find a SEPARATING instance where forward < backward.

The headline finding (HANDOFF Sec.5) is that backward usually beats forward.  But
backward is only a *removal-local* optimum: at gamma > 0 it can get trapped (no
single deletion stays feasible) at a set strictly larger than what forward's
build-from-empty path lands on.  This script random-searches tiny synthetic
ecosystems for such (seed, gamma) pairs, i.e. where

    |forward result| < |backward result|.

(At gamma = 0 backward is optimal, so we only sweep gamma > 0.)

Setup: N in [8..12] random points in R^2 or R^3, Y_fit = Y_eval = points (the
noiseless geometry, no honest split), gamma on a grid scaled to each instance's
mean pairwise distance.  No UTD19 bundle needed -- fully self-contained.

Run from project root:
    python experiments/experiment_forward_beats_backward.py [n_seeds]

Outputs:
    results/forward_beats_backward/separators.json   (all separators + smallest)
    results/forward_beats_backward/smallest_instance.png   (2D scatter)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from epcrc.coverage import CoverageFunctional
from epcrc.pruning import BackwardEliminationPruner, ForwardSelectionPruner

OUT_DIR = os.path.join(_root, "results", "forward_beats_backward")

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
N_RANGE = (8, 9, 10, 11, 12)
DIMS = (2, 3)
METRIC = "mean_abs"
N_GAMMA = 25  # gamma grid resolution per instance


def mean_pairwise_dist(pts: np.ndarray) -> float:
    """pts: (N, d). Mean Euclidean distance between distinct points."""
    n = pts.shape[0]
    tot, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += float(np.linalg.norm(pts[i] - pts[j]))
            cnt += 1
    return tot / max(cnt, 1)


def search():
    rng_master = np.random.default_rng(0)
    separators = []  # each: dict(seed, dim, N, gamma, n_fwd, n_bwd, fwd, bwd)

    for dim in DIMS:
        for N in N_RANGE:
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
                pts = rng.standard_normal((N, dim))  # (N, d)
                Y = pts.T.copy()  # (d, N): models are columns
                cov = CoverageFunctional(
                    Y_fit=Y, Y_eval=Y, model_names=[f"m{i}" for i in range(N)], metric=METRIC
                )

                scale = mean_pairwise_dist(pts)
                # gamma > 0 grid, scaled to instance; skip the degenerate ends.
                gammas = np.linspace(0.02, 0.6, N_GAMMA) * scale

                for g in gammas:
                    g = float(g)
                    fwd = ForwardSelectionPruner(cov, g).run()
                    bwd = BackwardEliminationPruner(cov, g).run()
                    n_f, n_b = len(fwd.kept_set), len(bwd.kept_set)
                    # both must be feasible (they are by construction); record gap
                    if n_f < n_b:
                        separators.append(
                            dict(
                                seed=int(seed),
                                dim=int(dim),
                                N=int(N),
                                gamma=g,
                                gamma_over_scale=g / scale,
                                n_fwd=int(n_f),
                                n_bwd=int(n_b),
                                gap=int(n_b - n_f),
                                fwd=sorted(int(x) for x in fwd.kept_set),
                                bwd=sorted(int(x) for x in bwd.kept_set),
                                E_fwd=float(fwd.coverage),
                                E_bwd=float(bwd.coverage),
                                points=pts.tolist(),
                            )
                        )

    return separators


def pick_smallest(separators):
    """Smallest = fewest models N, then largest gap, then dim=2 (plottable)."""
    if not separators:
        return None
    return sorted(
        separators,
        key=lambda s: (s["N"], -s["gap"], s["dim"], s["gamma"]),
    )[0]


def plot_instance(inst, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.array(inst["points"])
    if pts.shape[1] != 2:
        print(f"[plot] smallest instance is {pts.shape[1]}D; skipping scatter.")
        return

    fwd, bwd = set(inst["fwd"]), set(inst["bwd"])
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # all points
    ax.scatter(pts[:, 0], pts[:, 1], s=40, c="lightgray", zorder=1, label="dropped")
    for i, (x, y) in enumerate(pts):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=9)

    # backward-kept (trapped, larger): big hollow red circles
    b = np.array(sorted(bwd))
    ax.scatter(
        pts[b, 0], pts[b, 1], s=320, facecolors="none", edgecolors="red",
        linewidths=2.0, zorder=2, label=f"backward kept ({len(bwd)})",
    )
    # forward-kept (smaller): filled blue
    f = np.array(sorted(fwd))
    ax.scatter(
        pts[f, 0], pts[f, 1], s=70, c="blue", zorder=3, label=f"forward kept ({len(fwd)})",
    )

    ax.set_title(
        f"forward < backward  (N={inst['N']}, dim={inst['dim']}, "
        f"gamma={inst['gamma']:.3f}={inst['gamma_over_scale']:.2f}*scale)\n"
        f"|forward|={inst['n_fwd']}  <  |backward|={inst['n_bwd']}   "
        f"(backward got trapped)"
    )
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"[plot] wrote {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Searching {len(DIMS)}*{len(N_RANGE)}*{N_SEEDS} instances x {N_GAMMA} gammas ...")
    separators = search()
    print(f"Found {len(separators)} (instance, gamma) separators with |fwd| < |bwd|.")

    smallest = pick_smallest(separators)
    out = {
        "config": dict(n_seeds=N_SEEDS, N_range=list(N_RANGE), dims=list(DIMS),
                       metric=METRIC, n_gamma=N_GAMMA),
        "n_separators": len(separators),
        "smallest": smallest,
        # keep the full list lightweight: drop the bulky points array except for smallest
        "separators": [{k: v for k, v in s.items() if k != "points"} for s in separators],
    }
    out_path = os.path.join(OUT_DIR, "separators.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[json] wrote {out_path}")

    if smallest is not None:
        print("\n=== SMALLEST SEPARATING INSTANCE ===")
        print(f"  N={smallest['N']} dim={smallest['dim']} seed={smallest['seed']}")
        print(f"  gamma={smallest['gamma']:.4f}  ({smallest['gamma_over_scale']:.3f} * mean_pairwise)")
        print(f"  |forward|={smallest['n_fwd']}  kept={smallest['fwd']}  E={smallest['E_fwd']:.4f}")
        print(f"  |backward|={smallest['n_bwd']} kept={smallest['bwd']}  E={smallest['E_bwd']:.4f}")
        plot_instance(smallest, os.path.join(OUT_DIR, "smallest_instance.png"))
    else:
        print("No separator found -- widen the search (more seeds / finer gamma).")


if __name__ == "__main__":
    main()
