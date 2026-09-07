"""Contact-sheet renderer: draws every figure in figs.py to _png/ for review."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import doclib  # noqa: F401  (sets rcParams)
import figs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_png")
os.makedirs(OUT, exist_ok=True)

SIZES = {
    "fig_cheatsheet": 4.6,
    "fig_certification": 2.4,
    "fig_downstream": 2.4,
}

names = [n for n in dir(figs) if n.startswith("fig_")]
for n in sorted(names):
    func = getattr(figs, n)
    h = SIZES.get(n, 2.5)
    fig = plt.figure(figsize=(6.7, h))
    fig.patch.set_facecolor("white")
    func(fig, [0.0, 0.0, 1.0, 1.0])
    fig.savefig(os.path.join(OUT, n + ".png"), dpi=150)
    plt.close(fig)
    print("ok", n)
print("done:", len(names), "figures")
