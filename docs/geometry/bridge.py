"""One figure: how EPCRC and the LLM judge panel are the same problem.

Writes docs/EPCRC_TO_JUDGES.png.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from doclib import INK, MUTED, RULE, ACCENT, ACCENT2, ACCENT3, ACCENT4, HULLFILL
from figs import to_xy, hull_poly, _poly, dot, simplex_ax, bare_ax

OUT = os.path.join(HERE, "..", "EPCRC_TO_JUDGES.png")

# The same cloud is drawn in panels (a) and (c): that is the whole point.
CLOUD = np.array([[0.10, 0.22], [0.52, 0.05], [0.93, 0.34], [0.80, 0.86],
                  [0.30, 0.92], [0.05, 0.60],
                  [0.42, 0.40], [0.60, 0.55], [0.35, 0.62]])
KEPT = np.arange(6)
INSIDE = np.arange(6, 9)


def cloud_panel(ax, title, sub, subcol):
    ax.set_title(title, fontsize=9.6, color=INK, pad=14, linespacing=1.7)
    ax.add_patch(_poly(hull_poly(CLOUD[KEPT]), fc=HULLFILL, ec=ACCENT, lw=1.3,
                       zorder=2))
    ax.scatter(CLOUD[KEPT, 0], CLOUD[KEPT, 1], s=46, c=ACCENT, zorder=5,
               edgecolors="white", lw=0.9)
    ax.scatter(CLOUD[INSIDE, 0], CLOUD[INSIDE, 1], s=46, c=ACCENT2, zorder=5,
               marker="D", edgecolors="white", lw=0.9)
    ax.text(0.50, -0.14, sub, fontsize=8.2, color=subcol, ha="center",
            transform=ax.transAxes)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, 1.12)
    ax.axis("off")


def vector_strip(ax, y, blocks, block_w, colors, top_label, bot_label,
                 group=1, gap=0.012):
    """Draw a row of coordinate cells; group cells into blocks of `group`."""
    x = 0.0
    for b in range(blocks):
        for g in range(group):
            ax.add_patch(Rectangle((x, y), block_w, 0.13,
                                   fc=colors[g], ec="white", lw=1.1,
                                   transform=ax.transAxes, clip_on=False,
                                   zorder=3))
            x += block_w
        if group > 1:
            ax.plot([x - group * block_w, x], [y - 0.035, y - 0.035],
                    color=MUTED, lw=0.9, transform=ax.transAxes,
                    clip_on=False, zorder=3)
            ax.text(x - group * block_w / 2, y - 0.075, "item %d" % (b + 1),
                    fontsize=6.6, color=MUTED, ha="center",
                    transform=ax.transAxes, clip_on=False)
        else:
            ax.text(x - block_w / 2, y - 0.075, "query %d" % (b + 1),
                    fontsize=6.6, color=MUTED, ha="center",
                    transform=ax.transAxes, clip_on=False)
        x += gap
    ax.text(0.0, y + 0.185, top_label, fontsize=8.4, color=INK,
            transform=ax.transAxes, clip_on=False, fontweight="bold")
    ax.text(x + 0.02, y + 0.065, bot_label, fontsize=9.0, color=INK,
            va="center", transform=ax.transAxes, clip_on=False)


def main():
    fig = plt.figure(figsize=(11.5, 7.6))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.965, "EPCRC and the judge panel are the same geometry",
             fontsize=14.5, color=INK, ha="center", fontweight="bold")
    fig.text(0.5, 0.933, "only the number of coordinates per query changes",
             fontsize=9.8, color=MUTED, ha="center", style="italic")

    # ---- (a) EPCRC cloud ------------------------------------------------
    axa = fig.add_axes([0.055, 0.545, 0.25, 0.30])
    cloud_panel(axa, "(a)  EPCRC: a model is a point\n"
                "coordinates = its prediction on each query",
                "red = inside the hull = replaceable", ACCENT2)

    # ---- (b) the bridge -------------------------------------------------
    axb = fig.add_axes([0.375, 0.545, 0.585, 0.33])
    axb.axis("off")
    axb.set_title("(b)  the only difference: what one query returns",
                  fontsize=9.6, color=INK, pad=6, loc="left")
    vector_strip(axb, 0.58, 5, 0.085, [ACCENT],
                 "EPCRC model", r"$\in\ \mathbb{R}^{n}$", group=1)
    vector_strip(axb, 0.10, 4, 0.038, [ACCENT3, ACCENT4, ACCENT2],
                 "judge", r"$\in\ \mathbb{R}^{3n}$", group=3)
    axb.text(0.0, 0.44, "one number per query", fontsize=7.4, color=MUTED,
             transform=axb.transAxes)
    axb.text(0.0, -0.04, "three numbers per item:  P(A better),  P(B better),"
             "  P(tie)", fontsize=7.4, color=MUTED, transform=axb.transAxes)

    # ---- (c) judge cloud ------------------------------------------------
    axc = fig.add_axes([0.055, 0.105, 0.25, 0.30])
    cloud_panel(axc, "(c)  judge panel: identical picture\n"
                "coordinates = its three probabilities on each item",
                "same hull, same replaceability test", ACCENT2)

    # ---- (d) zoom into one item ----------------------------------------
    axd = simplex_ax(fig, [0.345, 0.035, 0.31, 0.43],
                     title="(d)  zoom: one item at a time")
    P = np.array([[0.84, 0.09, 0.07], [0.13, 0.79, 0.08], [0.24, 0.20, 0.56],
                  [0.52, 0.36, 0.12], [0.30, 0.48, 0.22], [0.46, 0.28, 0.26]])
    hull = hull_poly(to_xy(P[:3]))
    axd.add_patch(_poly(hull, fc=HULLFILL, ec=ACCENT, lw=1.3, zorder=2))
    for p in P[:3]:
        dot(axd, p, ACCENT, s=40)
    for p in P[3:]:
        dot(axd, p, ACCENT2, s=34, marker="D")
    axd.text(0.5, -0.20, "each judge is a point in the triangle",
             fontsize=7.6, color=MUTED, ha="center")
    axd.text(0.5, -0.255, "the hull casts a polygon shadow here",
             fontsize=7.6, color=ACCENT, ha="center")

    # ---- (e) what carries over -----------------------------------------
    axe = fig.add_axes([0.685, 0.075, 0.285, 0.36])
    axe.axis("off")
    axe.set_title("(e)  what carries over, what changes", fontsize=9.6,
                  color=INK, pad=6, loc="left")
    rows = [("convex hull of the kept set", "same", ACCENT3),
            ("inside the hull = removable", "same", ACCENT3),
            ("simplex weights, one vector per target", "same", ACCENT3),
            ("forward / backward / k-swap", "same", ACCENT3),
            ("exact MILP + cuts", "same", ACCENT3),
            ("error metric", "abs  ->  total variation", ACCENT2),
            ("one fit per target", "worst case over contexts", ACCENT2),
            ("headline claim", "removals do not compose (C1)", ACCENT2)]
    y = 0.90
    for lab, val, col in rows:
        axe.text(0.0, y, lab, fontsize=7.8, color=INK, va="center",
                 transform=axe.transAxes)
        axe.text(1.0, y, val, fontsize=7.8, color=col, va="center",
                 ha="right", transform=axe.transAxes)
        axe.plot([0, 1], [y - 0.052, y - 0.052], color=RULE, lw=0.6,
                 transform=axe.transAxes)
        y -= 0.113

    # ---- connective arrows ---------------------------------------------
    for xy, xytext in [((0.372, 0.70), (0.318, 0.70)),
                       ((0.372, 0.245), (0.318, 0.245))]:
        fig.patches.append(FancyArrowPatch(
            xytext, xy, transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=13, lw=1.4, color=MUTED))

    fig.savefig(OUT, dpi=190, facecolor="white")
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
