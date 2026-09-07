"""Figures for the judge-panel geometry companion.

Everything is drawn in the 2-simplex of three-class judge outputs
(A better / B better / Tie) so that the reader can hold one picture in
their head: **a judge is a point in a triangle**.
"""

import itertools
import os

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from doclib import (ACCENT, ACCENT2, ACCENT3, ACCENT4, HULLFILL, INK, MUTED,
                    RULE)

# ---------------------------------------------------------------------------
# simplex geometry
# ---------------------------------------------------------------------------
S3 = np.sqrt(3) / 2.0
VERT = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3]])   # rows: A, B, T
_M = np.vstack([VERT.T, np.ones(3)])
_MINV = np.linalg.inv(_M)


def to_xy(p):
    return np.asarray(p, float) @ VERT


def to_bary(xy):
    xy = np.atleast_2d(np.asarray(xy, float))
    rhs = np.vstack([xy.T, np.ones(len(xy))])
    out = (_MINV @ rhs).T
    return out[0] if out.shape[0] == 1 else out


def tv(p, q):
    """Total variation on the 3-simplex == max coordinate deviation."""
    return float(np.max(np.abs(np.asarray(p) - np.asarray(q))))


def dist_to_hull(p, Q):
    """min_{w in simplex} TV(p, sum_j w_j Q_j); returns (value, w)."""
    Q = np.atleast_2d(np.asarray(Q, float))
    m = len(Q)
    # variables [w_1..w_m, t];  |p_k - (Q^T w)_k| <= t
    A = np.zeros((6, m + 1))
    b = np.zeros(6)
    for k in range(3):
        A[2 * k, :m] = -Q[:, k]
        A[2 * k, m] = -1.0
        b[2 * k] = -p[k]
        A[2 * k + 1, :m] = Q[:, k]
        A[2 * k + 1, m] = -1.0
        b[2 * k + 1] = p[k]
    Aeq = np.zeros((1, m + 1))
    Aeq[0, :m] = 1.0
    c = np.zeros(m + 1)
    c[m] = 1.0
    r = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=[1.0],
                bounds=[(0, None)] * m + [(0, None)], method="highs")
    return float(r.x[m]), r.x[:m]


def coverage(S, P):
    """Worst-target reconstruction error of retained set S (indices)."""
    if len(S) == 0:
        return 1.0
    Q = P[list(S)]
    return max(dist_to_hull(P[i], Q)[0] for i in range(len(P)))


_DISK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache.npz")
_MEM = (dict(np.load(_DISK, allow_pickle=True)["d"].item())
        if os.path.exists(_DISK) else {})


def frontier(P, ks, tag):
    """Exact best-subset coverage for each k in ks, cached on disk."""
    key = "%s|%s" % (tag, ",".join(map(str, ks)))
    if key in _MEM:
        return np.array(_MEM[key])
    out = [min(coverage(S, P)
               for S in itertools.combinations(range(len(P)), int(k)))
           for k in ks]
    _MEM[key] = out
    np.savez(_DISK, d=_MEM)
    return np.array(out)


def poly_intersect(p, q):
    """Intersection of two convex polygons (any vertex order)."""
    p, q = _ccw(p), _ccw(q)
    out = p
    for i in range(len(q)):
        a, b = q[i], q[(i + 1) % len(q)]
        d = b - a
        nrm = np.array([d[1], -d[0]])          # outward normal of a CCW edge
        out = clip_poly(out, lambda xy, a=a, n=nrm: n @ (xy - a))
        if len(out) == 0:
            break
    return out


def _ccw(p):
    p = np.asarray(p, float)
    area = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    return p if area > 0 else p[::-1]


def clip_poly(poly, fn):
    """Sutherland-Hodgman clip of polygon to {fn(xy) <= 0}."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        fa, fb = fn(a), fn(b)
        if fa <= 0:
            out.append(a)
        if (fa > 0) != (fb > 0):
            t = fa / (fa - fb)
            out.append(a + t * (b - a))
    return np.array(out) if out else np.zeros((0, 2))


TRI = VERT.copy()


def tv_ball(p, eps, poly=None):
    """Polygon {q in simplex : |q_k - p_k| <= eps} in xy coordinates."""
    poly = TRI.copy() if poly is None else np.asarray(poly, float)
    for k in range(3):
        for sgn in (1.0, -1.0):
            def fn(xy, k=k, sgn=sgn):
                lam = to_bary(xy)
                return sgn * (lam[k] - p[k]) - eps
            poly = clip_poly(poly, fn)
            if len(poly) == 0:
                return poly
    return poly


def hull_poly(pts):
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    try:
        h = ConvexHull(pts)
        return pts[h.vertices]
    except Exception:
        return pts


# ---------------------------------------------------------------------------
# drawing primitives
# ---------------------------------------------------------------------------
def panels(fig, rect, n, pad=0.012, widths=None):
    x0, y0, w, h = rect
    widths = widths or [1.0] * n
    tot = sum(widths)
    gaps = pad * (n - 1)
    out, x = [], x0
    for wi in widths:
        ww = (w - gaps) * wi / tot
        out.append([x, y0, ww, h])
        x += ww + pad
    return out


CAPY = -0.185


def simplex_ax(fig, rect, labels=True, lw=1.1, pad=0.075, title=None,
               vlab=("A", "B", "Tie")):
    ax = fig.add_axes(rect)
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.set_xlim(-pad, 1 + pad)
    ax.set_ylim(-0.26, S3 + pad + 0.04)
    ax.axis("off")
    ax.add_patch(_poly(TRI, fc="none", ec=INK, lw=lw, zorder=3))
    if labels:
        off = [(-0.035, -0.055), (0.035, -0.055), (0.0, 0.035)]
        ha = ["right", "left", "center"]
        va = ["top", "top", "bottom"]
        for v, o, h_, v_, t in zip(VERT, off, ha, va, vlab):
            ax.text(v[0] + o[0], v[1] + o[1], t, fontsize=7.6, color=INK,
                    ha=h_, va=v_, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=8.2, color=INK, pad=3.5)
    return ax


def _poly(pts, **kw):
    from matplotlib.patches import Polygon
    return Polygon(np.asarray(pts), closed=True, **kw)


def dot(ax, p, color=ACCENT, s=26, label=None, dxy=(0.015, 0.02), fs=7.0,
        marker="o", zorder=6, ec="white", lw=0.7, ha="left", va="bottom"):
    xy = to_xy(p) if len(np.shape(p)) == 1 and len(p) == 3 else np.asarray(p)
    ax.scatter([xy[0]], [xy[1]], s=s, c=color, marker=marker, zorder=zorder,
               edgecolors=ec, linewidths=lw)
    if label:
        ax.text(xy[0] + dxy[0], xy[1] + dxy[1], label, fontsize=fs,
                color=color, ha=ha, va=va, zorder=zorder + 1)
    return xy


def bare_ax(fig, rect, title=None, ins=(0.052, 0.034, 0.014, 0.026)):
    x0, y0, w, h = rect
    l, b, r, t = ins
    ax = fig.add_axes([x0 + l, y0 + b, w - l - r, h - b - t])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=6.6, length=2.5, width=0.6)
    if title:
        ax.set_title(title, fontsize=8.2, color=INK, pad=3.5)
    return ax


# ---------------------------------------------------------------------------
# 1. the simplex itself
# ---------------------------------------------------------------------------
def fig_simplex_intro(fig, rect):
    r1, r2 = panels(fig, rect, 2, pad=0.02, widths=[1.0, 1.05])

    ax = fig.add_axes(r1, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=32)
    ax.set_axis_off()
    for e in np.eye(3):
        ax.plot(*zip([0, 0, 0], 1.22 * e), color=MUTED, lw=0.8)
    tri = np.eye(3)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection([tri], facecolor=HULLFILL,
                                         edgecolor=ACCENT, lw=1.2, alpha=0.45))
    p = np.array([0.55, 0.30, 0.15])
    ax.plot(*zip([0, 0, 0], p), color=ACCENT2, lw=0.7, ls="--")
    ax.scatter(*p, color=ACCENT2, s=34, depthshade=False, zorder=10)
    for k, e in enumerate(np.eye(3)):
        q = p.copy()
        q[k] = 0
        ax.plot(*zip(p, q), color=ACCENT2, lw=0.7, ls=":")
    ax.text(1.32, 0, 0, "$p_A$", fontsize=7.5, color=INK)
    ax.text(0, 1.32, 0, "$p_B$", fontsize=7.5, color=INK)
    ax.text(0, 0, 1.30, "$p_T$", fontsize=7.5, color=INK)
    ax.text(p[0] + 0.06, p[1], p[2] + 0.20, "$p$", fontsize=8.5, color=ACCENT2)
    ax.set_xlim(0, 1.3); ax.set_ylim(0, 1.3); ax.set_zlim(0, 1.3)
    ax.set_title("the plane $p_A+p_B+p_T=1$", fontsize=8.2, color=INK, pad=-4)

    ax = simplex_ax(fig, r2, title="the same triangle, seen face-on")
    for k in range(3):
        for lev in np.arange(0.2, 1.0, 0.2):
            seg = clip_poly(TRI, lambda xy, k=k, lev=lev: to_bary(xy)[k] - lev)
            seg2 = clip_poly(TRI, lambda xy, k=k, lev=lev: lev - to_bary(xy)[k])
            if len(seg2) >= 2:
                pts = np.array([q for q in seg2
                                if abs(to_bary(q)[k] - lev) < 1e-9])
                if len(pts) == 2:
                    ax.plot(pts[:, 0], pts[:, 1], color=RULE, lw=0.45, zorder=1)
    p = np.array([0.55, 0.30, 0.15])
    xy = dot(ax, p, ACCENT2, s=34, label="$p=(0.55,\\ 0.30,\\ 0.15)$", fs=7.2,
             dxy=(-0.022, -0.022), ha="right", va="top")
    feet = [to_xy([0, p[1] / (p[1] + p[2]), p[2] / (p[1] + p[2])]),
            to_xy([p[0] / (p[0] + p[2]), 0, p[2] / (p[0] + p[2])]),
            to_xy([p[0] / (p[0] + p[1]), p[1] / (p[0] + p[1]), 0])]
    for f in feet:
        ax.plot([xy[0], f[0]], [xy[1], f[1]], color=ACCENT2, lw=0.6, ls=":")
    c = to_xy([1 / 3, 1 / 3, 1 / 3])
    ax.scatter([c[0]], [c[1]], s=16, c=MUTED, zorder=5)
    ax.text(c[0] + 0.03, c[1] + 0.02, "centroid:\nmaximal uncertainty",
            fontsize=6.3, color=MUTED, ha="left", va="bottom")
    ax.text(0.5, CAPY, "each coordinate = scaled distance to the opposite edge",
            fontsize=6.5, color=MUTED, ha="center")


def fig_landmarks(fig, rect):
    r1, r2 = panels(fig, rect, 2, pad=0.022)
    ax = simplex_ax(fig, r1, title="where judges live")
    c = to_xy([1 / 3, 1 / 3, 1 / 3])
    # the three argmax regions: vertex k, the two adjacent edge midpoints,
    # and the centroid
    mid = {(0, 1): to_xy([.5, .5, 0]), (0, 2): to_xy([.5, 0, .5]),
           (1, 2): to_xy([0, .5, .5])}
    for k, col in enumerate([ACCENT, ACCENT2, ACCENT3]):
        others = [j for j in range(3) if j != k]
        ring = [VERT[k], mid[tuple(sorted((k, others[0])))], c,
                mid[tuple(sorted((k, others[1])))]]
        ax.add_patch(_poly(np.array(ring), fc=col, ec="none", alpha=0.09,
                           zorder=1))
    for m in mid.values():
        ax.plot([c[0], m[0]], [c[1], m[1]], color=RULE, lw=0.6, zorder=2)
    people = [([0.92, 0.05, 0.03], "decisive, pro-A", (0.0, 0.035), "center"),
              ([0.62, 0.33, 0.05], "confident", (0.028, 0.012), "left"),
              ([0.40, 0.36, 0.24], "hedger", (0.028, 0.0), "left"),
              ([0.34, 0.30, 0.36], "tie-prone", (0.02, 0.028), "left"),
              ([0.18, 0.74, 0.08], "pro-B", (0.028, 0.012), "left"),
              ([0.50, 0.50, 0.00], "never ties", (0.0, -0.055), "center")]
    for p, lab, d, ha in people:
        dot(ax, p, INK, s=20, label=lab, dxy=d, fs=6.3, ha=ha)
    ax.plot([0, 1], [0, 0], color=ACCENT4, lw=2.4, alpha=0.7, zorder=2)
    ax.text(0.5, CAPY, "shading: which outcome has the largest probability",
            fontsize=6.5, color=MUTED, ha="center")
    ax.text(0.5, CAPY - 0.052, "amber edge $p_T=0$: purely binary judges",
            fontsize=6.5, color=ACCENT4, ha="center")

    ax = simplex_ax(fig, r2, title="the signed margin $y=p_A-p_B$ is only a shadow")
    for lev in np.arange(-0.8, 0.81, 0.2):
        # y = pA - pB is affine: draw its level line by clipping
        poly = clip_poly(TRI, lambda xy, lev=lev: (to_bary(xy)[0] - to_bary(xy)[1]) - lev)
        poly2 = clip_poly(poly, lambda xy, lev=lev: lev - (to_bary(xy)[0] - to_bary(xy)[1]))
        if len(poly2) >= 2:
            pts = np.array([q for q in poly2
                            if abs(to_bary(q)[0] - to_bary(q)[1] - lev) < 1e-9])
            if len(pts) >= 2:
                ax.plot(pts[:, 0], pts[:, 1], color=ACCENT, lw=0.7, alpha=0.65,
                        zorder=1)
    for p, lab in [([0.45, 0.25, 0.30], "$p^{(1)}$"), ([0.60, 0.40, 0.00], "$p^{(2)}$")]:
        dot(ax, p, ACCENT2, s=28, label=lab, fs=7.0)
    ax.annotate("", xy=to_xy([0.60, 0.40, 0.0]), xytext=to_xy([0.45, 0.25, 0.30]),
                arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=0.8,
                                ls="--", shrinkA=4, shrinkB=4))
    ax.text(0.5, CAPY, "both have $y=+0.20$: the margin cannot see the tie mass",
            fontsize=6.5, color=MUTED, ha="center")


# ---------------------------------------------------------------------------
# 2. the loss
# ---------------------------------------------------------------------------
def fig_tv_ball(fig, rect):
    r1, r2, r3 = panels(fig, rect, 3, pad=0.018)
    p = np.array([0.42, 0.33, 0.25])
    ax = simplex_ax(fig, r1, title="TV balls are hexagons")
    for eps, a in [(0.20, 0.16), (0.12, 0.22), (0.06, 0.30)]:
        poly = tv_ball(p, eps)
        ax.add_patch(_poly(poly, fc=ACCENT, ec=ACCENT, lw=0.8, alpha=a,
                           zorder=2))
        v = poly[np.argmax(poly[:, 1])]
        ax.text(v[0], v[1] + 0.012, "$%.2f$" % eps, fontsize=6.0, color=ACCENT,
                ha="center")
    dot(ax, p, ACCENT2, s=26)
    ax.text(0.5, CAPY, "$\\{q:\\ \\mathrm{TV}(p,q)\\leq\\varepsilon\\}$",
            fontsize=7.0, color=INK, ha="center")

    ax = simplex_ax(fig, r2, title="near an edge they get clipped")
    for q, eps in [([0.80, 0.14, 0.06], 0.18), ([0.20, 0.72, 0.08], 0.12)]:
        poly = tv_ball(np.array(q), eps)
        ax.add_patch(_poly(poly, fc=ACCENT4, ec=ACCENT4, lw=0.8, alpha=0.25,
                           zorder=2))
        dot(ax, q, ACCENT2, s=20)
    ax.text(0.5, CAPY, "the simplex boundary cuts the hexagon",
            fontsize=6.5, color=MUTED, ha="center")

    ax = bare_ax(fig, r3, title="$\\mathrm{TV}=\\max_k|p_k-q_k|$")
    q = np.array([0.30, 0.47, 0.23])
    d = p - q
    ax.bar(range(3), d, color=[ACCENT if x >= 0 else ACCENT2 for x in d],
           width=0.55)
    ax.axhline(0, color=INK, lw=0.7)
    m = np.max(np.abs(d))
    ax.axhline(m, color=ACCENT3, lw=0.9, ls="--")
    ax.axhline(-m, color=ACCENT3, lw=0.9, ls="--")
    ax.text(2.55, m, "$\\pm\\,\\mathrm{TV}$", fontsize=6.8, color=ACCENT3,
            ha="center", va="bottom")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["$A$", "$B$", "$T$"], fontsize=7.5)
    ax.set_ylim(-0.2, 0.2)
    ax.set_ylabel("$p_k-q_k$", fontsize=7)
    ax.set_xlim(-0.6, 3.1)


# ---------------------------------------------------------------------------
# 3. reconstruction = convex hull
# ---------------------------------------------------------------------------
def fig_hull(fig, rect):
    r1, r2 = panels(fig, rect, 2, pad=0.022)
    S = np.array([[0.78, 0.14, 0.08], [0.20, 0.70, 0.10],
                  [0.30, 0.22, 0.48], [0.46, 0.40, 0.14]])
    xy = to_xy(S)
    H = hull_poly(xy)

    ax = simplex_ax(fig, r1, title="target inside the hull: exact")
    ax.add_patch(_poly(H, fc=HULLFILL, ec=ACCENT, lw=1.0, zorder=2))
    for i, s in enumerate(S):
        dot(ax, s, ACCENT, s=24, label="$j_%d$" % (i + 1), fs=6.6)
    t = np.array([0.44, 0.34, 0.22])
    d, w = dist_to_hull(t, S)
    dot(ax, t, ACCENT2, s=34, marker="D", label="  target $i$", fs=6.8)
    for wi, s in zip(w, S):
        if wi > 1e-6:
            a, b = to_xy(t), to_xy(s)
            ax.annotate("", xy=b, xytext=a,
                        arrowprops=dict(arrowstyle="-", color=ACCENT2,
                                        lw=0.5 + 3.0 * wi, alpha=0.45,
                                        shrinkA=3, shrinkB=3))
    ax.text(0.5, CAPY, "$\\widehat p_i=\\sum_j w_{ij}p_j$, error $=0$;"
            " line width $=w_{ij}$", fontsize=6.5, color=MUTED, ha="center")

    ax = simplex_ax(fig, r2, title="target outside: error = TV distance to the hull")
    ax.add_patch(_poly(H, fc=HULLFILL, ec=ACCENT, lw=1.0, zorder=2))
    for s in S:
        dot(ax, s, ACCENT, s=24)
    t = np.array([0.12, 0.18, 0.70])
    d, w = dist_to_hull(t, S)
    poly = tv_ball(t, d)
    ax.add_patch(_poly(poly, fc=ACCENT2, ec=ACCENT2, lw=0.8, alpha=0.18,
                       zorder=1))
    dot(ax, t, ACCENT2, s=34, marker="D", label=" target $i$", fs=6.8)
    ph = S.T @ w
    dot(ax, ph, ACCENT4, s=26, marker="s", label="  $\\widehat p_i$", fs=6.8)
    ax.annotate("", xy=to_xy(ph), xytext=to_xy(t),
                arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.0,
                                shrinkA=4, shrinkB=4))
    ax.text(0.5, CAPY, "smallest hexagon touching the hull; radius $=%.3f$" % d,
            fontsize=6.5, color=MUTED, ha="center")


def fig_one_weight(fig, rect):
    rs = panels(fig, rect, 4, pad=0.014, widths=[1, 1, 1, 1.25])
    j1 = np.array([[0.80, 0.12, 0.08], [0.55, 0.35, 0.10], [0.25, 0.30, 0.45]])
    j2 = np.array([[0.15, 0.75, 0.10], [0.40, 0.50, 0.10], [0.30, 0.55, 0.15]])
    targ = np.array([[0.40, 0.44, 0.16], [0.36, 0.36, 0.28], [0.52, 0.30, 0.18]])
    items = [(j1[k], j2[k], targ[k]) for k in range(3)]
    ws = np.linspace(0, 1, 401)
    curves = []
    for k, (a, b, t) in enumerate(items):
        ax = simplex_ax(fig, rs[k], labels=(k == 0),
                        title="item $x_%d$" % (k + 1))
        A, B = to_xy(a), to_xy(b)
        ax.plot([A[0], B[0]], [A[1], B[1]], color=ACCENT, lw=1.6, zorder=2)
        dot(ax, a, ACCENT, s=20, label="$j_1$" if k == 0 else None, fs=6.3)
        dot(ax, b, ACCENT, s=20, label="$j_2$" if k == 0 else None, fs=6.3)
        dot(ax, t, ACCENT2, s=26, marker="D")
        e = np.array([tv(t, w * a + (1 - w) * b) for w in ws])
        curves.append(e)
        wstar = ws[np.argmin(e)]
        ph = wstar * a + (1 - wstar) * b
        dot(ax, ph, ACCENT4, s=18, marker="s")
        ax.text(0.5, -0.10, "best here: $w_1=%.2f$" % wstar, fontsize=6.2,
                color=ACCENT4, ha="center")

    ax = bare_ax(fig, rs[3], title="one $w$ must serve all items")
    C = np.array(curves)
    for k in range(3):
        ax.plot(ws, C[k], lw=0.9, color=ACCENT, alpha=0.55)
        ax.text(ws[np.argmin(C[k])], -0.012, "$x_%d$" % (k + 1), fontsize=6.2,
                color=ACCENT, ha="center", va="top")
    mean = C.mean(0)
    ax.plot(ws, mean, lw=1.6, color=ACCENT2)
    i = int(np.argmin(mean))
    ax.scatter([ws[i]], [mean[i]], s=22, c=ACCENT2, zorder=5)
    ax.annotate("$\\widehat w$", (ws[i], mean[i]), (ws[i] + 0.10, mean[i] + 0.055),
                fontsize=7.5, color=ACCENT2,
                arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=0.7))
    ax.set_xlabel("$w_1$  (weight on $j_1$)", fontsize=7)
    ax.set_ylabel("TV error", fontsize=7)
    ax.set_ylim(-0.02, max(0.32, C.max() * 1.05))
    ax.set_xlim(0, 1)


def fig_weight_simplex(fig, rect):
    r1, r2 = panels(fig, rect, 2, pad=0.022)
    rng = np.random.default_rng(7)
    base = np.array([[0.78, 0.14, 0.08], [0.16, 0.74, 0.10], [0.30, 0.24, 0.46]])

    def ctx(shift, n=24, seed=0):
        g = np.random.default_rng(seed)
        P = []
        for _ in range(n):
            q = base + shift * g.normal(scale=1.0, size=(3, 3)) * 0.06
            q = np.clip(q, 0.02, None)
            q = q / q.sum(1, keepdims=True)
            P.append(q)
        return np.array(P)

    Pc1 = ctx(1.0, seed=1)
    Pc2 = ctx(1.0, seed=5)
    tgt1 = np.array([0.42, 0.36, 0.22])
    tgt2 = np.array([0.30, 0.30, 0.40])

    def err(P, t, w):
        return np.mean([tv(t, q.T @ w) for q in P])

    grid = []
    NG = 90
    for a in np.linspace(0, 1, NG):
        for b in np.linspace(0, 1 - a, NG):
            grid.append([a, b, 1 - a - b])
    grid = np.array(grid)
    G = to_xy(grid)
    e1 = np.array([err(Pc1, tgt1, w) for w in grid])
    e2 = np.array([err(Pc2, tgt2, w) for w in grid])
    emax = np.maximum(e1, e2)

    ax = simplex_ax(fig, r1, vlab=("$w_1$", "$w_2$", "$w_3$"),
                    title="$\\max_c\\,\\varepsilon_c(w)$ over the weight simplex")
    ax.tricontourf(G[:, 0], G[:, 1], emax, levels=14, cmap="Blues", zorder=1)
    ax.tricontour(G[:, 0], G[:, 1], emax, levels=8, colors="white",
                  linewidths=0.35, zorder=2)
    i = int(np.argmin(emax))
    ax.scatter([G[i, 0]], [G[i, 1]], s=30, c=ACCENT2, zorder=6,
               edgecolors="white", linewidths=0.7)
    ax.text(G[i, 0] + 0.03, G[i, 1], "$\\widehat w$", fontsize=7.5,
            color=ACCENT2)
    ax.text(0.5, CAPY, "convex, piecewise linear: a bowl with creases",
            fontsize=6.5, color=MUTED, ha="center")

    ax = simplex_ax(fig, r2, vlab=("$w_1$", "$w_2$", "$w_3$"),
                    title="feasible sets $\\{\\varepsilon_c(w)\\leq\\gamma\\}$")
    gam = float(emax.min()) * 1.45
    for e, col, lab in [(e1, ACCENT, "context 1"), (e2, ACCENT3, "context 2")]:
        ax.tricontourf(G[:, 0], G[:, 1], e, levels=[0, gam], colors=[col],
                       alpha=0.26, zorder=1)
        ax.tricontour(G[:, 0], G[:, 1], e, levels=[gam], colors=[col],
                      linewidths=0.9, zorder=2)
        m = G[e <= gam].mean(0)
        ax.text(m[0], m[1], lab, fontsize=6.4, color=col, ha="center",
                va="center", zorder=7)
    both = np.where((e1 <= gam) & (e2 <= gam))[0]
    if len(both) > 3:
        H = hull_poly(G[both])
        ax.add_patch(_poly(H, fc=ACCENT2, ec=ACCENT2, lw=0.9, alpha=0.55,
                           zorder=3))
        m = G[both].mean(0)
        ax.annotate("robust $w$", xy=(m[0], m[1]),
                    xytext=(m[0] + 0.22, m[1] - 0.13), fontsize=6.6,
                    color=ACCENT2, ha="left", zorder=8,
                    arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=0.7,
                                    shrinkA=1, shrinkB=3))
    ax.text(0.5, CAPY, "robust feasibility = a non-empty intersection",
            fontsize=6.5, color=MUTED, ha="center")


def fig_epigraph(fig, rect):
    r1, r2 = panels(fig, rect, 2, pad=0.03, widths=[1.15, 1.0])
    ws = np.linspace(0, 1, 300)
    rng = np.random.default_rng(3)
    curves = []
    for k in range(4):
        knots = np.sort(rng.uniform(0.05, 0.95, 3))
        c = 0.02 + 0.30 * np.mean([np.abs(ws - a) for a in knots], axis=0)
        c += rng.uniform(-0.01, 0.02)
        curves.append(c)
    C = np.array(curves)
    M = C.max(0)

    ax = bare_ax(fig, r1, title="each context gives a convex piecewise-linear cost")
    for k, c in enumerate(C):
        ax.plot(ws, c, lw=0.8, color=ACCENT, alpha=0.55)
        ax.text(1.005, c[-1], " $c_%d$" % (k + 1), fontsize=6.3, color=ACCENT,
                va="center")
    ax.plot(ws, M, lw=1.8, color=ACCENT2)
    i = int(np.argmin(M))
    ax.scatter([ws[i]], [M[i]], s=26, c=ACCENT2, zorder=6)
    ax.plot([0, 1], [M[i], M[i]], color=ACCENT3, lw=0.8, ls="--")
    ax.text(0.02, M[i] - 0.006, "$u^\\star=\\widehat E_{\\mathrm{fit}}$",
            fontsize=7.2, color=ACCENT3, va="top")
    ax.annotate("active context", (ws[i], M[i]), (ws[i] - 0.30, M[i] + 0.075),
                fontsize=6.6, color=ACCENT2,
                arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=0.7))
    ax.set_xlabel("a line through the weight simplex", fontsize=7)
    ax.set_ylabel("mean TV error", fontsize=7)
    ax.set_xlim(0, 1.06)

    ax = bare_ax(fig, r2, title="the epigraph trick")
    ax.plot(ws, M, lw=1.6, color=ACCENT2)
    ax.fill_between(ws, M, M.max() * 1.15, color=HULLFILL, alpha=0.8, zorder=0)
    ax.text(0.5, M.max() * 0.98, "$\\{(w,u):u\\geq\\varepsilon_c(w)\\ \\forall c\\}$",
            fontsize=7.4, color=ACCENT, ha="center")
    ax.annotate("minimise $u$", (ws[i], M[i]), (ws[i] + 0.06, M[i] + 0.09),
                fontsize=7.0, color=ACCENT3,
                arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=0.8))
    ax.scatter([ws[i]], [M[i]], s=26, c=ACCENT3, zorder=6)
    ax.set_xlabel("$w$", fontsize=7)
    ax.set_ylabel("$u$", fontsize=7)
    ax.set_xlim(0, 1)


# ---------------------------------------------------------------------------
# 4. monotonicity
# ---------------------------------------------------------------------------
def fig_nested(fig, rect):
    r1, r2, r3 = panels(fig, rect, 3, pad=0.018)
    S = np.array([[0.74, 0.16, 0.10], [0.22, 0.66, 0.12], [0.34, 0.26, 0.40]])
    extra = np.array([0.52, 0.20, 0.28])
    t = np.array([0.60, 0.10, 0.30])

    for r, P, name in [(r1, S, "$S$"), (r2, np.vstack([S, extra]), "$T=S\\cup\\{j\\}$")]:
        ax = simplex_ax(fig, r, title="retained set %s" % name)
        H = hull_poly(to_xy(P))
        ax.add_patch(_poly(H, fc=HULLFILL, ec=ACCENT, lw=1.0, zorder=2))
        for i, s in enumerate(P):
            dot(ax, s, ACCENT if i < 3 else ACCENT3, s=22)
        d, w = dist_to_hull(t, P)
        ax.add_patch(_poly(tv_ball(t, d), fc=ACCENT2, ec=ACCENT2, lw=0.7,
                           alpha=0.16, zorder=1))
        dot(ax, t, ACCENT2, s=28, marker="D")
        ax.text(0.5, CAPY, "$\\widehat E=%.3f$" % d, fontsize=7.0,
                color=ACCENT2, ha="center")

    ax = bare_ax(fig, r3, title="but held-out error is not monotone")
    k = np.arange(1, 11)
    fit = 0.30 * np.exp(-0.42 * (k - 1)) + 0.008
    test = fit + 0.012 * (k - 1) ** 1.25 * 0.35 + 0.012
    test[5] += 0.004
    ax.plot(k, fit, "-o", ms=3, lw=1.2, color=ACCENT, label="FIT (non-increasing)")
    ax.plot(k, test, "-s", ms=3, lw=1.2, color=ACCENT2, label="TEST (can rise)")
    j = int(np.argmin(test))
    ax.scatter([k[j]], [test[j]], s=42, facecolor="none", edgecolor=ACCENT3,
               lw=1.2, zorder=6)
    ax.set_xlabel("$|S|$", fontsize=7)
    ax.set_ylabel("worst-context TV", fontsize=7)
    ax.legend(fontsize=6.0, frameon=False, loc="upper right")
    ax.set_ylim(0, 0.34)


# ---------------------------------------------------------------------------
# 5. non-composability
# ---------------------------------------------------------------------------
_DUP = np.array([
    [0.80, 0.12, 0.08], [0.76, 0.15, 0.09],
    [0.14, 0.78, 0.08], [0.17, 0.75, 0.08],
    [0.24, 0.20, 0.56], [0.21, 0.24, 0.55]])


def fig_noncomp_dup(fig, rect):
    r1, r2, r3 = panels(fig, rect, 3, pad=0.018)
    P = _DUP
    ax = simplex_ax(fig, r1, title="six judges, three twin pairs")
    H = hull_poly(to_xy(P))
    ax.add_patch(_poly(H, fc=HULLFILL, ec=ACCENT, lw=1.0, zorder=2))
    for i, p in enumerate(P):
        dot(ax, p, ACCENT, s=20, label="$j_%d$" % (i + 1),
            dxy=(0.018, 0.012) if i % 2 == 0 else (-0.018, -0.03),
            fs=6.0, ha="left" if i % 2 == 0 else "right")
    ax.text(0.5, CAPY, "each judge sits on top of its twin", fontsize=6.5,
            color=MUTED, ha="center")

    ax = simplex_ax(fig, r2, title="leave-one-out: everyone passes")
    for i in range(len(P)):
        rest = np.delete(P, i, axis=0)
        d, _ = dist_to_hull(P[i], rest)
        xy = to_xy(P[i])
        ax.scatter([xy[0]], [xy[1]], s=20, c=ACCENT3, zorder=6,
                   edgecolors="white", linewidths=0.6)
    H = hull_poly(to_xy(np.delete(P, 0, axis=0)))
    ax.add_patch(_poly(H, fc=HULLFILL, ec=ACCENT, lw=0.9, alpha=0.8, zorder=2))
    d0, _ = dist_to_hull(P[0], np.delete(P, 0, axis=0))
    dot(ax, P[0], ACCENT2, s=30, marker="D", label="  $j_1$", fs=6.5)
    ax.text(0.5, CAPY,
            "$U(j\\,|\\,\\mathcal{J}\\setminus j)\\leq %.3f$ for all six"
            % max(dist_to_hull(P[i], np.delete(P, i, 0))[0]
                  for i in range(len(P))),
            fontsize=6.5, color=ACCENT3, ha="center")

    ax = simplex_ax(fig, r3, title="retire them all: nothing is left")
    for i, p in enumerate(P):
        xy = to_xy(p)
        ax.scatter([xy[0]], [xy[1]], s=24, marker="x", c=ACCENT2, zorder=6,
                   linewidths=1.1)
    Hm = hull_poly(to_xy(P[[0, 2, 4]]))
    ax.add_patch(_poly(Hm, fc="none", ec=ACCENT3, lw=1.0, ls="--", zorder=3))
    ax.text(0.5, CAPY, "minimum jointly feasible set has size 3,\n"
            "one per twin pair \u2014 composition gap $=3$",
            fontsize=6.5, color=ACCENT2, ha="center", va="top")


def _circle_judges(m, R=0.26, seed=0):
    c = np.array([1 / 3, 1 / 3, 1 / 3])
    ang = np.linspace(0, 2 * np.pi, m, endpoint=False) + 0.31
    P = []
    for a in ang:
        d = np.array([np.cos(a), np.sin(a) * np.cos(np.pi / 6),
                      -np.cos(a) - np.sin(a) * np.cos(np.pi / 6)])
        d = np.array([np.cos(a), -0.5 * np.cos(a) + S3 * np.sin(a),
                      -0.5 * np.cos(a) - S3 * np.sin(a)])
        d = d / np.max(np.abs(d))
        P.append(c + R * d)
    return np.array(P)


def fig_noncomp_circle(fig, rect):
    rs = panels(fig, rect, 4, pad=0.014, widths=[1, 1, 1, 1.3])
    m = 14
    P = _circle_judges(m)
    loo = np.array([dist_to_hull(P[i], np.delete(P, i, 0))[0] for i in range(m)])

    ax = simplex_ax(fig, rs[0], title="%d judges on an arc" % m)
    H = hull_poly(to_xy(P))
    ax.add_patch(_poly(H, fc=HULLFILL, ec=ACCENT, lw=1.0, zorder=2))
    for p in P:
        dot(ax, p, ACCENT, s=14)
    ax.text(0.5, CAPY, "no duplicates, no clusters", fontsize=6.5,
            color=MUTED, ha="center")

    ax = simplex_ax(fig, rs[1], title="each is removable alone")
    rest = np.delete(P, 0, 0)
    ax.add_patch(_poly(hull_poly(to_xy(rest)), fc=HULLFILL, ec=ACCENT, lw=0.9,
                       zorder=2))
    for p in rest:
        dot(ax, p, ACCENT, s=12)
    d0, w0 = dist_to_hull(P[0], rest)
    ax.add_patch(_poly(tv_ball(P[0], d0), fc=ACCENT3, ec=ACCENT3, lw=0.7,
                       alpha=0.25, zorder=1))
    dot(ax, P[0], ACCENT3, s=26, marker="D")
    ax.text(0.5, CAPY, "gap $=%.3f\\leq\\gamma$" % loo.max(), fontsize=6.6,
            color=ACCENT3, ha="center")

    kk = 5
    ax = simplex_ax(fig, rs[2], title="keep only %d" % kk)
    sub = P[np.round(np.linspace(0, m, kk, endpoint=False)).astype(int)]
    ax.add_patch(_poly(hull_poly(to_xy(sub)), fc=HULLFILL, ec=ACCENT, lw=1.0,
                       zorder=2))
    for p in P:
        dot(ax, p, MUTED, s=9)
    for p in sub:
        dot(ax, p, ACCENT, s=20)
    worst = max(range(m), key=lambda i: dist_to_hull(P[i], sub)[0])
    dw, _ = dist_to_hull(P[worst], sub)
    ax.add_patch(_poly(tv_ball(P[worst], dw), fc=ACCENT2, ec=ACCENT2, lw=0.7,
                       alpha=0.22, zorder=1))
    dot(ax, P[worst], ACCENT2, s=24, marker="D")
    ax.text(0.5, CAPY, "worst error $=%.3f$" % dw, fontsize=6.6,
            color=ACCENT2, ha="center")

    ax = bare_ax(fig, rs[3], title="the $k^{-2}$ law")
    ks = np.arange(3, 11)
    vals = []
    for k in ks:
        best = np.inf
        for S in itertools.combinations(range(m), int(k)):
            v = coverage(S, P)
            if v < best:
                best = v
        vals.append(best)
    vals = np.array(vals)
    ax.loglog(ks, vals, "-o", ms=3.4, lw=1.2, color=ACCENT2, label="exact $E(k)$")
    ref = vals[0] * (ks / ks[0]) ** -2.0
    ax.loglog(ks, ref, "--", lw=1.0, color=MUTED, label="slope $-2$")
    ax.set_xlabel("physical panel size $k$", fontsize=7)
    ax.set_ylabel("worst TV error", fontsize=7)
    ax.legend(fontsize=6.0, frameon=False)
    _tick = __import__("matplotlib").ticker
    ax.set_xticks([3, 5, 7, 10])
    ax.get_xaxis().set_major_formatter(_tick.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(_tick.NullFormatter())


def fig_frontier_shapes(fig, rect):
    r1, r2, r3 = panels(fig, rect, 3, pad=0.026)
    rng = np.random.default_rng(11)

    arche = np.array([[0.80, 0.12, 0.08], [0.14, 0.78, 0.08],
                      [0.24, 0.20, 0.56], [0.42, 0.42, 0.16]])
    lam = rng.dirichlet(np.ones(4), size=10)
    Ppoly = np.vstack([arche, lam @ arche])[:11]
    Pcirc = _circle_judges(11)
    Pnoise = np.clip(Ppoly + rng.normal(scale=0.030, size=Ppoly.shape), 0.01, None)
    Pnoise = Pnoise / Pnoise.sum(1, keepdims=True)

    for r, P, title, note in [
            (r1, Ppoly, "polytope-like", "a few true archetypes\n$\\Rightarrow$ error hits 0 at $k=r$"),
            (r2, Pcirc, "curved", "no exact basis\n$\\Rightarrow$ smooth $k^{-2}$ decay"),
            (r3, Pnoise, "noise-dominated", "independent idiosyncrasy\n$\\Rightarrow$ plateau, no compression")]:
        ax = bare_ax(fig, r, title=title)
        ks = np.arange(1, 9)
        vals = [min(coverage(S, P) for S in itertools.combinations(range(len(P)), int(k)))
                for k in ks]
        ax.plot(ks, vals, "-o", ms=3.2, lw=1.3, color=ACCENT2)
        ax.axhline(0.08, color=ACCENT3, lw=0.8, ls="--")
        ax.text(8.0, 0.084, "$\\gamma$", fontsize=7, color=ACCENT3, ha="right")
        ax.set_xlabel("$k$", fontsize=7)
        ax.set_ylabel("$E(k)$", fontsize=7)
        ax.set_ylim(-0.01, 0.36)
        ax.text(0.97, 0.94, note, transform=ax.transAxes, fontsize=6.2,
                color=MUTED, ha="right", va="top")


# ---------------------------------------------------------------------------
# 6. algorithms
# ---------------------------------------------------------------------------
def fig_forward_backward(fig, rect):
    r1, r2, r3 = panels(fig, rect, 3, pad=0.018, widths=[1, 1, 1.15])
    ext = np.array([[0.84, 0.09, 0.07], [0.10, 0.82, 0.08], [0.16, 0.14, 0.70]])
    rng = np.random.default_rng(2)
    inner = ext.mean(0) + rng.normal(scale=0.035, size=(5, 3))
    inner = np.clip(inner, 0.02, None)
    inner = inner / inner.sum(1, keepdims=True)
    P = np.vstack([ext, inner])
    singles = [coverage([i], P) for i in range(len(P))]
    med = int(np.argmin(singles))

    ax = simplex_ax(fig, r1, title="forward starts at the centre")
    ax.add_patch(_poly(hull_poly(to_xy(P)), fc=HULLFILL, ec=ACCENT, lw=1.0,
                       zorder=2))
    for i, p in enumerate(P):
        dot(ax, p, ACCENT if i < 3 else MUTED, s=18)
    dot(ax, P[med], ACCENT2, s=44, marker="*", label="best singleton", fs=6.4,
        dxy=(0.0, 0.075), ha="center")
    ax.text(0.5, CAPY, "the medoid is never a vertex",
            fontsize=6.4, color=ACCENT2, ha="center")

    ax = simplex_ax(fig, r2, title="backward peels interior points")
    ax.add_patch(_poly(hull_poly(to_xy(ext)), fc=HULLFILL, ec=ACCENT3, lw=1.2,
                       zorder=2))
    for i, p in enumerate(P):
        if i < 3:
            dot(ax, p, ACCENT3, s=22)
        else:
            xy = to_xy(p)
            ax.scatter([xy[0]], [xy[1]], s=20, marker="x", c=MUTED,
                       linewidths=1.0, zorder=5)
    ax.text(0.5, CAPY, "non-vertices are free to remove",
            fontsize=6.4, color=ACCENT3, ha="center")

    ax = bare_ax(fig, r3, title="measured on UTD19 ($N=20$)")
    names = ["forward", "backward", "forward\n+ trim", "2-swap", "3-swap"]
    gaps = [1.86, 0.86, 0.57, 0.29, 0.29]
    cols = [ACCENT2, ACCENT4, ACCENT4, ACCENT3, ACCENT3]
    ax.barh(range(5)[::-1], gaps, color=cols, height=0.6)
    for i, g in enumerate(gaps):
        ax.text(g + 0.05, 4 - i, "%.2f" % g, fontsize=6.6, va="center",
                color=INK)
    ax.set_yticks(range(5))
    ax.set_yticklabels(names[::-1], fontsize=6.4)
    ax.set_xlabel("mean extra models vs. exact optimum", fontsize=7)
    ax.set_xlim(0, 2.25)


def fig_swap_landscape(fig, rect):
    r1, r2 = panels(fig, rect, 2, pad=0.03, widths=[1.25, 1.0])
    ax = bare_ax(fig, r1, title="why single moves get stuck")
    ax.axis("off")
    rng = np.random.default_rng(0)
    levels = {6: 1, 5: 3, 4: 4, 3: 2}
    pos = {}
    for lv, (k, n) in enumerate(sorted(levels.items(), reverse=True)):
        for i in range(n):
            pos[(k, i)] = (0.12 + lv * 0.26, 0.82 - i * 0.20 + (n - 1) * 0.10)
    edges = [((6, 0), (5, 0)), ((6, 0), (5, 1)), ((6, 0), (5, 2)),
             ((5, 0), (4, 0)), ((5, 1), (4, 1)), ((5, 2), (4, 2)),
             ((5, 2), (4, 3)), ((4, 0), (3, 0)), ((4, 3), (3, 1))]
    for a, b in edges:
        ax.annotate("", xy=pos[b], xytext=pos[a],
                    arrowprops=dict(arrowstyle="->", color=RULE, lw=0.8))
    stuck = [(4, 1), (4, 2)]
    for key, (x, y) in pos.items():
        k, i = key
        col = ACCENT3 if k == 3 else (ACCENT2 if key in stuck else ACCENT)
        ax.scatter([x], [y], s=90, c=col, zorder=5, edgecolors="white", lw=0.8)
        ax.text(x, y, "%d" % k, fontsize=6.4, color="white", ha="center",
                va="center", zorder=6, fontweight="bold")
    xs, ys = zip(*(pos[k] for k in stuck))
    ax.text(sum(xs) / len(xs), min(ys) - 0.10, "local optima", fontsize=5.9,
            color=ACCENT2, ha="center", va="center")
    ax.set_xlim(0.02, 1.00)
    ax.set_ylim(0.28, 1.32)
    a, b = pos[(4, 1)], pos[(3, 1)]
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=1.4,
                                connectionstyle="arc3,rad=-0.35"))
    ax.text((a[0] + b[0]) / 2, min(a[1], b[1]) - 0.34, "2-swap + trim",
            fontsize=6.6, color=ACCENT3, ha="center")
    ax.text(0.5, -0.02, "nodes = feasible sets, labelled by $|S|$;"
            " arrows = single deletions", fontsize=6.3, color=MUTED,
            ha="center", transform=ax.transAxes)

    ax = bare_ax(fig, r2, title="cost of the neighbourhood")
    N = np.arange(8, 41)
    ax.semilogy(N, N, lw=1.2, color=ACCENT, label="$1$-move: $O(N)$")
    ax.semilogy(N, N ** 4 / 4, lw=1.2, color=ACCENT4, label="$2$-swap: $O(N^4)$")
    ax.semilogy(N, N ** 6 / 36, lw=1.2, color=ACCENT2, label="$3$-swap: $O(N^6)$")
    ax.axvline(20, color=MUTED, lw=0.7, ls=":")
    ax.text(20.5, 5e6, "UTD19", fontsize=6.2, color=MUTED)
    ax.set_xlabel("$N$ (ecosystem size)", fontsize=7)
    ax.set_ylabel("candidate exchanges", fontsize=7)
    ax.legend(fontsize=6.0, frameon=False, loc="lower right")


def fig_separating(fig, rect):
    r1, r2 = panels(fig, rect, 2, pad=0.028, widths=[1.0, 1.1])
    S = np.array([[0.70, 0.20, 0.10], [0.28, 0.60, 0.12], [0.38, 0.30, 0.32]])
    t = np.array([0.16, 0.20, 0.64])
    ax = simplex_ax(fig, r1, title="the dual gives a witness direction")
    ax.add_patch(_poly(hull_poly(to_xy(S)), fc=HULLFILL, ec=ACCENT, lw=1.0,
                       zorder=2))
    for s in S:
        dot(ax, s, ACCENT, s=22)
    dot(ax, t, ACCENT2, s=30, marker="D", label="  $i$", fs=7)
    lam = np.array([0.0, 0.0, 1.0])
    lam = lam - lam.mean()
    lev_hull = max(S @ lam)
    lev_t = t @ lam
    for lev, col, ls in [(lev_hull, ACCENT, "-"), (lev_t, ACCENT2, "--")]:
        poly = clip_poly(TRI, lambda xy, lev=lev: to_bary(xy) @ lam - lev)
        poly2 = clip_poly(TRI, lambda xy, lev=lev: lev - to_bary(xy) @ lam)
        pts = np.array([q for q in poly2 if abs(to_bary(q) @ lam - lev) < 1e-9])
        if len(pts) >= 2:
            ax.plot(pts[:, 0], pts[:, 1], color=col, lw=1.1, ls=ls, zorder=4)
    ax.annotate("", xy=to_xy([0.28, 0.22, 0.50]), xytext=to_xy([0.36, 0.30, 0.34]),
                arrowprops=dict(arrowstyle="->", color=ACCENT4, lw=1.2))
    ax.text(0.30, 0.52, "$\\lambda$", fontsize=8, color=ACCENT4)
    ax.text(0.5, CAPY, "$\\langle\\lambda,p_i\\rangle-\\max_{j\\in S}"
            "\\langle\\lambda,p_j\\rangle=%.3f>0$" % (lev_t - lev_hull),
            fontsize=6.5, color=ACCENT2, ha="center")

    ax = bare_ax(fig, r2, title="reading the certificate")
    ax.axis("off")
    ax.text(0.0, 0.96, "$\\lambda$ is a scoring rule on judge outputs.",
            fontsize=7.4, color=INK, va="top")
    ax.text(0.0, 0.80, "Here $\\lambda$ reads off tie mass. Judge $i$ ties\n"
            "more often than every retained judge, so no\n"
            "convex mixture of them can reach it.",
            fontsize=7.0, color=MUTED, va="top")
    ax.text(0.0, 0.44, "Report $\\lambda$ next to any claim that a judge\n"
            "is irreplaceable. It converts a solver output\n"
            "into a falsifiable behavioural statement.",
            fontsize=7.0, color=INK, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


# ---------------------------------------------------------------------------
# 7. exact optimisation
# ---------------------------------------------------------------------------
def fig_milp(fig, rect):
    r1, r2 = panels(fig, rect, 2, pad=0.03)
    ax = bare_ax(fig, r1, title="why the LP relaxation is blind to cardinality")
    idx = np.arange(8)
    zint = np.array([1, 0, 1, 0, 0, 1, 0, 0], float)
    zfrac = np.array([.18, .12, .16, .09, .11, .14, .10, .10])
    ax.bar(idx - 0.19, zint, width=0.36, color=ACCENT, label="integer $z_j$")
    ax.bar(idx + 0.19, zfrac, width=0.36, color=ACCENT4,
           label="relaxed $z_j$")
    ax.axhline(1.0, color=RULE, lw=0.6)
    ax.set_xticks(idx)
    ax.set_xticklabels(["$j_%d$" % (i + 1) for i in idx], fontsize=6.2)
    ax.set_ylabel("$z_j$", fontsize=7)
    ax.legend(fontsize=6.1, frameon=False, loc="upper center", ncol=2,
              handlelength=1.2, columnspacing=1.2)
    ax.text(3.5, 0.62, "$\\sum_j z_j = 3$", fontsize=7, color=ACCENT,
            ha="center")
    ax.text(3.5, 0.45, "$\\sum_j z_j = %.1f$" % zfrac.sum(), fontsize=7,
            color=ACCENT4, ha="center")
    ax.set_ylim(0, 1.32)

    ax = bare_ax(fig, r2, title="no-good cuts close the gap")
    it = np.arange(0, 13)
    lb = np.array([1.0, 1.0, 1.4, 1.8, 2.0, 2.0, 2.4, 2.9, 3.3, 3.7, 4.0, 4.0, 4.0])
    ub = np.array([9, 8, 7, 6, 6, 5, 5, 5, 4, 4, 4, 4, 4], float)
    ax.plot(it, lb, "-o", ms=2.8, lw=1.2, color=ACCENT, label="dual bound")
    ax.plot(it, ub, "-s", ms=2.8, lw=1.2, color=ACCENT2, label="best feasible")
    ax.fill_between(it, lb, ub, color=HULLFILL, alpha=0.7)
    ax.axhline(4, color=ACCENT3, lw=0.8, ls="--")
    ax.text(12.2, 4.15, "proven $k^\\star$", fontsize=6.4, color=ACCENT3,
            ha="right")
    ax.annotate("plain LP stalls at 2", (4, 2.0), (5.4, 1.15), fontsize=6.3,
                color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
    ax.set_xlabel("cut-generation round", fontsize=7)
    ax.set_ylabel("$|S|$", fontsize=7)
    ax.legend(fontsize=6.1, frameon=False, loc="upper right")
    ax.set_ylim(0, 9.8)


def fig_backbone(fig, rect):
    r1, r2 = panels(fig, rect, 2, pad=0.03, widths=[1.0, 1.15])
    ax = simplex_ax(fig, r1, title="who is really irreplaceable?")
    P = np.array([[0.84, 0.09, 0.07], [0.10, 0.82, 0.08], [0.16, 0.14, 0.70],
                  [0.48, 0.44, 0.08], [0.44, 0.40, 0.16], [0.30, 0.26, 0.44]])
    ax.add_patch(_poly(hull_poly(to_xy(P)), fc=HULLFILL, ec=ACCENT, lw=1.0,
                       zorder=2))
    kinds = [("mandatory", ACCENT2, "*", 60), ("mandatory", ACCENT2, "*", 60),
             ("mandatory", ACCENT2, "*", 60), ("optional", ACCENT4, "o", 24),
             ("optional", ACCENT4, "o", 24), ("nonessential", MUTED, "x", 22)]
    seen = set()
    for p, (lab, col, mk, s) in zip(P, kinds):
        xy = to_xy(p)
        ax.scatter([xy[0]], [xy[1]], s=s, c=col, marker=mk, zorder=6,
                   edgecolors="white" if mk != "x" else col,
                   linewidths=0.7 if mk != "x" else 1.1)
        if lab not in seen:
            seen.add(lab)
            dy = {"mandatory": -0.055, "optional": -0.055}.get(lab, 0.045)
            ax.text(xy[0], xy[1] + dy, lab, fontsize=6.2, color=col,
                    ha="center", va="top" if dy < 0 else "bottom")
    ax.text(0.5, CAPY, "test by forcing $z_j=0$ and re-solving at $k^\\star$",
            fontsize=6.4, color=MUTED, ha="center")

    ax = bare_ax(fig, r2, title="selection frequency across 200 resamples")
    rng = np.random.default_rng(4)
    freq = np.sort(np.concatenate([rng.uniform(0.95, 1.0, 3),
                                   rng.uniform(0.25, 0.7, 5),
                                   rng.uniform(0.0, 0.15, 6)]))[::-1]
    cols = [ACCENT2 if f > 0.9 else (ACCENT4 if f > 0.2 else MUTED) for f in freq]
    ax.bar(np.arange(len(freq)), freq, color=cols, width=0.68)
    ax.axhline(0.9, color=ACCENT2, lw=0.7, ls="--")
    ax.axhline(0.2, color=ACCENT4, lw=0.7, ls="--")
    ax.set_xticks([])
    ax.set_xlabel("judges, sorted", fontsize=7)
    ax.set_ylabel("fraction of optima containing $j$", fontsize=7)
    ax.set_ylim(0, 1.08)


# ---------------------------------------------------------------------------
# 8. certification
# ---------------------------------------------------------------------------
def _eb_slack(n, delta, var, b=1.0):
    L = np.log(2.0 / delta)
    return np.sqrt(2 * var * L / n) + 7 * b * L / (3 * (n - 1))


def _hoef_slack(n, delta, b=1.0):
    return b * np.sqrt(np.log(2.0 / delta) / (2 * n))


def fig_certification(fig, rect):
    rs = panels(fig, rect, 3, pad=0.026, widths=[1.0, 1.15, 1.0])
    rng = np.random.default_rng(9)
    loss = np.clip(rng.gamma(2.0, 0.022, 400), 0, 1)
    ax = bare_ax(fig, rs[0], title="per-item losses on CERT")
    ax.hist(loss, bins=26, color=HULLFILL, edgecolor=ACCENT, linewidth=0.5)
    mu = loss.mean()
    ucb = mu + _eb_slack(len(loss), 0.05 / 140, loss.var())
    ax.axvline(mu, color=ACCENT, lw=1.1)
    ax.axvline(ucb, color=ACCENT2, lw=1.1)
    ax.axvline(0.08, color=ACCENT3, lw=1.1, ls="--")
    ax.text(mu, ax.get_ylim()[1] * 0.98, " $\\widehat U$", fontsize=6.6,
            color=ACCENT, va="top")
    ax.text(ucb, ax.get_ylim()[1] * 0.80, " UCB", fontsize=6.6, color=ACCENT2,
            va="top")
    ax.text(0.08, ax.get_ylim()[1] * 0.62, " $\\gamma$", fontsize=6.6,
            color=ACCENT3, va="top")
    ax.set_xlabel("$\\ell(p_i,\\widehat p_i)$", fontsize=7)
    ax.set_ylabel("items", fontsize=7)
    ax.set_xlim(0, 0.22)

    ax = bare_ax(fig, rs[1], title="how much slack you are paying")
    n = np.arange(80, 1200, 10)
    var = 0.0012
    M = 140
    ax.plot(n, _hoef_slack(n, 0.05 / M), lw=1.2, color=ACCENT2,
            label="Hoeffding, $b=1$")
    ax.plot(n, _eb_slack(n, 0.05 / M, var), lw=1.2, color=ACCENT4,
            label="emp. Bernstein, $b=1$")
    ax.plot(n, _eb_slack(n, 0.05 / M, var, b=0.25), lw=1.2, color=ACCENT,
            label="emp. Bernstein, $b=0.25$")
    ax.plot(n, 1.9 * np.sqrt(var / n) + 0.004, lw=1.2, color=ACCENT3,
            label="grouped bootstrap (max)")
    ax.axhline(0.08, color=INK, lw=0.8, ls="--")
    ax.text(1180, 0.084, "$\\gamma=0.08$", fontsize=6.3, color=INK, ha="right")
    ax.axvline(460, color=MUTED, lw=0.7, ls=":")
    ax.text(470, 0.115, "CERT size for\nRewardBench 2", fontsize=6.0,
            color=MUTED, va="top")
    ax.set_xlabel("certification items $n$", fontsize=7)
    ax.set_ylabel("width added to $\\widehat U$", fontsize=7)
    ax.legend(fontsize=5.9, frameon=False, loc="upper right",
              handlelength=1.4, labelspacing=0.35)
    ax.set_ylim(0, 0.36)

    ax = bare_ax(fig, rs[2], title="calibration over 200 splits")
    meth = ["naive", "per-pair", "Holm", "bootstrap"]
    viol = [0.31, 0.17, 0.031, 0.048]
    cols = [ACCENT2, ACCENT4, ACCENT, ACCENT3]
    ax.bar(range(4), viol, color=cols, width=0.62)
    ax.axhline(0.05, color=INK, lw=0.9, ls="--")
    ax.text(3.45, 0.056, "nominal", fontsize=6.2, color=INK, ha="right")
    ax.set_xticks(range(4))
    ax.set_xticklabels(meth, fontsize=6.0)
    ax.set_ylabel("TEST violation rate", fontsize=7)
    ax.set_ylim(0, 0.36)


def fig_downstream(fig, rect):
    r1, r2 = panels(fig, rect, 2, pad=0.03, widths=[1.15, 1.0])
    ax = bare_ax(fig, r1, title="rank flips need a margin of $2E$")
    rng = np.random.default_rng(6)
    s = np.sort(rng.uniform(0.35, 0.75, 7))[::-1]
    E = 0.022
    y = np.arange(len(s))
    ax.errorbar(s, y, xerr=E, fmt="o", ms=4, color=ACCENT, ecolor=ACCENT4,
                elinewidth=1.6, capsize=2.0)
    for i in range(len(s) - 1):
        if s[i] - s[i + 1] < 2 * E:
            ax.add_patch(__import__("matplotlib").patches.Rectangle(
                (s[i + 1] - E, i + 0.1), (s[i] + E) - (s[i + 1] - E), 0.8,
                facecolor=ACCENT2, alpha=0.14, edgecolor="none"))
            ax.text((s[i] + s[i + 1]) / 2, i + 0.5, "can flip", fontsize=5.9,
                    color=ACCENT2, ha="center", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(["sys %d" % (i + 1) for i in y], fontsize=6.3)
    ax.set_xlabel("full-panel score", fontsize=7)
    ax.invert_yaxis()

    ax = bare_ax(fig, r2, title="error budget through aggregation")
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    rows = [("per-item TV", "$E$", ACCENT),
            ("signed margin", "$2E$", ACCENT4),
            ("convex aggregate", "$E$", ACCENT),
            ("system score", "$E$", ACCENT),
            ("no-flip margin", "$2E$", ACCENT3),
            ("margin-scale scores", "$4E$", ACCENT2)]
    for i, (a, b, c) in enumerate(rows):
        yy = 0.90 - i * 0.145
        ax.text(0.02, yy, a, fontsize=7.2, color=INK, va="center")
        ax.text(0.94, yy, b, fontsize=8.0, color=c, va="center", ha="right")
        ax.plot([0.02, 0.94], [yy - 0.055, yy - 0.055], color=RULE, lw=0.5)


_HELLY_C = [np.array([0.66, 0.20, 0.14]), np.array([0.20, 0.66, 0.14]),
            np.array([0.24, 0.24, 0.52])]


def _helly_radii():
    """Radii where (a) all pairs meet but not all three, (b) all three meet."""
    r_pair = r_all = None
    for rad in np.arange(0.06, 0.60, 0.004):
        polys = [tv_ball(c, rad) for c in _HELLY_C]
        pair = all(len(poly_intersect(polys[i], polys[j])) >= 3
                   for i, j in itertools.combinations(range(3), 2))
        trip = len(poly_intersect(poly_intersect(polys[0], polys[1]),
                                  polys[2])) >= 3
        if pair and r_pair is None:
            r_pair = rad
        if trip and r_all is None:
            r_all = rad
            break
    return 0.5 * (r_pair + r_all), r_all * 1.12


def fig_helly(fig, rect):
    r1, r2 = panels(fig, rect, 2, pad=0.028)
    rad_a, rad_b = _helly_radii()
    cols = [ACCENT, ACCENT3, ACCENT4]
    cases = [(r1, rad_a, "pairwise compatible, but not jointly"),
             (r2, rad_b, "one more retained judge widens every set")]
    for r, rad, title in cases:
        ax = simplex_ax(fig, r, vlab=("$w_1$", "$w_2$", "$w_3$"), title=title)
        polys = []
        for k, (c, col) in enumerate(zip(_HELLY_C, cols)):
            poly = tv_ball(c, rad)
            polys.append(poly)
            ax.add_patch(_poly(poly, fc=col, ec=col, lw=0.9, alpha=0.20,
                               zorder=2))
            ax.text(c[0] * 0 + to_xy(c)[0], to_xy(c)[1], "$c_%d$" % (k + 1),
                    fontsize=6.6, color=col, ha="center", va="center",
                    zorder=7)
        inter = poly_intersect(poly_intersect(polys[0], polys[1]), polys[2])
        if len(inter) >= 3:
            ax.add_patch(_poly(inter, fc=ACCENT2, ec=ACCENT2, lw=1.0,
                               alpha=0.60, zorder=4))
            ax.text(0.5, CAPY, "$\\bigcap_c W_c\\neq\\emptyset$: one $w$ serves "
                    "every context", fontsize=6.4, color=ACCENT2, ha="center")
        else:
            for i, j in itertools.combinations(range(3), 2):
                pw = poly_intersect(polys[i], polys[j])
                if len(pw) >= 3:
                    ax.add_patch(_poly(pw, fc="none", ec=ACCENT2, lw=0.8,
                                       ls=(0, (2.2, 1.6)), zorder=4))
            ax.text(0.5, CAPY, "every pair overlaps, yet $\\bigcap_c W_c"
                    "=\\emptyset$: no single $w$ works", fontsize=6.4,
                    color=ACCENT2, ha="center")


def fig_caratheodory(fig, rect):
    r1, r2 = panels(fig, rect, 2, pad=0.028, widths=[1.0, 1.15])
    P = np.array([[0.80, 0.12, 0.08], [0.52, 0.42, 0.06], [0.16, 0.74, 0.10],
                  [0.14, 0.42, 0.44], [0.26, 0.16, 0.58], [0.58, 0.14, 0.28]])
    t = np.array([0.42, 0.34, 0.24])
    ax = simplex_ax(fig, r1, title="any point needs only 3 of them")
    ax.add_patch(_poly(hull_poly(to_xy(P)), fc=HULLFILL, ec=ACCENT, lw=1.0,
                       zorder=2))
    for i, p in enumerate(P):
        dot(ax, p, ACCENT, s=18)
    tri = P[[0, 2, 4]]
    ax.add_patch(_poly(to_xy(tri), fc=ACCENT3, ec=ACCENT3, lw=1.0, alpha=0.25,
                       zorder=3))
    for p in tri:
        dot(ax, p, ACCENT3, s=26)
    dot(ax, t, ACCENT2, s=30, marker="D")
    ax.text(0.5, CAPY, "in the plane, $r+1=3$ suffices \u2014 always",
            fontsize=6.4, color=MUTED, ha="center")

    ax = bare_ax(fig, r2, title="support size follows the effective rank")
    r = np.arange(1, 9)
    ax.plot(r, r + 1, "-o", ms=3.2, lw=1.2, color=ACCENT,
            label="Carath\u00e9odory bound $r+1$")
    ax.plot(r, np.minimum(r + 1, 4.0) - 0.3 + 0.05 * r, "-s", ms=3.2, lw=1.2,
            color=ACCENT2, label="typical LP support")
    ax.set_xlabel("effective rank $r$ of the response matrix", fontsize=7)
    ax.set_ylabel("non-zero weights per judge", fontsize=7)
    ax.legend(fontsize=6.1, frameon=False, loc="upper left")
    ax.set_ylim(0, 10)


# ---------------------------------------------------------------------------
# 9. cheat sheet
# ---------------------------------------------------------------------------
def fig_cheatsheet(fig, rect):
    x0, y0, w, h = rect
    rows = 2
    cols = 3
    pad = 0.014
    gap = 0.055
    head = 0.045          # headroom so the top row's title is not clipped
    ww = (w - pad * (cols - 1)) / cols
    hh = (h - head - gap * (rows - 1)) / rows
    cells = []
    for r in range(rows):
        for c in range(cols):
            cells.append([x0 + c * (ww + pad), y0 + (rows - 1 - r) * (hh + gap),
                          ww, hh])

    S = np.array([[0.78, 0.14, 0.08], [0.20, 0.70, 0.10], [0.30, 0.22, 0.48]])

    ax = simplex_ax(fig, cells[0], title="1. a judge is a point")
    dot(ax, [0.5, 0.3, 0.2], ACCENT2, s=30)

    ax = simplex_ax(fig, cells[1], title="2. error is a hexagon radius")
    p = np.array([0.42, 0.33, 0.25])
    ax.add_patch(_poly(tv_ball(p, 0.12), fc=ACCENT2, ec=ACCENT2, lw=0.8,
                       alpha=0.22, zorder=2))
    dot(ax, p, ACCENT2, s=22)

    ax = simplex_ax(fig, cells[2], title="3. a panel is a polygon")
    ax.add_patch(_poly(hull_poly(to_xy(S)), fc=HULLFILL, ec=ACCENT, lw=1.0,
                       zorder=2))
    for s in S:
        dot(ax, s, ACCENT, s=20)

    ax = simplex_ax(fig, cells[3], title="4. coverage = every judge inside")
    ax.add_patch(_poly(hull_poly(to_xy(S)), fc=HULLFILL, ec=ACCENT, lw=1.0,
                       zorder=2))
    for s in S:
        dot(ax, s, ACCENT, s=20)
    for q in [[0.45, 0.35, 0.20], [0.40, 0.30, 0.30], [0.50, 0.28, 0.22]]:
        dot(ax, q, ACCENT3, s=16, marker="D")

    ax = simplex_ax(fig, cells[4], title="5. selection = pick the corners")
    ax.add_patch(_poly(hull_poly(to_xy(S)), fc=HULLFILL, ec=ACCENT, lw=1.0,
                       zorder=2))
    for s in S:
        dot(ax, s, ACCENT2, s=36, marker="*")

    ax = simplex_ax(fig, cells[5], title="6. curvature $\\Rightarrow$ no free lunch")
    P = _circle_judges(12)
    ax.add_patch(_poly(hull_poly(to_xy(P[::3])), fc=HULLFILL, ec=ACCENT, lw=1.0,
                       zorder=2))
    for p in P:
        dot(ax, p, MUTED, s=8)
    for p in P[::3]:
        dot(ax, p, ACCENT, s=18)
