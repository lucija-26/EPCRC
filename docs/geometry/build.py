"""Build JUDGE_PANEL_GEOMETRY.pdf -- a visual math companion to the
LLM judge-panel compression plan.

    python build.py            ->  ../JUDGE_PANEL_GEOMETRY.pdf
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import figs
from doclib import (ACCENT, ACCENT2, ACCENT3, HULLFILL, INK, MUTED, RULE, Doc,
                    measure)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "JUDGE_PANEL_GEOMETRY.pdf")

TITLE = "The Geometry of Judge-Panel Compression"
SUB = "a visual companion to the execution plan"


# ---------------------------------------------------------------------------
# cover
# ---------------------------------------------------------------------------
def cover(fig, doc):
    W, H = doc.W, doc.H
    ml = doc.ml / W
    fig.text(ml, 0.845, TITLE, fontsize=20.5, fontweight="bold", color=INK,
             ha="left", va="baseline")
    fig.add_artist(plt.Line2D([ml, 1 - doc.mr / W], [0.828, 0.828],
                              color=ACCENT, lw=1.6, transform=fig.transFigure))
    fig.text(ml, 0.795, SUB, fontsize=12.5, color=MUTED, ha="left",
             va="baseline", style="italic")

    ax = fig.add_axes([0.15, 0.255, 0.70, 0.40])
    ax.set_aspect("equal")
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.14, figs.S3 + 0.14)
    ax.axis("off")
    rng = np.random.default_rng(11)
    P = []
    for _ in range(17):
        q = np.abs(rng.normal(size=3)) ** 1.7 + 0.06
        P.append(q / q.sum())
    P = np.array(P)
    hull = figs.hull_poly(figs.to_xy(P))
    ax.add_patch(figs._poly(hull, fc=HULLFILL, ec=ACCENT, lw=1.4, zorder=2))
    ax.add_patch(figs._poly(figs.TRI, fc="none", ec=INK, lw=1.4, zorder=3))
    keep = set(np.argmin(np.linalg.norm(
        figs.to_xy(P)[:, None, :] - hull[None, :, :], axis=2), axis=0))
    for k, p in enumerate(P):
        if k in keep:
            figs.dot(ax, p, ACCENT, s=52, marker="*", lw=0.6)
        else:
            figs.dot(ax, p, MUTED, s=13, lw=0.5)
    for v, lab, o in zip(figs.VERT, ["A better", "B better", "tie"],
                         [(-0.02, -0.055), (0.02, -0.055), (0.0, 0.028)]):
        ax.text(v[0] + o[0], v[1] + o[1], lab, fontsize=9, color=INK,
                ha={-0.02: "right", 0.02: "left", 0.0: "center"}[o[0]],
                va="top" if o[1] < 0 else "bottom")
    ax.text(0.5, -0.125, "%d judges, %d kept: the rest are convex combinations"
            % (len(P), len(keep)), fontsize=9, color=MUTED, ha="center")

    body = ("Every idea in the plan -- reconstruction error, coverage, "
            "non-composability, the compression frontier, certification -- "
            "is a statement about points, polygons and distances in one "
            "triangle. This companion derives the mathematics and draws "
            "each step.")
    from doclib import Para
    Para(body, size=10.6, leading=1.5).draw(fig, doc.ml, 2.55, W - doc.ml - doc.mr)
    fig.text(ml, 0.085, "EPCRC special course", fontsize=9.5, color=MUTED,
             ha="left", va="baseline")
    fig.text(1 - doc.mr / W, 0.085, "built from figs.py -- every figure is "
             "computed, not drawn by hand", fontsize=8.2, color=MUTED,
             ha="right", va="baseline")


# ---------------------------------------------------------------------------
# table of contents
# ---------------------------------------------------------------------------
def make_toc(entries):
    def draw(fig, rect):
        x0, y0, w, h = rect
        ax = fig.add_axes(rect)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        n = len(entries)
        for k, (num, name, page) in enumerate(entries):
            y = 1 - (k + 0.6) / n
            ax.text(0.0, y, num, fontsize=9.6, color=ACCENT, va="center",
                    fontweight="bold")
            ax.text(0.062, y, name, fontsize=9.6, color=INK, va="center")
            wname = measure(name, 9.6)[0] / (w * 8.27) + 0.062
            ax.plot([wname + 0.012, 0.965], [y - 0.004, y - 0.004],
                    color=RULE, lw=0.5, ls=(0, (1.2, 1.8)))
            ax.text(1.0, y, str(page) if page else "", fontsize=9.6,
                    color=MUTED, va="center", ha="right")
    return draw


SECTIONS = [
    ("1", "A judge is a point in a triangle"),
    ("2", "The ruler: total variation"),
    ("3", "Reconstruction is a convex hull"),
    ("4", "One weight vector for all items"),
    ("5", "The projection is a linear program"),
    ("6", "Coverage of a set, and what is monotone"),
    ("7", "Why individual redundancy does not compose"),
    ("8", "The compression frontier"),
    ("9", "Search algorithms, seen geometrically"),
    ("10", "Certificates: the separating direction"),
    ("11", "Exact optimisation and the backbone"),
    ("12", "Honest certification on held-out data"),
    ("13", "Downstream preservation"),
    ("14", "Helly and Caratheodory"),
    ("15", "One page to remember"),
]


# ---------------------------------------------------------------------------
def build(page_of=None):
    d = Doc(running_title=TITLE)
    d.cover = cover

    entries = [(n, t, (page_of or {}).get("%s  %s" % (n, t), ""))
               for n, t in SECTIONS]
    d.h1("Contents")
    d.fig(make_toc(entries), 0.30 * len(SECTIONS))
    d.space(0.12)
    d.box("picture", [
        "Hold one image in your head for the whole document: **the triangle "
        "of three-class judge outputs.** A judge on an item is a point in it. "
        "A retained panel is a polygon inside it. Compression asks how few "
        "corners that polygon needs before every judge still sits (almost) "
        "inside.",
        "Everything else -- linear programs, MILPs, confidence bounds -- is "
        "bookkeeping attached to that picture."], title="The one picture")
    d.newpage()

    # -- 1 -----------------------------------------------------------------
    d.h1("A judge is a point in a triangle", "1")
    d.p("Fix an item $x$ and a context $c$. Judge $i$ does not return a "
        "verdict; it returns a probability over three exclusive outcomes: "
        "$A$ is better, $B$ is better, or the two tie. Write that output as")
    d.eq(r"p_{i,c}(x)=\left(p_A,\ p_B,\ p_T\right),\qquad p_k\geq 0,"
         r"\qquad p_A+p_B+p_T=1.")
    d.p("Those two conditions are the entire object. Non-negativity puts the "
        "point in the positive octant of $\\mathrm{R}^3$; the sum condition "
        "puts it on a plane. The intersection is a filled equilateral "
        "triangle, the **2-simplex** $\\Delta^2$. Three coordinates, two "
        "degrees of freedom -- which is exactly why you can draw it flat on "
        "paper and lose nothing.")
    d.fig(figs.fig_simplex_intro, 2.35, num=1, caption=(
        "Left: the plane $p_A+p_B+p_T=1$ cuts the positive octant in a "
        "triangle. The dashed ray is the direction of $p$; the dotted "
        "segments are its three coordinates. Right: the same triangle seen "
        "face-on. Barycentric coordinates make each $p_k$ the scaled "
        "distance from $p$ to the edge opposite vertex $k$, so the grid "
        "lines are the level sets $p_k=0.2,0.4,\\ldots$"))
    d.p("Reading the triangle takes ten seconds and then it is permanent. "
        "The three corners are the deterministic verdicts. The centroid "
        "$(1/3,1/3,1/3)$ is total indecision. Each edge is the face where "
        "one outcome has been ruled out; the bottom edge $p_T=0$ holds the "
        "purely binary judges that never allow a tie. Distance from an edge "
        "is probability mass.")
    d.fig(figs.fig_landmarks, 2.35, num=2, caption=(
        "Left: the landmarks of judge behaviour. A confident partisan lives "
        "near a corner, a hedger near the centroid, a tie-prone judge high "
        "up. Right: the signed margin $y=p_A-p_B$ is an affine function, so "
        "its level sets are straight lines; two very different judges can "
        "share one. Collapsing to $y$ projects the triangle onto a segment "
        "and destroys the tie coordinate."))
    d.box("key", [
        "The plan insists on three-class probabilities rather than a scalar "
        "margin, and Figure 2 is the reason. The map $p\\mapsto p_A-p_B$ is "
        "a projection of a 2-dimensional object onto a 1-dimensional one; "
        "an entire line of judges collapses to a point. Two judges that "
        "agree on who wins can disagree completely on how often nobody "
        "does, and a compressed panel that only preserves $y$ cannot be "
        "audited for that."])

    # -- 2 -----------------------------------------------------------------
    d.h1("The ruler: total variation", "2")
    d.p("To say a reconstruction is good we need a distance between two "
        "points of the triangle. The plan fixes total variation:")
    d.eq(r"\ell(p,\widehat p)=\frac{1}{2}\|p-\widehat p\|_1"
         r"=\frac{1}{2}\sum_{k\in\{A,B,T\}}|p_k-\widehat p_k|.", tag="2.1")
    d.p("Three properties earn it the job. It is a metric, so the triangle "
        "inequality is available. It lands in $[0,1]$ with a probabilistic "
        "meaning: $\\ell(p,\\widehat p)$ is the largest disagreement in "
        "probability that the two judges can have about any event. And it is "
        "bounded, which is what later lets an empirical Bernstein bound turn "
        "a sample mean into a certificate.")
    d.p("On the 3-simplex it has a shape worth knowing. Let "
        "$\\delta=p-\\widehat p$. Because both points sum to one, "
        "$\\delta_A+\\delta_B+\\delta_T=0$, so either one coordinate is "
        "positive and two are negative or the reverse. In both cases the "
        "lone coordinate carries exactly half of $\\|\\delta\\|_1$, which "
        "gives an identity that is special to three outcomes:")
    d.eq(r"\ell(p,\widehat p)=\max_{k}\,|p_k-\widehat p_k|.", tag="2.2")
    d.p("So the total-variation ball of radius $\\varepsilon$ is the set cut "
        "out by three pairs of parallel constraints "
        "$|q_k-p_k|\\leq\\varepsilon$. Six half-planes meeting the triangle: "
        "a regular hexagon, clipped to a pentagon or a quadrilateral when "
        "the centre sits near the boundary. It is convex and centrally "
        "symmetric -- the two facts the optimisation will lean on.")
    d.fig(figs.fig_tv_ball, 2.45, num=3, caption=(
        "Total-variation balls in the simplex. The three slabs "
        "$|q_k-p_k|\\leq\\varepsilon$ intersect in a hexagon (left, centre); "
        "near the boundary the simplex itself truncates it (right). Read "
        "$\\varepsilon$ as the tolerance $\\gamma$: the hexagon around a "
        "judge is the region of reconstructions you are willing to accept."))

    # -- 3 -----------------------------------------------------------------
    d.h1("Reconstruction is a convex hull", "3")
    d.p("Now the compression idea. Keep a subset $S\\subseteq\\mathcal{J}$ "
        "of judges physically, and rebuild every retired judge $i$ as a "
        "weighted blend of the survivors:")
    d.eq(r"\widehat p_{i,c}(x)=\sum_{j\in S}w_{ij}\,p_{j,c}(x),"
         r"\qquad w_i\in\Delta^{|S|-1}.", tag="3.1")
    d.p("The weights are non-negative and sum to one, so $\\widehat p_i$ is "
        "automatically a probability vector -- no renormalisation, no "
        "clipping. That is the payoff of insisting on a simplex constraint "
        "rather than free least-squares coefficients.")
    d.p("It also settles the reachability question immediately. As $w_i$ "
        "ranges over the simplex, $\\widehat p_{i}$ ranges over exactly the "
        "convex hull of the retained points:")
    d.eq(r"\left\{\sum_{j\in S}w_{ij}p_j\ :\ "
         r"w_i\in\Delta^{|S|-1}\right\}=\mathrm{conv}\{p_j:j\in S\}.")
    d.p("A retained set is therefore a **polygon** in the triangle, and the "
        "best possible single-item error for target $i$ is the "
        "total-variation distance from $p_i$ to that polygon: the radius of "
        "the smallest hexagon centred at $p_i$ that touches it. If $p_i$ "
        "lies inside, the error is zero -- which is also why a retained "
        "judge reconstructs itself perfectly, using the unit vector "
        "$w_{ii}=1$.")
    d.fig(figs.fig_hull, 2.5, num=4, caption=(
        "Left: the target sits inside the polygon spanned by the retained "
        "judges, so some convex combination hits it exactly. Right: the "
        "target sits outside; the error is the radius of the smallest "
        "total-variation hexagon centred on it that reaches the polygon, "
        "and the optimal blend is the touching point on the boundary."))
    d.box("picture", [
        "Compression is now a covering problem you can see: **choose few "
        "points of the cloud whose convex hull swallows the whole cloud, up "
        "to a hexagon of radius $\\gamma$.** Interior judges are free to "
        "retire. Extreme judges -- the ones on the boundary of the cloud -- "
        "are the ones that carry the panel."])

    # -- 4 -----------------------------------------------------------------
    d.h1("One weight vector for all items", "4")
    d.p("If that were the whole story the answer would just be the vertex "
        "set of the hull, computed once. The difficulty is that the weights "
        "are fitted per target judge but shared across every item. Judge "
        "$i$ gets one vector $w_i$, and it must work for all $x$ "
        "simultaneously. Per item the reconstruction problem is a "
        "projection; across items it is a compromise.")
    d.fig(figs.fig_one_weight, 2.45, num=5, caption=(
        "Three items, one weight. Each panel shows the reconstruction error "
        "for item $x$ as the weight moves along a one-parameter family of "
        "blends: a convex, piecewise-linear V. Individually each item wants "
        "a different minimiser; the fitted $w$ minimises their average "
        "(right), so no item is reconstructed as well as it could be alone."))
    d.p("Define, for a fixed target $i$ and context $c$, the mean "
        "reconstruction error as a function of the weights:")
    d.eq(r"\varepsilon_{i,c}(w)=\frac{1}{|Q_c|}\sum_{x\in Q_c}"
         r"\ell\!\left(p_{i,c}(x),\ \sum_{j\in S}w_j\,p_{j,c}(x)\right).",
         tag="4.1")
    d.p("This function is convex. Each summand is a norm of an affine "
        "function of $w$, hence convex; by identity (2.2) it is even a "
        "maximum of six affine functions, so each summand is piecewise "
        "linear; sums and maxima of convex piecewise-linear functions stay "
        "convex and piecewise linear. The landscape over the weight simplex "
        "is a bowl with straight creases, and it has no local minima that "
        "are not global.")
    d.p("The plan then fits one weight vector per target using the worst "
        "context rather than the average, so that a small stress context "
        "cannot be drowned out by a large clean one:")
    d.eq(r"\widehat w_i^{\mathrm{fit}}(S)\in\arg\min_{w\in\Delta^{|S|-1}}"
         r"\ \max_{c\in\mathcal{C}_{\mathrm{fit}}}\ \varepsilon_{i,c}(w).",
         tag="4.2")
    d.p("A maximum over finitely many convex piecewise-linear functions is "
        "again convex and piecewise linear, so (4.2) is still a "
        "well-behaved convex program. Geometrically, each context defines a "
        "convex sublevel set $W_{i,c}(\\gamma)=\\{w:\\varepsilon_{i,c}(w)"
        "\\leq\\gamma\\}$, and robust feasibility means these sets share a "
        "point.")
    d.fig(figs.fig_weight_simplex, 2.5, num=6, caption=(
        "The weight simplex for a three-judge retained set. Left: the "
        "robust objective $\\max_c\\varepsilon_c(w)$, a convex bowl with "
        "creases where the active context or the active item changes. "
        "Right: the per-context feasible sets at tolerance $\\gamma$ are "
        "convex; their intersection is the set of weights that survive "
        "every registered context."))

    # -- 5 -----------------------------------------------------------------
    d.h1("The projection is a linear program", "5")
    d.p("Convex and piecewise linear means linear-programmable. Two standard "
        "moves do it. First, split each absolute value: to encode "
        "$t\\geq|a|$ write the pair $t\\geq a$, $t\\geq -a$, which is exact at "
        "the optimum because we are minimising. Second, the epigraph trick: "
        "to minimise a maximum, introduce one scalar $u$ that sits above "
        "every branch and minimise $u$.")
    d.fig(figs.fig_epigraph, 2.3, num=7, caption=(
        "The epigraph trick. Left: three contexts give three convex "
        "piecewise-linear costs; their upper envelope is what (4.2) "
        "minimises. Right: minimising the envelope equals minimising a new "
        "variable $u$ constrained to lie above all of them -- and the "
        "region above a piecewise-linear convex curve is an intersection of "
        "half-planes, so the whole thing is linear."))
    d.p("Writing $n_c=|Q_c|$ and $r_k(x,w)=p_{i,c,k}(x)-\\sum_j w_j "
        "p_{j,c,k}(x)$ for the signed residual in outcome $k$, the fit for "
        "one target judge is")
    d.eq(r"\min_{w,\,t,\,u}\ u")
    d.eq(r"\mathrm{s.t.}\quad t_{x,k}\ \geq\ \pm\,r_k(x,w)\quad"
         r"\forall x,\ \forall k\in\{A,B,T\},")
    d.eq(r"\frac{1}{2n_c}\sum_{x\in Q_c}\sum_k t_{x,k}\ \leq\ u\quad"
         r"\forall c\in\mathcal{C}_{\mathrm{fit}},")
    d.eq(r"\sum_{j\in S}w_j=1,\qquad w\geq 0,\qquad t\geq 0.")
    d.p("The size is modest: $|S|+3n+1$ variables and $6n+|\\mathcal{C}|+1$ "
        "constraints for $n$ fitting items. HiGHS through SciPy solves it in "
        "milliseconds, which matters because the outer search will call this "
        "oracle hundreds of thousands of times -- once per target judge per "
        "candidate set.")
    d.box("note", [
        "Two accelerations are worth building in from the start, because "
        "they are pure wins. Cache every evaluated set under a sorted "
        "bitmask key, since greedy and swap search revisit sets constantly. "
        "And screen targets: if a cheap lower bound on target $i$'s error "
        "already exceeds the incumbent worst target, its LP never needs to "
        "be solved."])

    # -- 6 -----------------------------------------------------------------
    d.h1("Coverage of a set, and what is monotone", "6")
    d.p("Fitting handles one target. The quality of a retained set is the "
        "worst case over all of them:")
    d.eq(r"\widehat E_{\mathrm{fit}}(S)=\max_{i\in\mathcal{J}}"
         r"\ \max_{c\in\mathcal{C}_{\mathrm{fit}}}"
         r"\ \varepsilon_{i,c}\!\left(\widehat w_i^{\mathrm{fit}}(S)\right).",
         tag="6.1")
    d.p("A single number per subset, produced by $|\\mathcal{J}|$ linear "
        "programs. Note the two maxima are deliberate: coverage is a promise "
        "about every retired judge in every registered context, not an "
        "average that a few good reconstructions can rescue.")
    d.h3("The one thing that is monotone")
    d.p("If $S\\subseteq T$ then $\\widehat E_{\\mathrm{fit}}(T)\\leq"
        "\\widehat E_{\\mathrm{fit}}(S)$. The proof is one sentence: any "
        "$w\\in\\Delta^{|S|-1}$ extends to $\\Delta^{|T|-1}$ by padding with "
        "zeros, achieving the same objective, so the larger problem "
        "minimises over a superset of the feasible set. Geometrically, "
        "adding a judge can only enlarge the polygon.")
    d.p("This is what makes backward elimination sound and what lets a "
        "branch-and-bound prune. It is also the only free lunch available, "
        "and it comes with two warnings.")
    d.fig(figs.fig_nested, 2.4, num=8, caption=(
        "Nested sets, growing polygons, non-increasing fitting coverage. "
        "The last panel is the warning: held-out coverage is not monotone. "
        "A judge added because it helps on the fitting split can pull the "
        "blend away from the truth on new data."))
    d.box("warn", [
        "**Held-out error is not monotone.** Only the re-optimised "
        "objective, on the same data and the same loss, is guaranteed to "
        "decrease. Adding a judge fits the fitting split better by "
        "construction and can still generalise worse. Report "
        "$\\widehat E_{\\mathrm{TEST}}$ separately and never argue from "
        "monotonicity about it.",
        "**Coverage is not submodular.** The natural set function here has "
        "no diminishing-returns guarantee, so the classical "
        "$1-1/e$ greedy bound does not apply. Greedy is a heuristic and must "
        "be measured against exact optima, not assumed near-optimal."])

    # -- 7 -----------------------------------------------------------------
    d.h1("Why individual redundancy does not compose", "7")
    d.p("This is claim C1, and it is the scientific centre of the plan. The "
        "tempting procedure is: test each judge on its own, collect all the "
        "ones that pass, retire them together. Formally, compute the "
        "leave-one-out error $U(i\\mid\\mathcal{J}\\setminus\\{i\\})$, form "
        "the individually removable set")
    d.eq(r"R_\gamma=\{i:\ U(i\mid\mathcal{J}\setminus\{i\})\leq\gamma\},"
         r"\qquad S^{\mathrm{naive}}_\gamma=\mathcal{J}\setminus R_\gamma,")
    d.p("and hope that $\\widehat E(S^{\\mathrm{naive}}_\\gamma)\\leq\\gamma$. "
        "It does not follow, and the geometry says why in one line: "
        "$U(i\\mid\\mathcal{J}\\setminus\\{i\\})$ is measured against a "
        "polygon that still contains all the other candidates for removal. "
        "Each test is conducted in a world where its own alibi is still "
        "present.")
    d.h3("The clean counterexample: twins")
    d.p("Put six judges at three extreme positions, two at each. Every judge "
        "has an identical twin, so every leave-one-out error is exactly "
        "zero: all six are individually removable at any tolerance. Remove "
        "all six and nothing is left. The minimum jointly feasible set has "
        "size three -- one representative per extreme.")
    d.fig(figs.fig_noncomp_dup, 2.4, num=9, caption=(
        "Duplicated extremes. Each judge is perfectly reconstructed by its "
        "twin, so leave-one-out redundancy is universal, yet the three "
        "extremes are jointly irreplaceable. The composition gap is as "
        "large as it can be."))
    d.h3("The realistic version: gentle curvature")
    d.p("Twins are easy to dismiss as a pathology, so the plan also needs "
        "the smooth case. Place judges along a slightly curved arc -- no "
        "duplicates, no clusters, just a one-dimensional trend with a little "
        "curvature, which is what a family of related models actually looks "
        "like. Each judge lies close to the chord joining its neighbours, so "
        "each is individually removable. But keeping only $k$ of them "
        "replaces the arc by an inscribed polygon, and the deepest gap is "
        "the sagitta of an arc of half-angle $\\alpha\\approx\\pi/k$:")
    d.eq(r"\mathrm{gap}(k)\;\approx\;R\left(1-\cos\alpha\right)"
         r"\;\approx\;\frac{R\alpha^2}{2}\;=\;\Theta\!\left(k^{-2}\right).",
         tag="7.1")
    d.p("Inverting, reaching tolerance $\\gamma$ needs "
        "$k=\\Theta(\\gamma^{-1/2})$ retained judges. Curvature, not "
        "duplication, is what forces a panel to be large: the error of the "
        "naive simultaneous removal grows quadratically in how many judges "
        "you retire at once, even though every individual test passed.")
    d.fig(figs.fig_noncomp_circle, 2.55, num=10, caption=(
        "Judges on a gentle arc. Left: every judge is within $\\gamma$ of "
        "the chord through its neighbours, so all pass leave-one-out. "
        "Centre: keeping a few of them leaves a visible gap at the middle "
        "of each chord. Right: the measured worst error against $k$ follows "
        "the $k^{-2}$ law of (7.1)."))
    d.box("key", [
        "Individual redundancy is a statement about one polygon; joint "
        "removability is a statement about a different, smaller polygon. "
        "The gap between them is the composition gap, and reporting it "
        "across tolerances and split seeds is what E0 exists to do. If it "
        "turns out to be near zero on real judges, the plan says to report "
        "that honestly -- the geometry permits it when the cloud has no "
        "curvature."])

    # -- 8 -----------------------------------------------------------------
    d.h1("The compression frontier", "8")
    d.p("Two optimisation views appear in the plan and they are inverses of "
        "each other. The budget form asks how good a panel of $k$ judges can "
        "be; the tolerance form asks how few judges buy a promise "
        "$\\gamma$:")
    d.eq(r"E^\star(k)=\min_{|S|\leq k}\widehat E_{\mathrm{fit}}(S),"
         r"\qquad k^\star(\gamma)=\min\{|S|:\widehat E_{\mathrm{fit}}(S)"
         r"\leq\gamma\}.", tag="8.1")
    d.p("By monotonicity $E^\\star$ is non-increasing, so $k^\\star$ is its "
        "generalised inverse: $k^\\star(\\gamma)=\\min\\{k:E^\\star(k)\\leq"
        "\\gamma\\}$. One curve answers both questions, which is why the "
        "budget form is the one to plot and the tolerance form the one to "
        "solve exactly.")
    d.p("The shape of that curve is the headline result of E1, and it is "
        "worth predicting in advance what each shape would mean.")
    d.fig(figs.fig_frontier_shapes, 2.35, num=11, caption=(
        "Frontier shapes and their interpretation. A sharp elbow means a "
        "genuine low-dimensional basis: a few judges span the panel and the "
        "rest are blends. A slow decay means the cloud is genuinely "
        "high-dimensional and compression buys little. A step means "
        "discrete behavioural families, each needing exactly one delegate."))

    # -- 9 -----------------------------------------------------------------
    d.h1("Search algorithms, seen geometrically", "9")
    d.p("The outer problem -- choose $S$ -- is combinatorial: "
        "$2^{20}$ subsets for the Core-20 panel, each costing "
        "$|\\mathcal{J}|$ linear programs. The plan specifies four "
        "heuristics, and the triangle explains their systematic biases.")
    d.li("**Forward selection** starts from the best singleton. But the best "
         "single judge is the one closest to all the others -- the medoid, "
         "which sits in the middle of the cloud. A middle point is almost "
         "never a vertex of the final hull, so forward selection begins by "
         "committing to a judge that the optimum would discard.")
    d.li("**Backward elimination** starts from the full panel and peels. Its "
         "first removals are interior points, which change the hull not at "
         "all, so it never pays for its early moves. It stops when every "
         "remaining removal breaks feasibility.")
    d.li("**Forward-trim** runs forward to feasibility and then eliminates "
         "backwards, repeatedly -- it can undo the medoid mistake.")
    d.li("**$k$-swap with trim** exchanges $k$ retained judges for $k$ "
         "outsiders and re-trims, which is the only move that escapes the "
         "local optima single removals cannot.")
    d.fig(figs.fig_forward_backward, 2.45, num=12, caption=(
        "Left and centre: the two greedy directions on the same cloud. "
        "Forward anchors on the medoid; backward peels interior points for "
        "free. Right: measured mean excess over the proven optimum on this "
        "repository's UTD19 instance, in extra retained models."))
    d.box("warn", [
        "In this repository the measured ordering is: backward $0.86$, "
        "forward $1.86$, forward-trim $0.57$, $k$-swap $0.29$ extra models "
        "on average. Backward beats plain forward by more than a factor of "
        "two. Only forward-trim -- a hybrid -- beats backward, and it must "
        "never be reported under the label 'forward'. The swap methods are "
        "exactly optimal in the loose-tolerance regime."])
    d.p("Why single moves get stuck is also geometric. Removing one judge "
        "can be infeasible while removing two and adding one back is fine, "
        "because the pair was jointly propping up one side of the hull. "
        "Swaps see that; single-move neighbourhoods cannot.")
    d.fig(figs.fig_swap_landscape, 2.4, num=13, caption=(
        "Left: a set that is a local optimum for single removals but not "
        "for exchanges -- the swap crosses a barrier no single deletion can. "
        "Right: neighbourhood sizes. The $2$-swap enumeration is "
        "$O(N^4)$ candidate exchanges before oracle cost, the $3$-swap "
        "$O(N^6)$, which is why the latter is a diagnostic only."))

    # -- 10 ----------------------------------------------------------------
    d.h1("Certificates: the separating direction", "10")
    d.p("When a candidate panel fails, the linear program does not merely "
        "say no. Its dual hands back a reason, and the reason is a direction "
        "in the triangle.")
    d.p("Separation is the underlying fact: if a point lies outside a "
        "closed convex set, some hyperplane has the point strictly on one "
        "side and the whole set on the other. Applied to $p_i$ and "
        "$\\mathrm{conv}\\{p_j:j\\in S\\}$, and measured in the norm dual "
        "to total variation -- for which the unit ball is "
        "$\\{\\lambda:\\max_k\\lambda_k-\\min_k\\lambda_k\\leq 1\\}$ -- the "
        "distance has a variational form:")
    d.eq(r"\mathrm{dist}_{\mathrm{TV}}\!\left(p_i,\ \mathrm{conv}\,P_S\right)"
         r"=\max_{\lambda}\ \left[\langle\lambda,p_i\rangle-"
         r"\max_{j\in S}\langle\lambda,p_j\rangle\right],", tag="10.1")
    d.p("the maximum being over $\\lambda$ in that dual ball. Any single "
        "$\\lambda$ gives a valid lower bound on the error, so an optimal "
        "$\\lambda$ is a **certificate of infeasibility**: it proves, with "
        "three numbers, that no weights whatsoever can reconstruct judge "
        "$i$ from $S$ within tolerance. And it is interpretable -- "
        "$\\lambda$ is a contrast over the three outcomes, so it names the "
        "behaviour the panel has lost.")
    d.fig(figs.fig_separating, 2.4, num=14, caption=(
        "Left: the dual optimum is the direction in which the target sticks "
        "out furthest beyond the retained polygon. Right: reading the "
        "certificate -- the gap in (10.1) is the error, and the sign "
        "pattern of $\\lambda$ says which outcome the reconstruction "
        "systematically gets wrong."))
    d.box("note", [
        "Certificates make the search cheap as well as honest. A single "
        "$\\lambda$ found for one candidate set stays valid for every "
        "subset of it, so one dual vector can prune a whole branch of the "
        "subset lattice without any further LP solves."])

    # -- 11 ---------------------------------------------------------------
    d.h1("Exact optimisation and the backbone", "11")
    d.p("Heuristics need a yardstick. For tractable instances the tolerance "
        "problem becomes a mixed-integer linear program: keep the LP above, "
        "add a binary $z_j$ per judge, and couple weights to it.")
    d.eq(r"\min_{z,w,t}\ \sum_{j}z_j\quad\mathrm{s.t.}\quad"
         r"\sum_j w_{ij}=1,\quad 0\leq w_{ij}\leq z_j,")
    d.eq(r"\frac{1}{2n_c}\sum_{x\in Q_c}\sum_k t^{(i)}_{x,k}\leq\gamma"
         r"\qquad\forall i,\ \forall c,\qquad z\in\{0,1\}^{N}.")
    d.p("The coupling $w_{ij}\\leq z_j$ is the whole modelling trick: a "
        "retired judge can carry no weight. Because $w_{ij}\\leq1$ already, "
        "no big-$M$ constant is needed, which keeps the relaxation as tight "
        "as this formulation allows -- and that is still not very tight.")
    d.h3("Why the relaxation is weak, and what fixes it")
    d.p("Relax $z$ to $[0,1]$. Nothing stops the solver from setting every "
        "$z_j=1/N$ scaled just enough to support the fractional weights it "
        "wants, so the LP bound collapses to a small constant while the "
        "true optimum is much larger. In this repository the dual bound sat "
        "at $2.0$ for every tolerance -- a property of the formulation, not "
        "of the solver.")
    d.p("The fix that worked was to stop asking the relaxation for the "
        "bound. Certify small cardinalities by exhaustive subset "
        "enumeration, then add the cardinality cut $\\sum_j z_j\\geq k+1$ "
        "once size $k$ has been ruled out; each infeasible subset also "
        "yields a no-good cut. With those, all seven tolerances on the "
        "$N=20$ UTD19 instance were proven optimal.")
    d.fig(figs.fig_milp, 2.35, num=15, caption=(
        "Left: the fractional relaxation spreads mass over all judges and "
        "certifies almost nothing, so the dual bound flatlines. Right: "
        "exhaustive certification of small sizes plus cardinality and "
        "no-good cuts closes the gap."))
    d.h3("Never read irreplaceability off one optimum")
    d.p("An optimal solution names a set, not the set. If judges 3 and 7 "
        "are interchangeable, the solver returns whichever it saw first, "
        "and reading that as evidence of importance is a mistake. The plan "
        "requires the explicit test: force $z_j=0$ and re-solve at "
        "$k^\\star(\\gamma)$; if the problem becomes infeasible, $j$ is "
        "mandatory. Forcing $z_j=1$ tests whether $j$ appears in any "
        "optimum at all.")
    d.fig(figs.fig_backbone, 2.4, num=16, caption=(
        "Backbone analysis. Left: the three categories -- mandatory "
        "backbone, optional representative, nonessential. Right: selection "
        "frequency across resamples, which separates a judge that is "
        "genuinely irreplaceable from one that merely won a tie-break."))

    # -- 12 ---------------------------------------------------------------
    d.h1("Honest certification on held-out data", "12")
    d.p("Everything so far is empirical: $\\widehat E_{\\mathrm{fit}}(S)$ is "
        "a minimum computed on the data used to choose $S$, so it is "
        "optimistically biased twice over -- once by fitting the weights, "
        "once by selecting the set. A deployment promise needs held-out "
        "data and an upper confidence bound.")
    d.p("Freeze the set and the weights, then compute per-item losses on the "
        "certification split. They are bounded in $[0,1]$, which admits an "
        "empirical Bernstein bound: with probability $1-\\delta'$,")
    d.eq(r"U_{i,c}\ \leq\ \bar U_{i,c}+\sqrt{\frac{2\widehat V_{i,c}"
         r"\log(3/\delta')}{n}}+\frac{3\log(3/\delta')}{n},", tag="12.1")
    d.p("with $\\widehat V$ the empirical variance. The variance term is why "
        "this beats Hoeffding here: reconstruction losses are usually small "
        "and tightly concentrated, so $\\widehat V\\ll1/4$ and the bound "
        "shrinks accordingly.")
    d.p("There are $|\\mathcal{J}|\\times|\\mathcal{C}|$ judge-context pairs "
        "and the promise is about the worst of them, so the levels must be "
        "corrected simultaneously -- $\\delta'=\\delta/(|\\mathcal{J}|"
        "|\\mathcal{C}|)$ by Bonferroni, or a grouped bootstrap over base "
        "items for the maximum statistic. Certification is then the single "
        "condition")
    d.eq(r"\max_{i,c}\ \mathrm{UCB}_{1-\delta}\!\left(U_{i,c}\right)"
         r"\ \leq\ \gamma.", tag="12.2")
    d.fig(figs.fig_certification, 2.45, num=17, caption=(
        "Left: per-item losses on the certification split, with the sample "
        "mean and the resulting upper bound. Centre: the slack you pay as a "
        "function of $n$, Hoeffding against empirical Bernstein. Right: "
        "calibration -- a nominal 95% certificate should exceed $\\gamma$ "
        "on test in about 5% of repeated splits, and no more."))
    d.box("warn", [
        "The certificate is relative to the registered contexts and the "
        "item distribution. It says nothing about arbitrary future tasks or "
        "unregistered interventions, and the write-up must say so. "
        "'Certified at $\\gamma$ over $\\mathcal{C}$' is the claim; "
        "'safe to retire' is not."])

    # -- 13 ---------------------------------------------------------------
    d.h1("Downstream preservation", "13")
    d.p("A panel exists to produce an aggregate. If the aggregation is "
        "linear, $A(P)=\\sum_i a_iP_i$, then reconstruction error passes "
        "through it without amplification:")
    d.eq(r"\left\|A(P)-A(\widehat P)\right\|\ \leq\ \sum_i|a_i|\,"
         r"\left\|P_i-\widehat P_i\right\|\ \leq\ \|a\|_1\,"
         r"\widehat E(S).", tag="13.1")
    d.p("For a normalised non-negative aggregation $\\|a\\|_1=1$, so the "
        "aggregate error is bounded by the worst per-judge error -- exactly "
        "the quantity coverage controls. That is the formal reason for "
        "defining coverage as a maximum rather than a mean.")
    d.p("The consequence that matters for system ranking follows from the "
        "triangle inequality. If two systems differ by margin $m$ under the "
        "true panel and each aggregate moves by at most $E$, their order "
        "can only flip if $m\\leq 2E$:")
    d.eq(r"m>2\,\widehat E(S)\ \Longrightarrow\ \mathrm{sign}\,"
         r"(A(\widehat P)_1-A(\widehat P)_2)="
         r"\mathrm{sign}\,(A(P)_1-A(P)_2).", tag="13.2")
    d.fig(figs.fig_downstream, 2.35, num=18, caption=(
        "Left: two system scores with margin $m$; each can move by $E$, so "
        "a flip requires $m\\leq2E$. Right: the error budget through "
        "aggregation -- how many pairs in a leaderboard fall inside the "
        "$2E$ band, which is what E2 must measure rather than assume."))
    d.box("note", [
        "The bound is worst-case and therefore loose: it assumes every "
        "judge errs in the same direction. Errors that behave like "
        "independent noise cancel, so measured flip rates should be far "
        "below the bound. Report both -- the guarantee and the empirical "
        "rate."])

    # -- 14 ---------------------------------------------------------------
    d.h1("Helly and Caratheodory", "14")
    d.p("Two classical theorems say something concrete about this problem, "
        "and both are visible in the triangle.")
    d.h3("Caratheodory: how sparse a reconstruction can be")
    d.p("In $\\mathrm{R}^r$, every point of the convex hull of a set is a "
        "convex combination of at most $r+1$ of its points. The simplex "
        "$\\Delta^2$ has $r=2$, so on a single item three retained judges "
        "always suffice, no matter how many are kept.")
    d.p("That does not mean $r=3$ sparse weights suffice in general, and "
        "the reason is important. The weights must serve all items at once, "
        "so the relevant dimension is not $2$ but the effective rank of the "
        "stacked response matrix across items and contexts. Caratheodory "
        "gives the per-item floor; the empirical support size tracks the "
        "effective rank, which is what E6 measures.")
    d.fig(figs.fig_caratheodory, 2.4, num=19, caption=(
        "Left: in the plane any hull point is a blend of three vertices -- "
        "the triangle containing it. Right: across many items the required "
        "support grows with the effective rank of the response matrix, not "
        "with the number of retained judges."))
    d.h3("Helly: when robustness across contexts is possible at all")
    d.p("Helly's theorem: for a finite family of convex sets in "
        "$\\mathrm{R}^d$, if every $d+1$ of them have a common point then "
        "all of them do. The per-context feasible sets $W_{i,c}(\\gamma)$ "
        "are convex and live in the weight simplex, of dimension "
        "$d=|S|-1$. So robust feasibility across all contexts can be "
        "checked $|S|$ sets at a time, and pairwise compatibility is not "
        "enough once $|S|>2$.")
    d.fig(figs.fig_helly, 2.4, num=20, caption=(
        "Three context-feasible sets in a two-dimensional weight simplex. "
        "Left: every pair overlaps, yet all three share no point -- no "
        "single weight vector is robust. Right: retaining one more judge "
        "widens every set until a common weight exists. Helly says checking "
        "$d+1=3$ at a time is enough here."))
    d.p("The practical reading is a design rule. Robustness failures are "
        "not caused by one impossible context; they are caused by a small "
        "group of contexts whose demands cannot be met simultaneously. "
        "Helly bounds the size of that group by $|S|$, so the diagnostic is "
        "to search for a minimal infeasible group rather than to inspect "
        "contexts one at a time.")

    # -- 15 ---------------------------------------------------------------
    d.newpage()
    d.h1("One page to remember", "15")
    d.p("If everything else fades, these six pictures are the argument.")
    d.fig(figs.fig_cheatsheet, 4.5, num=21, caption=(
        "The whole document in six panels. A judge is a point; the "
        "tolerance is a hexagon around it; a retained panel is a polygon; "
        "coverage means every judge lies within a hexagon of that polygon; "
        "good selection picks corners, not centres; and curvature is what "
        "makes individual redundancy fail to compose."))
    d.space(0.06)
    d.h3("The chain of definitions, in order")
    d.li(r"$p_{i,c}(x)\in\Delta^2$ -- a judge on an item is a point in a "
         r"triangle.")
    d.li(r"$\ell=\frac{1}{2}\|\cdot\|_1=\max_k|\cdot|$ -- error is the radius "
         r"of a hexagon.")
    d.li(r"$\widehat p_i=\sum_{j\in S}w_{ij}p_j$ -- reconstruction is a "
         r"point of the polygon $\mathrm{conv}\,P_S$.")
    d.li(r"$\varepsilon_{i,c}(w)$ -- convex, piecewise linear, hence a "
         r"linear program.")
    d.li(r"$\widehat E(S)=\max_i\max_c\varepsilon_{i,c}"
         r"(\widehat w_i(S))$ -- one number per subset, monotone in $S$ on "
         r"fitting data only.")
    d.li(r"$k^\star(\gamma)$ and $E^\star(k)$ -- inverse views of the same "
         r"frontier.")
    d.li(r"$\max_{i,c}\mathrm{UCB}_{1-\delta}\leq\gamma$ -- the only "
         r"statement that may be called a certificate.")
    d.space(0.10)
    d.rule()
    d.p("Built from figs.py; every figure is computed from the geometry it "
        "illustrates, not drawn by hand. Regenerate with "
        "__python build.py__.", size=8.4, color=MUTED)
    return d


if __name__ == "__main__":
    tmp = os.path.join(HERE, "_scratch.pdf")
    pages = build().render(tmp)
    doc = build(pages)
    doc.render(OUT)
    print("wrote", os.path.abspath(OUT))
    sys.exit(0)
