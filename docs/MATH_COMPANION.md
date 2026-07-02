# Math companion — every formula in the project, written out completely

**How to use this file:** upload it to Claude (claude.ai) and say:
*"Walk me through this section by section. Explain slowly, with small worked
numeric examples for every formula, and render all the math. Quiz me at the
end of each section before moving on."*
Claude in the app renders LaTeX, so all formulas below will display properly.

Notation used everywhere: $N$ models, $J = \{1,\dots,N\}$, kept set
$S \subseteq J$, tolerance $\gamma \ge 0$, eval sample of $n$ query rows.

---

## 1. The objects everything is built from

**Response matrices.** Each model $j$ is reduced to a column vector of its
scalar outputs on shared query points:

$$Y^{\text{fit}} \in \mathbb{R}^{m \times N}, \qquad Y^{\text{eval}} \in \mathbb{R}^{n \times N}.$$

Row = a query, column = a model. Two separate samples (fit vs eval) is the
*honest split*: weights are learned on one, judged on the other.

**Routing weights (the certificate).** To substitute model $i$ using kept set
$S$, find simplex weights by least squares **on the fit sample**:

$$w_i^\*(S) \;=\; \arg\min_{w \in \Delta^{|S|-1}} \; \big\| Y^{\text{fit}}_{:,i} - Y^{\text{fit}}_{:,S}\, w \big\|_2^2,
\qquad \Delta^{|S|-1} = \Big\{ w \in \mathbb{R}^{|S|} : w_j \ge 0,\; \textstyle\sum_j w_j = 1 \Big\}.$$

**Uniqueness / substitution error**, judged **on the eval sample** with mean
absolute error:

$$U(i \mid S) \;=\; \frac{1}{n} \sum_{t=1}^{n} \Big| \, Y^{\text{eval}}_{t,i} - \big( Y^{\text{eval}}_{:,S}\, w_i^\*(S) \big)_t \Big|.$$

**Coverage error** = the worst-substituted model:

$$\mathcal{E}(S) \;=\; \max_{i \in J} U(i \mid S).$$

(If $i \in S$ it substitutes itself, $U = 0$.) Key structural fact:
$\mathcal{E}$ is **monotone**: $S \subseteq S' \Rightarrow \mathcal{E}(S') \le \mathcal{E}(S)$
(a bigger hull is closer to everything) — in population; the finite-sample
version can wiggle because $w$ is fit on one sample and judged on another.

**The two optimization problems.**

$$\text{(minimal set)} \quad \min_{S \subseteq J} |S| \;\; \text{s.t.}\;\; \mathcal{E}(S) \le \gamma
\qquad\qquad
\text{(budget)} \quad \min_{|S| \le k} \mathcal{E}(S).$$

Everything in the repo is about the left one.

---

## 2. The algorithms as formulas

**Backward elimination.** Start $S = J$. Repeat: remove the best feasible
single model,

$$j^\* = \arg\min_{j \in S} \mathcal{E}(S \setminus \{j\}) \quad \text{subject to} \quad \mathcal{E}(S \setminus \{j\}) \le \gamma,$$

stop when no single removal is feasible. The stopping point is a
*single-deletion local optimum*.

**Why backward gets stuck one step from better solutions.** Monotonicity
gives, for any two models $a, b \in S$:

$$\mathcal{E}(S \setminus \{a, b\}) \;\ge\; \mathcal{E}(S \setminus \{a\}) \;>\; \gamma,$$

so once no *single* removal works, no *pure multi-removal* can ever work —
you must **add something back** while removing more. That is exactly the:

**k-swap reduction move.** For $k = 1, 2, \dots, k_{\max}$: remove any $k$
models from $S$, add any $k-1$ from outside,

$$S' \;=\; \big( S \setminus R \big) \cup A, \qquad R \subseteq S,\; |R| = k, \qquad A \subseteq J \setminus S,\; |A| = k-1,$$

accept the first $S'$ with $\mathcal{E}(S') \le \gamma$ (note $|S'| = |S| - 1$),
restart at $k=1$. Termination is guaranteed by the lexicographic potential

$$\Phi(S) = \big( |S|,\; \mathcal{E}(S) \big),$$

which strictly decreases with every accepted move. $k=1$ is just a backward
step; $k=2$ (remove 2, add 1) is the smallest move backward cannot make.
Empirical result: seeded from backward, $k=2$ is optimal on ~99% of
instances, and $k=3$ fixed every observed failure (all failures were optima
sharing **zero** models with the backward seed).

---

## 3. The MILP, written out completely

Variables: for all $j, i \in J$ and rows $t = 1..n$,

$$z_j \in \{0,1\} \;(\text{keep model } j), \qquad w_{ij} \ge 0 \;(\text{weight of } j \text{ in } i\text{'s recipe}), \qquad e_{ti} \ge 0 \;(\text{residual envelope}).$$

$$
\begin{aligned}
\min_{z, w, e} \quad & \sum_{j=1}^{N} z_j \\[4pt]
\text{s.t.} \quad
& \sum_{j=1}^{N} w_{ij} = 1 && \forall i && \text{(weights on the simplex)}\\[2pt]
& w_{ij} \le z_j && \forall i, j && \text{(route only to kept models)}\\[2pt]
& e_{ti} \ge \; Y^{\text{eval}}_{t,i} - \sum_{j} Y^{\text{eval}}_{t,j} w_{ij} && \forall t, i && \text{(abs. value, side 1)}\\[2pt]
& e_{ti} \ge -\Big( Y^{\text{eval}}_{t,i} - \sum_{j} Y^{\text{eval}}_{t,j} w_{ij} \Big) && \forall t, i && \text{(abs. value, side 2)}\\[2pt]
& \frac{1}{n} \sum_{t=1}^{n} e_{ti} \le \gamma && \forall i && \text{(mean-abs error budget)}
\end{aligned}
$$

Three tricks to internalize:

1. **$w_{ij} \le z_j$** — if $z_j = 0$, the weight is forced to 0. One linear
   inequality implements "pruned models can't be used." (Works because
   $w_{ij} \le 1$ always; this is a big-M constraint with $M = 1$.)
2. **The two $e$-inequalities** — together they say $e_{ti} \ge |r_{ti}|$,
   turning an absolute value into two linear rows. Example: residual $r = -3$
   gives $e \ge -3$ and $e \ge 3$, i.e. $e \ge 3 = |-3|$.
3. **"max ≤ γ" is free** — $\max_i U(i|S) \le \gamma$ is the same as
   "$\le \gamma$ for every $i$", one budget row per model.

**Sparse variant (Open Problem 5, eq. 11 of the paper):** add binaries
$u_{ij} \in \{0,1\}$ with

$$w_{ij} \le u_{ij} \;\;\forall i,j, \qquad \sum_{j} u_{ij} \le r \;\;\forall i,$$

so each certificate uses at most $r$ models.

**What the MILP optimum means.** It picks $w$ freely against
$Y^{\text{eval}}$ — no honest split — so it answers "*does any certificate
exist*" (oracle). Every protocol-feasible $S$ is oracle-feasible, hence the
sandwich

$$\big|\text{OPT}_{\text{oracle}}\big| \;\le\; \big|\text{OPT}_{\text{protocol}}\big| \;\le\; \big|\text{backward+kswap}\big|,$$

and the left–middle gap is the **price of honest splitting** (measured on
UTD19: large, e.g. 4 vs ~8.7 at $\gamma = 60$).

---

## 4. Beta (task-error preservation), with full derivations

Ground truth $y^\* \in \mathbb{R}^n$ at the eval queries; task loss $L$
(RMSE/MAE/MSE). Define, with $\hat{y}_i = Y^{\text{eval}}_{:,S} w_i^\*(S)$
the substitute's predictions:

$$\delta(i \mid S) \;=\; L(\hat{y}_i,\, y^\*) \;-\; L\big(Y^{\text{eval}}_{:,i},\, y^\*\big), \qquad B(S) = \max_{i} \delta(i \mid S).$$

**Bound 1 (triangle inequality; drift is capped by $\gamma$).** For any norm
loss in the *same metric* as the fidelity term:

$$L(\hat{y}_i, y^\*) \;\le\; L\big(Y_{:,i}, y^\*\big) + L\big(\hat{y}_i, Y_{:,i}\big) \;=\; L\big(Y_{:,i}, y^\*\big) + U(i \mid S) \;\le\; L\big(Y_{:,i}, y^\*\big) + \gamma,$$

hence $\delta(i|S) \le \gamma$ automatically. (Careful: MAE-fidelity does
**not** bound RMSE-drift — keep both in the same metric.)

**Bound 2 (Jensen; a blend is at least as good as its worst ingredient).**
For convex $\ell$ (e.g. $\ell(r) = r^2$ or $|r|$), simplex weights, per query $t$:

$$\ell\Big( \sum_{j \in S} w_j \big(Y_{t,j} - y^\*_t\big) \Big) \;\le\; \sum_{j \in S} w_j \, \ell\big(Y_{t,j} - y^\*_t\big).$$

Average over $t$:

$$L\Big( \sum_j w_j Y_{:,j},\; y^\* \Big) \;\le\; \sum_{j \in S} w_j \, L\big(Y_{:,j},\, y^\*\big) \;\le\; \max_{j \in S} L\big(Y_{:,j},\, y^\*\big).$$

Numeric picture: truth 100, kept models predict 90 and 110 (each error 10) —
any weighted average lands in $[90, 110]$ (error $\le 10$), and the 50/50 mix
hits 100 exactly (error 0, *better than both*). This is why $\delta$ is often
negative.

**The design consequence (quality by construction).** Pre-filter the
keep-eligible pool

$$K_\beta \;=\; \big\{\, j \in J \;:\; L\big(Y^{\text{eval}}_{:,j},\, y^\*\big) \le \beta \,\big\},$$

then solve the ordinary problem restricted to it:

$$\min |S| \quad \text{s.t.} \quad S \subseteq K_\beta, \qquad \max_{i \in J} U(i \mid S) \le \gamma.$$

By Bound 2 **every** certificate over $S \subseteq K_\beta$ automatically has
task error $\le \beta$ — no constraint inside the loop, monotone theory
untouched. Possible new outcome: infeasibility (an archetype fails the
quality bar) — a legitimate finding, not a bug. Anchor $\beta$ to the
persistence-forecaster RMSE or a percentile of kept-model pooled-eval RMSE,
never a raw number.

---

## 5. Risk-controlled pruning (UCB)

$U(i|S)$ is a *sample mean* of $n$ i.i.d. residual magnitudes
$|R_{t,i}|$, so the CLT gives a confidence bound. With union bound over the
$N$ models (test each at level $\delta/N$):

$$\mathrm{UCB}(i \mid S) \;=\; \underbrace{\frac{1}{n}\sum_t |R_{t,i}|}_{\hat U(i|S)} \;+\; z_{1 - \delta/N} \cdot \frac{\mathrm{sd}\big(|R_{\cdot,i}|\big)}{\sqrt{n}},
\qquad \mathcal{E}_{\mathrm{UCB}}(S) = \max_i \mathrm{UCB}(i \mid S).$$

Certified rule: remove $j$ only if
$\mathcal{E}_{\mathrm{UCB}}(S \setminus \{j\}) \le \gamma$. Then for each
**fixed** $S$, $\;\mathbb{P}\big(\mathcal{E}(S) \le \gamma\big) \ge 1 - \delta$
(asymptotically). What remains open (the paper's OP1): validity **uniform
over the adaptive sequence** of sets the algorithm visits — the union bound
covers models, not the exponentially many sets a run could reach.

---

## 6. Carathéodory: the geometry that predicts your numbers

**Exact Carathéodory.** If the model columns lie in a $d$-dimensional affine
subspace, then any point of their convex hull is a convex combination of at
most $d + 1$ of them. Consequence: **sparse certificates with $r = d+1$ never
increase the optimum.** Verified: on $d=2$ synthetic instances, the sparse
MILP with $r = 3$ matched the unconstrained optimum on every trial.

**Approximate Carathéodory (Barman).** For any point $x$ in the hull of
points with $\|v_j\|_2 \le \rho$, there is a combination of only

$$r \;=\; O\!\big( \rho^2 / \varepsilon^2 \big)$$

of them with $\|x - \hat{x}\|_2 \le \varepsilon$ — **independent of the
dimension**. Consequence: even in high intrinsic dimension, tolerance
$\gamma$ buys certificate sparsity $r = O(\rho^2/\gamma^2)$. This directly
answers the paper's OP5 geometric question ("when can a few archetypes
cover"), and it's *testable* with the sparse MILP: plot optimum vs $r$ and
compare the knee to $d+1$ (estimate $d$ by PCA on $Y^{\text{eval}}$).

**Related literature to cite (positions the whole project):** the pruning
problem is *representative selection / archetype selection*: separable NMF
and the Successive Projection Algorithm (Arora et al. 2012; Gillis &
Vavasis), "Sparse Modeling for Finding Representative Objects" (Elhamifar,
Sapiro & Vidal, CVPR 2012), and coresets / $\varepsilon$-kernels for hull
approximation (Agarwal, Har-Peled & Varadarajan). SPA is ~20 lines and
provable under near-separability — a natural extra baseline against
backward+k-swap.
