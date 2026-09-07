# Algorithms and MILP formulation

Ecosystem pruning: select the smallest set of models that can substitute for
every model in the ecosystem within a tolerance $\gamma$.

## 1. Problem statement

| symbol | meaning |
|---|---|
| $\mathcal M = \{1,\dots,N\}$ | the model ecosystem; $N = 20$ for UTD19 |
| $S \subseteq \mathcal M$ | the kept (representative) set |
| $Y \in \mathbb{R}^{n \times N}$ | predictions; column $j$ is model $j$'s output on $n$ queries |
| $\gamma$ | tolerance on substitution error |

Model $i$ is substituted by a blend of the kept models with weights on the
simplex, $w \ge 0$ and $\sum_{j \in S} w_j = 1$. The reachable predictions are
therefore the convex hull of the kept models.

The substitution error of model $i$ under $S$, and the coverage of $S$, are

$$U(i \mid S) = \bigl\lVert\, y_i - Y_S w_i \,\bigr\rVert_{\mathrm{mae}},
\qquad
E(S) = \max_{i \in \mathcal M} U(i \mid S),$$

with $U(i \mid S) = 0$ for $i \in S$. The set $S$ is feasible iff
$E(S) \le \gamma$. The objective is

$$\min_{S \subseteq \mathcal M} |S| \quad \text{subject to} \quad E(S) \le \gamma.$$

Two properties of $E$ govern the algorithm design. It is a maximum over
residuals, hence not submodular, so greedy selection carries no $1 - 1/e$
guarantee. It is also non-monotone in the finite-sample regime described below,
so adding a model can increase $E$.

### Routing semantics

The weights $w_i$ are obtained in one of two ways.

**Protocol.** Fit $w_i$ by least squares on a sample $Y_{\mathrm{fit}}$ and score
the residual on a held-out sample $Y_{\mathrm{eval}}$. This is realisable by a
deployed router and is the semantics of all algorithms in §2. A larger hull gives
$w_i$ more freedom to overfit $Y_{\mathrm{fit}}$, which is the source of the
non-monotonicity.

**Oracle.** Choose $w_i$ directly against $Y_{\mathrm{eval}}$. This is the
semantics of the MILP in §3.

The oracle is strictly more permissive, so

$$|OPT_{\mathrm{oracle}}| \;\le\; |OPT_{\mathrm{protocol}}| \;\le\; |S_{\mathrm{algorithm}}|.$$

The MILP optimum is thus a certified lower bound against which the algorithms are
measured.

## 2. Algorithms

All algorithms return a protocol-feasible set and differ only in the search.

### 2.1 Forward selection (`forward`)

Start from the empty set and add the model that reduces coverage most, stopping
at the first feasible prefix.

```
S = {}
while E(S) > gamma:
    j* = argmin_{j not in S} E(S + {j})
    S = S + {j*}
return S
```

Since $E$ is non-monotone, a model added early may be made redundant by later
additions. The returned prefix is therefore not necessarily locally minimal; on
UTD19 it retains up to 3 models that could be removed without violating $\gamma$.

### 2.2 Forward selection with trimming (`forward_trim`)

Forward selection followed by the single-deletion trim of §2.3.

```
S = forward_selection()
repeat:
    D = { j in S : E(S - {j}) <= gamma }
    if D is empty: break
    S = S - { argmin_{j in D} E(S - {j}) }
return S
```

The result is a single-deletion optimum. This is a forward/backward hybrid and is
reported separately from §2.1.

### 2.3 Backward elimination (`backward`)

Start from the full ecosystem and repeatedly delete the model whose removal
leaves coverage lowest, while feasibility holds.

```
S = M
repeat:
    D = { j in S : E(S - {j}) <= gamma }
    if D is empty: break
    S = S - { argmin_{j in D} E(S - {j}) }
return S
```

The result is minimal by construction. Every decision is taken on a large set, so
each projection is well-conditioned, in contrast to forward selection where the
first decisions are made on an under-determined projection.

### 2.4 $k$-swap (`backward_kswap2`, `backward_kswap3`)

Backward elimination terminates at a single-deletion optimum, which need not be
globally minimal. The $k$-swap neighbourhood removes $k$ models and reinserts
$k-1$, giving a net reduction of one while permitting the composition of $S$ to
change.

```
S = backward_elimination()          # seed
repeat:
    for k = 1 .. K:
        for each R subset of S,   |R| = k:
            for each A subset of M\S, |A| = k-1:
                S' = (S - R) + A
                if E(S') <= gamma:
                    S = S'; restart from k = 1
    if no move accepted: break
return S
```

The case $k = 1$ is a single deletion, so $k = 2$ is the first move not already
available to backward elimination. A pure swap variant (remove $k$, add $k$)
exists but is disabled by default: each accepted reduction move decreases $|S|$
by one, which bounds the iteration count by $N$, and pure swaps preserve $|S|$
and remove that bound.

### 2.5 Priority-queue $k$-swap (`pq_kswap`)

The neighbourhood of §2.4 explored through a priority queue keyed by a cached
gain estimate. Candidates whose optimistic stale gain cannot improve on the
incumbent are skipped without re-evaluation. The worst case is unchanged; the
measured saving is a factor of 3 to 4.

### 2.6 Complexity

Counting evaluations of $E(S)$, with $m$ the size of the returned set and $K$ the
maximum swap order:

| algorithm | evaluations |
|---|---|
| `forward`, `forward_trim` | $O(Nm)$; the trim contributes $O(m^2)$ |
| `backward` | $O(N^2)$, independent of $\gamma$ |
| `backward_kswap`, `pq_kswap` | $O(N^{2K})$ |

One evaluation of $E(S)$ requires $N - |S|$ simplex-constrained least-squares
solves of size $n \times |S|$. Derivations are in `docs/COMPLEXITY.md`.

## 3. MILP formulation

The oracle-routing variant is solved exactly as a mixed-integer linear program
(`epcrc/milp.py`, `milp_min_representative_set`).

### Variables

| variable | type | meaning |
|---|---|---|
| $z_j$, $j = 1,\dots,N$ | binary | model $j$ is kept |
| $w_{ij}$ | continuous, $\ge 0$ | weight target $i$ places on model $j$ |
| $e_{ti}$ | continuous, $\ge 0$ | residual envelope for target $i$ at row $t$ |

### Program

$$\min \sum_{j=1}^{N} z_j$$

subject to, for every target $i = 1,\dots,N$:

$$\sum_{j=1}^{N} w_{ij} = 1 \tag{1}$$

$$w_{ij} \le z_j, \qquad j = 1,\dots,N \tag{2}$$

$$\pm\Bigl( (Yw_i)_t - y_{ti} \Bigr) \le e_{ti},
\qquad t = 1,\dots,n \tag{3}$$

$$\frac{1}{n}\sum_{t=1}^{n} e_{ti} \le \gamma \tag{4}$$

Constraint (1) places each routing vector on the simplex. Constraint (2) links
the routing to the selection, permitting weight only on kept models. Constraint
(3) is the standard linearisation of the absolute value; at the optimum
$e_{ti} = |y_{ti} - (Yw_i)_t|$, since the objective pushes $e$ downward.
Constraint (4) imposes the mean-absolute-error budget. For a maximum-error
metric, (4) is replaced by the variable bound $e_{ti} \le \gamma$.

At $N = 20$ and $n = 1500$ the program has 20 binary variables, 400 routing
variables, 30 000 envelope variables and approximately 60 000 envelope rows.

A sparsity restriction $\lVert w_i \rVert_0 \le r$ is available through
`max_support=r`, adding binaries $u_{ij}$ with $w_{ij} \le u_{ij}$ and
$\sum_j u_{ij} \le r$.

### Strength of the relaxation

The linear relaxation of this formulation is weak. Relaxing $z_j$ to $[0,1]$ and
setting $z_j = 0.05$ for all $j$ gives an objective of
$\sum_j z_j = 20 \times 0.05 = 1$, while $w_{ij} \le 0.05$ still permits
$\sum_j w_{ij} = 1$. Fractional selections are therefore nearly free and the
relaxation optimum lies near 2, against a true optimum of 13 at $\gamma = 40$.
On the full ecosystem Gurobi held a dual bound of 2.0 at every $\gamma$ with a
gap of 75–80% and reached the time limit.

Three strengthenings were tested and do not improve the bound. Cuts on $w$ are
redundant, since the envelope rows are already tight in $w$. Disaggregating the
linking constraint is not applicable, as (2) is already the disaggregated form
rather than $\sum_i w_{ij} \le N z_j$. Solver-side measures — MIPFocus, extended
time limits, and warm starting — do not move the bound, since the weakness lies
entirely in the fractional $z$.

### Combinatorial certification

Oracle feasibility decomposes over targets: $S$ is feasible iff for each $i$
independently

$$\min_{w \in \Delta} \bigl\lVert Y_S w - y_i \bigr\rVert_{\mathrm{mae}} \le \gamma,$$

which is a small $L_1$ linear program requiring approximately 14 ms at
$n = 1500$. This yields the following procedure (`certify_lower_bound`).

1. Enumerate all subsets of size $1,\dots,k_{\max}$ — 6195 subsets for $N = 20$
   and $k_{\max} = 4$ — testing each by the LPs above, in parallel.
2. If a subset of size $k$ is feasible, it is optimal, since every smaller size
   has been refuted by exhaustion.
3. If no subset is feasible, $|OPT| \ge k_{\max} + 1$ is proved. This bound is
   supplied to Gurobi as the cardinality cut $\sum_j z_j \ge L$ together with a
   warm start, which closes the gap.

The certifier agrees with exact HiGHS on 18 of 18 small instances ($N = 9$,
$n = 60$). Solve times on UTD19 at $N = 20$:

| $\gamma$ | 40 | 60 | 80 | 100 | 120 | 140 | 160 |
|---|---|---|---|---|---|---|---|
| time | 343 s | 223 s | 22 s | 1.3 s | 0.7 s | 0.7 s | 0.7 s |
| proof | MILP | MILP | enumeration | enumeration | enumeration | enumeration | enumeration |

## 4. Results

Full UTD19 ecosystem, $N = 20$, $n_{\mathrm{fit}} = n_{\mathrm{eval}} = 1500$,
mean absolute error. All seven optima are proved.

| $\lvert S\rvert$ | $\gamma$=40 | 60 | 80 | 100 | 120 | 140 | 160 | mean gap |
|---|---|---|---|---|---|---|---|---|
| `forward` | 17 | 8 | 7 | 4 | 4 | 3 | 2 | 1.86 |
| `backward` | 14 | 7 | 6 | 4 | 3 | 2 | 2 | 0.86 |
| `forward_trim` | 14 | 7 | 5 | 3 | 3 | 2 | 2 | 0.57 |
| `backward_kswap2` | 14 | 7 | 4 | 3 | 2 | 2 | 2 | 0.29 |
| `backward_kswap3` | 14 | 7 | 4 | 3 | 2 | 2 | 2 | 0.29 |
| `pq_kswap` | 14 | 7 | 4 | 3 | 2 | 2 | 2 | 0.29 |
| oracle optimum | 13 | 6 | 4 | 3 | 2 | 2 | 2 | — |

Backward elimination attains a mean gap of 0.86 against 1.86 for forward
selection. Trimming forward selection reduces its gap to 0.57, with strict wins
over backward elimination at $\gamma = 80$ and $\gamma = 100$; the two are
distinct algorithms and both are reported.

All three $k$-swap variants attain the oracle bound exactly for $\gamma \ge 80$.
At $N = 20$, $K = 3$ returns the same sets as $K = 2$ at every $\gamma$ while
costing a factor $N^2$ more, so $K = 2$ is used by default.

At $\gamma = 40$ and $\gamma = 60$ all methods lie exactly one above the oracle
bound. Since the oracle relaxes the fit/evaluation split, the protocol optimum at
these tolerances may itself be 14 and 7.

## 5. Code map

| file | contents |
|---|---|
| `epcrc/coverage.py` | $U(i \mid S)$, $E(S)$, substitution certificates |
| `epcrc/geometry.py` | simplex-constrained projection |
| `epcrc/pruning.py` | the algorithms of §2 |
| `epcrc/milp.py` | the MILP of §3, `is_feasible_set`, `certify_lower_bound` |
| `experiments/run_all.py` | the sweep; writes `results/sweep.json` |
| `experiments/audit_algorithms.py` | feasibility and minimality checks |
| `experiments/verify_sweep.py` | independent re-verification of the optima |
| `docs/COMPLEXITY.md` | complexity derivations |
