# Certified Compression of LLM Judge Panels

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Paper Status](https://img.shields.io/badge/Paper-Under%20Review-red)](https://openreview.net/)

> Reference implementation of **EPCRC** (Ecosystem Pruning via Convex Routing
> Coverage), the joint behavioural coverage condition for deciding which judges
> in an LLM evaluation panel can be retired together.

## Overview

Panels of LLM judges are expensive, so practitioners prune them. Pruning is
normally decided one judge at a time: hold a judge out, measure how well the
remaining panel reproduces it, and drop the judges whose error is small. That
procedure is unsound. Each certificate is verified against a *different*
retained set, so certificates do not compose, and judges that are individually
redundant need not be jointly redundant.

This repository replaces the per-judge test with a joint condition on the
three-class comparison simplex. A retained subset `S` is admissible at
tolerance `gamma` when every dropped judge is matched within `gamma` in total
variation by a convex combination of the judges kept, in the worst case over
judges and over prompt contexts:

```
E(S) = max      min       max     (1/n) sum  TV( p_jc(i),  sum   w_k p_kc(i) )
       j not in S  w in Delta(S)   c              i              k in S
```

Checking the condition costs one linear program per dropped judge. `E` is
monotone, so a single evaluation at any subset is a floor on every subset
inside it, which is what turns an exhaustive search into a certificate rather
than a heuristic.

### A three-judge example

Judges 1 and 2 always answer "A is better"; judge 3 always answers "B is
better". Hold out judge 1 and judge 2 reproduces it exactly, so its error is 0.
Hold out judge 2 and judge 1 reproduces it, error 0 again. Both one-at-a-time
checks pass, yet dropping both leaves only judge 3, whose answers are the
opposite: total variation 1, the worst possible. Each check assumed the other
judge would still be there. `E(S)` evaluates the set that is actually kept, so
it rejects this deletion.

### Headline results

Measured on a fully scored response tensor of 20 open-weight judges, 7 prompt
contexts and 3724 comparison pairs, with all 521,360 cells scored and none
imputed:

| Finding | Number |
| --- | --- |
| Joint deletion breaks the tolerance | 86 of 95 audited cases |
| Overdeletion by the one-at-a-time audit at the predeclared `gamma = 0.2` | one judge: the 17 it returns violate the tolerance, 18 is the smallest that does not |
| Coverage vs. the best of 11 baselines, worst-judge worst-context TV | 0.363 vs. 0.404 (pivoted QR) |
| Significant losses to any baseline | all at panels of 4 judges or fewer |
| Backward elimination reaches the certified optimum | 80% of instances, vs. 20% for forward selection |
| Backbone recovery on planted synthetic panels | 105 of 105 runs |

Two negatives are reported with the same weight. The compression ratio
predeclared before the data existed is *unreachable* on this panel rather than
merely unmet, since removing the single most redundant judge already costs
0.130 against a declared band of 0.08 to 0.10. And under benchmark shift the
selected subset transfers (mean Jaccard 0.78 against the oracle) while its
mixing weights do not and must be recalibrated.

## Repository layout

```text
.
├── epcrc/                     # the library
│   ├── coverage.py            # E(S), substitution certificates, the general functional
│   ├── judge.py               # the three-class judge instance: TV, worst context, minimax weights
│   ├── geometry.py            # convex projection and the NNLS/simplex solvers
│   ├── pruning.py             # backward, forward, k-swap and priority-queue pruners
│   ├── certification.py       # empirical Bernstein and naive certificates
│   ├── reconstruction.py      # weight-fitting rules and their scoring
│   ├── selection_baselines.py # the 11 comparator selection rules and the low-rank floor
│   ├── panel.py               # the judge registry, split loading and the response tensor
│   ├── rewardbench.py         # RewardBench 2 to judged pairs
│   ├── judgebench.py          # JudgeBench to judged pairs (transfer benchmark)
│   ├── prompts.py             # the 7 registered prompt contexts and their hashes
│   ├── scoring.py             # label scoring and normalisation
│   ├── downstream.py          # aggregation to one verdict per item
│   ├── synthetic_judges.py    # planted-structure judge generators
│   ├── report.py              # result JSON to paper tables
│   └── figures.py             # result JSON to paper figures
│
├── experiments/               # one script per experiment, named as in the execution plan
│   ├── build_rewardbench_pairs.py   # RewardBench 2 -> data/pairs_seed*.jsonl
│   ├── build_judgebench_pairs.py    # JudgeBench   -> data/judgebench_pairs_seed*.jsonl
│   ├── score_panel.py               # run the judges behind quality gates G0, G1, G2 (GPU)
│   ├── experiment_e0_noncomposability.py    # C1: certificates do not compose
│   ├── experiment_e1_compression_frontier.py# C2: the physical-to-virtual frontier
│   ├── experiment_c3_baselines.py           # C3: coverage vs. 11 selection baselines
│   ├── experiment_c4_stress_specialists.py  # C4: stress specialists, multi-context selection
│   ├── experiment_c5_downstream.py          # C5: downstream verdict preservation
│   ├── experiment_c6_exchange.py            # C6: exact optimality and exchange structure
│   ├── experiment_c7_certification.py       # C7: certification reliability
│   ├── experiment_e6_sparse_cost.py         # C8: sparse certificates and cost-aware panels
│   ├── experiment_e7_transfer.py            # E7: cross-benchmark transfer to JudgeBench
│   ├── experiment_e8_synthetic.py           # E8: planted structure and sample complexity
│   ├── experiment_backbone.py               # the backbone by pruned downward enumeration
│   └── export_results.py                    # pack every result into one checkable zip
│
├── notebooks/                 # executed, with outputs saved, so the numbers are readable here
│   ├── 01_judge_behaviour.ipynb        # what the 20 judges actually do
│   ├── 02_c1_noncomposability.ipynb    # Figure 1 and the 86 of 95
│   ├── 03_c2_compression_frontier.ipynb# the frontier and why the declared band is unreachable
│   └── 04_results_package.ipynb        # builds the delivery zip
│
├── tests/                     # 334 tests, including solver cross-checks against an exact LP
├── data/                      # split manifests with SHA-256 over the item ids (pairs are regenerated)
├── DECISIONS.md               # every scientific choice, recorded before results were inspected
├── requirements.txt           # CPU: all analysis, tables and figures
└── requirements-scoring.txt   # GPU: only needed to run the judges themselves
```

`results/` is not versioned. It is roughly 150 MB of machine output, all of it
reproducible from the scripts below. The committed notebooks keep their
executed outputs, so every reported number is readable without rerunning
anything.

## Installation

```bash
git clone https://github.com/<user>/EPCRC.git
cd EPCRC
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Core requirements are numpy, scipy, pandas, scikit-learn, joblib and
matplotlib. Running the judges additionally needs `requirements-scoring.txt`.
On Blackwell (sm_100) the default PyPI torch wheel does not build for the
architecture, so install torch from the cu128 index before that file.

## Reproducing the results

Scoring the panel is the only step that needs a GPU, and it is the only step
that is expensive. Everything after it is CPU analysis over the cached tensor.

```bash
# 1. Build the judged pairs and the split manifests (CPU, minutes)
python experiments/build_rewardbench_pairs.py --all-seeds
python experiments/build_judgebench_pairs.py

# 2. Score the 20 judges under 7 contexts (GPU, hours). One gate per call:
#    G0 checks the protocol, G1 the label extractor, G2 writes the tensor.
python -u experiments/score_panel.py --gate g0 --panel core20
python -u experiments/score_panel.py --gate g1
python -u experiments/score_panel.py --gate g2 --panel core20 \
        --items 4000 --batch-size 64 --scores-only

#    The E7 transfer set is scored into its own cache, never over the above.
python -u experiments/score_panel.py --gate g2 --panel core20 \
        --dataset judgebench --items 620 --batch-size 64 --scores-only

# 3. Each claim, in any order (CPU)
python experiments/experiment_e0_noncomposability.py     --real --panel core20
python experiments/experiment_e1_compression_frontier.py --panel core20
python experiments/experiment_c3_baselines.py            --panel core20
python experiments/experiment_c4_stress_specialists.py   --panel core20
python experiments/experiment_c5_downstream.py           --panel core20
python experiments/experiment_c6_exchange.py             --panel core20
python experiments/experiment_c7_certification.py        --panel core20
python experiments/experiment_e6_sparse_cost.py          --panel core20
python experiments/experiment_e7_transfer.py             --panel core20
python experiments/experiment_backbone.py                --panel core20
python experiments/experiment_e8_synthetic.py

# 4. Tables, figures and the delivery zip
python experiments/export_results.py --panel core20
```

```bash
pytest -q          # 334 tests
```

## Usage

`JudgeResponses` holds the response tensor for one split.
`JudgeCoverageFunctional` fits replacement weights on one split and evaluates
`E(S)` on another, which is what keeps a certificate from being read off the
items it was fitted to.

```python
from epcrc.panel import load_panel, scores_dir
from epcrc.judge import JudgeCoverageFunctional
from epcrc.rewardbench import PRIMARY_SEED

panel = load_panel(PRIMARY_SEED, scores_dir("core20"), split_seed=20260818)
cov = JudgeCoverageFunctional(
    panel.splits["FIT"], panel.splits["CERT"], panel.judge_ids
)

# Is this hand-picked subset admissible at gamma = 0.2?
kept = {0, 2, 3, 5, 7, 8, 11, 12, 14, 16}
error, per_judge = cov.compute_coverage(kept)
print(error, error <= 0.2)

# Which absent judge is the binding constraint?
print(cov.find_bottleneck(kept))
```

Searching for a panel rather than checking one goes through a pruner, which
takes the functional and the tolerance and returns the retained set together
with the error that certifies it:

```python
from epcrc.pruning import BackwardEliminationPruner

result = BackwardEliminationPruner(cov, tolerance_gamma=0.2).run()
print(sorted(result.kept_set), result.coverage)
```

To audit your own panel, build a `JudgeResponses` from per-context blocks of
per-item class probabilities, each of shape `(n_items, n_judges, 3)` with the
last axis summing to one, and the same code applies.
