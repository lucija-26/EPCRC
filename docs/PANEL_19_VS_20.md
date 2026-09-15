# What changed when J07 joined the panel

The Core-20 panel is prescribed by plan section 10. J07 was behind a gated
Hugging Face repository when the first full pass was run, so every preliminary
number was produced on the other nineteen judges. Once access came through the
whole chain was re-run on all twenty.

Both runs are kept. The 19-judge files are in `results/archive_19judge/`; the
20-judge files carry the registry name in `results/`. This note is the
comparison, because the comparison is more informative than either run alone:
it answers whether any preliminary conclusion was an artefact of the missing
judge.

**It was not. No claim changes verdict, and no reported ordering changes.**

Both runs share the pairs seed 20260818 and the same five split seeds, so the
items and partitions are identical and the only moving part is the panel.

---

## 1. J07 is the second-most redundant judge in the panel

Backward elimination on the twenty-judge panel discards J08 first and J07
second. The consequence is sharp:

**The subset chosen at every budget from 1 to 18 judges is identical in the two
runs, judge for judge.** Not similar — the same set.

| k | subset selected by coverage backward elimination (identical in both runs) |
| ---: | --- |
| 1 | J04 |
| 2 | J04, J09 |
| 3 | J04, J09, J10 |
| 5 | J02, J04, J09, J10, J14 |
| 10 | J01, J02, J04, J06, J09, J10, J13, J14, J18, J19 |
| 14 | J01, J02, J03, J04, J05, J06, J09, J10, J12, J13, J14, J15, J18, J19 |
| 18 | all but J07 and J08 |

The two runs first select differently at k = 19, where the 19-judge panel has no
choice left to make.

## 2. The errors are identical to four decimals up to k = 17

| k | 19-judge | 20-judge | delta |
| ---: | ---: | ---: | ---: |
| 2 | 0.6747 | 0.6747 | 0.0000 |
| 5 | 0.5127 | 0.5127 | 0.0000 |
| 10 | 0.3478 | 0.3478 | 0.0000 |
| 14 | 0.2792 | 0.2792 | 0.0000 |
| 17 | 0.2070 | 0.2070 | 0.0000 |
| 18 | 0.1316 | 0.1560 | +0.0245 |
| 19 | 0.0000 | 0.1299 | +0.1299 |

Worst-judge worst-context total variation on the held-out TEST split.

Up to k = 17 the agreement is exact. At k = 18 the *selected panel is still the
same*, and the error rises only because J07 has joined the set of judges that
must be reconstructed — and it is slightly harder to reconstruct than the rest.
At k = 19 the two runs are answering different questions: for the 19-judge panel
that is the whole panel, which reconstructs itself at zero error by
construction.

This is the reason the preliminary numbers were sound. Every claim that reads
the frontier below k = 18 reads bit-identical values.

## 3. Claim by claim

### C1 — non-composability: confirmed in both, stronger at twenty

On the predeclared tolerance grid (fixed before any judge was scored):

| gamma | 19: judges certified removable | 19: joint deletion breaks budget | 20: removable | 20: breaks budget |
| ---: | ---: | :--- | ---: | :--- |
| 0.020 – 0.120 | 0 | not a composition test | 0 | not a composition test |
| 0.150 | 1 | not a composition test | 1 | not a composition test |
| 0.200 | 2 | 5 of 5 seeds | **3** | 5 of 5 seeds |

A composition test needs at least two individually-removable judges, so only
gamma = 0.20 is one. The twentieth judge adds a third removable judge there,
which makes the test harder to pass and it still fails in every seed.

Across the data-driven leave-one-out breakpoints:

| | cases with >= 2 removable | joint deletion broke the budget | mean composition gap |
| --- | ---: | ---: | ---: |
| 19-judge | 90 | 84 | +2.40 judges |
| 20-judge | 95 | 86 | +2.37 judges |

### C2 — compression frontier: fails as declared in both

The evidence matrix in plan section 31 asks for Core-20 reduced to 6-10 judges
at worst-context error 0.08-0.10. Measured at k = 10: **0.348 in both runs**,
over three times the band.

The twentieth judge cannot rescue this and the shortfall is not a matter of
tuning. Removing the single most redundant judge from the full panel already
costs 0.130. That is a floor under every smaller panel, so no panel below twenty
reaches 0.10 at all. The judges are less mutually redundant than the planning
target assumed. Reported as measured.

### C3 — coverage selection beats the baselines: same ranking, lower errors

Mean worst-judge worst-context TV over k in [2, N-1] and five split seeds.

| method | 19-judge | 20-judge | rank 19 | rank 20 |
| --- | ---: | ---: | ---: | ---: |
| coverage_backward | 0.3750 | 0.3627 | 1 | 1 |
| coverage_forward | 0.4110 | 0.4014 | 2 | 2 |
| pivoted_qr | 0.4181 | 0.4035 | 3 | 3 |
| farthest_first | 0.4490 | 0.4404 | 4 | 4 |
| hierarchical | 0.4642 | 0.4560 | 6 | 5 |
| correlation_medoid | 0.4604 | 0.4590 | 5 | 6 |
| kmedoids | 0.4710 | 0.4725 | 7 | 7 |
| cost_ascending | 0.4734 | 0.4867 | 8 | 8 |
| leverage | 0.5379 | 0.5298 | 9 | 9 |
| cost_descending | 0.5559 | 0.5551 | 11 | 10 |
| one_per_family | 0.5517 | 0.5565 | 10 | 11 |
| top_accuracy | 0.5844 | 0.5823 | 12 | 12 |

Ten of twelve ranks are unchanged. The two that move are adjacent pairs
separated by less than 0.005 TV — hierarchical against correlation_medoid, and
cost_descending against one_per_family — which is noise at this resolution, not
a reordering.

Coverage backward elimination is first in both runs. Every method improves on
the larger panel, coverage included, and its margin over the nearest rival,
pivoted_qr, narrows marginally from 0.0431 to 0.0408 TV — a change well inside
the bootstrap width, and in any case the head-to-head paired test below is what
settles the comparison.

The verdict on C3 is decided by the paired bootstrap of plan section 34.2, not
by this table. On the twenty-judge panel that is 1080 comparisons — five seeds
by eighteen budgets by twelve baselines — with 31 losses, every one of them at
k <= 4. From k = 5 upward coverage does not lose to any baseline in any seed.

### C4 — stress specialists: identical finding, larger effect

| | 19-judge | 20-judge |
| --- | ---: | ---: |
| mean advantage over `clean_pipeline` | +0.304 TV | +0.329 TV |
| budgets where robust wins in all 5 seeds | 9 of 9 | 9 of 9 |
| mean advantage over `clean_select` | +0.045 TV | +0.060 TV |
| budgets unanimous against `clean_select` | 4 of 9 | 5 of 9 |
| specialists flagged in every seed | J02, J05, J06, J10, J15, J20 | J02, J05, J06, J10, J15, J20 |
| specialists per seed | 8, 6, 8, 7, 7 | 8, 6, 8, 7, 7 |

The specialist set is unchanged: **J07 is not a stress specialist under any
split seed.** All six recurring specialists bind on the bias-resistant rubric
context and two of them also on the correctness rubric, which is the mechanism
the claim predicts rather than a correlate — a pipeline that only ever sees
clean items is blind to exactly the contexts where specialisation lives.

The two baseline arms must never be pooled. Pooled they average to roughly 30%
and read as borderline. Separated they say something specific: robust *fitting*
buys most of the reduction, and robust *selection* alone buys little below
k = 18. C4 as stated is about selection, so this is a partial result for the
claim and a strong result for the deployed pipeline.

---

## Practical consequence

Anything that reads only k <= 17 does not need to be re-run against the
twenty-judge panel; it would return the same bits. The twentieth judge matters
at the top of the budget range and nowhere else.
