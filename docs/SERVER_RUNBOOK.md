# Core-20 server runbook

Everything needed to produce the full results package on the UCloud B200 job,
in the order it has to happen.

The laptop cannot do this: Core-20 is ~316 GB of bf16 weights. Core-8 already
ran on the small machine and is what licensed the Core-20 attempt.

**Rule of thumb: nothing starts the GPU until step 3 passes.** Gate G0 is cheap
and catches the failures that otherwise waste hours.

---

## 0. Before starting the job

On the laptop, confirm the tree is committed and the tests pass:

```bash
.venv/bin/python -m pytest tests/ -q
git status --porcelain          # should be empty
git rev-parse HEAD              # note this; it lands in the manifest
```

Push the branch so the server can pull it.

---

## 1. Start the job

UCloud → Jobs → **gpu-nvidia-b200**

| Field | Value | Why |
|---|---|---|
| Hours | 24 | Billed on actual runtime; this is a ceiling, not a charge |
| Nodes | 1 | Nothing here is multi-node |
| Machine | 1 full B200 | Not a MIG slice — a slice will not hold a 14B judge comfortably |
| Drive | mount Member Files (⌘⌥S) | Otherwise everything vanishes when the job ends |
| SSH | enable (⌘⌥C) | Needed to run anything interactively |

Then connect:

```bash
ssh -p <port> ucloud@ssh.cloud.sdu.dk
```

The port is shown on the running job's page.

---

## 2. Set up the environment

`/work` itself is rewritten per job — the job scripts and logs at its top level
belong to the current job only. The *persistent* member-files drive is the
folder inside it, `/work/Lucija`. Everything expensive goes there.

```bash
cd /work/Lucija               # persistent; /work alone is not
git clone https://github.com/lucija-26/EPCRC.git EPCRC   # or: cd EPCRC && git pull
cd EPCRC
git checkout judges

python3 -m venv .venv
.venv/bin/pip install -U pip wheel
.venv/bin/pip install -r requirements.txt

# The B200 is sm_100, which the default PyPI torch wheel does not build for.
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements-scoring.txt
.venv/bin/pip install pytest nbconvert nbformat ipykernel   # tests and step 7
```

Confirm the GPU is actually visible to torch before going any further:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"
# expect: True (10, 0)
```

Put the model cache on the persistent drive too, or the 316 GB of weights is
lost every time the job restarts and the disk check measures the wrong
filesystem:

```bash
export HF_HOME=/work/Lucija/hf
```

Five of the twenty repos are gated (J07, J08 Llama; J09, J10 Gemma; J19 Aya).
Log in once — the token is written into `$HF_HOME/token`, so it survives job
restarts along with the cache and does not have to be re-exported:

```bash
.venv/bin/hf auth login          # paste a read token from hf.co/settings/tokens
```

The licence for each gated repo must also be accepted in a browser, with the
same account the token belongs to. G0 names any repo still missing access.

Then build the pair files, which are derived data and deliberately not in git:

```bash
.venv/bin/python -u experiments/build_rewardbench_pairs.py --all-seeds
```

---

## 3. Gate G0 — fail cheaply

```bash
.venv/bin/python -u experiments/score_panel.py --gate g0 --panel core20
```

This checks, without loading a single weight:

- every one of the 20 repos is reachable and not gated behind an unaccepted licence
- free disk exceeds the weight budget (~316 GB)
- the prompt set builds and the pair file matches its seed

**Do not continue past a G0 failure.** A gated repo found at hour 6 costs six
hours; found now it costs a browser click.

---

## 4. Gates G1 and G2

```bash
.venv/bin/python -u experiments/score_panel.py --gate g1 --panel core20
.venv/bin/python -u experiments/score_panel.py --gate g2 --panel core20 --max-exhaustive 0
```

G1 is a single judge on a handful of items — it proves the parsing and scoring
path works end to end. G2 sweeps decoding settings on a small sample.

`--max-exhaustive 0` matters for the same reason it does in E1: G2 finishes by
enumerating every subset to confirm the greedy pruners never beat the true
optimum. That is instant at N=8 and never finishes at N=20, and the scoring
itself will already have completed by the time it stalls. The exact-optimum
rows are then absent from `g2.json`, which `exhaustive_enumerated: false`
records.

G2 is resumable per (judge, context) block, so a rerun after a fix re-scores
nothing. `--judges J07` scores a single judge once its licence comes through.

The gate writes its sample to `results/core20/scores_gate/`, separate from the
production blocks in section 5. They must not share a directory: the cache is
keyed on (judge, context) alone, so a gate run landing in `scores/` would find
the 3724-pair blocks "not matching" its 100 items and overwrite all of them.

---

## 5. Score the panel

The long step. Run it detached so a dropped SSH connection does not kill it:

There is no separate "score" subcommand: G2 *is* the scoring pass, and `--items`
sets how much of the pair file it covers. `--items` above the pair count means
all of it. Bare `score_panel.py --panel core20` would silently re-run G0,
because `--gate` defaults to `g0`.

```bash
tmux new -s score
.venv/bin/python -u experiments/score_panel.py --gate g2 --panel core20 \
    --items 4000 --scores-only --batch-size 64 2>&1 | tee score.log
# detach with ctrl-b d, reattach with: tmux attach -t score
```

`--batch-size 64` is not optional at this scale. The default of 8 leaves a B200
mostly idle — measured on 256 pairs, Phi-3.5-mini goes 1.7 → 17.0 pairs/s from
bs=8 to bs=32, and Qwen2.5-14B goes 5.1 → 8.2 at bs=64 before regressing at 128.
Small judges saturate at 32 and the 14B peaks at 64, so 64 is the single best
choice for a mixed panel. It turns a ~18–27 hour pass, which would not fit in
the job, into ~6–7 hours at a sustained ~21 pairs/s.

`--scores-only` stops once every block is written. Without it the run ends by
driving the pruners over all 3724 pairs, which costs many hours and re-tests the
pipeline rather than the data — G2 on the 100-item sample already did that.

Writes one JSON per (judge, context) into `results/core20/scores/`. The run is
resumable: an existing block is skipped, so an interruption costs only the
judge that was in flight.

Progress check from another shell:

```bash
ls results/core20/scores/*.json | wc -l     # target: 20 judges x 7 contexts = 140
```

---

## 6. Run the claims

All CPU — no GPU needed once scoring is done. Everything here reads the cached
response tensor from step 5, so the GPU can be idle for the whole of this
section. Run detached.

```bash
nohup sh -c '
.venv/bin/python -u experiments/experiment_e0_noncomposability.py --real --panel core20
.venv/bin/python -u experiments/experiment_e1_compression_frontier.py --panel core20 --max-exhaustive 0
.venv/bin/python -u experiments/experiment_c4_stress_specialists.py --panel core20
.venv/bin/python -u experiments/experiment_c5_downstream.py --panel core20
.venv/bin/python -u experiments/experiment_c3_baselines.py --panel core20 --max-exhaustive 0
' > claims.log 2>&1 &
```

C3 runs last deliberately: it is the slowest — 6090 s (1 h 41 m) on the
19-judge panel — and the ones before it are what the notebooks read first, so
a failure surfaces early rather than after the long wait.

C6 and C7 both parallelise internally, so they go in their own chain rather
than inside the one above, and `--workers` should be set to the cores the
machine actually has:

```bash
nohup sh -c '
.venv/bin/python -u experiments/experiment_c6_exchange.py --panel core20 --workers 16
.venv/bin/python -u experiments/experiment_c7_certification.py --panel core20 --workers 16
' > exchange_certification.log 2>&1 &
```

These two are the long poles on a laptop and the reason for running them here.
C7 re-partitions the items 120 times and re-certifies every budget on each,
which is roughly ten minutes per repetition at eight workers; C6 enumerates
subsets of a 12-to-16 judge subpanel to prove the optimum. Both scale almost
linearly with cores, so the server turns an overnight job into an hour or two.

Notes:

- Enumerating every subset to find the true optimum is 2^20 evaluations per
  budget: instant on Core-8, never finishing here. E0 and C3 are already safe
  by default, because `select_exhaustive` returns nothing once the panel
  exceeds `--max-exhaustive`, which defaults to 10. Passing `0` only makes the
  intent explicit. Either way the exhaustive row is absent from the Core-20
  tables, which is expected and is stated in the summary.
- The one place this was *not* guarded was `score_panel.py`, where G2 finished
  all its GPU work and then enumerated forever — hence `--max-exhaustive` there
  too, and the `exhaustive_enumerated` flag in `g2.json`.
- E0 also sweeps the leave-one-out breakpoints, which is where C1 gets its
  evidence when the predeclared gamma grid is too tight for the panel.
- C3 parallelises over subsets; it deduplicates to the unique subsets first,
  so adding baselines costs far less than it looks.
- C6 uses tolerance points outside the plan's section 23 grid, because that
  grid is unreachable on real judges. The deviation is recorded in the result
  file and repeated in the summary, so it does not have to be remembered here.
- C7's split grid is 120 repetitions, not the five predeclared seeds. Five
  seeds cannot resolve a 5% violation rate at all. The five predeclared seeds
  lead the list, so the primary setting stays inside the grid.

---

## 6b. Score the E7 transfer benchmark

The second and last GPU step. E7 asks whether a basis picked on RewardBench 2
still works on a benchmark it was never selected for, so the same 20 judges have
to answer JudgeBench's 620 pairs under the same 7 contexts.

```bash
.venv/bin/python -u experiments/build_judgebench_pairs.py --all-seeds

tmux new -s transfer
.venv/bin/python -u experiments/score_panel.py --gate g2 --panel core20 \
    --dataset judgebench --items 620 --scores-only --batch-size 64 \
    2>&1 | tee transfer.log
```

`--dataset judgebench` is not optional and is not cosmetic. Blocks are cached as
`{judge}__{context}.json` with no benchmark in the name, so without it this run
would overwrite `results/core20/scores/` — the blocks every other claim is
computed from. With it, the blocks go to `results/core20_judgebench/scores/` and
the pairs come from `data/judgebench_pairs_seed*.jsonl`.

620 pairs is about a sixth of the in-domain pass, so expect roughly an hour.
The weights are already on disk from step 5, so do not pass `--evict` here.

Then, on CPU:

```bash
.venv/bin/python -u experiments/experiment_e7_transfer.py --panel core20
```

Two limits travel with every E7 number and are recorded in the result file:
JudgeBench has no ties, so nothing here says whether the tie corner survives
compression; and 620 pairs means the calibration curve runs out of data before
it flattens.

---

## 7. Build the package

```bash
EPCRC_PANEL=core20 .venv/bin/python -m nbconvert --to notebook --execute \
    --inplace notebooks/04_results_package.ipynb
.venv/bin/python -u experiments/export_results.py --panel core20
```

The notebook reads `EPCRC_PANEL` and defaults to `core8`, so nothing has to be
edited by hand — and the server's working tree stays clean apart from the
executed outputs.

Produces `results/export/epcrc_results_core20_<date>.zip` containing
`tables/`, `figures/`, `raw/`, `notebooks/`, `SUMMARY.md` and a `manifest.json`
with a SHA-256 per file plus the git commit.

Copy it off the job before shutting down — the container filesystem does not
survive, only the mounted drive does.

---

## 8. Shut the job down

Stop it from the UCloud jobs page. Billing is by actual runtime, so stopping
early is the whole saving.

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| G0 says a repo is gated | Licence not accepted for that model | Accept it on the model's HF page with the same account as `HF_TOKEN` |
| A repo that worked before returns 401 | The command ran in a shell that never got `HF_HOME` — `nohup sh -c ...`, `ssh host "..."` and cron do not read your profile | Export `HF_HOME=/work/Lucija/hf` inside the command itself, not before it |
| Out of memory on a 14B judge | Another judge's weights still resident | Add `--evict` to drop weights between judges |
| Scoring restarted from zero | `HF_HOME` was not on `/work` | Re-export it; already-written score blocks are still skipped |
| E1 appears to hang | `--max-exhaustive` left at its default | Rerun with `--max-exhaustive 0` |
| Export says the package is partial | An experiment has not been run | The warning names which one |

---

## What Core-20 is expected to change

Core-8 is a real but small panel, and two of its limits are sample-size limits
rather than findings:

- **C1** had no bite on the predeclared gamma grid, because the smallest
  leave-one-out error was 0.241 while the grid stops at 0.20. With 20 judges
  and several within-family pairs, leave-one-out errors should fall
  substantially and the declared grid should start doing work.
- **C3** could not separate coverage from the strongest geometric baseline —
  their bootstrap intervals overlap. At N=8 there are few distinct subsets to
  choose between; at N=20 there are many, which is where a selection rule can
  actually show an advantage.

Both are stated as open in the current summary. Core-20 is what settles them,
either way — and a negative result stated clearly is still a result.
