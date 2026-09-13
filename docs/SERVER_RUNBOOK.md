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

```bash
cd /work                      # the mounted drive, NOT the container filesystem
git clone <repo> EPCRC        # or: cd EPCRC && git pull
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

Put the model cache on the mounted drive too, or downloads are lost on restart
and the disk check measures the wrong filesystem:

```bash
export HF_HOME=/work/hf
export HF_TOKEN=<your token>        # needed for the gated Llama and Gemma repos
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
.venv/bin/python -u experiments/score_panel.py --gate g2 --panel core20
```

G1 is a single judge on a handful of items — it proves the parsing and scoring
path works end to end. G2 sweeps decoding settings on a small sample.

---

## 5. Score the panel

The long step. Run it detached so a dropped SSH connection does not kill it:

```bash
tmux new -s score
.venv/bin/python -u experiments/score_panel.py --panel core20 2>&1 | tee score.log
# detach with ctrl-b d, reattach with: tmux attach -t score
```

Writes one JSON per (judge, context) into `results/core20/scores/`. The run is
resumable: an existing block is skipped, so an interruption costs only the
judge that was in flight.

Progress check from another shell:

```bash
ls results/core20/scores/*.json | wc -l     # target: 20 judges x 7 contexts = 140
```

---

## 6. Run the three claims

All CPU, all fast — no GPU needed once scoring is done.

```bash
.venv/bin/python -u experiments/experiment_e0_noncomposability.py --real --panel core20
.venv/bin/python -u experiments/experiment_e1_compression_frontier.py --panel core20 --max-exhaustive 0
.venv/bin/python -u experiments/experiment_c3_baselines.py --panel core20
```

Notes:

- `--max-exhaustive 0` is **required** at N=20. Enumerating every subset is
  2^20 evaluations per budget; it finishes on Core-8 and never finishes here.
  The exhaustive row simply will not appear in the Core-20 tables, which is
  expected and is stated in the summary.
- E0 also sweeps the leave-one-out breakpoints, which is where C1 gets its
  evidence when the predeclared gamma grid is too tight for the panel.
- C3 parallelises over subsets; it deduplicates to the unique subsets first,
  so adding baselines costs far less than it looks.

---

## 7. Build the package

```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace \
    notebooks/04_results_package.ipynb        # set PANEL = "core20" first
.venv/bin/python -u experiments/export_results.py --panel core20
```

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
