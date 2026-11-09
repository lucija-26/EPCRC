# Render frozen supplementary results

These three renderers were moved out of result directories so their source can
be versioned without copying response caches, analysis outputs or figures. They
use the CPU dependencies in `supplementary/requirements.txt` and perform no model
inference, fitting or resampling. Paths below are relative to the repository;
default inputs are resolved from the script location, independently of the
working directory. Use `--help` for explicit input/output overrides.

| Script | Required results | Default figure directory | Previous source location |
| --- | --- | --- | --- |
| `render_core20.py` | `results/supplementary/real/core20/exp02_reconstruction/results.json`, `exp04_selection_benchmarks/results.json`, `exp05_measured_cost/results.json`, plus the reconstruction TEST-loss NPZ referenced by its JSON | `results/supplementary/real/core20/figures/` | `results/supplementary/real/core20/figures/render_core20.py` |
| `plot_dependency.py` | Main dependency and five-split fixed-case JSONs, their exact response NPZ, and five original split manifests | `results/supplementary/real/core20/figures_dependency/` | `results/supplementary/real/core20/figures_dependency/plot_dependency.py` |
| `render_reconstruction.py` | `results/supplementary/real/core20_convergence1000/exp02_reconstruction/results.json` and `audit_status.json` | `results/supplementary/real/core20_convergence1000/figures/` | `results/supplementary/real/core20_convergence1000/figures/render_reconstruction.py` |

Run from the repository root after obtaining or regenerating the required
artifacts:

```bash
python supplementary/figures/render_core20.py
python supplementary/figures/plot_dependency.py
python supplementary/figures/render_reconstruction.py
```

`render_core20.py` accepts `--run-dir` and `--output-dir`. It retains every measured
cost point and labels only frozen CERT-selected operating points. Its underlying
renderer requires real results with a common response-cache hash.

`plot_dependency.py` accepts `--main`, `--fixed`, `--responses`,
`--split-manifest-dir` and `--output`. Defaults are the main report under
`results/supplementary/real/core20/exp01_dependency_significance/`, the fixed-case
report at `results/supplementary/real/core20_fixed_case/results.json`, the response
cache at `results/supplementary/real/core20/responses.npz`, and manifests in
`data/`. Reports can retain their original absolute provenance paths: the
renderer locates manifest basenames in the supplied directory and still verifies
every recorded SHA-256, including the response cache. Do not edit report hashes
to accommodate relocation. Complete reports, all five published split seeds,
the J07/J08/J11 trio, first primary split, gamma 0.20, alpha 0.05, and matching
fixed-case bootstrap counts remain required.

`render_reconstruction.py` accepts `--run-dir` and `--output-dir`. It is the
specific frozen maxiter-1000 audit figure, with eight declared supports and
19/54 unconverged selected gate fits; its original result/status hash and content
checks are retained. Existing output artifacts are not overwritten. The figure
shows TEST point estimates; its CSV intervals are pointwise and do not establish
simultaneous curve bands.

The original plot content and statistical caveats are preserved: five overlapping
partitions are sensitivity checks, conditional bands do not cover all partitions
and roles simultaneously, sufficient composition bounds do not prove universal
joint failure, and TEST tolerance lines do not authorize selecting budgets on
TEST. Each renderer records source hashes and exports exact plotted data with
PNG/PDF/SVG figures. Result data is intentionally absent from this source-only
migration.
