"""Run every analysis entrypoint on explicitly synthetic data, without GPUs."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'supplementary' / 'smoke'
OUT.mkdir(parents=True, exist_ok=True)
commands = [
    ('exp01_dependency_significance', ['--bootstrap', '200', '--plot']),
    ('exp02_reconstruction', ['--budgets', '2', '4', '--bootstrap', '100', '--maxiter', '30']),
    ('exp03_ecosystem_replication', ['--k', '2', '3', '--bootstrap', '100']),
    ('exp04_selection_benchmarks', ['--k', '2', '3', '--bootstrap', '100', '--random-repeats', '2']),
    ('exp05_measured_cost', ['--ks', '2', '4', '6', '--timing-repeats', '3']),
]
# exp02 has a directory output because it also saves fitted parameters/losses.
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
for name, options in commands:
    output = OUT / name if name == 'exp02_reconstruction' else OUT / (name + '.json')
    subprocess.run([sys.executable, str(ROOT / 'supplementary' / name / 'run.py'), '--demo',
                    *options, '--output', str(output)], cwd=ROOT, env=env, check=True)
print('All five synthetic CLI checks passed; these are not empirical paper results.')
