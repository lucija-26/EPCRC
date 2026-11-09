"""Refine every frozen exp04 comparison without selecting or refitting models.

Example:
  python supplementary/exp04_selection_benchmarks/refine_intervals.py \
    --results RUN/exp04_selection_benchmarks/results.json \
    --losses RUN/exp04_selection_benchmarks/results.losses.npz \
    --input RUN/responses.npz --output RUN/exp04_selection_benchmarks/refined.json

Bootstrap source-item groups jointly across contexts, endpoints, and methods.
Batching bounds the large matrix product; only B x family_size differences are
retained. Original artifacts are read-only and an existing output is refused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from supplementary.common import load_npz


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def choose_repetitions(requested, family_size, alpha):
    if family_size < 1 or not 0 < alpha < 1:
        raise ValueError('Need a nonempty comparison family and 0 < alpha < 1')
    minimum = math.ceil(40 * family_size / alpha)  # >=20 draws per Bonferroni tail.
    if requested is None:
        return max(100000, minimum)
    if requested < minimum:
        raise ValueError(f'--bootstrap must be >= {minimum} for >=20 expected draws per Bonferroni tail')
    return requested


def prepare_inputs(results_path, losses_path, input_path):
    paths = {'results': Path(results_path).resolve(), 'losses': Path(losses_path).resolve(),
             'responses': Path(input_path).resolve()}
    hashes = {name: sha256(path) for name, path in paths.items()}
    report = json.loads(paths['results'].read_text())
    if report.get('experiment') != 'selection_benchmarks':
        raise ValueError('Expected an original exp04 selection_benchmarks report')
    if report.get('protocol', {}).get('reference') != 'backward':
        raise ValueError('The frozen report must predeclare backward as its reference')
    if report.get('metadata', {}).get('input_sha256') != hashes['responses']:
        raise ValueError('Original report does not match the supplied responses SHA256')
    data = load_npz(paths['responses'])
    groups = np.asarray(data.groups['TEST']).astype(str)
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError('At least two TEST source-item groups are required')
    group_sizes = np.bincount(inverse)
    rows = report.get('rows', [])
    if not rows:
        raise ValueError('Original report has no frozen comparisons')
    keys, by_k = [], {}
    for row in rows:
        k, method = row['k'], row['method']
        kept = row['kept_indices']
        if (not isinstance(k, int) or not 1 <= k <= len(data.judge_ids)
                or len(kept) != k or len(set(kept)) != k
                or any(not isinstance(j, int) or not 0 <= j < len(data.judge_ids) for j in kept)):
            raise ValueError('Invalid frozen retained set')
        key = f'k{k}__{method}'
        if key in keys:
            raise ValueError('Duplicate frozen k/method comparison')
        keys.append(key)
        by_k.setdefault(k, set()).add(method)
    method_sets = list(by_k.values())
    if any('backward' not in methods or methods != method_sets[0] for methods in method_sets):
        raise ValueError('Every budget must contain the same complete method family including backward')
    candidate_indices = [i for i, row in enumerate(rows) if row['method'] != 'backward']
    family_size = len(candidate_indices)
    if family_size < 1:
        raise ValueError('No non-reference comparisons to refine')
    reference_indices = [keys.index(f"k{rows[i]['k']}__backward") for i in candidate_indices]
    shape = (len(data.context_names), len(groups), len(data.judge_ids))
    components = shape[0] * shape[2]
    sums = np.zeros((len(unique), len(rows) * components))
    observed = np.empty(len(rows))
    with np.load(paths['losses'], allow_pickle=False) as archive:
        if set(archive.files) != set(keys) | {'test_groups'}:
            raise ValueError('Loss archive must contain every frozen row exactly once and no extra comparisons')
        if not np.array_equal(archive['test_groups'].astype(str), groups):
            raise ValueError('Loss TEST groups do not align with the supplied responses')
        for i, (key, row) in enumerate(zip(keys, rows)):
            loss = np.asarray(archive[key], dtype=float)
            if loss.shape != shape or not np.isfinite(loss).all() or loss.min() < 0 or loss.max() > 1 + 1e-6:
                raise ValueError(f'Invalid loss axes or TV values: {key}')
            if np.any(np.abs(loss[:, :, row['kept_indices']]) > 1e-12):
                raise ValueError(f'Retained endpoints must have zero reconstruction loss: {key}')
            observed[i] = loss.mean(axis=1).max()
            if not np.isclose(observed[i], row['test_tv'], rtol=1e-10, atol=1e-12):
                raise ValueError(f'Frozen TEST risk disagrees with per-item losses: {key}')
            np.add.at(sums[:, i * components:(i + 1) * components], inverse,
                      loss.transpose(1, 0, 2).reshape(len(groups), components))
    differences = observed[candidate_indices] - observed[reference_indices]
    for column, i in enumerate(candidate_indices):
        prior = rows[i].get('paired_test_bootstrap', {})
        if prior.get('family_size') != family_size:
            raise ValueError('Original bootstrap family size does not match all frozen comparisons')
        if not np.isclose(prior.get('difference_method_minus_backward', np.nan), differences[column],
                          rtol=1e-10, atol=1e-12):
            raise ValueError('Original paired difference disagrees with frozen losses')
    return {'paths': paths, 'hashes': hashes, 'report': report, 'rows': rows, 'keys': keys,
            'group_sizes': group_sizes, 'group_sums': sums, 'components': components,
            'candidate_indices': candidate_indices, 'reference_indices': reference_indices,
            'observed_risks': observed, 'observed_differences': differences,
            'family_size': family_size, 'n_test_items': len(groups), 'n_test_groups': len(unique)}


def risks_from_counts(multiplicities, group_sums, group_sizes, n_metrics, components):
    """Exactly reproduce resampled item means, including unequal group sizes."""
    counts = np.asarray(multiplicities, dtype=float)
    denominators = counts @ group_sizes
    if np.any(denominators <= 0):
        raise ValueError('Every bootstrap draw must contain items')
    means = (counts @ group_sums) / denominators[:, None]
    return means.reshape(len(counts), n_metrics, components).max(axis=2)


def bootstrap_differences(prepared, repetitions, seed, batch_size=128, progress=None):
    if repetitions < 2 or batch_size < 1:
        raise ValueError('Need repetitions >=2 and batch_size >=1')
    rng = np.random.default_rng(seed)
    group_count = prepared['n_test_groups']
    probabilities = np.full(group_count, 1.0 / group_count)
    differences = np.empty((repetitions, prepared['family_size']))
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        multiplicities = rng.multinomial(group_count, probabilities, size=stop - start)
        risks = risks_from_counts(multiplicities, prepared['group_sums'], prepared['group_sizes'],
                                  len(prepared['rows']), prepared['components'])
        differences[start:stop] = (risks[:, prepared['candidate_indices']]
                                  - risks[:, prepared['reference_indices']])
        if progress:
            progress(stop, repetitions)
    return differences


def summarize(prepared, draws, alpha):
    observed = prepared['observed_differences']
    family = prepared['family_size']
    pointwise = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    tail = alpha / (2 * family)
    bonferroni = np.quantile(draws, [tail, 1 - tail], axis=0)
    max_deviations = np.max(np.abs(draws - observed[None, :]), axis=1)
    radius = float(np.quantile(max_deviations, 1 - alpha))
    rows = []
    for column, (i, reference) in enumerate(zip(prepared['candidate_indices'], prepared['reference_indices'])):
        original = prepared['rows'][i]
        rows.append({'k': original['k'], 'method': original['method'], 'seed': original.get('seed'),
                     'kept_indices': original['kept_indices'], 'kept_ids': original.get('kept_ids'),
                     'reference': 'backward', 'test_tv': float(prepared['observed_risks'][i]),
                     'reference_test_tv': float(prepared['observed_risks'][reference]),
                     'difference_method_minus_backward': float(observed[column]),
                     'pointwise_percentile_interval': pointwise[:, column].tolist(),
                     'bonferroni_percentile_interval': bonferroni[:, column].tolist(),
                     'simultaneous_centered_max_abs_interval':
                         [float(observed[column] - radius), float(observed[column] + radius)]})
    return rows, radius


def refine(results, losses, input_path, output, bootstrap=None, seed=20260818,
           batch_size=128, alpha=.05, progress=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    if output in {Path(p).resolve() for p in (results, losses, input_path)}:
        raise ValueError('Output must be separate from all original inputs')
    started = time.perf_counter()
    prepared = prepare_inputs(results, losses, input_path)
    repetitions = choose_repetitions(bootstrap, prepared['family_size'], alpha)
    draws = bootstrap_differences(prepared, repetitions, seed, batch_size, progress)
    rows, radius = summarize(prepared, draws, alpha)
    if any(sha256(path) != prepared['hashes'][name] for name, path in prepared['paths'].items()):
        raise RuntimeError('An original input changed during refinement; refusing to save mixed provenance')
    result = {
        'experiment': 'selection_benchmark_interval_refinement',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_artifacts': {name: {'path': str(path), 'sha256': prepared['hashes'][name]}
                             for name, path in prepared['paths'].items()},
        'source_script_sha256': sha256(__file__),
        'source_metadata': prepared['report']['metadata'],
        'configuration': {'seed': seed, 'bootstrap_repetitions': repetitions, 'batch_size': batch_size,
                          'alpha': alpha, 'family_size': prepared['family_size'],
                          'expected_replicates_per_bonferroni_tail': repetitions * alpha / (2 * prepared['family_size']),
                          'n_test_items': prepared['n_test_items'], 'n_test_groups': prepared['n_test_groups'],
                          'numpy_version': np.__version__, 'python_version': platform.python_version()},
        'protocol': {
            'family': 'All original non-backward rows, including every random repeat and full-panel zero difference; no filtering.',
            'conditioning': 'FIT/CERT data, retained sets, weights, methods and TEST losses frozen; no refitting or new selection.',
            'sampling': 'Uniform multinomial resampling of TEST source-item groups; identical draws across every context, endpoint and comparison.',
            'statistic': 'Method minus backward maximum context/endpoint mean TV; maxima recomputed in each draw.',
            'pointwise_interval': 'Percentile bootstrap at alpha/2 and 1-alpha/2.',
            'bonferroni_interval': 'Percentile bootstrap at alpha/(2F) and 1-alpha/(2F); approximate familywise coverage via Bonferroni.',
            'simultaneous_interval': 'Observed differences plus/minus the (1-alpha) quantile of the maximum absolute centered bootstrap difference over the entire family.',
            'interpretation': 'Negative differences favor the named method. Intervals are conditional bootstrap approximations, not finite-sample certificates or uncertainty over ecosystems.',
            'monte_carlo': 'At least 20 expected draws per Bonferroni tail improves numerical resolution; it does not eliminate bootstrap or finite-sample approximation error.',
        },
        'simultaneous_centered_max_abs_radius': radius,
        'elapsed_seconds': time.perf_counter() - started,
        'rows': rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects original artifacts and concurrent refinement runs.
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--losses', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='Original complete responses.npz')
    parser.add_argument('--output', type=Path, required=True, help='New JSON path; existing files are refused')
    parser.add_argument('--bootstrap', type=int, help='Default max(100000, ceil(40*family_size/alpha))')
    parser.add_argument('--seed', type=int, default=20260818)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=.05)
    args = parser.parse_args()
    last_report = 0

    def progress(done, total):
        nonlocal last_report
        if done == total or done - last_report >= 10000:
            print(f'Frozen comparison bootstrap: {done}/{total}', flush=True)
            last_report = done

    result = refine(args.results, args.losses, args.input, args.output, args.bootstrap,
                    args.seed, args.batch_size, args.alpha, progress)
    print(f"Saved {len(result['rows'])} unchanged comparisons to {args.output}", flush=True)


if __name__ == '__main__':
    main()
