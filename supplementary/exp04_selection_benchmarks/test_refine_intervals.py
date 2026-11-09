import json

import numpy as np
import pytest

from supplementary.exp04_selection_benchmarks.refine_intervals import (
    bootstrap_differences, choose_repetitions, prepare_inputs, refine,
    risks_from_counts, sha256, summarize,
)


def frozen_artifacts(tmp_path):
    responses = tmp_path / 'responses.npz'
    losses_path = tmp_path / 'results.losses.npz'
    results = tmp_path / 'results.json'
    # Unequal source-group sizes make group-means averaging observably wrong.
    groups = np.array(['f0', 'f1', 'c0', 'c1', 't0', 't0', 't1', 't2', 't2', 't2'])
    split = np.array(['FIT'] * 2 + ['CERT'] * 2 + ['TEST'] * 6)
    probabilities = np.full((2, 10, 3, 3), 1 / 3)
    np.savez(responses, probabilities=probabilities, groups=groups,
             pair_ids=np.array([f'p{i}' for i in range(10)]),
             judge_ids=np.array(['J0', 'J1', 'J2']), context_names=np.array(['a', 'b']),
             split=split, metadata_json=np.array(json.dumps({'synthetic': True})))
    rng = np.random.default_rng(4)
    arrays = {'test_groups': groups[split == 'TEST']}
    rows = []
    for k in (1, 3):
        for method in ('backward', 'qr'):
            kept = list(range(k))
            loss = rng.uniform(0, .6, (2, 6, 3))
            loss[:, :, kept] = 0
            arrays[f'k{k}__{method}'] = loss
            row = {'k': k, 'method': method, 'seed': 7, 'kept_indices': kept,
                   'test_tv': float(loss.mean(1).max())}
            if method != 'backward':
                reference = arrays[f'k{k}__backward']
                row['paired_test_bootstrap'] = {
                    'family_size': 2,
                    'difference_method_minus_backward': row['test_tv'] - float(reference.mean(1).max())}
            rows.append(row)
    np.savez(losses_path, **arrays)
    results.write_text(json.dumps({'experiment': 'selection_benchmarks',
                                  'protocol': {'reference': 'backward'},
                                  'metadata': {'input_sha256': sha256(responses), 'synthetic': True},
                                  'rows': rows}))
    return results, losses_path, responses


def test_cluster_matrix_equals_literal_resampling_with_unequal_groups(tmp_path):
    paths = frozen_artifacts(tmp_path)
    prepared = prepare_inputs(*paths)
    counts = np.array([[2, 1, 0], [0, 0, 3], [0, 2, 1], [1, 1, 1]])
    actual = risks_from_counts(counts, prepared['group_sums'], prepared['group_sizes'],
                              len(prepared['rows']), prepared['components'])
    with np.load(paths[1]) as archive:
        groups = archive['test_groups']
        unique = np.unique(groups)
        expected = []
        for row in counts:
            indices = np.concatenate([np.tile(np.flatnonzero(groups == g), int(n))
                                      for g, n in zip(unique, row)])
            expected.append([archive[key][:, indices].mean(1).max() for key in prepared['keys']])
    assert actual == pytest.approx(np.asarray(expected))


def test_batching_preserves_common_draws_and_full_panel_rows(tmp_path):
    prepared = prepare_inputs(*frozen_artifacts(tmp_path))
    small = bootstrap_differences(prepared, 101, 123, 7)
    large = bootstrap_differences(prepared, 101, 123, 1000)
    assert small == pytest.approx(large)
    assert np.array_equal(small[:, 1], np.zeros(101))
    rows, radius = summarize(prepared, small, .05)
    expected_radius = np.quantile(np.abs(small - prepared['observed_differences']).max(1), .95)
    assert radius == pytest.approx(expected_radius)
    assert len(rows) == prepared['family_size'] == 2
    assert rows[1]['k'] == 3
    assert rows[1]['bonferroni_percentile_interval'] == [0., 0.]


def test_repetition_budget_resolves_all_bonferroni_tails():
    assert choose_repetitions(None, 80, .05) == 100000
    assert choose_repetitions(None, 200, .05) == 160000
    with pytest.raises(ValueError, match='>= 64000'):
        choose_repetitions(2000, 80, .05)


def test_refinement_preserves_inputs_records_provenance_and_refuses_overwrite(tmp_path):
    paths = frozen_artifacts(tmp_path)
    hashes = [sha256(p) for p in paths]
    output = tmp_path / 'refined.json'
    result = refine(*paths, output, bootstrap=1600, seed=17, batch_size=19)
    assert [sha256(p) for p in paths] == hashes
    assert len(result['rows']) == 2
    assert result['configuration']['expected_replicates_per_bonferroni_tail'] == 20
    assert result['source_artifacts']['responses']['sha256'] == hashes[2]
    assert json.loads(output.read_text())['configuration']['seed'] == 17
    with pytest.raises(FileExistsError, match='overwrite'):
        refine(*paths, output, bootstrap=1600)


@pytest.mark.parametrize('problem', ['groups', 'missing_comparison', 'risk', 'family', 'responses_hash'])
def test_refuses_misaligned_or_changed_frozen_evidence(tmp_path, problem):
    results, losses, responses = frozen_artifacts(tmp_path)
    report = json.loads(results.read_text())
    with np.load(losses) as z:
        arrays = {key: z[key] for key in z.files}
    if problem == 'groups':
        arrays['test_groups'] = arrays['test_groups'][::-1]
    elif problem == 'missing_comparison':
        del arrays['k3__qr']
    elif problem == 'risk':
        report['rows'][0]['test_tv'] += .1
    elif problem == 'family':
        report['rows'][1]['paired_test_bootstrap']['family_size'] = 1
    elif problem == 'responses_hash':
        report['metadata']['input_sha256'] = '0' * 64
    np.savez(losses, **arrays)
    results.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        prepare_inputs(results, losses, responses)
