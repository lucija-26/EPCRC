"""Explicit resplitting and fixed-case regression tests; tiny synthetic CPU data."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

from epcrc.panel import CORE20
from epcrc.rewardbench import build_pairs, grouped_split, split_manifest
from supplementary.exp01_dependency_significance.fixed_case import (
    FIXED_JUDGES, load_explicit_split, main, run_fixed_case,
)


@pytest.fixture
def tiny_panel(tmp_path):
    pairs = build_pairs([{'id': str(i), 'subset': 'Focus', 'prompt': f'Question {i}',
                          'chosen': ['yes'], 'rejected': ['no']} for i in range(16)], seed=20260818)
    rows = [asdict(pair) for pair in pairs]
    groups = np.asarray([pair.base_item_id for pair in pairs])
    pairs_path = tmp_path / 'pairs.jsonl'
    pairs_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    first = {'FIT': groups[:8].tolist(), 'CERT': groups[8:12].tolist(), 'TEST': groups[12:].tolist()}
    second = {'FIT': groups[4:12].tolist(), 'CERT': groups[12:].tolist(), 'TEST': groups[:4].tolist()}
    first_path, second_path = tmp_path / 'first.json', tmp_path / 'second.json'
    first_path.write_text(json.dumps({'seed': 20260818, 'splits': first}))
    second_path.write_text(json.dumps({'seed': 20260819, 'splits': second}))
    assignment = np.array(['FIT'] * 8 + ['CERT'] * 4 + ['TEST'] * 4)
    rng = np.random.default_rng(22)
    probabilities = rng.dirichlet([.8, .8, .8], size=(2, len(rows), 20))
    input_path = tmp_path / 'panel.npz'
    fields = dict(probabilities=probabilities, groups=groups,
                  pair_ids=np.array([pair.pair_id for pair in pairs]),
                  judge_ids=np.array(list(CORE20)), model_ids=np.array(list(CORE20.values())),
                  context_names=np.array(['clean', 'context2']), split=assignment,
                  metadata_json=np.array(json.dumps({'synthetic': True, 'split_seed': 20260818})))
    np.savez_compressed(input_path, **fields)
    return SimpleNamespace(input=input_path, pairs_path=pairs_path, pairs=pairs, fields=fields,
                           first=first_path, second=second_path, groups=groups, tmp=tmp_path)


def test_explicit_manifest_replaces_stored_split(tiny_panel):
    t = tiny_panel
    first = load_explicit_split(t.input, t.pairs_path, t.first)
    second = load_explicit_split(t.input, t.pairs_path, t.second)
    assert first.metadata['input_stored_split_matches_this_manifest'] is True
    assert second.metadata['input_stored_split_matches_this_manifest'] is False
    assert first.groups['TEST'].tolist() == t.groups[12:].tolist()
    assert second.groups['TEST'].tolist() == t.groups[:4].tolist()
    assert np.array_equal(second.splits['FIT'].blocks[0], t.fields['probabilities'][0, 4:12])
    assert second.metadata['split_seed'] == 20260819
    assert second.metadata['split_source'] == 'explicit_verified_manifest'
    assert first.metadata['split_group_sha256'] != second.metadata['split_group_sha256']


def test_original_summary_manifest_checks_actual_group_partition(tiny_panel):
    t = tiny_panel
    assignment = grouped_split(t.pairs, seed=20260820)
    path = t.tmp / 'original_summary.json'
    path.write_text(json.dumps(split_manifest(t.pairs, assignment, 20260820)))
    data = load_explicit_split(t.input, t.pairs_path, path)
    assert set(data.groups['TEST']) == set(assignment['TEST'])
    assert data.metadata['split_manifest_seed'] == 20260820
    assert data.metadata['split_group_sha256'] == json.loads(path.read_text())['split_sha256']


def test_case_never_reselects_trio_when_individual_errors_fail_threshold(tiny_panel):
    t = tiny_panel
    data = load_explicit_split(t.input, t.pairs_path, t.first)
    result = run_fixed_case(data, gamma=0, bootstrap=100, bootstrap_seed=1)
    assert result['fixed_removed_ids'] == list(FIXED_JUDGES)
    assert len(result['kept_ids']) == 17
    assert not result['splits']['CERT']['all_individuals_within_gamma_point_estimate']
    assert result['splits']['CERT']['joint_max_tv'] > 0
    for split in ['CERT', 'TEST']:
        assert result['splits'][split]['bootstrap']['n_groups'] == 4
        assert [r['judge_id'] for r in result['splits'][split]['per_judge']] == list(FIXED_JUDGES)
    assert 'Validation-split' in result['splits']['CERT']['bootstrap']['analysis_role']
    assert 'Held-out' in result['splits']['TEST']['bootstrap']['analysis_role']


def test_pair_order_mismatch_cannot_be_silently_resplit(tiny_panel):
    t = tiny_panel
    t.fields['pair_ids'] = t.fields['pair_ids'][::-1]
    np.savez_compressed(t.input, **t.fields)
    with pytest.raises(ValueError, match='pair order'):
        load_explicit_split(t.input, t.pairs_path, t.first)


def test_duplicate_partitions_rejected_even_with_different_filename_seed(tiny_panel):
    t = tiny_panel
    duplicate = json.loads(t.first.read_text())
    duplicate['seed'] = 999
    t.second.write_text(json.dumps(duplicate))
    output = t.tmp / 'fixed.json'
    with pytest.raises(SystemExit):
        main(['--input', str(t.input), '--pairs', str(t.pairs_path), '--split-manifests',
              str(t.first), str(t.second), '--bootstrap', '100', '--output', str(output)])
    assert not output.exists()


def test_cli_preserves_all_declared_splits_and_primary_designation(tiny_panel):
    t = tiny_panel
    output = t.tmp / 'fixed.json'
    main(['--input', str(t.input), '--pairs', str(t.pairs_path), '--split-manifests',
          str(t.first), str(t.second), '--bootstrap', '100', '--output', str(output)])
    report = json.loads(output.read_text())
    assert report['status'] == 'complete'
    assert [row['analysis_role'] for row in report['split_reports']] == ['primary', 'overlapping_split_sensitivity']
    assert [row['metadata']['split_manifest_seed'] for row in report['split_reports']] == [20260818, 20260819]
    assert all(row['metadata']['synthetic'] for row in report['split_reports'])
