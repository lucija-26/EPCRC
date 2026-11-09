"""Small synthetic fixtures; no real experiment or expensive subset search."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from supplementary import common
from supplementary.exp02_reconstruction import run as reconstruction
from supplementary.exp02_reconstruction.convergence_audit import (
    BUDGETS, STAGE, audit, prepare_sources, replay_selector,
)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    input_path = tmp_path / 'responses.npz'
    # Twenty identical endpoints make non-gate/gate fitting immediate. Two
    # source groups in each stored split suffice for this software-only check.
    probability = np.tile([.2, .3, .5], (1, 12, 20, 1))
    ids = [f'J{i:02}' for i in range(20)]
    np.savez(input_path, probabilities=probability, groups=np.array([f'g{i // 2}' for i in range(12)]),
             pair_ids=np.array([f'p{i}' for i in range(12)]), judge_ids=np.array(ids),
             model_ids=np.array([f'synthetic/{j}' for j in ids]), context_names=np.array(['clean']),
             split=np.array(['FIT'] * 4 + ['CERT'] * 4 + ['TEST'] * 4),
             metadata_json=np.array(json.dumps({'synthetic': True, 'split_seed': 17})))
    runs = [tmp_path / 'main', tmp_path / 'k17']
    with monkeypatch.context() as context:
        context.setattr(common, 'select_subset', lambda fit, k, method, seed: list(range(k)))
        for directory, budgets in zip(runs, [[k for k in BUDGETS if k != 17], [17]]):
            output = directory / STAGE
            args = SimpleNamespace(input=input_path, demo=False, output=output, seed=17, budgets=budgets,
                                   selection='backward', regularizations=[.01], temperatures=[1.],
                                   maxiter=3, bootstrap=2, gammas=[.12, .2])
            reconstruction.run(args)
            command = ['python', str(Path(reconstruction.__file__).resolve()), '--input', str(input_path),
                       '--output', str(output), '--seed', '17', '--bootstrap', '2',
                       '--budgets', *map(str, budgets)]
            stage = {'status': 'complete', 'artifact_sha256': common.sha256_file(output / 'results.json'),
                     'command': command, 'code_signature': {
                         'script_sha256': common.sha256_file(reconstruction.__file__),
                         'common_sha256': common.sha256_file(common.__file__),
                         'merged_sha256': common.sha256_file(input_path)}}
            (directory / 'status.json').write_text(json.dumps({'stages': {STAGE: stage}}))
    return input_path, runs


def mutate_report(run_dir, change):
    path = run_dir / STAGE / 'results.json'
    report = json.loads(path.read_text())
    change(report)
    path.write_text(json.dumps(report))
    status_path = run_dir / 'status.json'
    status = json.loads(status_path.read_text())
    status['stages'][STAGE]['artifact_sha256'] = common.sha256_file(path)
    status_path.write_text(json.dumps(status))


def test_complete_audit_changes_only_iteration_budget_and_marks_replayed_times(sources, tmp_path):
    input_path, runs = sources
    original_selector = common.select_subset
    original_hashes = {p: common.sha256_file(p) for run in runs for p in run.rglob('*') if p.is_file()}
    output = tmp_path / 'audit'
    result = audit(input_path, runs, output)
    assert common.select_subset is original_selector
    assert result['configuration']['maxiter'] == 1000
    assert result['replayed_frozen_supports'] is True
    assert result['non_gate_consistency_verification']['passed'] is True
    assert result['non_gate_consistency_verification']['largest_non_gate_test_loss_difference'] == 0
    assert [row['k'] for row in result['budgets']] == BUDGETS
    for row in result['budgets']:
        assert row['retained_indices'] == list(range(row['k']))
        assert 'selection_seconds' not in row
        assert row['source_selection_seconds'] >= 0
        assert row['replayed_support_lookup_seconds'] >= 0
    status = json.loads((output / 'audit_status.json').read_text())
    assert status['status'] == 'complete'
    assert status['result_sha256'] == common.sha256_file(output / 'results.json')
    assert all(common.sha256_file(p) == digest for p, digest in original_hashes.items())
    with pytest.raises(FileExistsError, match='new output'):
        audit(input_path, runs, output)


def test_selector_restored_when_run_raises_and_failures_are_explicit(sources, tmp_path, monkeypatch):
    input_path, runs = sources
    original_selector = common.select_subset
    def fail(args):
        raise RuntimeError('intentional fixture failure')
    monkeypatch.setattr(reconstruction, 'run', fail)
    output = tmp_path / 'failed_audit'
    with pytest.raises(RuntimeError, match='fixture failure'):
        audit(input_path, runs, output)
    assert common.select_subset is original_selector
    assert json.loads((output / 'audit_status.json').read_text())['status'] == 'failed'


def test_replay_rejects_changed_fit_and_unfrozen_budget(sources):
    prepared = prepare_sources(*sources)
    original = common.select_subset
    with replay_selector(prepared):
        fit = prepared['data'].splits['FIT']
        assert common.select_subset(fit, 17, method='backward', seed=17) == list(range(17))
        with pytest.raises(ValueError, match='method, seed or budget'):
            common.select_subset(fit, 3, method='backward', seed=17)
        changed = fit.subset(list(range(20)))
        changed.blocks[0][0, 0] = [1, 0, 0]
        with pytest.raises(ValueError, match='identical frozen FIT'):
            common.select_subset(changed, 17, method='backward', seed=17)
    assert common.select_subset is original


@pytest.mark.parametrize('field,value', [
    ('judge_ids', ['wrong'] * 20),
    ('contexts', ['wrong']),
    ('selection_method', 'pivoted_qr'),
    ('gammas', [.12, .3]),
])
def test_inconsistent_source_protocol_is_rejected(sources, field, value):
    input_path, runs = sources
    mutate_report(runs[1], lambda report: report.update({field: value}))
    with pytest.raises(ValueError):
        prepare_sources(input_path, runs)


@pytest.mark.parametrize('field', ['regularizations', 'temperatures'])
def test_hyperparameter_grid_must_be_identical(sources, field):
    input_path, runs = sources
    mutate_report(runs[1], lambda report: report['configuration'].update({field: [.5, 1.]}))
    with pytest.raises(ValueError, match='configuration'):
        prepare_sources(input_path, runs)


def test_checkpoint_or_unbound_report_is_not_a_complete_source(sources):
    input_path, runs = sources
    status_path = runs[0] / 'status.json'
    status = json.loads(status_path.read_text())
    status['stages'][STAGE]['status'] = 'running'
    status_path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match='not complete'):
        prepare_sources(input_path, runs)
    status['stages'][STAGE]['status'] = 'complete'
    status['stages'][STAGE]['artifact_sha256'] = '0' * 64
    status_path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match='SHA256'):
        prepare_sources(input_path, runs)


def test_non_gate_change_cannot_be_reported_as_success(sources, tmp_path, monkeypatch):
    input_path, runs = sources
    original = reconstruction.run
    def changed(args):
        result = original(args)
        result['budgets'][0]['methods']['temperature_convex']['test']['worst_endpoint_context_tv'] += .1
        return result
    monkeypatch.setattr(reconstruction, 'run', changed)
    with pytest.raises(ValueError, match='non-gate numeric result changed'):
        audit(input_path, runs, tmp_path / 'changed_audit')
