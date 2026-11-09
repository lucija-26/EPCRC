import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from supplementary.runtime.pipeline import Runner, contexts_for_manifest, make_plan, parse_args, preflight_gpu
from supplementary.runtime.progress import inspect


def fixture_args(tmp_path, n=4, extra=()):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'name': 'fixture', 'interface': 'nli', 'contexts': ['clean'],
                                   'endpoints': [{'judge_id': f'N{i}', 'model_id': f'fixture/{i}', 'revision': 'a' * 40}
                                                 for i in range(n)]}))
    pairs = tmp_path / 'pairs.jsonl'
    pairs.write_text(json.dumps({'pair_id': 'p', 'base_item_id': 'g', 'text': 'fixture', 'text_pair': 'fixture', 'gold': 0}) + '\n')
    return parse_args(['--manifest', str(manifest), '--pairs', str(pairs), '--cache', str(tmp_path / 'cache'),
                       '--output', str(tmp_path / 'run'), '--gpu', '2', '--score-only', *extra])


def test_plan_shards_keep_complete_manifest_and_explicit_truncation(tmp_path):
    args = fixture_args(tmp_path, extra=['--judges', 'N0', 'N2', '--allow-truncation'])
    plan = make_plan(args)
    assert plan['score'][-3:] == ['--judges', 'N0', 'N2']
    assert '--allow-truncation' in plan['score']
    assert len(plan['manifest']['endpoints']) == 4
    assert plan['budgets'] == [1, 2, 3, 4]
    assert '--judges' not in plan['merge']


def test_interpreter_paths_preserve_virtualenv_symlinks(tmp_path):
    gpu_python = tmp_path / 'gpu-venv' / 'bin' / 'python'
    cpu_python = tmp_path / 'cpu-venv' / 'bin' / 'python'
    for executable in [gpu_python, cpu_python]:
        executable.parent.mkdir(parents=True)
        executable.symlink_to(Path(sys.executable).resolve())
    args = fixture_args(tmp_path, extra=['--python', str(gpu_python), '--cpu-python', str(cpu_python)])
    plan = make_plan(args)
    assert args.python == gpu_python.absolute()
    assert args.cpu_python == cpu_python.absolute()
    assert plan['score'][0] == str(gpu_python.absolute())
    assert plan['analyses'][0]['command'][0] == str(cpu_python.absolute())


def test_sharded_analysis_is_rejected_before_launch(tmp_path):
    args = fixture_args(tmp_path)
    with pytest.raises(SystemExit):
        parse_args(['--manifest', str(args.manifest), '--pairs', str(args.pairs), '--cache', str(args.cache),
                    '--output', str(args.output), '--gpu', '2', '--judges', 'N0'])


def test_failed_child_is_logged_and_not_reported_complete(tmp_path):
    args = fixture_args(tmp_path)
    runner = Runner(args, make_plan(args))
    with pytest.raises(RuntimeError, match='exit 7'):
        runner.stage('fixture_failure', [sys.executable, '-c', 'print("expected fixture failure"); raise SystemExit(7)'])
    state = json.loads((args.output / 'status.json').read_text())
    assert state['stages']['fixture_failure']['status'] == 'failed'
    assert state['active_child'] is None
    assert 'expected fixture failure' in Path(state['stages']['fixture_failure']['log']).read_text()


def test_restart_reuses_only_unchanged_artifact_and_code(tmp_path):
    args = fixture_args(tmp_path)
    runner = Runner(args, make_plan(args))
    result = tmp_path / 'result.json'
    cmd = [sys.executable, '-c', f'from pathlib import Path; Path({str(result)!r}).write_text("{{}}")']
    runner.stage('fixture_success', cmd, artifact=result, skip_valid=True, code_signature='v1')
    runner.stage('fixture_success', cmd, artifact=result, skip_valid=True, code_signature='v1')
    assert runner.state['stages']['fixture_success']['attempt'] == 1
    runner.stage('fixture_success', cmd, artifact=result, skip_valid=True, code_signature='v2')
    assert runner.state['stages']['fixture_success']['attempt'] == 2


def test_busy_gpu_is_rejected_without_killing_processes(monkeypatch):
    def fake_run(command, **kwargs):
        if '--query-gpu=index,uuid,name,memory.free,memory.total' in command:
            return SimpleNamespace(stdout='2, GPU-fixture, B200, 180000, 183359\n')
        return SimpleNamespace(stdout='GPU-fixture, 123, 1000\n')
    monkeypatch.setattr('supplementary.runtime.pipeline.subprocess.run', fake_run)
    with pytest.raises(RuntimeError, match='already has compute processes'):
        preflight_gpu(2, 24)


def test_progress_preserves_missing_models_in_denominator(tmp_path):
    args = fixture_args(tmp_path)
    args.cache.mkdir()
    (args.cache / 'N0__clean.json').write_text(json.dumps({
        'judge_id': 'N0', 'model_id': 'fixture/0', 'revision': 'a' * 40, 'context': 'clean',
        'synthetic': False, 'pair_ids': ['p'], 'probabilities': [[.2, .3, .5]]}))
    report = inspect(args.manifest, args.cache)
    assert report['completed_blocks'] == 1
    assert report['expected_blocks'] == 4
    assert report['expected_endpoints'] == 4


def test_judge_context_registry_is_read_without_ml_imports():
    names = contexts_for_manifest({'interface': 'judge'})
    assert len(names) == 7
    assert names[0] == 'I0_clean'
