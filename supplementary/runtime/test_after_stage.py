import json
from pathlib import Path
import sys

import pytest

from supplementary.runtime.after_stage import execute, inspect_stage, parse_args


def test_completed_score_is_ready_even_if_later_analysis_failed():
    condition, row = inspect_stage({'status': 'failed', 'current_stage': 'analysis',
                                   'stages': {'score': {'status': 'complete'}}}, 'score')
    assert condition == 'ready'


@pytest.mark.parametrize('status', ['failed', 'interrupted', 'failed_missing_artifact'])
def test_failed_score_stops_queue(status):
    assert inspect_stage({'stages': {'score': {'status': status}}}, 'score')[0] == 'failed'


def test_running_or_absent_score_waits():
    assert inspect_stage({'status': 'running', 'stages': {'score': {'status': 'running'}}}, 'score')[0] == 'waiting'
    assert inspect_stage({'status': 'starting'}, 'score')[0] == 'waiting'


def test_success_runs_child_and_records_result_without_sleep(tmp_path):
    upstream = tmp_path / 'upstream.json'
    upstream.write_text(json.dumps({'stages': {'score': {'status': 'complete'}}}))
    output, child_result = tmp_path / 'queue.json', tmp_path / 'result.txt'
    args = parse_args(['--wait-status', str(upstream), '--output', str(output), '--', sys.executable, '-c',
                       f'from pathlib import Path; Path({str(child_result)!r}).write_text("done")'])
    assert execute(args) == 0
    assert child_result.read_text() == 'done'
    assert json.loads(output.read_text())['status'] == 'complete'


def test_failure_never_launches_child(tmp_path):
    upstream = tmp_path / 'upstream.json'
    upstream.write_text(json.dumps({'stages': {'score': {'status': 'failed', 'returncode': 2}}}))
    output, child_result = tmp_path / 'queue.json', tmp_path / 'must_not_exist.txt'
    args = parse_args(['--wait-status', str(upstream), '--output', str(output), '--', sys.executable, '-c',
                       f'from pathlib import Path; Path({str(child_result)!r}).touch()'])
    assert execute(args) == 3
    assert not child_result.exists()
    assert json.loads(output.read_text())['status'] == 'upstream_failed'


def test_polling_cannot_busy_loop(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(['--wait-status', str(tmp_path / 'status.json'), '--poll-seconds', '1', '--', sys.executable, '-c', 'pass'])
