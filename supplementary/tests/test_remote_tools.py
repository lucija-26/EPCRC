"""Remote helper portability without opening SSH connections or querying GPUs."""
import json
import subprocess
import sys
from unittest.mock import Mock

import pytest

from supplementary.runtime import launch_remote
from supplementary.tools import check_gpus


def test_launcher_requires_explicit_destination(monkeypatch, tmp_path):
    run = Mock()
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['launch_remote', '--job-dir', str(tmp_path), '--', 'true'])
    with pytest.raises(SystemExit) as error:
        launch_remote.main()
    assert error.value.code == 2
    run.assert_not_called()


@pytest.mark.parametrize('custom', [False, True])
def test_launcher_uses_selected_host_and_optional_paths(monkeypatch, tmp_path, custom):
    run = Mock(return_value=subprocess.CompletedProcess([], 0, '', ''))
    monkeypatch.setattr(subprocess, 'run', run)
    argv = ['launch_remote', '--host', 'gpu-alias', '--without-hf-auth', '--job-dir', str(tmp_path)]
    if custom:
        argv += ['--port', '2222', '--identity', '/tmp/test-key', '--hf-hub-cache', '/remote/cache']
    monkeypatch.setattr(sys, 'argv', argv + ['--', 'python', 'experiment.py'])
    with pytest.raises(SystemExit) as error:
        launch_remote.main()
    assert error.value.code == 0
    command = run.call_args.args[0]
    payload = json.loads(run.call_args.kwargs['input'])
    assert command[-2] == 'gpu-alias'
    assert payload['argv'] == ['python', 'experiment.py']
    assert payload['cwd'] == str(launch_remote.ROOT)
    assert 'HF_TOKEN' not in payload['env']
    if custom:
        assert command[command.index('-p') + 1] == '2222'
        assert command[command.index('-i') + 1] == '/tmp/test-key'
        assert payload['env']['HF_HUB_CACHE'] == '/remote/cache'
    else:
        assert '-i' not in command and '-p' not in command
        assert 'HF_HUB_CACHE' not in payload['env']


def test_gpu_inventory_defaults_to_local_only(monkeypatch, tmp_path):
    run = Mock(return_value=subprocess.CompletedProcess([], 0, 'local-node\n', ''))
    monkeypatch.setattr(subprocess, 'run', run)
    output = tmp_path / 'inventory.json'
    monkeypatch.setattr(sys, 'argv', ['check_gpus', '--output', str(output)])
    check_gpus.main()
    assert run.call_count == 1
    assert run.call_args.args[0][0] == 'bash'
    assert json.loads(output.read_text())['nodes'][0]['node'] == 'current'


def test_gpu_inventory_requires_host_for_ports(monkeypatch):
    run = Mock()
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['check_gpus', '--ports', '2222'])
    with pytest.raises(SystemExit) as error:
        check_gpus.main()
    assert error.value.code == 2
    run.assert_not_called()


def test_gpu_inventory_uses_explicit_destination(monkeypatch):
    run = Mock(return_value=subprocess.CompletedProcess([], 0, 'remote-node\n', ''))
    monkeypatch.setattr(subprocess, 'run', run)
    result = check_gpus.probe(2222, None, 'user@gpu-host')
    command = run.call_args.args[0]
    assert command[-2] == 'user@gpu-host'
    assert command[command.index('-p') + 1] == '2222'
    assert result['reachable'] is True
