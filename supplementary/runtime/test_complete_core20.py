import hashlib
import json
from pathlib import Path

import pytest

from supplementary.runtime import complete_core20 as module


def test_weight_head_checks_actual_file_and_does_not_forward_token(monkeypatch):
    calls = []
    def request(url, method, token=None, redirect=False):
        calls.append((url, method, token, redirect))
        if method == 'GET':
            return 200, json.dumps({'siblings': [{'rfilename': 'model-00001-of-00002.safetensors'}]}).encode(), None
        if 'huggingface.co/' in url:
            return 302, b'', 'https://cdn.example/weight?signature=redacted'
        return 200, b'', None
    monkeypatch.setattr(module, 'safe_request', request)
    result = module.probe_weight({'model_id': 'fixture/model', 'revision': 'a' * 40}, 'fixture-secret')
    assert result['ready']
    assert calls[1][1] == 'HEAD'
    assert calls[-1][2] is None
    assert 'fixture-secret' not in json.dumps(result)


def test_denied_weight_is_pending_not_success(monkeypatch):
    monkeypatch.setattr(module, 'safe_request', lambda *args, **kwargs: (403, b'', None))
    result = module.probe_weight({'model_id': 'fixture/model', 'revision': 'a' * 40}, 'fixture-secret')
    assert not result['ready'] and result['http_status'] == 403


def test_existing_shard_failure_is_not_silently_waited_out(tmp_path):
    path = tmp_path / 'status.json'
    path.write_text(json.dumps({'status': 'failed', 'error': 'fixture failure', 'stages': {'score': {'status': 'failed'}}}))
    with pytest.raises(RuntimeError, match='shard stopped'):
        module.check_shards([path], [('J01', 'clean')])


def test_completed_shard_missing_block_is_an_error(tmp_path):
    path = tmp_path / 'status.json'
    path.write_text(json.dumps({'status': 'complete_score_only', 'stages': {
        'score': {'status': 'complete', 'command': ['python', 'score', '--judges', 'J01']}}}))
    with pytest.raises(RuntimeError, match='assigned blocks are missing'):
        module.check_shards([path], [('J01', 'clean')])


def test_block_inspection_keeps_full_missing_denominator(tmp_path):
    endpoint = {'judge_id': 'J07', 'model_id': 'fixture/model', 'revision': 'a' * 40}
    protocol = {'manifest_sha256': 'm', 'pairs_sha256': 'p', 'endpoint': endpoint}
    fingerprint = hashlib.sha256(json.dumps(dict(protocol, context='clean', pair_ids=['p0']), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    block = dict(judge_id='J07', model_id='fixture/model', revision='a' * 40, context='clean',
                 synthetic=False, pair_ids=['p0'], probabilities=[[.2, .3, .5]], protocol=protocol, fingerprint=fingerprint)
    (tmp_path / 'J07__clean.json').write_text(json.dumps(block))
    assert module.inspect_blocks({'endpoints': [endpoint]}, ['clean', 'stress'], tmp_path, ['p0'], 'm', 'p') == [('J07', 'stress')]
    block['probabilities'] = [[float('nan'), 0, 1]]
    (tmp_path / 'J07__clean.json').write_text(json.dumps(block))
    with pytest.raises(RuntimeError, match='Invalid existing probabilities'):
        module.inspect_blocks({'endpoints': [endpoint]}, ['clean'], tmp_path, ['p0'], 'm', 'p')
