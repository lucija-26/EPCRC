"""Wait for pinned Llama access, finish the full core20 cache, then analyse it."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import ROOT, atomic_json, contexts_for_manifest, digest, lock_file, now, preflight_gpu

REAL = ROOT / 'results/supplementary/real'
GATED = ['J07', 'J08']


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def safe_request(url, method, token=None, redirect=False):
    """Return code/body/Location; never include credentials or signed URLs in errors/logs."""
    headers = {'User-Agent': 'EPCRC-core20-completion/1'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    opener = urllib.request.build_opener() if redirect else urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(urllib.request.Request(url, headers=headers, method=method), timeout=45) as response:
            return response.status, response.read() if method == 'GET' else b'', response.headers.get('Location')
    except urllib.error.HTTPError as error:
        return error.code, b'', error.headers.get('Location')
    except Exception as error:
        raise RuntimeError(f'Hugging Face request failed ({type(error).__name__}); no credential details recorded') from None


def probe_weight(endpoint, token):
    repo, revision = endpoint['model_id'], endpoint['revision']
    metadata_url = f'https://huggingface.co/api/models/{repo}/revision/{revision}'
    code, payload, _ = safe_request(metadata_url, 'GET', token)
    if code in {401, 403, 404}:
        return {'ready': False, 'http_status': code, 'check': 'pinned_model_metadata'}
    if code != 200:
        raise RuntimeError(f'Pinned model metadata probe returned HTTP {code}')
    try:
        files = [entry['rfilename'] for entry in json.loads(payload)['siblings']]
    except (KeyError, TypeError, ValueError):
        raise RuntimeError('Pinned model metadata has no valid file inventory') from None
    weights = sorted(f for f in files if f.endswith('.safetensors')) or sorted(f for f in files if f.endswith('.bin'))
    if not weights:
        raise RuntimeError('Pinned model has no actual weight file to test; refusing metadata-only access inference')
    filename = weights[0]
    url = f'https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(filename, safe="/")}'
    code, _, location = safe_request(url, 'HEAD', token)
    if code in {301, 302, 303, 307, 308}:
        if not location:
            raise RuntimeError('Weight HEAD redirect has no target')
        target = urllib.parse.urljoin(url, location)
        parsed = urllib.parse.urlparse(target)
        if parsed.scheme != 'https' or '/login' in parsed.path:
            return {'ready': False, 'http_status': 403, 'check': 'weight_head_login_redirect'}
        # CDN redirects must not receive the Hugging Face bearer token.
        code, _, _ = safe_request(target, 'HEAD', None, redirect=True)
    if code in {401, 403, 404}:
        return {'ready': False, 'http_status': code, 'check': 'actual_pinned_weight_head', 'filename': filename}
    if code not in {200, 206}:
        raise RuntimeError(f'Actual pinned weight HEAD returned HTTP {code}')
    return {'ready': True, 'http_status': code, 'check': 'actual_pinned_weight_head', 'filename': filename}


def inspect_blocks(manifest, contexts, cache, pair_ids, manifest_hash, pairs_hash):
    missing = []
    for endpoint in manifest['endpoints']:
        for context in contexts:
            path = cache / f"{endpoint['judge_id']}__{context}.json"
            if not path.exists():
                missing.append((endpoint['judge_id'], context))
                continue
            block = json.loads(path.read_text())
            protocol = block.get('protocol', {})
            if (block.get('judge_id') != endpoint['judge_id'] or block.get('model_id') != endpoint['model_id']
                    or block.get('revision') != endpoint['revision'] or block.get('context') != context
                    or block.get('synthetic') is not False or block.get('pair_ids') != pair_ids
                    or protocol.get('manifest_sha256') != manifest_hash or protocol.get('pairs_sha256') != pairs_hash
                    or protocol.get('endpoint') != endpoint):
                raise RuntimeError(f'Existing cache metadata/inputs mismatch: {path}')
            values = block.get('probabilities', [])
            if (len(values) != len(pair_ids) or any(len(row) != 3 or not all(math.isfinite(x) and x >= 0 for x in row)
                                                 or abs(sum(row) - 1) > 1e-6 for row in values)):
                raise RuntimeError(f'Invalid existing probabilities: {path}')
            import hashlib
            expected = hashlib.sha256(json.dumps(dict(protocol, context=context, pair_ids=pair_ids),
                                                  sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if block.get('fingerprint') != expected:
                raise RuntimeError(f'Existing cache fingerprint mismatch: {path}')
    return missing


def check_shards(paths, missing):
    for path in paths:
        if not path.exists():
            continue
        state = json.loads(path.read_text())
        score = state.get('stages', {}).get('score', {})
        if state.get('status') in {'failed', 'interrupted'} and score.get('status') != 'complete':
            raise RuntimeError(f'Existing core20 shard stopped: {path}; error={state.get("error")}; log={score.get("log")}')
        if score.get('status') == 'complete':
            command = score.get('command', [])
            judges = command[command.index('--judges') + 1:] if '--judges' in command else None
            absent = [(j, c) for j, c in missing if judges is None or j in judges]
            if absent:
                raise RuntimeError(f'Shard reports completion but {len(absent)} assigned blocks are missing: {path}')


def available_gpu():
    for gpu in (0, 1, 2):
        try:
            import platform
            with lock_file(ROOT / f'supplementary/runtime/locks/{platform.node()}.gpu{gpu}.lock'):
                preflight_gpu(gpu, 40)
            return gpu
        except RuntimeError:
            continue
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT / 'supplementary/exp00_response_cache/core20.locked.json')
    p.add_argument('--pairs', type=Path, default=ROOT / 'results/supplementary/data/rewardbench.jsonl')
    p.add_argument('--cache', type=Path, default=ROOT / 'results/supplementary/cache/core20')
    p.add_argument('--split-manifest', type=Path, default=ROOT / 'data/split_seed20260818.json')
    p.add_argument('--output', type=Path, default=REAL / 'core20_coordinator')
    p.add_argument('--llama-output', type=Path, default=REAL / 'core20_llama')
    p.add_argument('--analysis-output', type=Path, default=REAL / 'core20')
    p.add_argument('--shard-status', nargs='+', type=Path, default=[REAL / f'core20_shard{i}/status.json' for i in (0, 1)])
    p.add_argument('--gpu-python', type=Path, default=ROOT / '.venv-gpu/bin/python')
    p.add_argument('--cpu-python', type=Path, default=ROOT / '.venv/bin/python')
    p.add_argument('--timeout-hours', type=float, default=24)
    p.add_argument('--poll-seconds', type=float, default=300)
    args = p.parse_args(argv)
    if not math.isfinite(args.timeout_hours) or args.timeout_hours <= 0 or not math.isfinite(args.poll_seconds) or args.poll_seconds < 10:
        p.error('Use positive finite timeout and poll-seconds >= 10')
    for key in ['manifest', 'pairs', 'cache', 'split_manifest', 'output', 'llama_output', 'analysis_output', 'gpu_python', 'cpu_python']:
        setattr(args, key, getattr(args, key).absolute())
    return args


def run(args):
    manifest = json.loads(args.manifest.read_text())
    endpoints = {e['judge_id']: e for e in manifest['endpoints']}
    if len(endpoints) != 20 or not set(GATED) <= set(endpoints):
        raise RuntimeError('Coordinator requires the complete original 20-endpoint manifest')
    contexts = contexts_for_manifest(manifest)
    with args.pairs.open() as handle:
        pair_ids = [json.loads(line)['pair_id'] for line in handle if line.strip()]
    protocol = {'manifest_sha256': digest(args.manifest), 'pairs_sha256': digest(args.pairs),
                'split_manifest_sha256': digest(args.split_manifest), 'batch_size': 16, 'repeats': 1,
                'max_tokens': 8192, 'dtype': 'bfloat16', 'allow_truncation': False}
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('Coordinator inputs/protocol changed; use a new output directory')
    atomic_json(protocol_path, protocol)
    deadline = time.monotonic() + args.timeout_hours * 3600
    status = {'status': 'starting', 'pid': os.getpid(), 'started_at': now(), 'expected_blocks': len(endpoints) * len(contexts),
              'deadline_utc': (datetime.now(timezone.utc) + timedelta(hours=args.timeout_hours)).isoformat(),
              'remaining_access_endpoint_ids': GATED, 'authenticated_requests': bool(os.environ.get('HF_TOKEN'))}
    child = None

    def save():
        status['updated_at'] = now()
        atomic_json(args.output / 'status.json', status)

    def child_run(name, command):
        nonlocal child
        log_path = args.output / f'{name}.{int(time.time())}.log'
        with log_path.open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status.update(status=name, child_pid=child.pid, child_log=str(log_path), command=command)
            save()
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Coordinator deadline reached while owned child was running')
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
            code = child.returncode
            status.update(child_returncode=code, child_pid=None)
            child = None
            save()
            return code

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    base = [str(args.cpu_python), '-u', str(ROOT / 'supplementary/runtime/pipeline.py'), '--manifest', str(args.manifest),
            '--pairs', str(args.pairs), '--cache', str(args.cache), '--python', str(args.gpu_python), '--cpu-python', str(args.cpu_python),
            '--batch-size', '16', '--repeats', '1', '--max-tokens', '8192', '--dtype', 'bfloat16']
    try:
        save()
        while time.monotonic() < deadline:
            missing = inspect_blocks(manifest, contexts, args.cache, pair_ids, protocol['manifest_sha256'], protocol['pairs_sha256'])
            status.update(completed_blocks=status['expected_blocks'] - len(missing), missing_blocks=missing)
            if not missing:
                command = base + ['--analyses-only', '--split-manifest', str(args.split_manifest), '--output', str(args.analysis_output),
                                  '--relationship', 'original_panel', '--benchmark', 'RewardBench2-pinned', '--bootstrap', '2000',
                                  '--budgets', '2', '4', '10', '16', '18', '19', '20']
                code = child_run('running_full_core20_analysis', command)
                if code:
                    raise RuntimeError(f'Full core20 merge/analysis failed (exit {code}); see {status["child_log"]}')
                status.update(status='complete', completed_at=now())
                save()
                return 0
            check_shards(args.shard_status, missing)
            missing_llama = sorted({j for j, _ in missing if j in GATED})
            if missing_llama:
                checks = {j: probe_weight(endpoints[j], os.environ.get('HF_TOKEN')) for j in missing_llama}
                status['access_checks'] = checks
                status['remaining_access_endpoint_ids'] = [j for j, row in checks.items() if not row['ready']]
                if all(row['ready'] for row in checks.values()):
                    gpu = available_gpu()
                    if gpu is not None:
                        command = base + ['--score-only', '--judges', *GATED, '--gpu', str(gpu), '--min-free-gib', '40',
                                          '--output', str(args.llama_output)]
                        code = child_run('scoring_authorized_llama', command)
                        if code:
                            child_status = args.llama_output / 'status.json'
                            detail = json.loads(child_status.read_text()).get('error', '') if child_status.exists() else ''
                            if not any(text in detail for text in ['Already locked', 'already has compute processes', 'less than required']):
                                raise RuntimeError(f'Llama scoring failed (exit {code}); see {status["child_log"]}')
                        else:
                            continue
                    status['status'] = 'pending_idle_gpu'
                else:
                    status['status'] = 'pending_model_authorization'
            else:
                status['status'] = 'pending_remaining_core20_blocks'
            save()
            print(f'[{now()}] {status["status"]}; {status["completed_blocks"]}/{status["expected_blocks"]} blocks', flush=True)
            time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
        raise TimeoutError('24-hour completion window expired; authorization or required blocks remain pending')
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        status.update(status='pending_timeout' if isinstance(error, TimeoutError) else 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=str(error), finished_at=now(), child_pid=None)
        save()
        print(status['error'], file=sys.stderr, flush=True)
        return 124 if isinstance(error, TimeoutError) else 130 if isinstance(error, KeyboardInterrupt) else 1


def main(argv=None):
    args = parse_args(argv)
    try:
        with lock_file(args.output / '.coordinator.lock'):
            return run(args)
    except Exception as error:
        atomic_json(args.output / f'startup_failure.{os.getpid()}.json', {'error': str(error), 'utc': now()})
        print(error, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
