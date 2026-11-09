"""Durable single-host score/merge/analyse runner. This script installs nothing."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = ROOT / 'supplementary/exp00_response_cache/run.py'
VERIFIER = ROOT / 'supplementary/runtime/verify_merged.py'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def process_start(pid):
    try:
        # Fields after '(comm)' begin at stat field 3; starttime is field 22.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, PermissionError, IndexError):
        return None


def contexts_for_manifest(manifest):
    """Read the literal registered context names without importing the ML stack."""
    if 'contexts' in manifest:
        return manifest['contexts']
    if manifest['interface'] == 'nli':
        return ['clean']
    tree = ast.parse((ROOT / 'epcrc/prompts.py').read_text())
    names, registry = {}, None
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id == 'Context':
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names[target.id] = ast.literal_eval(node.value.args[0])
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == 'REGISTERED_CONTEXTS':
            registry = node.value
    if not isinstance(registry, (ast.Tuple, ast.List)):
        raise ValueError('Cannot read registered contexts statically; add explicit contexts to the locked manifest')
    return [names[element.id] for element in registry.elts]


@contextmanager
def lock_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Already locked by another runner: {path}') from None
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({'pid': os.getpid(), 'host': platform.node(), 'utc': now()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def preflight_gpu(index, min_free_gib):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free,memory.total',
                             '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True)
    found = None
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(',')]
        if int(values[0]) == index:
            found = dict(index=index, uuid=values[1], name=values[2], free_mib=float(values[3]), total_mib=float(values[4]))
    if found is None:
        raise RuntimeError(f'Physical GPU {index} not found on {platform.node()}')
    if found['free_mib'] < min_free_gib * 1024:
        raise RuntimeError(f'GPU {index} has {found["free_mib"]:.0f} MiB free, less than required {min_free_gib:g} GiB')
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'],
                            text=True, capture_output=True, check=True)
    occupants = [line.strip() for line in active.stdout.splitlines() if line.split(',')[0].strip() == found['uuid']]
    if occupants:
        raise RuntimeError(f'GPU {index} already has compute processes; no existing process will be stopped: {occupants}')
    return found


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['manifest', 'pairs', 'cache', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--split-manifest', type=Path)
    p.add_argument('--gpu', type=int, help='Physical nvidia-smi GPU index on this host; score sees it as cuda:0')
    p.add_argument('--python', type=Path, default=ROOT / '.venv-gpu/bin/python', help='Existing GPU Python; never installed by this runner')
    p.add_argument('--cpu-python', type=Path, default=ROOT / '.venv/bin/python')
    p.add_argument('--judges', nargs='+', help='Disjoint scoring shard; requires --score-only')
    stage = p.add_mutually_exclusive_group()
    stage.add_argument('--score-only', action='store_true')
    stage.add_argument('--analyses-only', action='store_true', help='Require all raw blocks; merge/verify and analyse without taking a GPU')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default='bfloat16')
    p.add_argument('--allow-truncation', action='store_true')
    p.add_argument('--limit', type=int, default=0, help='Explicit debug item prefix; 0 uses all supplied rows')
    p.add_argument('--min-free-gib', type=float, default=24)
    p.add_argument('--budgets', nargs='+', type=int)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20260818)
    p.add_argument('--selection-methods', nargs='+', choices=['backward', 'forward', 'qr', 'kmedoids', 'random', 'swap1', 'swap2'])
    p.add_argument('--exact-budget', type=int, default=10000)
    p.add_argument('--experiments', nargs='+', type=int, choices=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    p.add_argument('--relationship', choices=['original_panel', 'independent_ecosystem', 'same_panel_benchmark_shift', 'subpanel'])
    p.add_argument('--benchmark', help='Explicit dataset name/version for the replication provenance')
    p.add_argument('--assembly-protocol', help='Defaults to the locked manifest note; provenance only, not proof of independence')
    p.add_argument('--detach', action='store_true', help='Start a new session and return its PID; logs go to output/runner.log')
    p.add_argument('--dry-run', action='store_true', help='Print commands and immutable protocol without GPU checks or writes')
    args = p.parse_args(argv)
    for name in ['manifest', 'pairs', 'cache', 'output', 'split_manifest', 'python', 'cpu_python']:
        value = getattr(args, name)
        if value is not None:
            # Resolving a venv's Python symlink selects the system interpreter
            # and discards its environment. Preserve that invocation path.
            setattr(args, name, value.absolute() if name in {'python', 'cpu_python'} else value.resolve())
    if args.gpu is None and not args.analyses_only:
        p.error('--gpu is required unless --analyses-only')
    if args.judges and not args.score_only:
        p.error('--judges may shard scoring only with --score-only; analysis requires the complete manifest')
    if (args.batch_size < 1 or args.repeats < 1 or args.max_tokens < 1 or args.limit < 0
            or args.bootstrap < 2 or args.min_free_gib <= 0 or args.exact_budget < 0):
        p.error('Invalid numeric budget or protocol setting')
    if not args.score_only and 3 in args.experiments and (not args.relationship or not args.benchmark):
        p.error('exp03 requires explicit --relationship and --benchmark provenance')
    return args


def make_plan(args):
    manifest = json.loads(args.manifest.read_text())
    endpoints = manifest['endpoints']
    endpoint_ids = [e['judge_id'] for e in endpoints]
    if len(set(endpoint_ids)) != len(endpoint_ids) or len(endpoint_ids) < 2:
        raise ValueError('Manifest requires unique endpoints and N >= 2')
    for endpoint in endpoints:
        rev = endpoint.get('revision', '')
        if len(rev) != 40 or any(c not in '0123456789abcdef' for c in rev):
            raise ValueError('A fully locked immutable manifest is required; runner will not resolve or replace revisions')
    if args.judges and (not set(args.judges) <= set(endpoint_ids) or len(set(args.judges)) != len(args.judges)):
        raise ValueError('Unknown or duplicate endpoint in scoring shard')
    n = len(endpoints)
    budgets = args.budgets or (list(range(1, n + 1)) if n <= 8 else sorted({2, 4, n // 2, max(1, n - 4), n - 2, n - 1, n}))
    if any(k < 1 or k > n for k in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError('Budgets must be distinct and lie between 1 and N')
    methods = args.selection_methods or (['backward', 'forward', 'qr', 'kmedoids', 'random', 'swap1', 'swap2']
                                           if n <= 8 else ['backward', 'forward', 'qr', 'kmedoids', 'random'])
    if 'backward' not in methods or len(set(methods)) != len(methods):
        raise ValueError('Selection methods must be distinct and include backward')
    if len(set(args.experiments)) != len(args.experiments):
        raise ValueError('Duplicate analysis experiment')
    merged = args.output / 'responses.npz'
    score = [str(args.python), '-u', str(COLLECTOR), 'score', '--manifest', str(args.manifest), '--pairs', str(args.pairs),
             '--output', str(args.cache), '--device', 'cuda:0', '--batch-size', str(args.batch_size), '--repeats', str(args.repeats),
             '--max-tokens', str(args.max_tokens), '--dtype', args.dtype, '--limit', str(args.limit)]
    if args.allow_truncation:
        score.append('--allow-truncation')
    if args.judges:
        score += ['--judges', *args.judges]
    merge = [str(args.python), '-u', str(COLLECTOR), 'merge', '--manifest', str(args.manifest), '--pairs', str(args.pairs),
             '--cache', str(args.cache), '--output', str(merged), '--limit', str(args.limit)]
    verify = [str(args.cpu_python), '-u', str(VERIFIER), '--manifest', str(args.manifest), '--pairs', str(args.pairs),
              '--cache', str(args.cache), '--input', str(merged), '--limit', str(args.limit)]
    if args.split_manifest:
        merge += ['--split-manifest', str(args.split_manifest)]
        verify += ['--split-manifest', str(args.split_manifest)]
    names = {1: 'exp01_dependency_significance', 2: 'exp02_reconstruction', 3: 'exp03_ecosystem_replication',
             4: 'exp04_selection_benchmarks', 5: 'exp05_measured_cost'}
    analysis = []
    for number in args.experiments:
        name = names[number]
        script = ROOT / 'supplementary' / name / 'run.py'
        directory = args.output / name
        result = directory / 'results.json'
        cmd = [str(args.cpu_python), '-u', str(script)]
        if number == 3:
            cmd += ['--manifest', str(args.output / 'replication_manifest.json')]
        else:
            cmd += ['--input', str(merged)]
        cmd += ['--seed', str(args.seed), '--output', str(directory if number == 2 else result)]
        if number in [1, 2, 3, 4]:
            cmd += ['--bootstrap', str(args.bootstrap)]
        if number in [2, 3, 4, 5]:
            cmd += [{2: '--budgets', 3: '--k', 4: '--k', 5: '--ks'}[number], *map(str, budgets)]
        if number == 4:
            cmd += ['--methods', *methods, '--exact-budget', str(args.exact_budget)]
        if number == 5:
            cmd += ['--timing-cache', str(args.cache)]
        analysis.append({'stage': name, 'command': cmd, 'artifact': str(result), 'script_sha256': digest(script)})
    protocol = {'manifest': str(args.manifest), 'manifest_sha256': digest(args.manifest), 'pairs': str(args.pairs),
                'pairs_sha256': digest(args.pairs), 'cache': str(args.cache), 'limit': args.limit,
                'split_manifest_sha256': digest(args.split_manifest) if args.split_manifest else None,
                'batch_size': args.batch_size, 'repeats': args.repeats, 'max_tokens': args.max_tokens,
                'dtype': args.dtype, 'allow_truncation': args.allow_truncation,
                'collector_sha256': digest(COLLECTOR), 'scoring_sha256': digest(ROOT / 'epcrc/scoring.py'),
                'prompts_sha256': digest(ROOT / 'epcrc/prompts.py')}
    replication = {'ecosystems': [{'name': manifest['name'], 'input': str(merged), 'relationship': args.relationship,
                   'benchmark': args.benchmark, 'assembly_protocol': args.assembly_protocol or manifest.get('note', 'See locked manifest.'),
                   'model_ids': [e['model_id'] for e in endpoints]}]}
    return {'protocol': protocol, 'manifest': manifest, 'score': score, 'merge': merge, 'verify': verify,
            'analyses': analysis, 'replication': replication, 'budgets': budgets, 'selection_methods': methods,
            'merged': str(merged)}


class Runner:
    def __init__(self, args, plan):
        self.args, self.plan = args, plan
        self.path = args.output / 'status.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {'stages': {}, 'created_at': now()}
        self.env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                        NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', PYTHONUNBUFFERED='1')
        previous = self.state.get('active_child')
        if (previous and previous.get('host') == platform.node() and previous.get('start') is not None
                and process_start(previous['pid']) == previous.get('start')):
            raise RuntimeError(f'Previous child PID {previous["pid"]} is still alive; refusing duplicate execution')
        if previous and previous.get('host') != platform.node():
            raise RuntimeError('Unresolved previous child on another host; resume on its recorded host to check it safely')
        self.state.update(status='starting', runner_pid=os.getpid(), runner_process_start=process_start(os.getpid()),
                          host=platform.node(), attempt_started_at=now(), active_child=None, error=None)
        self.save()

    def save(self):
        self.state['updated_at'] = now()
        atomic_json(self.path, self.state)

    def stage(self, name, command, artifact=None, skip_valid=False, env=None, code_signature=None):
        previous = self.state['stages'].get(name)
        if (skip_valid and previous and previous.get('status') == 'complete'
                and previous.get('command') == command and artifact and Path(artifact).exists()
                and previous.get('code_signature') == code_signature
                and previous.get('artifact_sha256') == digest(artifact)):
            print(f'[{now()}] reuse validated stage {name}', flush=True)
            return
        attempt = int(previous.get('attempt', 0)) + 1 if previous else 1
        log = self.args.output / 'logs' / f'{name}.{attempt:03d}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        row = {'status': 'running', 'started_at': now(), 'attempt': attempt, 'command': command, 'log': str(log),
               'code_signature': code_signature}
        self.state['stages'][name] = row
        self.state.update(status='running', current_stage=name)
        self.save()
        print(f'[{now()}] {name}: {shlex.join(command)}', flush=True)
        with log.open('w') as handle:
            handle.write(shlex.join(command) + '\n')
            handle.flush()
            child = subprocess.Popen(command, cwd=ROOT, env=env or self.env, stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            self.state['active_child'] = {'pid': child.pid, 'start': process_start(child.pid), 'host': platform.node()}
            self.save()
            try:
                code = child.wait()
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                row.update(status='interrupted', finished_at=now(), returncode=child.returncode)
                self.state['active_child'] = None
                self.save()
                raise
        row.update(status='complete' if code == 0 else 'failed', finished_at=now(), returncode=code)
        self.state['active_child'] = None
        if code == 0 and artifact:
            if not Path(artifact).exists():
                row['status'] = 'failed_missing_artifact'
                self.save()
                raise RuntimeError(f'{name} exited successfully but required artifact is absent: {artifact}')
            row['artifact_sha256'] = digest(artifact)
        self.save()
        if code:
            raise RuntimeError(f'{name} failed with exit {code}; see {log}')

    def run(self):
        args, plan = self.args, self.plan
        if not args.analyses_only:
            gpu_lock = ROOT / 'supplementary/runtime/locks' / f'{platform.node()}.gpu{args.gpu}.lock'
            # These locks also prevent overlapping endpoint shards on different GPUs.
            with ExitStack() as locks:
                locks.enter_context(lock_file(gpu_lock))
                for judge in sorted(args.judges or [e['judge_id'] for e in plan['manifest']['endpoints']]):
                    locks.enter_context(lock_file(args.cache / '.runner_locks' / f'{judge}.lock'))
                self.state['gpu_preflight'] = preflight_gpu(args.gpu, args.min_free_gib)
                self.save()
                self.stage('score', plan['score'], env=dict(self.env, CUDA_VISIBLE_DEVICES=str(args.gpu)))
        if args.score_only:
            self.state.update(status='complete_score_only', current_stage=None)
            self.save()
            return
        manifest = plan['manifest']
        contexts = contexts_for_manifest(manifest)
        expected = [args.cache / f"{e['judge_id']}__{c}.json" for e in manifest['endpoints'] for c in contexts]
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            self.state['missing_blocks'] = missing
            self.save()
            raise RuntimeError(f'Complete-manifest gate failed: {len(missing)} raw blocks missing. No merge or analysis started.')
        if not Path(plan['merged']).exists():
            self.stage('merge', plan['merge'], artifact=plan['merged'])
        self.stage('verify_complete_merged_cache', plan['verify'])
        atomic_json(args.output / 'replication_manifest.json', plan['replication'])
        for analysis in plan['analyses']:
            code_signature = {'script_sha256': analysis['script_sha256'],
                              'common_sha256': digest(ROOT / 'supplementary/common.py'),
                              'merged_sha256': digest(plan['merged']),
                              'replication_manifest_sha256': digest(args.output / 'replication_manifest.json')}
            self.stage(analysis['stage'], analysis['command'], artifact=analysis['artifact'], skip_valid=True,
                       code_signature=code_signature)
        self.state.update(status='complete', current_stage=None, completed_at=now())
        self.save()


def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, '-u', str(Path(__file__).resolve()), *[x for x in (argv or sys.argv[1:]) if x != '--detach']]
        with (args.output / 'runner.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        atomic_json(args.output / 'launch.json', {'pid': child.pid, 'host': platform.node(), 'utc': now(), 'command': command,
                                                'process_start': process_start(child.pid), 'log': str(args.output / 'runner.log')})
        print(json.dumps({'launched_pid': child.pid, 'status': str(args.output / 'status.json')}))
        return 0
    runner = None
    try:
        with lock_file(args.output / '.runner.lock'):
            protocol_file = args.output / 'protocol.json'
            if protocol_file.exists() and json.loads(protocol_file.read_text()) != plan['protocol']:
                raise RuntimeError('Output protocol differs from this invocation. Use a new output/cache for any inference protocol change.')
            atomic_json(protocol_file, plan['protocol'])
            atomic_json(args.output / 'plan.json', plan)
            runner = Runner(args, plan)
            def interrupted(signum, frame):
                raise KeyboardInterrupt(f'Received signal {signum}')
            signal.signal(signal.SIGTERM, interrupted)
            runner.run()
        return 0
    except BaseException as error:
        if runner is not None:
            runner.state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
            runner.save()
        else:
            atomic_json(args.output / f'launch_failure.{os.getpid()}.json', {'utc': now(), 'error': str(error), 'host': platform.node()})
        print(f'Pipeline stopped: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
