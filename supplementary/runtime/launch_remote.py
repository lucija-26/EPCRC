"""Launch a durable shared-workspace job; credentials travel only over SSH stdin.

The token is read from the user's standard Hugging Face configuration, passed to
the remote process environment, and never stored in the workspace or command line.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = '''
import json, os, pathlib, subprocess, sys
p = json.load(sys.stdin)
cwd = pathlib.Path(p['cwd'])
job = pathlib.Path(p['job_dir'])
job.mkdir(parents=True, exist_ok=True)
record = job / 'launch.json'
if record.exists():
    prior = json.loads(record.read_text())
    try:
        os.kill(prior['pid'], 0)
        raise SystemExit('Previous job PID still exists; inspect it before relaunching')
    except ProcessLookupError: pass
env = dict(os.environ, **p['env'])
with (job / 'launcher.log').open('ab') as log:
    process = subprocess.Popen(p['argv'], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
public = {'pid': process.pid, 'argv': p['argv'], 'cwd': str(cwd), 'job_dir': str(job),
          'hf_auth_present': bool(env.get('HF_TOKEN'))}
record.write_text(json.dumps(public, indent=2) + '\\n')
print(json.dumps(public))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True, help='SSH destination (user@host or SSH config alias)')
    p.add_argument('--port', type=int, help='SSH port; otherwise use the SSH configuration')
    p.add_argument('--identity', type=Path, help='Optional SSH private-key path')
    p.add_argument('--hf-hub-cache', help='Optional Hugging Face cache directory on the remote host')
    p.add_argument('--job-dir', type=Path, required=True)
    p.add_argument('--without-hf-auth', action='store_true')
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    argv = args.command[1:] if args.command and args.command[0] == '--' else args.command
    if not argv: p.error('Supply a command after --')
    env = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false',
           'PYTHONUNBUFFERED': '1'}
    if args.hf_hub_cache:
        env['HF_HUB_CACHE'] = args.hf_hub_cache
    if not args.without_hf_auth:
        from huggingface_hub import get_token
        token = get_token()
        if token: env['HF_TOKEN'] = token
    payload = {'cwd': str(ROOT), 'job_dir': str(args.job_dir.resolve()), 'argv': argv, 'env': env}
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12']
    if args.identity:
        command += ['-i', str(args.identity), '-o', 'IdentitiesOnly=yes']
    if args.port is not None:
        command += ['-p', str(args.port)]
    command += [args.host, 'python -c ' + shlex.quote(BOOTSTRAP)]
    # No shell interpolation of payload, and no payload logging.
    result = subprocess.run(command, input=json.dumps(payload), text=True, capture_output=True)
    print(result.stdout, end='')
    if result.stderr: print(result.stderr, end='')
    raise SystemExit(result.returncode)


if __name__ == '__main__': main()
