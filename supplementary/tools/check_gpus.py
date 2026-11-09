"""Read-only local/SSH GPU inventory; does not reserve devices or stop processes."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from supplementary.common import save_json

REMOTE_QUERY = 'hostname; nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits'


def probe(port, identity, host=None):
    if port is None:
        command = ['bash', '-c', REMOTE_QUERY]
    else:
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '-o', 'StrictHostKeyChecking=accept-new']
        if identity: command += ['-i', str(identity), '-o', 'IdentitiesOnly=yes']
        command += ['-p', str(port), host, REMOTE_QUERY]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        lines = result.stdout.strip().splitlines()
        gpus = []
        for line in lines[1:]:
            columns = [c.strip() for c in line.split(',')]
            if len(columns) == 6 and columns[0].isdigit():
                index, name, uuid, total, used, utilization = columns
                gpus.append({'index': int(index), 'name': name, 'uuid': uuid, 'total_MiB': int(total),
                             'used_MiB': int(used), 'free_MiB': int(total) - int(used),
                             'gpu_utilization_percent': int(utilization),
                             'idle_candidate': int(used) < 1024 and int(utilization) == 0})
        return {'node': 'current' if port is None else str(port), 'hostname': lines[0] if lines else None,
                'reachable': result.returncode == 0, 'gpus': gpus,
                'raw_output': result.stdout, 'error': result.stderr or None}
    except subprocess.TimeoutExpired:
        return {'node': str(port), 'reachable': False, 'error': 'SSH/query timeout', 'gpus': []}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', help='Optional SSH destination; omitted means local inventory only')
    p.add_argument('--identity', type=Path)
    p.add_argument('--ports', type=int, nargs='+', default=[], help='Remote ports on --host (default: 22)')
    p.add_argument('--output', type=Path, default=Path('supplementary/gpu_status.json'))
    args = p.parse_args()
    if args.ports and not args.host:
        p.error('--ports requires --host')
    ports = (args.ports or [22]) if args.host else []
    with ThreadPoolExecutor(max_workers=5) as pool:
        nodes = list(pool.map(lambda port: probe(port, args.identity, args.host), [None] + ports))
    payload = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'nodes': nodes,
               'note': 'Momentary availability only; zero utilization does not imply unallocated memory. Recheck before launching.'}
    save_json(args.output, payload)
    for node in nodes:
        print(node['node'], node.get('hostname'), [(g['index'], g['free_MiB'], g['idle_candidate']) for g in node['gpus']])


if __name__ == '__main__': main()
