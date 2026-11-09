"""Read-only completed block counts, durable runner status, and recent stage logs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import contexts_for_manifest, now, process_start


def inspect(manifest_path, cache, outputs=(), log_lines=12):
    manifest = json.loads(Path(manifest_path).read_text())
    contexts = contexts_for_manifest(manifest)
    rows, issues = [], []
    for endpoint in manifest['endpoints']:
        complete, item_counts = [], []
        for context in contexts:
            path = Path(cache) / f"{endpoint['judge_id']}__{context}.json"
            if not path.exists():
                continue
            try:
                block = json.loads(path.read_text())
                n = len(block.get('pair_ids', []))
                if (block.get('judge_id') != endpoint['judge_id'] or block.get('context') != context
                        or block.get('model_id') != endpoint['model_id'] or block.get('revision') != endpoint['revision']
                        or block.get('synthetic') is not False or n == 0 or len(block.get('probabilities', [])) != n):
                    raise ValueError('Metadata/count mismatch')
                complete.append(context)
                item_counts.append(n)
            except (ValueError, KeyError) as error:
                issues.append({'path': str(path), 'error': str(error)})
        rows.append({'judge_id': endpoint['judge_id'], 'model_id': endpoint['model_id'],
                     'completed_context_blocks': len(complete), 'expected_context_blocks': len(contexts),
                     'missing_contexts': [c for c in contexts if c not in complete], 'item_counts': sorted(set(item_counts))})
    runners = []
    for output in outputs:
        path = Path(output) / 'status.json'
        if not path.exists():
            runners.append({'output': str(output), 'status': 'no_status_yet'})
            continue
        state = json.loads(path.read_text())
        stage = state.get('stages', {}).get(state.get('current_stage'), {})
        log = Path(stage['log']) if stage.get('log') else None
        tail = []
        if log is not None and log.exists():
            from collections import deque
            with log.open(errors='replace') as handle:
                tail = [line.rstrip('\n') for line in deque(handle, maxlen=log_lines)]
        import platform
        pid = state.get('runner_pid')
        locally_alive = (process_start(pid) == state.get('runner_process_start') and process_start(pid) is not None
                         if pid and state.get('host') == platform.node() else None)
        runners.append({'output': str(output), 'status': state.get('status'), 'host': state.get('host'),
                        'runner_pid': pid, 'runner_alive_on_current_host': locally_alive,
                        'active_child': state.get('active_child'), 'current_stage': state.get('current_stage'),
                        'error': state.get('error'), 'updated_at': state.get('updated_at'), 'log_tail': tail})
    return {'checked_at_utc': now(), 'manifest': manifest['name'], 'cache': str(cache),
            'completed_blocks': sum(r['completed_context_blocks'] for r in rows),
            'expected_blocks': len(rows) * len(contexts),
            'fully_scored_endpoints': sum(r['completed_context_blocks'] == len(contexts) for r in rows),
            'expected_endpoints': len(rows), 'endpoints': rows, 'invalid_blocks': issues, 'runners': runners,
            'scope': 'Progress count validates basic metadata only; complete merge/verify remains required before analysis.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--runs', nargs='*', type=Path, default=[])
    p.add_argument('--log-lines', type=int, default=12)
    args = p.parse_args()
    print(json.dumps(inspect(args.manifest, args.cache, args.runs, args.log_lines), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
