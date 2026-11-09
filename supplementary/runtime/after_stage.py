"""Wait for a durable pipeline stage, then run one command without installing anything."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def inspect_stage(document, stage):
    """Only the requested stage must succeed; later analysis failures do not block it."""
    row = document.get('stages', {}).get(stage)
    if row and row.get('status') == 'complete':
        return 'ready', row
    if row and row.get('status') in {'failed', 'interrupted', 'failed_missing_artifact'}:
        return 'failed', row
    if document.get('status') in {'failed', 'interrupted'}:
        return 'failed', {'status': document['status'], 'error': document.get('error'),
                          'current_stage': document.get('current_stage'), 'requested_stage': row}
    return 'waiting', row or {'status': 'stage_not_started'}


def atomic_json(path, payload):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


@contextmanager
def queue_lock(output):
    if output is None:
        yield
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(output.name + '.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another queue owns {output}') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def recent_log(row, count=20):
    path = row.get('log')
    if not path:
        return []
    try:
        with Path(path).open(errors='replace') as handle:
            return [line.rstrip('\n') for line in deque(handle, maxlen=count)]
    except OSError:
        return []


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-status', type=Path, required=True)
    parser.add_argument('--stage', default='score')
    parser.add_argument('--timeout-hours', type=float, default=72)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--output', type=Path, help='Durable queue status JSON; a sibling lock prevents duplicate queues')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='Command after --; executed as argv, never through a shell')
    args = parser.parse_args(argv)
    if args.command and args.command[0] == '--':
        args.command = args.command[1:]
    if not args.command:
        parser.error('Provide the child command after --')
    if not math.isfinite(args.timeout_hours) or not math.isfinite(args.poll_seconds) or args.timeout_hours <= 0 or args.poll_seconds < 10:
        parser.error('timeout-hours must be positive and poll-seconds must be at least 10')
    args.wait_status = args.wait_status.absolute()
    if args.output is not None:
        args.output = args.output.absolute()
        if args.output == args.wait_status:
            parser.error('--output must not overwrite --wait-status')
    return args


def execute(args):
    status = {'status': 'waiting', 'queue_pid': os.getpid(), 'host': platform.node(), 'created_at': now(),
              'wait_status': str(args.wait_status), 'wait_stage': args.stage, 'command': args.command,
              'timeout_hours': args.timeout_hours, 'poll_seconds': args.poll_seconds}
    child = None

    def save():
        status['updated_at'] = now()
        atomic_json(args.output, status)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    deadline = time.monotonic() + args.timeout_hours * 3600
    previous = None
    try:
        save()
        while True:
            if time.monotonic() >= deadline:
                status.update(status='timed_out', finished_at=now(), error=f'Timeout waiting for {args.stage}')
                save()
                print(status['error'], file=sys.stderr, flush=True)
                return 124
            try:
                document = json.loads(args.wait_status.read_text())
                condition, detail = inspect_stage(document, args.stage)
            except FileNotFoundError:
                condition, detail = 'waiting', {'status': 'status_file_not_created'}
            except json.JSONDecodeError as error:
                # Atomic upstream writes make this unusual; record it and allow a bounded recovery.
                condition, detail = 'waiting', {'status': 'unreadable_status_json', 'error': str(error)}
            status['last_observed_stage'] = detail
            status['last_checked_at'] = now()
            save()
            transition = (condition, detail.get('status'))
            if transition != previous:
                print(f'[{now()}] requested stage {args.stage}: {condition} ({detail.get("status")})', flush=True)
                previous = transition
            if condition == 'failed':
                status.update(status='upstream_failed', finished_at=now(), upstream_log_tail=recent_log(detail))
                save()
                print(json.dumps({'upstream_failure': detail, 'log_tail': status['upstream_log_tail']}, ensure_ascii=False),
                      file=sys.stderr, flush=True)
                return 3
            if condition == 'ready':
                break
            time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
        status.update(status='launching', dependency_completed_at=now())
        save()
        child = subprocess.Popen(args.command, stdin=subprocess.DEVNULL, start_new_session=True)
        status.update(status='running_child', child_pid=child.pid, child_started_at=now())
        save()
        code = child.wait()
        status.update(status='complete' if code == 0 else 'child_failed', child_returncode=code, finished_at=now())
        save()
        return code if code >= 0 else 128 - code
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=str(error), finished_at=now(), child_returncode=child.returncode if child else None)
        save()
        print(f'Queue stopped: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


def main(argv=None):
    args = parse_args(argv)
    try:
        with queue_lock(args.output):
            return execute(args)
    except RuntimeError as error:
        print(error, file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
