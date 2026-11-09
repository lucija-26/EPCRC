"""Check that a merged tensor is the complete exact manifest/input/cache inventory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from supplementary.common import load_npz, sha256_file
from epcrc.prompts import REGISTERED_CONTEXTS


def verify(manifest_path, pairs_path, cache, merged, split_manifest=None, limit=0):
    manifest = json.loads(Path(manifest_path).read_text())
    with Path(pairs_path).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if limit:
        rows = rows[:limit]
    data = load_npz(merged)
    contexts = manifest.get('contexts', [c.name for c in REGISTERED_CONTEXTS] if manifest['interface'] == 'judge' else ['clean'])
    endpoints = manifest['endpoints']
    if data.judge_ids != [e['judge_id'] for e in endpoints] or data.context_names != contexts:
        raise ValueError('Merged tensor does not contain the complete ordered manifest axes')
    if data.metadata.get('manifest') != manifest or data.metadata.get('synthetic') is not False:
        raise ValueError('Merged tensor has different manifest provenance or is synthetic')
    protocol = data.metadata.get('collection_protocol', {})
    if (protocol.get('manifest_sha256') != sha256_file(manifest_path)
            or protocol.get('pairs_sha256') != sha256_file(pairs_path)
            or protocol.get('limit') != limit):
        raise ValueError('Merged tensor belongs to different inputs or limit')
    if split_manifest is not None and data.metadata.get('split_manifest_sha256') != sha256_file(split_manifest):
        raise ValueError('Merged tensor belongs to a different explicit split manifest')
    with np.load(merged, allow_pickle=False) as z:
        if z['pair_ids'].astype(str).tolist() != [row['pair_id'] for row in rows]:
            raise ValueError('Merged tensor item order does not match the complete requested inputs')
    expected = {f"{e['judge_id']}__{c}.json" for e in endpoints for c in contexts}
    inventory = data.metadata.get('cache_inventory', [])
    if len(inventory) != len(expected) or {Path(e['path']).name for e in inventory} != expected:
        raise ValueError('Merged tensor inventory is incomplete or has duplicate blocks')
    for entry in inventory:
        path = Path(cache) / Path(entry['path']).name
        if entry['sha256'] != sha256_file(path):
            raise ValueError(f'Cache block changed after merge: {path}')
    return {'verified': True, 'endpoints': len(endpoints), 'contexts': len(contexts),
            'items': len(rows), 'input_sha256': sha256_file(merged)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['manifest', 'pairs', 'cache', 'input']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--split-manifest', type=Path)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    print(json.dumps(verify(args.manifest, args.pairs, args.cache, args.input, args.split_manifest, args.limit)))


if __name__ == '__main__':
    main()
