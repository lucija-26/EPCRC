"""Pinned, resumable response collection on disjoint GPUs; no synthetic fallback."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request
from importlib.metadata import version, PackageNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from supplementary.common import save_json, sha256_file, load_npz
from epcrc.prompts import REGISTERED_CONTEXTS, LABELS, render, canonicalize, prompt_sha256
from epcrc.rewardbench import read_pairs
from epcrc.scoring import LabelScorer


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def package_versions():
    names = ['numpy', 'torch', 'transformers', 'tokenizers', 'accelerate', 'datasets', 'huggingface_hub']
    result = {}
    for name in names:
        try: result[name] = version(name)
        except PackageNotFoundError: result[name] = None
    return result


def validate_manifest(manifest, locked=True):
    if manifest.get('interface') not in {'judge', 'nli'} or not manifest.get('name'):
        raise ValueError('Manifest needs a name and judge/nli interface')
    endpoints = manifest.get('endpoints')
    if not isinstance(endpoints, list) or len(endpoints) < 2:
        raise ValueError('At least two endpoints are required')
    ids = [e.get('judge_id', '') for e in endpoints]
    if len(ids) != len(set(ids)) or any(not i or '/' in i or '\\' in i or '__' in i for i in ids):
        raise ValueError('Endpoint IDs must be unique safe names without double underscores')
    valid_contexts = [c.name for c in REGISTERED_CONTEXTS] if manifest['interface'] == 'judge' else ['clean']
    contexts = manifest.get('contexts', valid_contexts)
    if not contexts or len(set(contexts)) != len(contexts) or not set(contexts) <= set(valid_contexts):
        raise ValueError('Contexts must be unique registered names')
    for endpoint in endpoints:
        if not endpoint.get('model_id'): raise ValueError('Missing model_id')
        revision = endpoint.get('revision', '')
        if locked and (len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision)):
            raise ValueError('Run lock first; formal inference requires full immutable model revisions')
        if manifest['interface'] == 'nli' and sorted(endpoint.get('canonical_indices', [])) != [0, 1, 2]:
            raise ValueError('NLI endpoint requires a canonical class permutation')
    return contexts


def validate_probabilities(values, n, path):
    p = np.asarray(values, dtype=float)
    if (p.shape != (n, 3) or not np.isfinite(p).all() or p.min() < 0
            or not np.allclose(p.sum(-1), 1, atol=1e-6, rtol=0)):
        raise ValueError(f'Invalid probabilities: {path}')
    return p


def nli_label_audit(id2label, endpoint):
    order = endpoint['canonical_indices']
    if sorted(order) != [0, 1, 2]: raise ValueError('Invalid canonical class permutation')
    audit = {'id2label': id2label, 'canonical_indices': order,
             'canonical_labels': ['entailment', 'neutral', 'contradiction']}
    labels = [str(id2label.get(i, id2label.get(str(i), ''))).lower() for i in order]
    if set(labels) == set(audit['canonical_labels']):
        if labels != audit['canonical_labels']:
            raise ValueError('Manifest canonical_indices disagrees with model.config.id2label')
        audit['mapping_validation'] = 'verified against explicit config labels'
    else:
        if not endpoint.get('label_source'):
            raise ValueError('Generic config labels require a pinned label_source in the manifest')
        audit['mapping_validation'] = 'declared mapping; config labels are not semantic labels'
        audit['label_source'] = endpoint['label_source']
    return audit


def split_assignment(path, rows, interface):
    """Read explicit groups, or reconstruct and verify an original summary."""
    from epcrc.rewardbench import JudgedPair, grouped_split, split_manifest
    payload = read_json(path)
    if 'split_sha256' in payload:
        if interface != 'judge': raise ValueError('Original judge split summaries cannot partition NLI rows')
        pairs = [JudgedPair(**row) for row in rows]
        assignment = grouped_split(pairs, seed=payload['seed'])
        actual = split_manifest(pairs, assignment, payload['seed'])
        if (actual['split_sha256'] != payload['split_sha256']
                or actual['n_pairs'] != payload.get('n_pairs')):
            raise ValueError('Original split summary does not match these pairs; use its complete input or explicit assignments')
    else:
        assignment = payload.get('splits', payload)
    if set(assignment) != {'FIT', 'CERT', 'TEST'} or any(not isinstance(v, list) for v in assignment.values()):
        raise ValueError('Split file needs FIT/CERT/TEST group lists or a verified original summary')
    members = [g for values in assignment.values() for g in values]
    if any(not isinstance(g, str) for g in members) or len(members) != len(set(members)):
        raise ValueError('Split groups must be strings and occur in exactly one split')
    membership = {g: name for name, values in assignment.items() for g in values}
    if any(row['base_item_id'] not in membership for row in rows):
        raise ValueError('Split assignment does not cover every source group')
    return np.asarray([membership[row['base_item_id']] for row in rows])


def lock_manifest(args):
    manifest = read_json(args.manifest)
    validate_manifest(manifest, locked=False)
    for endpoint in manifest['endpoints']:
        repo, revision = endpoint['model_id'], endpoint.get('revision') or 'main'
        request = urllib.request.Request(f'https://huggingface.co/api/models/{repo}/revision/{revision}')
        if os.environ.get('HF_TOKEN'):
            request.add_header('Authorization', 'Bearer ' + os.environ['HF_TOKEN'])
        with urllib.request.urlopen(request, timeout=45) as response:
            info = json.load(response)
        endpoint['revision'] = info['sha']
        if len(endpoint['revision']) != 40: raise ValueError('Hub did not return a full commit')
    manifest['locked_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save_json(args.output, manifest)


def materialize(args):
    """All dataset versions and selected rows are serialized before scoring."""
    from datasets import load_dataset
    from epcrc.rewardbench import build_pairs, grouped_split
    from dataclasses import asdict
    destinations = [args.output, args.output.with_suffix('.split.json'), args.output.with_suffix('.provenance.json')]
    if any(path.exists() for path in destinations):
        raise FileExistsError('Input or sidecar already exists; use a fresh destination')
    assignment = None
    if args.kind == 'rewardbench':
        dataset_id, config, split = 'allenai/reward-bench-2', None, 'test'
        rows = load_dataset(dataset_id, revision=args.revision, split=split)
        pairs = build_pairs(rows, seed=args.seed)
        output = [asdict(p) for p in pairs]
        assignment = grouped_split(pairs, seed=args.seed)
    else:
        dataset_id, config, split = 'nyu-mll/glue', 'mnli', 'validation_matched'
        rows = load_dataset(dataset_id, config, revision=args.revision, split=split)
        feature = getattr(rows, 'features', {}).get('label')
        if feature is not None and list(feature.names) != ['entailment', 'neutral', 'contradiction']:
            raise ValueError('Unexpected MNLI class order; canonicalize it explicitly before collection')
        output = []
        for index, row in enumerate(rows):
            if row['label'] < 0: continue
            premise_hash = hashlib.sha256(row['premise'].encode()).hexdigest()
            output.append({'pair_id': f'mnli/{row.get("idx", index)}', 'base_item_id': premise_hash,
                           'text': row['premise'], 'text_pair': row['hypothesis'], 'gold': int(row['label'])})
    if args.limit:
        unique = sorted({row['base_item_id'] for row in output})
        chosen = set(np.random.default_rng(args.seed).permutation(unique)[:args.limit])
        output = [row for row in output if row['base_item_id'] in chosen]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not output: raise ValueError('Dataset preparation yielded no rows')
    args.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in output))
    if assignment is not None:
        present = {row['base_item_id'] for row in output}
        save_json(args.output.with_suffix('.split.json'),
                  {name: [g for g in values if g in present] for name, values in assignment.items()})
    save_json(args.output.with_suffix('.provenance.json'), {
        'dataset_id': dataset_id, 'config': config, 'split': split, 'revision': args.revision,
        'seed': args.seed, 'limit_source_groups': args.limit, 'n_rows': len(output),
        'file_sha256': sha256_file(args.output), 'dataset_fingerprint': rows._fingerprint,
        'versions': package_versions(),
        'canonical_labels': list(LABELS) if args.kind == 'rewardbench' else ['entailment', 'neutral', 'contradiction']})


def load_inputs(path, interface):
    # JSONL records end at physical newline bytes. str.splitlines() also splits
    # valid U+2028/U+2029 characters inside JSON strings written ensure_ascii=False.
    with open(path, encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows: raise ValueError('Input file is empty')
    ids = [r['pair_id'] for r in rows]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate pair IDs')
    if any(not r.get('base_item_id') for r in rows): raise ValueError('Missing source group')
    if interface == 'judge': return rows, read_pairs(str(path))
    if interface != 'nli': raise ValueError('Supported interfaces: judge, nli')
    return rows, None


def score(args):
    import torch
    import transformers
    manifest = read_json(args.manifest)
    requested = validate_manifest(manifest)
    endpoints = manifest['endpoints']
    ids = [e['judge_id'] for e in endpoints]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate endpoint IDs')
    if args.judges:
        unknown = set(args.judges) - set(ids)
        if unknown: raise ValueError(f'Unknown endpoints: {unknown}')
        endpoints = [e for e in endpoints if e['judge_id'] in args.judges]
    interface = manifest['interface']
    rows, pairs = load_inputs(args.pairs, interface)
    if args.limit: rows, pairs = rows[:args.limit], None if pairs is None else pairs[:args.limit]
    context_map = {c.name: c for c in REGISTERED_CONTEXTS}
    pair_ids = [r['pair_id'] for r in rows]
    args.output.mkdir(parents=True, exist_ok=True)
    hardware = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'],
                              text=True, capture_output=True).stdout.strip() if args.device.startswith('cuda') else 'CPU'
    selected_device = {}
    if args.device.startswith('cuda'):
        properties = torch.cuda.get_device_properties(args.device)
        selected_device = {'selected_device_name': torch.cuda.get_device_name(args.device),
                           'selected_device_total_memory_bytes': properties.total_memory,
                           'selected_device_uuid': str(getattr(properties, 'uuid', 'unavailable'))}
    for endpoint in endpoints:
        revision = endpoint.get('revision', '')
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('Run lock first; formal inference requires full immutable model revisions')
        protocol = {'manifest_sha256': sha256_file(args.manifest), 'pairs_sha256': sha256_file(args.pairs),
                    'interface': interface, 'endpoint': endpoint, 'max_tokens': args.max_tokens,
                    'dtype': args.dtype, 'label_prefix': ' ', 'limit': args.limit,
                    'allow_truncation': args.allow_truncation,
                    'batch_size': args.batch_size, 'repeats': args.repeats,
                    'versions': package_versions(),
                    'scoring_source_sha256': sha256_file(__file__),
                    'original_scoring_sha256': sha256_file(Path(__file__).resolve().parents[2] / 'epcrc/scoring.py'),
                    'prompt_source_sha256': sha256_file(Path(__file__).resolve().parents[2] / 'epcrc/prompts.py')}
        pending = []
        for context in requested:
            path = args.output / f'{endpoint["judge_id"]}__{context}.json'
            expected = fingerprint(dict(protocol, context=context, pair_ids=pair_ids))
            if path.exists():
                cached = read_json(path)
                if cached.get('fingerprint') != expected:
                    raise ValueError(f'Incompatible existing cache: {path}; choose a fresh output directory')
                if (cached.get('pair_ids') != pair_ids or cached.get('judge_id') != endpoint['judge_id']
                        or cached.get('model_id') != endpoint['model_id'] or cached.get('revision') != revision
                        or cached.get('context') != context):
                    raise ValueError(f'Invalid existing cache identity: {path}')
                validate_probabilities(cached['probabilities'], len(rows), path)
                continue
            pending.append((context, path, expected))
        if not pending: continue
        started = time.perf_counter()
        if interface == 'judge':
            scorer = LabelScorer(endpoint['model_id'], revision=revision, device=args.device,
                                 dtype=args.dtype, max_prompt_tokens=args.max_tokens)
            scorer.load()
            audit = scorer.audit_tokenization().to_dict()
            if not audit['consistent_boundary']: raise ValueError(f'Label boundary audit failed: {audit}')
            tokenizer, model = scorer.tokenizer, scorer.model
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(endpoint['model_id'], revision=revision)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                endpoint['model_id'], revision=revision, torch_dtype=getattr(torch, args.dtype)).to(args.device).eval()
            if model.config.num_labels != 3: raise ValueError('NLI model must output exactly three classes')
            order = endpoint['canonical_indices']
            audit = nli_label_audit(model.config.id2label, endpoint)
        token_limits = [args.max_tokens]
        for candidate in (getattr(tokenizer, 'model_max_length', None),
                          getattr(model.config, 'max_position_embeddings', None)):
            if isinstance(candidate, int) and 0 < candidate < 10**8: token_limits.append(candidate)
        effective_max_tokens = min(token_limits)
        if interface == 'judge':
            continuation = max(audit['n_continuation_tokens'].values())
            if continuation > 1: effective_max_tokens = min(effective_max_tokens, min(token_limits) - continuation)
            if effective_max_tokens < 1: raise ValueError('No model context budget remains for the prompt')
            scorer.max_prompt_tokens = effective_max_tokens
        audit.update(requested_max_tokens=args.max_tokens, effective_max_tokens=effective_max_tokens,
                     truncation_side=getattr(tokenizer, 'truncation_side', 'unknown'),
                     padding_side=getattr(tokenizer, 'padding_side', 'unknown'),
                     chat_template_sha256=fingerprint(getattr(tokenizer, 'chat_template', None)))
        load_seconds = time.perf_counter() - started
        def synchronize():
            if args.device.startswith('cuda'): torch.cuda.synchronize(args.device)
        for context_name, path, expected in pending:
            if interface == 'judge':
                ctx = context_map[context_name]
                prompts = [render(p.prompt, p.response_a, p.response_b, context=ctx) for p in pairs]
                text_hashes = [prompt_sha256(t) for t in prompts]
                token_lengths = [len(tokenizer(scorer.apply_chat_template(t), add_special_tokens=False,
                                               truncation=False)['input_ids']) for t in prompts]
                def infer(start, end):
                    raw = scorer.score_prompts(prompts[start:end])
                    return np.asarray([canonicalize(tuple(row), ctx) for row in raw])
            else:
                text_hashes = [fingerprint([r['text'], r['text_pair']]) for r in rows]
                token_lengths = [len(tokenizer(r['text'], r['text_pair'], truncation=False)['input_ids']) for r in rows]
                def infer(start, end):
                    batch = tokenizer([r['text'] for r in rows[start:end]], [r['text_pair'] for r in rows[start:end]],
                                      padding=True, truncation=True, max_length=effective_max_tokens, return_tensors='pt').to(args.device)
                    with torch.inference_mode():
                        return model(**batch).logits.float().softmax(-1).cpu().numpy()[:, order]
            if any(length > effective_max_tokens for length in token_lengths) and not args.allow_truncation:
                raise ValueError('Input exceeds the model token budget. Use --allow-truncation only for a declared truncated-input protocol.')
            # Warmup is explicit, excluded from warm inference and recorded separately.
            warmup_start = time.perf_counter()
            infer(0, min(len(rows), args.batch_size)); synchronize()
            warmup_seconds = time.perf_counter() - warmup_start
            if args.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats(args.device)
            probabilities, measurements = [], []
            for repeat in range(args.repeats):
                for start in range(0, len(rows), args.batch_size):
                    end = min(len(rows), start + args.batch_size)
                    synchronize(); tick = time.perf_counter()
                    values = infer(start, end)
                    synchronize(); seconds = time.perf_counter() - tick
                    if repeat == 0: probabilities.extend(values.tolist())
                    measurements.append({'repeat': repeat, 'start': start, 'end': end, 'n_items': end-start, 'seconds': seconds,
                                         'input_tokens_untruncated': sum(token_lengths[start:end])})
            validate_probabilities(probabilities, len(rows), path)
            save_json(path, {'judge_id': endpoint['judge_id'], 'model_id': endpoint['model_id'], 'revision': revision,
                             'context': context_name, 'fingerprint': expected, 'protocol': protocol,
                             'pair_ids': pair_ids, 'probabilities': probabilities, 'prompt_sha256': text_hashes,
                             'tokenization': audit, 'n_truncated_prompts': sum(n > effective_max_tokens for n in token_lengths),
                             'input_token_lengths_untruncated': token_lengths,
                             'timing': {'load_seconds': load_seconds, 'warmup_seconds': warmup_seconds,
                                        'batch_size': args.batch_size, 'repeats': args.repeats, 'batches': measurements,
                                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None,
                                        'scope': 'warm serialized tokenization + device inference + CPU transfer; excludes model load'},
                             'environment': {'hostname': platform.node(), 'hardware': hardware, 'device': args.device,
                                             'torch': torch.__version__, 'transformers': transformers.__version__,
                                             'python': platform.python_version(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                                             **selected_device},
                             'synthetic': False})
            print(f'Saved {path}', flush=True)
        if interface == 'judge': del scorer
        del model, tokenizer
        gc.collect()
        if args.device.startswith('cuda'): torch.cuda.empty_cache()


def merge(args):
    manifest = read_json(args.manifest)
    contexts = validate_manifest(manifest)
    if args.output.exists(): raise FileExistsError(args.output)
    rows, _ = load_inputs(args.pairs, manifest['interface'])
    if args.limit: rows = rows[:args.limit]
    judges = [e['judge_id'] for e in manifest['endpoints']]
    ids = [r['pair_id'] for r in rows]
    tensor, inventory = [], []
    common_protocol = None
    for context in contexts:
        block = []
        for endpoint in manifest['endpoints']:
            path = args.cache / f'{endpoint["judge_id"]}__{context}.json'
            data = read_json(path)  # Missing any endpoint/context is fatal.
            if (data['pair_ids'] != ids or data['model_id'] != endpoint['model_id']
                    or data['revision'] != endpoint['revision'] or data.get('judge_id') != endpoint['judge_id']
                    or data.get('context') != context or data.get('synthetic') is not False):
                raise ValueError(f'Misaligned model, revision or item order: {path}')
            protocol = data['protocol']
            required_protocol = {'manifest_sha256', 'pairs_sha256', 'interface', 'endpoint', 'max_tokens',
                                 'dtype', 'label_prefix', 'limit', 'allow_truncation', 'batch_size', 'repeats',
                                 'versions', 'scoring_source_sha256', 'original_scoring_sha256', 'prompt_source_sha256'}
            if not required_protocol <= set(protocol):
                raise ValueError(f'Incomplete protocol metadata in {path}')
            if protocol['endpoint'] != endpoint or protocol['interface'] != manifest['interface']:
                raise ValueError(f'Protocol endpoint/interface mismatch: {path}')
            if protocol['manifest_sha256'] != sha256_file(args.manifest) or protocol['pairs_sha256'] != sha256_file(args.pairs):
                raise ValueError(f'Cache does not belong to this locked manifest/input: {path}')
            if protocol['limit'] != args.limit:
                raise ValueError(f'Cache item limit differs: {path}')
            if data['fingerprint'] != fingerprint(dict(protocol, context=context, pair_ids=ids)):
                raise ValueError(f'Cache fingerprint invalid: {path}')
            shared = {key: value for key, value in protocol.items() if key != 'endpoint'}
            if common_protocol is None: common_protocol = shared
            elif common_protocol != shared:
                raise ValueError(f'Cache protocols/software versions differ across endpoints or contexts: {path}')
            if len(data.get('prompt_sha256', [])) != len(rows):
                raise ValueError(f'Per-item prompt hashes missing: {path}')
            block.append(validate_probabilities(data['probabilities'], len(rows), path))
            inventory.append({'path': str(path.resolve()), 'sha256': sha256_file(path)})
        tensor.append(np.stack(block, axis=1))
    provenance_path = args.pairs.with_suffix('.provenance.json')
    provenance = read_json(provenance_path) if provenance_path.exists() else None
    if provenance and provenance.get('file_sha256') != sha256_file(args.pairs):
        raise ValueError('Prepared data provenance checksum does not match the input')
    metadata = {'synthetic': False, 'ecosystem': manifest['name'],
                'interface': manifest['interface'], 'manifest': manifest, 'cache_inventory': inventory,
                'collection_protocol': common_protocol, 'input_provenance': provenance,
                'canonical_labels': list(LABELS) if manifest['interface'] == 'judge' else ['entailment', 'neutral', 'contradiction']}
    fields = {'probabilities': np.asarray(tensor), 'groups': np.asarray([r['base_item_id'] for r in rows]),
              'pair_ids': np.asarray(ids), 'judge_ids': np.asarray(judges), 'context_names': np.asarray(contexts),
              'model_ids': np.asarray([e['model_id'] for e in manifest['endpoints']]),
              'gold': np.asarray([LABELS.index(r['gold_label']) if manifest['interface'] == 'judge' else r['gold'] for r in rows]),
              'metadata_json': np.asarray(json.dumps(metadata))}
    if args.split_manifest:
        fields['split'] = split_assignment(args.split_manifest, rows, manifest['interface'])
        metadata['split_manifest_sha256'] = sha256_file(args.split_manifest)
        metadata['split_seed'] = read_json(args.split_manifest).get('seed')
        fields['metadata_json'] = np.asarray(json.dumps(metadata))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f'.{os.getpid()}.tmp')
    try:
        with open(temporary, 'wb') as f: np.savez_compressed(f, **fields)
        load_npz(temporary)
        temporary.replace(args.output)
    finally:
        if temporary.exists(): temporary.unlink()
    save_json(args.output.with_suffix('.inventory.json'), {'sha256': sha256_file(args.output), 'blocks': inventory})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    lock = sub.add_parser('lock'); lock.add_argument('--manifest', type=Path, required=True); lock.add_argument('--output', type=Path, required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--kind', choices=['rewardbench', 'nli'], required=True)
    prep.add_argument('--revision', required=True, help='Immutable dataset commit SHA')
    prep.add_argument('--output', type=Path, required=True); prep.add_argument('--seed', type=int, default=20260818)
    prep.add_argument('--limit', type=int, default=0, help='Number of source groups, 0=all')
    for name in ['score', 'merge']:
        a = sub.add_parser(name)
        a.add_argument('--manifest', type=Path, required=True); a.add_argument('--pairs', type=Path, required=True)
        a.add_argument('--output', type=Path, required=True); a.add_argument('--limit', type=int, default=0)
        if name == 'score':
            a.add_argument('--judges', nargs='+'); a.add_argument('--batch-size', type=int, default=4)
            a.add_argument('--repeats', type=int, default=1); a.add_argument('--max-tokens', type=int, default=8192)
            a.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default='bfloat16')
            a.add_argument('--device', default='cuda:0')
            a.add_argument('--allow-truncation', action='store_true',
                           help='Declare truncated-input scoring; default rejects inputs beyond model token budget')
        else:
            a.add_argument('--cache', type=Path, required=True); a.add_argument('--split-manifest', type=Path)
    args = p.parse_args()
    if (getattr(args, 'batch_size', 1) < 1 or getattr(args, 'repeats', 1) < 1
            or getattr(args, 'limit', 0) < 0 or getattr(args, 'max_tokens', 1) < 1):
        p.error('Batch size/repeats must be positive; limit must be nonnegative')
    if args.command == 'prepare' and (len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision)):
        p.error('Dataset revision must be a full immutable 40-character commit SHA')
    {'lock': lock_manifest, 'prepare': materialize, 'score': score, 'merge': merge}[args.command](args)


if __name__ == '__main__': main()
