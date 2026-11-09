"""CPU-only tests for scientific cache integrity; no model or dataset downloads."""
from dataclasses import asdict
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from epcrc.rewardbench import build_pairs, grouped_split, split_manifest
from supplementary.common import load_npz, predict_convex, demo_dataset, sha256_file
from supplementary.exp00_response_cache.run import (
    fingerprint, materialize, merge, nli_label_audit, split_assignment, score, load_inputs,
)


def dump(path, value):
    path.write_text(json.dumps(value))


def raw_reward_rows():
    return [{'subset': domain, 'id': str(i), 'prompt': f'Prompt {i}',
             'chosen': ['good'], 'rejected': ['bad', 'worse']}
            for domain in ['Focus', 'Math'] for i in range(12)]


class FakeDataset(list):
    _fingerprint = 'fake-dataset-fingerprint'
    features = {}


def test_prepare_rewardbench_emits_explicit_group_assignment(tmp_path, monkeypatch):
    fake = FakeDataset(raw_reward_rows())
    calls = []
    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return fake
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=load))
    output = tmp_path / 'pairs.jsonl'
    args = SimpleNamespace(kind='rewardbench', revision='a' * 40, output=output, seed=19, limit=0)
    materialize(args)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assignment = json.loads(output.with_suffix('.split.json').read_text())
    assert set(assignment) == {'FIT', 'CERT', 'TEST'}
    assert set(sum(assignment.values(), [])) == {r['base_item_id'] for r in rows}
    assert len(sum(assignment.values(), [])) == len(set(sum(assignment.values(), [])))
    assert calls[0][1]['revision'] == 'a' * 40
    provenance = json.loads(output.with_suffix('.provenance.json').read_text())
    assert provenance['file_sha256'] == sha256_file(output)
    before = output.with_suffix('.split.json').read_bytes()
    with pytest.raises(FileExistsError): materialize(args)
    assert before == output.with_suffix('.split.json').read_bytes()


def test_prepare_nli_groups_shared_premises_and_checks_labels(tmp_path, monkeypatch):
    fake = FakeDataset([{'idx': i, 'premise': f'p{i // 2}', 'hypothesis': f'h{i}', 'label': i % 3}
                        for i in range(12)])
    fake.features = {'label': SimpleNamespace(names=['entailment', 'neutral', 'contradiction'])}
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=lambda *a, **k: fake))
    args = SimpleNamespace(kind='nli', revision='a' * 40, output=tmp_path / 'nli.jsonl', seed=3, limit=3)
    materialize(args)
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert len(rows) == 6
    assert len({r['base_item_id'] for r in rows}) == 3
    fake.features['label'].names = ['contradiction', 'neutral', 'entailment']
    args.output = tmp_path / 'invalid.jsonl'
    with pytest.raises(ValueError, match='class order'): materialize(args)
    assert not args.output.exists()


@pytest.mark.parametrize('interface', ['judge', 'nli'])
def test_jsonl_preserves_unicode_line_separators_inside_strings(tmp_path, interface):
    text = 'line one\u2028line two\u2029paragraph two'
    if interface == 'judge':
        raw = raw_reward_rows()[0]
        raw['prompt'] = text
        rows = [asdict(pair) for pair in build_pairs([raw])]
        field = 'prompt'
    else:
        rows = [{'pair_id': 'p0', 'base_item_id': 'g0', 'text': text, 'text_pair': 'hypothesis', 'gold': 0}]
        field = 'text'
    path = tmp_path / 'unicode.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    loaded, pairs = load_inputs(path, interface)
    assert loaded == rows
    assert loaded[0][field] == text
    if interface == 'judge': assert pairs[0].prompt == text


def test_original_split_summary_is_reconstructed_and_verified(tmp_path):
    pairs = build_pairs(raw_reward_rows(), seed=19)
    assignment = grouped_split(pairs, seed=19)
    path = tmp_path / 'summary.json'
    dump(path, split_manifest(pairs, assignment, 19))
    labels = split_assignment(path, [asdict(p) for p in pairs], 'judge')
    assert set(labels) == {'FIT', 'CERT', 'TEST'}
    with pytest.raises(ValueError, match='does not match'):
        split_assignment(path, [asdict(p) for p in pairs[:-1]], 'judge')


def test_explicit_assignment_rejects_repeated_groups(tmp_path):
    path = tmp_path / 'split.json'
    dump(path, {'FIT': ['x'], 'CERT': ['x'], 'TEST': ['y']})
    with pytest.raises(ValueError, match='exactly one'):
        split_assignment(path, [{'base_item_id': 'x'}, {'base_item_id': 'y'}], 'nli')


def create_cache(tmp_path):
    manifest = {'name': 'cpu_mock', 'interface': 'nli', 'contexts': ['clean'], 'endpoints': [
        {'judge_id': f'N{i}', 'model_id': f'model/{i}', 'revision': str(i) * 40,
         'canonical_indices': [0, 1, 2]} for i in range(2)]}
    manifest_path = tmp_path / 'manifest.json'
    dump(manifest_path, manifest)
    rows = [{'pair_id': f'p{i}', 'base_item_id': f'g{i // 2}', 'text': 'premise',
             'text_pair': 'hypothesis', 'gold': i % 3} for i in range(24)]
    pairs_path = tmp_path / 'pairs.jsonl'
    pairs_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    cache = tmp_path / 'blocks'; cache.mkdir()
    for endpoint in manifest['endpoints']:
        protocol = {'manifest_sha256': sha256_file(manifest_path), 'pairs_sha256': sha256_file(pairs_path),
                    'interface': 'nli', 'endpoint': endpoint, 'max_tokens': 512, 'dtype': 'float32',
                    'label_prefix': ' ', 'limit': 0, 'allow_truncation': False, 'batch_size': 4,
                    'repeats': 1, 'versions': {'torch': 'mock', 'transformers': 'mock'},
                    'scoring_source_sha256': 'code1', 'original_scoring_sha256': 'code2',
                    'prompt_source_sha256': 'code3'}
        ids = [r['pair_id'] for r in rows]
        payload = {'judge_id': endpoint['judge_id'], 'model_id': endpoint['model_id'],
                   'revision': endpoint['revision'], 'context': 'clean', 'synthetic': False,
                   'pair_ids': ids, 'protocol': protocol,
                   'fingerprint': fingerprint(dict(protocol, context='clean', pair_ids=ids)),
                   'probabilities': [[.2, .3, .5]] * len(rows), 'prompt_sha256': ['hash'] * len(rows)}
        dump(cache / f"{endpoint['judge_id']}__clean.json", payload)
    return SimpleNamespace(manifest=manifest_path, pairs=pairs_path, cache=cache,
                           output=tmp_path / 'panel.npz', limit=0, split_manifest=None)


def test_merge_aligns_all_models_and_groups_without_gpu(tmp_path):
    args = create_cache(tmp_path)
    merge(args)
    data = load_npz(args.output)
    assert data.judge_ids == ['N0', 'N1']
    assert data.metadata['collection_protocol']['versions']['torch'] == 'mock'
    groups = [set(data.groups[s]) for s in ['FIT', 'CERT', 'TEST']]
    assert all(not (groups[i] & groups[j]) for i in range(3) for j in range(i + 1, 3))
    assert args.output.with_suffix('.inventory.json').exists()


@pytest.mark.parametrize('field,value', [('max_tokens', 256), ('dtype', 'bfloat16'),
                                       ('versions', {'torch': 'different'})])
def test_merge_rejects_mixed_protocols_even_with_valid_fingerprint(tmp_path, field, value):
    args = create_cache(tmp_path)
    path = args.cache / 'N1__clean.json'
    payload = json.loads(path.read_text())
    payload['protocol'][field] = value
    payload['fingerprint'] = fingerprint(dict(payload['protocol'], context='clean', pair_ids=payload['pair_ids']))
    dump(path, payload)
    with pytest.raises(ValueError, match='protocols/software versions differ'): merge(args)
    assert not args.output.exists()


@pytest.mark.parametrize('kind', ['nan', 'negative', 'misaligned'])
def test_merge_rejects_bad_cache_without_final_output(tmp_path, kind):
    args = create_cache(tmp_path)
    path = args.cache / 'N1__clean.json'
    payload = json.loads(path.read_text())
    if kind == 'nan': payload['probabilities'][0] = [float('nan'), 0, 1]
    elif kind == 'negative': payload['probabilities'][0] = [-.1, .5, .6]
    else: payload['pair_ids'] = payload['pair_ids'][::-1]
    dump(path, payload)
    with pytest.raises(ValueError): merge(args)
    assert not args.output.exists()


def test_load_npz_rejects_group_leakage_and_records_stored_split(tmp_path):
    args = create_cache(tmp_path)
    merge(args)
    with np.load(args.output, allow_pickle=False) as original:
        fields = {k: original[k] for k in original.files}
    fields['split'] = np.array(['FIT'] * 8 + ['CERT'] * 8 + ['TEST'] * 8)
    stored = tmp_path / 'stored.npz'
    np.savez_compressed(stored, **fields)
    data = load_npz(stored, seed=999)
    assert data.metadata['split_source'] == 'stored'
    assert data.metadata['split_seed'] is None
    assert data.metadata['requested_split_seed'] == 999
    fields['split'][1] = 'CERT'
    np.savez_compressed(stored, **fields)
    with pytest.raises(ValueError, match='leaks'): load_npz(stored)


def test_nli_mapping_checks_semantic_order_and_discloses_generic_config():
    config = {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'}
    assert nli_label_audit(config, {'canonical_indices': [2, 1, 0]})['canonical_labels'][0] == 'entailment'
    with pytest.raises(ValueError, match='disagrees'):
        nli_label_audit(config, {'canonical_indices': [0, 1, 2]})
    with pytest.raises(ValueError, match='label_source'):
        nli_label_audit({0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2'}, {'canonical_indices': [0, 1, 2]})


def test_prediction_weight_axis_matches_sorted_retained_endpoints():
    data = demo_dataset(0)
    first, _ = predict_convex(data.splits['FIT'], data.splits['TEST'], [3, 1])
    second, _ = predict_convex(data.splits['FIT'], data.splits['TEST'], [1, 3])
    assert all(np.allclose(a, b) for a, b in zip(first, second))


def test_cpu_mock_scoring_canonicalizes_and_rejects_changed_resume_protocol(tmp_path, monkeypatch):
    args = create_cache(tmp_path)
    for path in args.cache.glob('*.json'): path.unlink()
    manifest = json.loads(args.manifest.read_text())
    manifest['endpoints'][1]['canonical_indices'] = [2, 1, 0]
    dump(args.manifest, manifest)

    class FakeArray:
        def __init__(self, values): self.values = np.asarray(values)
        def float(self): return self
        def softmax(self, axis):
            p = np.exp(self.values - self.values.max(axis=axis, keepdims=True))
            return FakeArray(p / p.sum(axis=axis, keepdims=True))
        def cpu(self): return self
        def numpy(self): return self.values

    class FakeBatch(dict):
        def to(self, device): return self

    class FakeTokenizer:
        model_max_length = 512
        padding_side = 'right'
        truncation_side = 'right'
        chat_template = None
        def __call__(self, text, text_pair=None, **kwargs):
            if isinstance(text, list): return FakeBatch(n_items=len(text))
            return {'input_ids': [1, 2, 3]}

    class FakeModel:
        def __init__(self, model_id):
            self.reverse = model_id.endswith('/1')
            labels = ['contradiction', 'neutral', 'entailment'] if self.reverse else ['entailment', 'neutral', 'contradiction']
            self.config = SimpleNamespace(num_labels=3, id2label=dict(enumerate(labels)), max_position_embeddings=512)
        def to(self, device): return self
        def eval(self): return self
        def __call__(self, n_items):
            logits = [3, 2, 1] if self.reverse else [1, 2, 3]
            return SimpleNamespace(logits=FakeArray(np.tile(logits, (n_items, 1))))

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        __version__='mock', float32='float32', inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        __version__='mock', AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer()),
        AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=lambda model_id, **k: FakeModel(model_id))))
    score_args = SimpleNamespace(manifest=args.manifest, pairs=args.pairs, output=args.cache,
                                 judges=None, device='cpu', max_tokens=512, dtype='float32',
                                 batch_size=5, repeats=2, limit=0, allow_truncation=False)
    score(score_args)
    first = json.loads((args.cache / 'N0__clean.json').read_text())
    second = json.loads((args.cache / 'N1__clean.json').read_text())
    assert np.allclose(first['probabilities'], second['probabilities'])
    assert first['tokenization']['effective_max_tokens'] == 512
    assert len(first['timing']['batches']) == 10  # Two repeats of 24 items / batch 5.
    merge(args)
    assert args.output.exists()
    score(score_args)  # Identical collection resumes without loading anything.
    score_args.repeats = 3
    with pytest.raises(ValueError, match='Incompatible existing cache'): score(score_args)
