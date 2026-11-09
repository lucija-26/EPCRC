"""Shared strict response-cache format and FIT-only subset selection.

NPZ (no pickled objects): probabilities[C,I,J,3], groups[I], pair_ids[I],
judge_ids[J], context_names[C], optional model_ids[J], split[I], gold[I],
metadata_json (scalar Unicode JSON). All contexts use the same item order.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from epcrc.judge import JudgeResponses, JudgeCoverageFunctional
from epcrc.selection_baselines import select_pivoted_qr, select_kmedoids


@dataclass
class Dataset:
    splits: dict
    groups: dict
    judge_ids: list
    context_names: list
    metadata: dict = field(default_factory=dict)
    gold: dict = field(default_factory=dict)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, payload):
    def convert(x):
        if isinstance(x, np.ndarray): return convert(x.tolist())
        if isinstance(x, np.generic): return convert(x.item())
        if isinstance(x, Path): return str(x)
        if isinstance(x, dict): return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (tuple, list, set)): return [convert(v) for v in x]
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(convert(payload), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def grouped_split(groups, seed):
    unique = np.unique(groups)
    if len(unique) < 6:
        raise ValueError('Need at least six independent source groups for FIT/CERT/TEST')
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_fit, n_cert = len(unique) // 2, max(1, len(unique) // 4)
    membership = {str(g): 'FIT' if i < n_fit else 'CERT' if i < n_fit + n_cert else 'TEST'
                  for i, g in enumerate(shuffled)}
    return np.asarray([membership[str(g)] for g in groups])


def load_npz(path, seed=20260818):
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        required = {'probabilities', 'groups', 'pair_ids', 'judge_ids', 'context_names'}
        if not required.issubset(z.files):
            raise ValueError(f'Missing NPZ fields: {sorted(required - set(z.files))}')
        p = np.asarray(z['probabilities'], dtype=float)
        groups, ids = z['groups'].astype(str), z['pair_ids'].astype(str)
        judges, contexts = z['judge_ids'].astype(str).tolist(), z['context_names'].astype(str).tolist()
        metadata = json.loads(str(z['metadata_json'].item())) if 'metadata_json' in z else {}
        if not isinstance(metadata, dict):
            raise ValueError('metadata_json must encode an object')
        model_ids = z['model_ids'].astype(str).tolist() if 'model_ids' in z else metadata.get('model_ids')
        stored_split = 'split' in z.files
        assignment = z['split'].astype(str) if stored_split else grouped_split(groups, seed)
        gold = np.asarray(z['gold']) if 'gold' in z else None
    if p.ndim != 4 or p.shape[-1] != 3:
        raise ValueError('probabilities must have shape (contexts, items, endpoints, 3)')
    c, n, j, _ = p.shape
    if groups.shape != (n,) or ids.shape != (n,) or assignment.shape != (n,):
        raise ValueError('groups/pair_ids/split must align with items')
    if (not isinstance(judges, list) or not isinstance(contexts, list)
            or any(not isinstance(v, str) for v in judges + contexts)
            or len(judges) != j or len(contexts) != c or j < 2 or c < 1 or n < 1):
        raise ValueError('Endpoint/context names do not match tensor axes')
    if any(not str(v).strip() for v in [*groups, *ids, *judges, *contexts]):
        raise ValueError('Group, pair, endpoint and context identifiers must be nonempty')
    if len(set(judges)) != j or len(set(contexts)) != c or len(set(ids)) != n:
        raise ValueError('Duplicate endpoint/context/pair IDs')
    if not np.isfinite(p).all() or p.min() < 0 or not np.allclose(p.sum(-1), 1, atol=1e-6, rtol=0):
        raise ValueError('Response cache contains invalid probabilities; no imputation allowed')
    if set(assignment) != {'FIT', 'CERT', 'TEST'}:
        raise ValueError('Exactly three nonempty splits FIT/CERT/TEST are required')
    for group in np.unique(groups):
        if len(set(assignment[groups == group])) != 1:
            raise ValueError(f'Source group {group!r} leaks across data splits')
    if model_ids is not None and len(model_ids) != j:
        raise ValueError('model_ids do not align with endpoints')
    if gold is not None and (gold.shape != (n,) or not np.isin(gold, [0, 1, 2]).all()):
        raise ValueError('gold must have one canonical three-class label per item')
    metadata = dict(metadata, input_path=str(path.resolve()), input_sha256=sha256_file(path),
                    split_seed=metadata.get('split_seed') if stored_split else int(seed),
                    requested_split_seed=int(seed),
                    split_source='stored' if stored_split else 'grouped_random_50_25_25',
                    model_ids=model_ids)
    return Dataset(
        splits={s: JudgeResponses([b[assignment == s] for b in p], contexts) for s in ['FIT', 'CERT', 'TEST']},
        groups={s: groups[assignment == s] for s in ['FIT', 'CERT', 'TEST']},
        judge_ids=judges, context_names=contexts, metadata=metadata,
        gold={} if gold is None else {s: gold[assignment == s] for s in ['FIT', 'CERT', 'TEST']})


def demo_dataset(seed=20260818):
    rng = np.random.default_rng(seed)
    splits, groups, gold = {}, {}, {}
    for split in ['FIT', 'CERT', 'TEST']:
        # Six endpoints, including two almost-duplicate endpoints. Synthetic only.
        latent = rng.dirichlet([.6, .6, .6], size=(24, 3))
        blocks = []
        for c in range(2):
            endpoints = np.stack([latent[:, 0], latent[:, 1], latent[:, 2],
                                  .98 * latent[:, 0] + .02 * latent[:, 1],
                                  .65 * latent[:, 0] + .35 * latent[:, 2],
                                  .1 * latent[:, 0] + .9 * latent[:, 1]], axis=1)
            endpoints = .98 * endpoints + .02 * rng.dirichlet([1, 1, 1], size=(24, 6))
            blocks.append(endpoints)
        splits[split] = JudgeResponses(blocks, ['clean', 'matched_context'])
        groups[split] = np.asarray([f'{split}_{i // 2}' for i in range(24)])
        gold[split] = blocks[0][:, 0].argmax(-1)
    return Dataset(splits, groups, [f'D{i}' for i in range(6)], ['clean', 'matched_context'],
                   {'synthetic': True, 'demo': True, 'seed': seed,
                    'model_ids': [f'synthetic/{i}' for i in range(6)],
                    'warning': 'Software verification only; not evidence about the paper.'}, gold)


def add_data_arguments(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=Path, help='Complete aligned NPZ response cache')
    group.add_argument('--demo', action='store_true', help='Synthetic software check; never a paper result')
    parser.add_argument('--seed', type=int, default=20260818)


def load_dataset(args):
    return demo_dataset(args.seed) if args.demo else load_npz(args.input, args.seed)


def select_subset(fit, k, method='backward', seed=0):
    """Fixed-cardinality search. Every objective evaluation is on FIT only.

    swap1/swap2 are fixed-cardinality exchanges, capped at four accepted rounds;
    they are NOT the original paper's cardinality-reducing exchange algorithm.

    kmedoids uses 25 random restarts and alternating cluster medoids. Each
    medoid owns itself when distances tie, preserving exactly k distinct
    endpoints even when responses are duplicated. It remains a heuristic.
    """
    n = fit.n_judges
    if not 1 <= k <= n: raise ValueError(f'k must be in [1,{n}]')
    if method not in {'qr', 'kmedoids', 'random', 'backward', 'forward', 'swap1', 'swap2'}:
        raise ValueError(method)
    if k == n: return list(range(n))
    if method == 'qr': return select_pivoted_qr(fit)[k]
    if method == 'kmedoids':
        from epcrc.selection_baselines import pairwise_tv
        distance = pairwise_tv(fit)
        rng = np.random.default_rng(seed)
        best, best_cost = None, np.inf
        for _ in range(25):
            medoids = rng.choice(n, size=k, replace=False).tolist()
            for _ in range(100):
                assignment = distance[:, medoids].argmin(axis=1)
                # This changes only zero-distance ties and prevents an empty
                # cluster's old medoid becoming another cluster's new medoid.
                assignment[medoids] = np.arange(k)
                moved = []
                for cluster in range(k):
                    members = np.flatnonzero(assignment == cluster)
                    costs = distance[np.ix_(members, members)].sum(axis=1)
                    moved.append(int(members[np.argmin(costs)]))
                if sorted(moved) == sorted(medoids):
                    break
                medoids = moved
            cost = float(distance[:, medoids].min(axis=1).sum())
            candidate = sorted(medoids)
            if cost < best_cost or (cost == best_cost and (best is None or candidate < best)):
                best, best_cost = candidate, cost
        return best
    if method == 'random': return sorted(np.random.default_rng(seed).choice(n, k, replace=False).tolist())
    cov = JudgeCoverageFunctional(fit, fit)
    def score(s): return cov.compute_coverage(set(s))[0]
    if method == 'forward':
        kept = set()
        while len(kept) < k:
            kept.add(min((j for j in range(n) if j not in kept), key=lambda j: (score(kept | {j}), j)))
    else:
        kept = set(range(n))
        while len(kept) > k:
            kept.remove(min(kept, key=lambda j: (score(kept - {j}), j)))
    if method.startswith('swap'):
        width = int(method[-1])
        for _ in range(4):
            current, best = score(kept), kept
            for r in range(1, min(width, len(kept), n - len(kept)) + 1):
                for drop in combinations(sorted(kept), r):
                    for add in combinations(sorted(set(range(n)) - kept), r):
                        candidate = (kept - set(drop)) | set(add)
                        value = score(candidate)
                        if value < current - 1e-10:
                            current, best = value, candidate
            if best == kept: break
            kept = best
    return sorted(kept)


def predict_convex(fit, evaluate, kept):
    kept = list(kept)
    if not kept or len(kept) != len(set(kept)):
        raise ValueError('kept must contain distinct retained endpoints')
    kept = sorted(kept)  # Certificate weights use increasing endpoint order.
    cov = JudgeCoverageFunctional(fit, fit)
    weights = np.zeros((fit.n_judges, len(kept)))
    for j in range(fit.n_judges):
        weights[j] = cov.compute_certificate(j, set(kept)).weights
    predictions = [np.einsum('nkd,jk->njd', b[:, kept], weights) for b in evaluate.blocks]
    return predictions, weights
