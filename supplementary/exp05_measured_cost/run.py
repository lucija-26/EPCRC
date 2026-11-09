"""Cost/fidelity frontiers using measured endpoint service work, not parameter counts."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from epcrc.judge import JudgeCoverageFunctional, total_variation
from supplementary.common import add_data_arguments, load_dataset, save_json, select_subset, sha256_file


def context_distribution(context_names, weights=None):
    if weights is None:
        return np.full(len(context_names), 1 / len(context_names))
    if set(weights) != set(context_names):
        raise ValueError('Context weights must name every cached context exactly once')
    values = np.asarray([weights[c] for c in context_names], dtype=float)
    if not np.isfinite(values).all() or values.min() < 0 or not np.isclose(values.sum(), 1, atol=1e-8):
        raise ValueError('Context weights must be finite nonnegative probabilities summing to one')
    return values


def read_costs(cache, data, context_weights=None):
    """Mean warm serialized service seconds per item, averaged over declared contexts.

    A batch's seconds/items is work per item, NOT request latency. Models must have
    the same ordered benchmark workload. Cross-node comparability is recorded.
    """
    costs, details, hosts, hardware = [], [], set(), set()
    mixture = context_distribution(data.context_names, context_weights)
    device_signatures = set()
    reference_ids, common_batch_size, reference_protocol = None, None, None
    prompts_by_context = {}
    one_pass_service_seconds, repeated_service_seconds, warmup_service_seconds = 0., 0., 0.
    expected_items = sum(len(g) for g in data.groups.values())
    for endpoint in data.judge_ids:
        context_costs = []
        for context in data.context_names:
            path = Path(cache) / f'{endpoint}__{context}.json'
            block = json.loads(path.read_text())
            if block.get('synthetic'): raise ValueError('Synthetic timing is not measured GPU cost')
            if block['judge_id'] != endpoint or block['context'] != context: raise ValueError('Cache axes mismatch')
            model_ids = data.metadata.get('model_ids')
            if model_ids and block['model_id'] != model_ids[data.judge_ids.index(endpoint)]:
                raise ValueError('Timing model does not match response cache')
            inventory = data.metadata.get('cache_inventory', [])
            matches = [entry for entry in inventory if Path(entry['path']).name == path.name]
            known = matches[0] if len(matches) == 1 else None
            if known is None or known['sha256'] != sha256_file(path):
                raise ValueError('Timings must come from exact response-cache inventory; merge these blocks first')
            if reference_ids is None: reference_ids = block['pair_ids']
            if reference_ids != block['pair_ids']: raise ValueError('Endpoints timed on different input orders')
            protocol_keys = ('pairs_sha256', 'interface', 'dtype', 'max_tokens', 'label_prefix',
                             'scoring_source_sha256', 'original_scoring_sha256', 'prompt_source_sha256')
            protocol = block.get('protocol', {})
            signature_protocol = tuple(protocol.get(key) for key in protocol_keys)
            if any(value is None for value in signature_protocol):
                raise ValueError('Missing comparable scoring protocol provenance')
            if reference_protocol is None: reference_protocol = signature_protocol
            if signature_protocol != reference_protocol:
                raise ValueError('Timing endpoints used different scoring workload/settings')
            if len(reference_ids) != expected_items or len(set(reference_ids)) != expected_items:
                raise ValueError('Timing workload must cover the response tensor exactly once per repeat')
            prompt_hashes = block.get('prompt_sha256')
            if not isinstance(prompt_hashes, list) or len(prompt_hashes) != expected_items:
                raise ValueError('Missing per-item input fingerprints')
            if context in prompts_by_context and prompts_by_context[context] != prompt_hashes:
                raise ValueError('Endpoints received different inputs within the same context')
            prompts_by_context[context] = prompt_hashes
            environment = block['environment']
            if not environment['device'].startswith('cuda'):
                raise ValueError('GPU cost experiment requires measurements on a CUDA device')
            device_name = environment.get('selected_device_name')
            device_memory = environment.get('selected_device_total_memory_bytes')
            if not device_name or not device_memory:
                raise ValueError('Missing selected-device hardware identity; recollect using the current collector')
            signature = (device_name, int(device_memory), environment.get('torch'), environment.get('transformers'))
            device_signatures.add(signature)
            if len(device_signatures) > 1:
                raise ValueError('Use the same selected GPU hardware and torch/transformers versions for comparable costs')
            timing = block['timing']
            if common_batch_size is None: common_batch_size = timing['batch_size']
            if common_batch_size != timing['batch_size']: raise ValueError('Use the same batch size for all endpoints')
            measurements = timing['batches']
            if not measurements or any(not np.isfinite(m['seconds']) or m['seconds'] <= 0 or m['n_items'] <= 0 for m in measurements):
                raise ValueError('Missing or invalid timing measurements')
            batch_size, n_repeats = timing['batch_size'], timing['repeats']
            if batch_size < 1 or n_repeats < 1:
                raise ValueError('Invalid batch size or repetition count')
            expected_ranges = [(r, s, min(s + batch_size, expected_items))
                               for r in range(n_repeats) for s in range(0, expected_items, batch_size)]
            measured_ranges = [(m['repeat'], m['start'], m['end']) for m in measurements]
            if measured_ranges != expected_ranges or any(m['n_items'] != m['end'] - m['start'] for m in measurements):
                raise ValueError('Timing batches must cover the same complete ordered workload each repeat')
            value = sum(m['seconds'] for m in measurements) / sum(m['n_items'] for m in measurements)
            context_costs.append(value)
            one_pass_service_seconds += value * expected_items
            repeated_service_seconds += sum(m['seconds'] for m in measurements)
            warmup = float(timing['warmup_seconds'])
            if not np.isfinite(warmup) or warmup < 0:
                raise ValueError('Invalid warmup timing')
            warmup_service_seconds += warmup
            hosts.add(block['environment']['hostname']); hardware.add(block['environment']['hardware'])
            details.append({'judge_id': endpoint, 'context': context, 'seconds_per_item': value,
                            'batch_seconds_p50': float(np.median([m['seconds'] for m in measurements])),
                            'batch_seconds_p95': float(np.quantile([m['seconds'] for m in measurements], .95)),
                            'repeats': timing['repeats'], 'batch_size': timing['batch_size'],
                            'load_seconds': timing['load_seconds'], 'warmup_seconds': timing['warmup_seconds'],
                            'hardware': block['environment']['hardware'], 'device': block['environment']['device'],
                            'selected_device_name': device_name, 'selected_device_total_memory_bytes': device_memory,
                            'source_sha256': sha256_file(path)})
        costs.append(float(np.dot(mixture, context_costs)))
    return np.asarray(costs), {'per_endpoint_context': details, 'hosts': sorted(hosts),
                               'hardware_strings': sorted(hardware), 'batch_size': common_batch_size,
                               'cross_host_measurements': len(hosts) > 1,
                               'context_distribution': dict(zip(data.context_names, mixture.tolist())),
                               'one_full_cache_collection_service_seconds_estimate': one_pass_service_seconds,
                               'all_recorded_repeats_service_seconds': repeated_service_seconds,
                               'all_recorded_warmups_service_seconds': warmup_service_seconds,
                               'warning': 'Serialized tokenization + inference + CPU transfer service work, not pure GPU kernel time, online p95 latency, queueing, parallel makespan, or energy. Cross-host CPU contention remains a confound.'}


def cost_aware_subset(fit, k, costs):
    """FIT-only greedy error increase per unit of saved measured endpoint work."""
    cov = JudgeCoverageFunctional(fit, fit)
    kept = set(range(fit.n_judges))
    while len(kept) > k:
        current = cov.compute_coverage(kept)[0]
        drop = min(kept, key=lambda j: ((cov.compute_coverage(kept - {j})[0] - current) / costs[j], j))
        kept.remove(drop)
    return sorted(kept)


def timed_reconstruction(blocks, kept, weights, repeats, mixture=None):
    # The input endpoint probabilities already exist; this measures ONLY mixing.
    if len(kept) == weights.shape[0]:
        return 0.  # Identity panel needs no replacement computation.
    for b in blocks: np.einsum('nkd,jk->njd', b[:, kept], weights)
    durations = []
    for b in blocks:
        context_durations = []
        for _ in range(repeats):
            tick = time.perf_counter()
            np.einsum('nkd,jk->njd', b[:, kept], weights)
            context_durations.append(time.perf_counter() - tick)
        durations.append(float(np.median(context_durations) / len(b)))
    return float(np.dot(mixture if mixture is not None else np.full(len(blocks), 1 / len(blocks)), durations))


def run(data, costs, ks, methods, gamma=.20, repeats=20, gpu_hour_rate=None, requests=None,
        context_weights=None, collection_service_seconds=None, recorded_audit_service_seconds=None):
    costs = np.asarray(costs, dtype=float)
    if len(costs) != len(data.judge_ids) or not np.isfinite(costs).all() or min(costs) <= 0:
        raise ValueError('One finite positive measured cost per endpoint required')
    if repeats < 1 or any(k < 1 or k > len(costs) for k in ks): raise ValueError('Invalid repeats or budget')
    if not ks or not methods or len(set(ks)) != len(ks) or len(set(methods)) != len(methods):
        raise ValueError('Provide nonempty distinct k values and methods')
    if not np.isfinite(gamma) or not 0 <= gamma <= 1: raise ValueError('gamma must be in [0, 1]')
    if requests is not None and requests < 0: raise ValueError('requests must be nonnegative')
    if gpu_hour_rate is not None and (not np.isfinite(gpu_hour_rate) or gpu_hour_rate < 0): raise ValueError('Invalid hourly rate')
    mixture = context_distribution(data.context_names, context_weights)
    if collection_service_seconds is None:
        if context_weights is not None:
            raise ValueError('Nonuniform context mixture requires the actual unweighted collection service total')
        collection_service_seconds = sum(len(g) for g in data.groups.values()) * len(data.context_names) * float(sum(costs))
    if not np.isfinite(collection_service_seconds) or collection_service_seconds < 0:
        raise ValueError('Invalid one-pass collection service cost')
    if recorded_audit_service_seconds is not None and (not np.isfinite(recorded_audit_service_seconds) or recorded_audit_service_seconds < 0):
        raise ValueError('Invalid recorded audit service cost')
    fit, test = data.splits['FIT'], data.splits['TEST']
    all_cost = float(sum(costs))
    rows = []
    for method in methods:
        for k in ks:
            tick = time.perf_counter()
            if method == 'cost_aware': kept = cost_aware_subset(fit, k, costs)
            elif method == 'cheapest': kept = sorted(np.argsort(costs)[:k].tolist())
            else: kept = sorted(select_subset(fit, k, method=method))
            fit_cov = JudgeCoverageFunctional(fit, fit)
            weights = np.stack([fit_cov.compute_certificate(j, set(kept)).weights for j in range(fit.n_judges)])
            cpu_fit_seconds = time.perf_counter() - tick
            pred = [np.einsum('nkd,jk->njd', b[:, kept], weights) for b in test.blocks]
            losses = np.asarray([total_variation(b, p) for b, p in zip(test.blocks, pred)])
            cert_pred = [np.einsum('nkd,jk->njd', b[:, kept], weights) for b in data.splits['CERT'].blocks]
            cert_tv = max(float(total_variation(b, p).mean(0).max()) for b, p in zip(data.splits['CERT'].blocks, cert_pred))
            removed = sorted(set(range(len(costs))) - set(kept))
            mixing = timed_reconstruction(test.blocks, kept, weights, repeats, mixture)
            if removed:
                retained_cost = float(costs[kept].sum())
                saved = all_cost - retained_cost
            else:
                # Different summation orders must not invent savings for identity.
                retained_cost, saved = all_cost, 0.
            row = {'method': method, 'k': k, 'kept_indices': kept, 'kept_ids': [data.judge_ids[i] for i in kept],
                   'test_worst_endpoint_context_tv': float(losses.mean(1).max()), 'cert_worst_endpoint_context_tv': cert_tv,
                   'cert_within_gamma': cert_tv <= gamma, 'gamma': gamma,
                   'removed_verdict_agreement': float(np.mean([np.mean(b[:, removed].argmax(-1) == p[:, removed].argmax(-1))
                                                              for b, p in zip(test.blocks, pred)])) if removed else 1.,
                   'full_endpoint_service_seconds_per_item': all_cost,
                   'retained_endpoint_service_seconds_per_item': retained_cost,
                   'endpoint_service_work_saving_fraction': saved / all_cost,
                   'reconstruction_cpu_seconds_per_item': mixing,
                   'serial_warm_time_saving_fraction_estimate': (saved - mixing) / all_cost,
                   'selection_and_fit_cpu_seconds': cpu_fit_seconds,
                   'cost_kind': 'synthetic_demo_cost' if data.metadata.get('synthetic') else 'measured_endpoint_time_accounting',
                   'latency_scope': 'Additive endpoint service times plus measured CPU mixing; not observed online full-panel latency.'}
            # Existing audit tensors are an upfront cost, not free deployment information.
            row['one_full_cache_collection_service_seconds_estimate'] = collection_service_seconds
            row['service_work_break_even_requests_estimate'] = collection_service_seconds / saved if saved > 0 else None
            row['serial_time_break_even_requests_estimate'] = ((collection_service_seconds + cpu_fit_seconds) / (saved - mixing)
                                                              if saved > mixing else None)
            if recorded_audit_service_seconds is not None:
                row['recorded_audit_service_seconds_excluding_loads'] = recorded_audit_service_seconds
                row['recorded_audit_service_break_even_requests_lower_bound'] = recorded_audit_service_seconds / saved if saved > 0 else None
            if gpu_hour_rate is not None:
                # This is a user-supplied rate for measured single-GPU endpoint work.
                row['user_supplied_gpu_hour_rate'] = gpu_hour_rate
                row['estimated_gpu_cost_saved_per_item'] = saved * gpu_hour_rate / 3600
                if requests is not None:
                    row['estimated_gpu_cost_net_after_collection'] = (requests * saved - collection_service_seconds) * gpu_hour_rate / 3600
            rows.append(row)
    # Budget choice uses CERT, never the best-looking TEST row.
    selected = {}
    for method in methods:
        candidates = [r for r in rows if r['method'] == method and r['cert_within_gamma']]
        selected[method] = min(candidates, key=lambda r: r['retained_endpoint_service_seconds_per_item']) if candidates else None
    return {'metadata': data.metadata, 'rows': rows, 'cert_selected_operating_points': selected,
            'context_distribution': dict(zip(data.context_names, mixture.tolist())),
            'pricing_assumption': 'Optional rate applies to one exclusively allocated single-GPU worker during measured serialized service, including host preprocessing. This is amortized worker-time accounting, not an observed cloud invoice or API price.',
            'limitations': ['No energy or electricity-price claims without energy measurements.',
                            'Measured service includes tokenization and host transfers; it is not pure GPU kernel work.',
                            'CPU selection time is reported separately; financial net does not price CPU mixing or fitting.',
                            'Cold loads, network/queueing, concurrent serving and nonlinear gate overhead require a separate live service benchmark.',
                            'Collection estimate excludes repeated timing runs, model loading and downloads; actual audit bill can be higher.']}


def main():
    p = argparse.ArgumentParser(description=__doc__); add_data_arguments(p)
    p.add_argument('--timing-cache', type=Path); p.add_argument('--ks', type=int, nargs='+', default=[2, 4, 6])
    p.add_argument('--methods', nargs='+', choices=['backward', 'qr', 'cheapest', 'cost_aware'], default=['backward', 'qr', 'cheapest', 'cost_aware'])
    p.add_argument('--gamma', type=float, default=.20); p.add_argument('--timing-repeats', type=int, default=20)
    p.add_argument('--gpu-hour-rate', type=float, help='User-specified currency units per single-GPU hour; never guessed')
    p.add_argument('--context-weights', type=Path, help='JSON map of context names to predeclared request probabilities summing to one')
    p.add_argument('--requests', type=int); p.add_argument('--output', type=Path, default=Path('results/supplementary/exp05_cost.json'))
    args = p.parse_args(); data = load_dataset(args)
    context_weights = json.loads(args.context_weights.read_text()) if args.context_weights else None
    if args.demo:
        costs = np.linspace(.01, .06, len(data.judge_ids))
        timings = {'synthetic': True, 'warning': 'Invented costs for software tests, NOT GPU measurements.',
                   'one_full_cache_collection_service_seconds_estimate': sum(len(g) for g in data.groups.values()) * len(data.context_names) * float(costs.sum())}
    else:
        if not args.timing_cache: p.error('--timing-cache is required for real data')
        costs, timings = read_costs(args.timing_cache, data, context_weights)
    if args.gpu_hour_rate is not None and (not np.isfinite(args.gpu_hour_rate) or args.gpu_hour_rate < 0): p.error('Invalid hourly rate')
    recorded_service = (timings['all_recorded_repeats_service_seconds'] + timings['all_recorded_warmups_service_seconds']
                        if not args.demo else None)
    result = run(data, costs, args.ks, args.methods, args.gamma, args.timing_repeats, args.gpu_hour_rate, args.requests,
                 context_weights, timings.get('one_full_cache_collection_service_seconds_estimate'), recorded_service)
    result['timing_provenance'] = timings
    save_json(args.output, result)
    fields = [k for k, v in result['rows'][0].items() if not isinstance(v, list)]
    with args.output.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(result['rows'])
    print(args.output)


if __name__ == '__main__': main()
