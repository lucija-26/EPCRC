import numpy as np
import pytest
import json
from types import SimpleNamespace
from epcrc.judge import JudgeResponses
from supplementary.common import demo_dataset
from supplementary.common import sha256_file
from supplementary.exp05_measured_cost.run import run, cost_aware_subset, read_costs


def test_full_panel_is_identity_and_saves_no_endpoint_work():
    data = demo_dataset(4)
    result = run(data, np.arange(1, 7.), [6], ['backward'], repeats=1)
    row = result['rows'][0]
    assert row['test_worst_endpoint_context_tv'] == 0
    assert row['endpoint_service_work_saving_fraction'] == 0
    assert row['service_work_break_even_requests_estimate'] is None
    assert row['serial_warm_time_saving_fraction_estimate'] == 0
    assert row['reconstruction_cpu_seconds_per_item'] == 0


@pytest.mark.parametrize('method', ['backward', 'qr', 'cheapest', 'cost_aware'])
def test_eight_endpoint_identity_has_no_roundoff_savings_or_break_even(method):
    data = demo_dataset(4)
    data.splits = {split: JudgeResponses([np.concatenate([b, b[:, :2]], axis=1)
                                         for b in responses.blocks], responses.context_names)
                   for split, responses in data.splits.items()}
    data.judge_ids += ['D6', 'D7']
    data.metadata['model_ids'] += ['synthetic/6', 'synthetic/7']
    costs = np.array([.01, .02, .03, .04, .05, .06, .07, .08])
    # Eight elements trigger NumPy's different summation order: old code
    # reported a positive 1-ulp saving and a huge, fictitious break-even.
    assert float(sum(costs)) > float(costs.sum())
    row = run(data, costs, [8], [method], repeats=1, gpu_hour_rate=3600,
              requests=1000, recorded_audit_service_seconds=100)['rows'][0]
    assert row['retained_endpoint_service_seconds_per_item'] == row['full_endpoint_service_seconds_per_item']
    assert row['endpoint_service_work_saving_fraction'] == 0
    assert row['serial_warm_time_saving_fraction_estimate'] == 0
    assert row['reconstruction_cpu_seconds_per_item'] == 0
    assert row['estimated_gpu_cost_saved_per_item'] == 0
    assert row['service_work_break_even_requests_estimate'] is None
    assert row['serial_time_break_even_requests_estimate'] is None
    assert row['recorded_audit_service_break_even_requests_lower_bound'] is None


def test_cost_selection_and_rate_use_measured_work():
    data = demo_dataset(4)
    result = run(data, np.arange(1, 7.), [2], ['cheapest'], repeats=1, gpu_hour_rate=3600)
    row = result['rows'][0]
    assert row['kept_indices'] == [0, 1]
    assert row['estimated_gpu_cost_saved_per_item'] == 18
    assert 'cost_kind' in row


def test_invalid_cost_not_silently_replaced_by_parameter_count():
    with pytest.raises(ValueError):
        run(demo_dataset(), [0] * 6, [2], ['cheapest'])


def timing_fixture(tmp_path):
    data = SimpleNamespace(judge_ids=['J0', 'J1'], context_names=['clean', 'stress'],
                           groups={'FIT': ['a', 'b'], 'CERT': ['c'], 'TEST': ['d']},
                           metadata={'model_ids': ['fixture/model0', 'fixture/model1'], 'cache_inventory': []})
    for j, endpoint in enumerate(data.judge_ids):
        for context in data.context_names:
            path = tmp_path / f'{endpoint}__{context}.json'
            seconds_per_item = (2 if context == 'clean' else 6) + 2 * j
            block = {'judge_id': endpoint, 'model_id': data.metadata['model_ids'][j], 'context': context,
                     'synthetic': False, 'pair_ids': ['a', 'b', 'c', 'd'],
                     'prompt_sha256': [f'{context}{x}' for x in range(4)],
                     'protocol': {key: 'fixture' for key in ('pairs_sha256', 'interface', 'dtype', 'max_tokens', 'label_prefix',
                                                           'scoring_source_sha256', 'original_scoring_sha256', 'prompt_source_sha256')},
                     'timing': {'batch_size': 4, 'repeats': 2, 'load_seconds': 1, 'warmup_seconds': 1,
                                'batches': [{'repeat': r, 'start': 0, 'end': 4, 'n_items': 4, 'seconds': seconds_per_item * 4}
                                            for r in range(2)]},
                     'environment': {'hostname': 'unit-test', 'hardware': 'fixture', 'device': 'cuda:0',
                                     'selected_device_name': 'fixtureGPU', 'selected_device_total_memory_bytes': 100,
                                     'torch': 'fixture', 'transformers': 'fixture'}}
            path.write_text(json.dumps(block))
            data.metadata['cache_inventory'].append({'path': str(path), 'sha256': sha256_file(path)})
    return data


def update_fixture(data, path, change):
    block = json.loads(path.read_text())
    change(block)
    path.write_text(json.dumps(block))
    for entry in data.metadata['cache_inventory']:
        if entry['path'] == str(path): entry['sha256'] = sha256_file(path)


def test_context_weighting_does_not_reweight_full_collection_cost(tmp_path):
    data = timing_fixture(tmp_path)
    costs, details = read_costs(tmp_path, data, {'clean': .75, 'stress': .25})
    assert costs == pytest.approx([3, 5])
    assert details['one_full_cache_collection_service_seconds_estimate'] == 80
    assert details['all_recorded_repeats_service_seconds'] == 160


def test_timing_inventory_tampering_is_rejected(tmp_path):
    data = timing_fixture(tmp_path)
    path = tmp_path / 'J0__clean.json'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='inventory'):
        read_costs(tmp_path, data)


@pytest.mark.parametrize('change, message', [
    (lambda b: b['environment'].update(selected_device_name='differentGPU'), 'hardware'),
    (lambda b: b['timing']['batches'].pop(), 'complete ordered workload'),
    (lambda b: b['timing']['batches'][0].update(seconds=float('nan')), 'invalid timing'),
    (lambda b: b['prompt_sha256'].__setitem__(0, 'different-input'), 'different inputs'),
])
def test_incomparable_or_incomplete_measurements_are_rejected(tmp_path, change, message):
    data = timing_fixture(tmp_path)
    update_fixture(data, tmp_path / 'J1__clean.json', change)
    with pytest.raises(ValueError, match=message):
        read_costs(tmp_path, data)


def test_operating_point_does_not_use_test_fidelity():
    data = demo_dataset(4)
    before = run(data, np.arange(1, 7.), [2, 3, 6], ['cheapest'], repeats=1)
    for block in data.splits['TEST'].blocks:
        block[:] = 1 / 3
    after = run(data, np.arange(1, 7.), [2, 3, 6], ['cheapest'], repeats=1)
    a = before['cert_selected_operating_points']['cheapest']
    b = after['cert_selected_operating_points']['cheapest']
    assert a['kept_indices'] == b['kept_indices']
    assert a['cert_worst_endpoint_context_tv'] == b['cert_worst_endpoint_context_tv']
