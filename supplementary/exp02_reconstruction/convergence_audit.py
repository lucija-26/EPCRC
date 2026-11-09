"""Repeat exp02 with maxiter=1000 and every previously frozen core20 support.

Only the gate optimizer's iteration budget changes. Source exp02 stages must
be complete and hash-verified. A separate output directory is required; this
process never edits the original reports or the existing reconstruction code.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from supplementary import common
from supplementary.exp02_reconstruction import run as reconstruction

BUDGETS = [2, 4, 10, 16, 17, 18, 19, 20]
MAXITER = 1000
NON_GATE = ('global_convex', 'context_convex', 'temperature_convex')
STAGE = 'exp02_reconstruction'


def read_snapshot(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def require_equal(actual, expected, message):
    if actual != expected:
        raise ValueError(message)


def flag_values(command, flag):
    if command.count(flag) != 1:
        raise ValueError(f'Source command must contain exactly one {flag}')
    rest = command[command.index(flag) + 1:]
    stop = next((i for i, token in enumerate(rest) if token.startswith('--')), len(rest))
    return rest[:stop]


def prepare_sources(input_path, source_runs):
    input_path = Path(input_path).resolve()
    input_hash = common.sha256_file(input_path)
    if len(source_runs) != 2 or len({Path(p).resolve() for p in source_runs}) != 2:
        raise ValueError('Supply the two distinct completed main/core20_k17 source runs')
    source_records, frozen, first = [], {}, None
    for run_dir in source_runs:
        run_dir = Path(run_dir).resolve()
        directory = run_dir / STAGE
        status_path, report_path = run_dir / 'status.json', directory / 'results.json'
        status, status_hash = read_snapshot(status_path)
        stage = status.get('stages', {}).get(STAGE, {})
        if stage.get('status') != 'complete':
            raise ValueError(f'Source exp02 stage is not complete: {run_dir}')
        report, report_hash = read_snapshot(report_path)
        require_equal(stage.get('artifact_sha256'), report_hash, 'Completed source report SHA256 mismatch')
        require_equal(report.get('experiment'), STAGE, 'Expected an exp02 reconstruction report')
        require_equal(report.get('metadata', {}).get('input_sha256'), input_hash, 'Source response SHA256 mismatch')
        signature = stage.get('code_signature', {})
        require_equal(signature.get('merged_sha256'), input_hash, 'Source stage response SHA256 mismatch')
        require_equal(signature.get('script_sha256'), common.sha256_file(reconstruction.__file__),
                      'Original reconstruction source changed since source analysis')
        require_equal(signature.get('common_sha256'), common.sha256_file(common.__file__),
                      'Original common source changed since source analysis')
        command = stage.get('command', [])
        declared = [int(k) for k in flag_values(command, '--budgets')]
        if len(set(declared)) != len(declared):
            raise ValueError('Duplicate source command budgets')
        require_equal([row['k'] for row in report.get('budgets', [])], sorted(declared),
                      'Source report does not contain its complete declared budget list')
        require_equal(flag_values(command, '--seed'), [str(report['seed'])], 'Source seed differs from command')
        require_equal(Path(flag_values(command, '--output')[0]).resolve(), directory,
                      'Source command output is not the declared report directory')
        source_input = Path(flag_values(command, '--input')[0]).resolve()
        require_equal(common.sha256_file(source_input), input_hash, 'Source command response file differs')
        require_equal(flag_values(command, '--bootstrap'), [str(report['configuration']['bootstrap'])],
                      'Source bootstrap count differs from command')
        if report.get('selection_method') not in {'backward', 'pivoted_qr'}:
            raise ValueError('Unknown source selection method')
        if type(report['seed']) is not int or not 0 < report['configuration']['maxiter'] < MAXITER:
            raise ValueError('Need an integer seed and a smaller positive original iteration budget')
        if first is None:
            first = report
        for key in ('judge_ids', 'contexts', 'seed', 'selection_method', 'configuration', 'gammas'):
            require_equal(report.get(key), first.get(key), f'Source reports disagree on {key}')
        for field in ('cert_selected_budget_at_gamma', 'observed_fit_budget_at_gamma'):
            if len(report.get(field, [])) != len(report['gammas']) * len(reconstruction.METHODS):
                raise ValueError('Source report is missing final budget-selection summaries')
        record = {'run_dir': str(run_dir), 'report': {'path': str(report_path), 'sha256': report_hash},
                  'status_snapshot': {'path': str(status_path), 'sha256': status_hash},
                  'source_input': {'path': str(source_input), 'sha256': input_hash},
                  'completed_stage': stage, 'budgets': sorted(declared), 'artifacts': []}
        for row in report['budgets']:
            k, kept = row['k'], row['retained_indices']
            if (k in frozen or k not in BUDGETS or len(kept) != k or len(set(kept)) != k
                    or any(type(j) is not int or not 0 <= j < 20 for j in kept)):
                raise ValueError('Source budgets must be disjoint with valid frozen retained sets')
            require_equal(row.get('retained_judge_ids'), [report['judge_ids'][j] for j in kept],
                          'Frozen retained IDs/order disagree')
            require_equal(set(row.get('methods', {})), set(reconstruction.METHODS),
                          'Source report lacks a reconstruction method')
            artifacts = {}
            for label, field, expected in [('maps', 'fitted_maps_file', f'k{k}_maps.json'),
                                           ('losses', 'test_losses_file', f'k{k}_test_losses.npz')]:
                require_equal(row[field], expected, 'Unexpected source artifact path')
                path = directory / expected
                artifacts[label] = path
                record['artifacts'].append({'path': str(path), 'sha256': common.sha256_file(path)})
            frozen[k] = {'row': row, 'source_report': str(report_path), **artifacts}
        source_records.append(record)
    require_equal(sorted(frozen), BUDGETS, 'Sources must cover all eight predeclared budgets, including k=17')
    data = common.load_npz(input_path, seed=first['seed'])
    require_equal(len(data.judge_ids), 20, 'This audit requires the full twenty-endpoint panel')
    require_equal(data.judge_ids, first['judge_ids'], 'Input model order differs from source reports')
    require_equal(data.context_names, first['contexts'], 'Input contexts differ from source reports')
    for record in source_records:
        report = json.loads(Path(record['report']['path']).read_text())
        for field in ('model_ids', 'manifest', 'collection_protocol', 'split_source', 'split_seed',
                      'requested_split_seed', 'split_manifest_sha256'):
            require_equal(report['metadata'].get(field), data.metadata.get(field),
                          f'Source metadata differs from response input: {field}')
    return {'input': input_path, 'input_sha256': input_hash, 'data': data, 'sources': source_records,
            'frozen': frozen, 'reference_report': first}


@contextmanager
def replay_selector(prepared):
    """Process-local replacement, restored on success and every exception."""
    original = common.select_subset
    reference = prepared['reference_report']
    expected_method = 'qr' if reference['selection_method'] == 'pivoted_qr' else reference['selection_method']
    fit_reference = prepared['data'].splits['FIT']
    validated_fit = None

    def select(fit, k, method='backward', seed=0):
        nonlocal validated_fit
        if method != expected_method or seed != reference['seed'] or k not in prepared['frozen']:
            raise ValueError('Replay request differs from frozen method, seed or budget')
        if fit is not validated_fit:
            if (fit.context_names != fit_reference.context_names or len(fit.blocks) != len(fit_reference.blocks)
                    or any(not np.array_equal(a, b) for a, b in zip(fit.blocks, fit_reference.blocks))):
                raise ValueError('Replay is only valid for the identical frozen FIT data')
            validated_fit = fit
        return list(prepared['frozen'][k]['row']['retained_indices'])

    common.select_subset = select
    try:
        yield
    finally:
        common.select_subset = original


def assert_numeric_tree(actual, expected, label):
    if isinstance(expected, dict):
        # In-memory diagnostics use integer endpoint keys; JSON stores strings.
        actual = {str(key): value for key, value in actual.items()}
        expected = {str(key): value for key, value in expected.items()}
        require_equal(set(actual), set(expected), f'{label}: keys changed')
        for key in expected:
            assert_numeric_tree(actual[key], expected[key], f'{label}/{key}')
    elif isinstance(expected, (list, tuple)):
        require_equal(len(actual), len(expected), f'{label}: length changed')
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_numeric_tree(a, b, f'{label}/{i}')
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
            raise ValueError(f'{label}: non-gate numeric result changed')
    else:
        require_equal(actual, expected, f'{label}: non-gate result changed')


def verify_non_gate(prepared, result, output):
    require_equal([r['k'] for r in result['budgets']], BUDGETS, 'Audit omitted a frozen budget')
    largest_loss_difference = 0.
    for row in result['budgets']:
        source = prepared['frozen'][row['k']]
        old = source['row']
        require_equal(row['retained_indices'], old['retained_indices'], 'Audit support changed')
        for method in NON_GATE:
            for field in ('test', 'cert', 'fit', 'fit_diagnostics', 'paired_difference_vs_global'):
                assert_numeric_tree(row['methods'][method][field], old['methods'][method][field],
                                    f"k{row['k']}/{method}/{field}")
        new_maps = json.loads((output / row['fitted_maps_file']).read_text())
        old_maps = json.loads(source['maps'].read_text())
        for method in NON_GATE:
            assert_numeric_tree(new_maps[method], old_maps[method], f"k{row['k']}/{method}/maps")
        with np.load(output / row['test_losses_file'], allow_pickle=False) as new_losses, \
                np.load(source['losses'], allow_pickle=False) as old_losses:
            require_equal(set(new_losses.files), set(old_losses.files), 'Loss archive keys changed')
            if not np.array_equal(new_losses['test_base_item_groups'], old_losses['test_base_item_groups']):
                raise ValueError('TEST group order changed')
            for method in NON_GATE:
                for c in range(len(result['contexts'])):
                    key = f'{method}_tv_context_{c}'
                    a, b = new_losses[key], old_losses[key]
                    if a.shape != b.shape or not np.allclose(a, b, rtol=1e-10, atol=1e-12):
                        raise ValueError(f'Non-gate TEST losses changed: {key}')
                    largest_loss_difference = max(largest_loss_difference, float(np.max(np.abs(a - b))))
        row['replayed_support_lookup_seconds'] = row.pop('selection_seconds')
        row['source_selection_seconds'] = old['selection_seconds']
        row['source_selection_report'] = source['source_report']
        row['replayed_frozen_supports'] = True
    return {'passed': True, 'methods': list(NON_GATE), 'rtol': 1e-10, 'atol': 1e-12,
            'largest_non_gate_test_loss_difference': largest_loss_difference,
            'scope': 'FIT/CERT/TEST metrics, paired intervals, fitted maps and every TEST loss; timing excluded.'}


def audit(input_path, source_runs, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Audit requires a new output directory: {output}')
    prepared = prepare_sources(input_path, source_runs)
    original = prepared['reference_report']
    configuration = original['configuration']
    args = SimpleNamespace(input=prepared['input'], demo=False, output=output, seed=original['seed'],
                           budgets=BUDGETS, selection=original['selection_method'],
                           regularizations=configuration['regularizations'], temperatures=configuration['temperatures'],
                           maxiter=MAXITER, bootstrap=configuration['bootstrap'], gammas=original['gammas'])
    provenance = {'replayed_frozen_supports': True, 'budgets': BUDGETS,
                  'input': {'path': str(prepared['input']), 'sha256': prepared['input_sha256']},
                  'sources': prepared['sources'], 'source_configuration': configuration,
                  'only_optimizer_change': {'maxiter': {'before': configuration['maxiter'], 'after': MAXITER}},
                  'selection_timing_scope': 'Support lookup is not a new cold-start selection. Original per-budget selection times are retained separately.',
                  'evaluation_scope': 'All frozen supports and hyperparameter candidates are rerun uniformly; CERT still chooses gate regularization. TEST does not select supports, candidates or iteration budgets.',
                  'source_files': {str(Path(path).resolve()): common.sha256_file(path)
                                   for path in [__file__, reconstruction.__file__, common.__file__, ROOT / 'epcrc/judge.py']}}
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / 'audit_status.json'
    common.save_json(output / 'audit_protocol.json', provenance)
    common.save_json(status_path, {'status': 'running', 'replayed_frozen_supports': True})
    try:
        with replay_selector(prepared):
            result = reconstruction.run(args)
        verification = verify_non_gate(prepared, result, output)
        for record in prepared['sources']:
            for item in [record['report'], record['source_input'], *record['artifacts']]:
                require_equal(common.sha256_file(item['path']), item['sha256'], 'A frozen source changed during audit')
            status = json.loads(Path(record['status_snapshot']['path']).read_text())
            stage = status.get('stages', {}).get(STAGE, {})
            require_equal(stage.get('status'), 'complete', 'Source exp02 stage no longer complete')
            require_equal(stage.get('artifact_sha256'), record['report']['sha256'], 'Source stage result binding changed')
        require_equal(common.sha256_file(prepared['input']), prepared['input_sha256'], 'Response input changed during audit')
        for path, digest in provenance['source_files'].items():
            require_equal(common.sha256_file(path), digest, 'Audit or fitting source changed during execution')
        result.update(replayed_frozen_supports=True, optimization_audit=provenance,
                      non_gate_consistency_verification=verification)
        common.save_json(output / 'results.json', result)
        common.save_json(status_path, {'status': 'complete', 'replayed_frozen_supports': True,
                                      'result_sha256': common.sha256_file(output / 'results.json'),
                                      'non_gate_consistency_verification': verification})
        return result
    except BaseException as error:
        common.save_json(status_path, {'status': 'failed', 'replayed_frozen_supports': True,
                                      'error': f'{type(error).__name__}: {error}'})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--source-runs', type=Path, nargs=2, required=True,
                        help='Completed core20 and core20_k17 pipeline directories')
    parser.add_argument('--output', type=Path, required=True, help='New reconstruction-result directory')
    args = parser.parse_args()
    result = audit(args.input, args.source_runs, args.output)
    print(f"Completed maxiter={MAXITER} audit of {len(result['budgets'])} frozen supports: {args.output}", flush=True)


if __name__ == '__main__':
    main()
