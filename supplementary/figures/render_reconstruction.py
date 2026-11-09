"""Render frozen convergence-audit outputs only; performs no fitting or resampling."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from supplementary.common import sha256_file
from supplementary.runtime import plot_results
import matplotlib
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path,
                        default=ROOT / 'results/supplementary/real/core20_convergence1000')
    parser.add_argument('--output-dir', type=Path,
                        help='Defaults to <run-dir>/figures; existing artifacts are not overwritten')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = (args.output_dir if args.output_dir is not None else run / 'figures').resolve()
    source_path = run / 'exp02_reconstruction/results.json'
    status_path = run / 'exp02_reconstruction/audit_status.json'
    source = json.loads(source_path.read_text())
    status = json.loads(status_path.read_text())
    source_hash = sha256_file(source_path)
    assert status['status'] == 'complete' and status['result_sha256'] == source_hash
    assert source['configuration']['maxiter'] == 1000 and source['replayed_frozen_supports']
    assert source['metadata']['synthetic'] is False
    assert [row['k'] for row in source['budgets']] == [2, 4, 10, 16, 17, 18, 19, 20]
    chosen = [diag['trials'][diag['selected_trial']] for row in source['budgets']
              for diag in row['methods']['nonlinear_gate']['fit_diagnostics'].values()]
    n_failed = sum(not trial['converged'] for trial in chosen)
    assert (n_failed, len(chosen)) == (19, 54)

    output.mkdir(parents=True, exist_ok=True)
    plot_results.style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 7.0), gridspec_kw={'width_ratios': [1.12, 1]})
    fig.subplots_adjust(left=.075, right=.97, top=.77, bottom=.31, wspace=.31)
    fig.suptitle('Core20 reconstruction: optimizer-budget audit', x=.075, ha='left',
                 y=.965, fontsize=17, fontweight='bold')
    fig.text(.075, .905, 'maxiter = 1000  |  20 models · 7 contexts · TEST: 926 pairs / 463 source groups',
             fontsize=10.5, color='#4D535C')
    fig.text(.075, .86, '19 / 54 selected gate fits remain unconverged; all eight frozen supports are shown.',
             fontsize=10.4, color='#9A394B', fontweight='bold')

    plot_results.reconstruction_plot(axes[0], source)
    axes[0].set_title('(a) All eight measured budgets', loc='left', pad=12)
    handles, labels = axes[0].get_legend_handles_labels()
    labels = ['Temperature-adjusted convex' if label == 'Temperature calibration' else label for label in labels]
    axes[0].legend(handles, labels, loc='upper right', fontsize=8.6)
    zoom = copy.deepcopy(source)
    zoom['budgets'] = [row for row in source['budgets'] if row['k'] >= 16]
    plot_results.reconstruction_plot(axes[1], zoom)
    axes[1].set_title('(b) Detail at k = 16–20', loc='left', pad=12)
    axes[1].get_legend().remove()
    axes[1].set_ylabel('TEST worst endpoint/context mean TV')
    axes[1].set_ylim(-.004, .255)
    axes[1].set_xlim(15.85, 20.3)
    for gamma in [.20, .12]:
        axes[1].axhline(gamma, color='#777D86', linewidth=.9, linestyle=(0, (3, 3)), zorder=0)
        axes[1].text(20.28, gamma + .003, f'γ = {gamma:.2f}', ha='right', va='bottom',
                     color='#646A73', fontsize=8.2)

    caption = (
        'Shared supports were selected on FIT and replayed unchanged. CERT tunes map parameters and selects k.\n'
        'CERT-selected k: γ = 0.20 → global / temperature / context: 18; gate: 17.\n'
        '                         γ = 0.12 → global / temperature: 20; context / gate: 19.\n'
        'Point estimates only; no confidence bands. Tolerance lines do not authorize TEST-based budget selection.\n'
        'Finite optimizer budgets and discrete shared supports do not establish an optimized or globally minimal frontier.'
    )
    fig.text(.075, .055, caption, fontsize=9.0, linespacing=1.65, color='#4D535C')
    stem = 'reconstruction_convergence1000'
    for suffix in ('png', 'pdf', 'svg'):
        path = output / f'{stem}.{suffix}'
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.18)
    plt.close(fig)

    choices = {(r['gamma'], r['method']): r['selected_k'] for r in source['cert_selected_budget_at_gamma']}
    table = []
    for row in source['budgets']:
        for method, values in row['methods'].items():
            trials = [d['trials'][d['selected_trial']] for d in values['fit_diagnostics'].values()] if method == 'nonlinear_gate' else []
            interval = values['paired_difference_vs_global']['percentile_95_ci']
            table.append({'k': row['k'], 'method': method, 'maxiter': 1000,
                          'retained_ids': '|'.join(row['retained_judge_ids']),
                          'fit_worst_tv': values['fit']['worst_endpoint_context_tv'],
                          'cert_worst_tv': values['cert']['worst_endpoint_context_tv'],
                          'test_worst_tv': values['test']['worst_endpoint_context_tv'],
                          'removed_verdict_agreement': values['test']['removed_verdict_agreement'],
                          'pointwise_difference_ci_low': interval[0], 'pointwise_difference_ci_high': interval[1],
                          'gate_selected_fits': len(trials) if method == 'nonlinear_gate' else '',
                          'gate_selected_unconverged': sum(not t['converged'] for t in trials) if method == 'nonlinear_gate' else '',
                          'cert_selected_at_gamma_0_12': choices[(.12, method)] == row['k'],
                          'cert_selected_at_gamma_0_20': choices[(.20, method)] == row['k']})
    csv_path = output / 'reconstruction_data.csv'
    with csv_path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_result': {'path': str(source_path), 'sha256': source_hash},
        'source_status': {'path': str(status_path), 'sha256': sha256_file(status_path)},
        'response_cache_sha256': source['metadata']['input_sha256'],
        'plotting_code': [{'path': str(Path(path).resolve()), 'sha256': sha256_file(path)}
                          for path in [__file__, plot_results.__file__]],
        'formats': ['png', 'pdf', 'svg'], 'png_dpi': 300, 'matplotlib_version': matplotlib.__version__,
        'budgets': [row['k'] for row in source['budgets']], 'maxiter': 1000,
        'selected_gate_fits': len(chosen), 'unconverged_selected_gate_fits': n_failed,
        'caption': caption,
        'scope': 'Independent reconstruction figure for the completed maxiter=1000 audit; original maxiter=150 figures and data are unchanged.',
        'uncertainty': 'Only observed TEST point estimates are plotted. CSV includes original pointwise paired-bootstrap intervals, not simultaneous curve bands.',
        'artifacts': [{'path': str(path), 'sha256': sha256_file(path)}
                      for path in [csv_path, *(output / f'{stem}.{suffix}' for suffix in ('png', 'pdf', 'svg'))]],
    }
    with (output / 'figure_sources.json').open('x') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(f'Saved 32 data rows, PNG/PDF/SVG and SHA manifest to {output}')


if __name__ == '__main__':
    main()
