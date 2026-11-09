"""Render completed core20 reports; label selected cost points to avoid crowding."""
import argparse
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from supplementary.runtime import plot_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path,
                        default=ROOT / 'results/supplementary/real/core20')
    parser.add_argument('--output-dir', type=Path,
                        help='Defaults to <run-dir>/figures')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = (args.output_dir if args.output_dir is not None else run / 'figures').resolve()
    original = plot_results.cost_plot

    def selected_cost_labels(ax, source):
        note = original(ax, source)
        for label in list(ax.texts):
            label.remove()
        seen = set()
        for row in source['cert_selected_operating_points'].values():
            if row is None:
                continue
            x = row['test_worst_endpoint_context_tv']
            y = 100 * row['endpoint_service_work_saving_fraction']
            key = (row['k'], x, y)
            if key in seen:
                continue
            seen.add(key)
            ax.scatter([x], [y], marker='o', s=38, color='#202329', zorder=5)
            ax.annotate(f"CERT: k={row['k']}", (x, y), xytext=(13, 20),
                        textcoords='offset points', fontsize=8.5,
                        arrowprops={'arrowstyle': '-', 'color': '#50555D', 'lw': .7},
                        bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none', 'pad': 1.5})
        return note + '\nBlack points and labels: CERT-selected budgets at gamma=0.20.'

    previous_argv = sys.argv
    plot_results.cost_plot = selected_cost_labels
    sys.argv = [__file__, '--run-dir', str(run), '--output-dir', str(output)]
    try:
        plot_results.main()
    finally:
        plot_results.cost_plot = original
        sys.argv = previous_argv
    manifest_path = output / 'figure_sources.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['renderers_sha256'] = {
        str(Path(p).resolve().relative_to(ROOT)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in [__file__, plot_results.__file__]
    }
    manifest['cost_labels'] = 'All measured points retained; only frozen CERT-selected operating points labeled.'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
