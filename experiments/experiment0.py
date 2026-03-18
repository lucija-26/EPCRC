"""Experiment 0: Backward elimination on SST-2 sentiment models.

Three BERT-family models fine-tuned on SST-2, pruned via Section 4.1
backward elimination with token-masking interventions.

Run from project root:
    python -m experiments.experiment0
    python experiments/experiment0.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for both invocation styles
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from epcrc.backward_elimination_hf import run_backward_elimination_real_models

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
MODEL_IDS = [
    "textattack/bert-base-uncased-SST-2",
    "textattack/distilbert-base-uncased-SST-2",
    "textattack/roberta-base-SST-2",
]
MAX_SAMPLES = 80
GAMMA = 1.0
METRIC = "mean_abs"  # one of: mean_abs, rmse, max
FIT_DOSES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
EVAL_DOSES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BATCH_SIZE = 16
OUTPUT_JSON = None  # e.g. "results/experiment0_summary.json"


def main() -> None:
    run_backward_elimination_real_models(
        model_ids=MODEL_IDS,
        max_samples=MAX_SAMPLES,
        gamma=GAMMA,
        doses_fit=FIT_DOSES,
        doses_eval=EVAL_DOSES,
        metric=METRIC,
        batch_size=BATCH_SIZE,
        output_json=OUTPUT_JSON,
    )


if __name__ == "__main__":
    main()
