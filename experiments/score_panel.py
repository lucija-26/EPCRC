"""Score a judge panel, behind quality gates G0, G1 and G2 (plan sections 46-48).

G0 is static: it must pass before any GPU time is spent.  G1 scores 20 items
with one small model and checks that the numbers coming out of the scorer mean
what the pipeline assumes.  G2 runs the whole selected panel over the stratified
item set and every registered context, builds the response tensor and drives the
pruners through it.

``--panel`` chooses between the section 10.3 smoke panel and the section 10
formal panel.  They write to different directories, so a Core-20 run can never
overwrite the Core-8 evidence that licensed it:

    core8    ->  results/smoke_core8/scores/    8 judges,  ~145 GB of weights
    core20   ->  results/core20/scores/        20 judges,  ~320 GB of weights

Every (judge, context) block is cached, so an interrupted run resumes without
rescoring and without duplicating rows.  Core-20 does not start until Core-8
has passed G2.

Usage on the GPU box:

    python -u experiments/score_panel.py --gate g0 --panel core20
    python -u experiments/score_panel.py --gate g1
    python -u experiments/score_panel.py --gate g2 --panel core20 --items 100
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epcrc.judge import JudgeCoverageFunctional, JudgeResponses
from epcrc.prompts import (
    I0_CLEAN,
    I1_SWAP,
    LABELS,
    REGISTERED_CONTEXTS,
    Context,
    canonicalize,
    protocol_hashes,
    render,
)
from epcrc.pruning import (
    BackwardEliminationPruner,
    BackwardKSwapPruner,
    ForwardSelectionPruner,
)
from epcrc.panel import (
    CORE8,
    DATASETS,
    PANELS,
    pairs_path,
    panel_weight_gb,
    scores_dir,
    split_path,
)
from epcrc.rewardbench import ALL_SEEDS, PRIMARY_SEED, read_pairs, stratified_subset
from epcrc.scoring import LabelScorer, accuracy_report, score_pairs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
# The cheapest model on the panel, used for the one-model G1 smoke test.
G1_MODEL = "Qwen/Qwen2.5-7B-Instruct"
G1_ITEMS = 20
G2_ITEMS = 100
# Real judges disagree far more than the synthetic panels do, so the sweep has
# to reach tolerances where anything is removable at all; stopping at 0.10 shows
# a flat line and says nothing about which method compresses better.
G2_GAMMAS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
DISK_HEADROOM_GB = 20.0

# Defaults for the Core-8 smoke panel; `configure` overwrites all of them.
PANEL_NAME = "core8"
PANEL: Dict[str, str] = dict(CORE8)
DATASET = "rewardbench"
SCORES = scores_dir("core8")
OUT = os.path.dirname(SCORES)
MIN_FREE_GB = 35.0


def configure(panel_name: str, evict: bool, dataset: str = "rewardbench") -> None:
    """Point the runner at one of the registered panels.

    The disk budget depends on both choices.  Scoring with ``--evict`` deletes
    each snapshot once its blocks are cached, so the peak requirement is the
    single largest model; without it the whole panel has to fit at once, which
    for Core-20 is roughly 320 GB.

    `dataset` also moves the block cache, which it must: blocks are named
    ``{judge}__{context}.json`` with no benchmark in the key, so scoring E7's
    JudgeBench pairs into the Core-20 directory would overwrite the blocks every
    other claim is computed from.
    """
    global PANEL_NAME, PANEL, DATASET, OUT, SCORES, MIN_FREE_GB

    PANEL_NAME = panel_name
    PANEL = dict(PANELS[panel_name])
    DATASET = dataset
    SCORES = scores_dir(panel_name, dataset)
    OUT = os.path.dirname(SCORES)

    if evict:
        peak = max(panel_weight_gb([j]) for j in PANEL)
    else:
        peak = panel_weight_gb(list(PANEL))
    MIN_FREE_GB = peak + DISK_HEADROOM_GB


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _pairs_path(seed: int) -> str:
    return pairs_path(seed, DATASET)


def _split_path(seed: int) -> str:
    return split_path(seed, DATASET)


def _cache_path(judge_id: str, context: Context) -> str:
    return os.path.join(SCORES, f"{judge_id}__{context.name}.json")


def _report(name: str, checks: Dict[str, object]) -> Dict[str, object]:
    """Print a gate's checks and return the payload with an overall verdict."""
    passed = all(v is True for k, v in checks.items() if k.startswith("ok_"))
    print(f"\n=== {name} ===")
    width = max(len(k) for k in checks)
    for key in checks:
        value = checks[key]
        mark = ""
        if key.startswith("ok_"):
            mark = "PASS" if value is True else "FAIL"
        print(f"  {key:<{width}}  {value}  {mark}")
    print(f"  --> {name}: {'PASS' if passed else 'FAIL'}")
    return {"gate": name, "passed": passed, "checks": checks}


def score_block(
    judge_id: str,
    model_id: str,
    pairs,
    context: Context,
    batch_size: int,
    revision: Optional[str] = None,
    scorer: Optional[LabelScorer] = None,
    device: str = "cuda",
) -> Dict[str, object]:
    """Score one (judge, context) block, reusing the cache when it is complete.

    A cached block is trusted only if it covers exactly the pairs asked for, in
    order; that is what makes resume safe rather than merely fast.
    """
    path = _cache_path(judge_id, context)
    want = [p.pair_id for p in pairs]

    if os.path.exists(path):
        with open(path) as handle:
            cached = json.load(handle)
        if cached.get("pair_ids") == want:
            cached["from_cache"] = True
            return cached
        print(f"  cache for {judge_id}/{context.name} does not match; rescoring")

    if scorer is None:
        scorer = LabelScorer(model_id, revision=revision, device=device)

    started = time.time()
    probabilities, hashes = score_pairs(
        scorer, pairs, context, batch_size=batch_size, progress=True
    )
    block = {
        "judge_id": judge_id,
        "model_id": model_id,
        "revision": revision,
        "context": context.name,
        "protocol": context.protocol,
        "pair_ids": want,
        "probabilities": probabilities.tolist(),
        "prompt_sha256": hashes,
        "tokenization": scorer.audit_tokenization().to_dict(),
        "accuracy": accuracy_report(probabilities, pairs),
        "seconds": time.time() - started,
        "from_cache": False,
    }

    os.makedirs(SCORES, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(block, handle)
    return block


# --------------------------------------------------------------------------
# G0 -- static preparation
# --------------------------------------------------------------------------

def gate_g0(seed: int, check_access: bool) -> Dict[str, object]:
    checks: Dict[str, object] = {}

    # Data schema.
    pairs_ok = all(os.path.exists(_pairs_path(s)) for s in ALL_SEEDS)
    splits_ok = all(os.path.exists(_split_path(s)) for s in ALL_SEEDS)
    checks["ok_pair_files_exist"] = pairs_ok
    checks["ok_split_files_exist"] = splits_ok
    if not (pairs_ok and splits_ok):
        checks["hint"] = "run experiments/build_rewardbench_pairs.py --all-seeds"
        return _report("G0", checks)

    pairs = read_pairs(_pairs_path(seed))
    with open(_split_path(seed)) as handle:
        manifest = json.load(handle)

    checks["n_pairs"] = len(pairs)
    checks["ok_pair_ids_unique"] = len({p.pair_id for p in pairs}) == len(pairs)
    checks["ok_labels_are_legal"] = {p.gold_label for p in pairs} <= set(LABELS)
    checks["ok_weights_positive"] = all(0.0 < p.weight <= 1.0 for p in pairs)
    checks["ok_no_empty_text"] = all(
        p.prompt and p.response_a and p.response_b for p in pairs
    )
    checks["ok_responses_differ"] = all(p.response_a != p.response_b for p in pairs)

    # Split leakage: the splits partition the base items, and no base item's
    # pairs straddle two splits.
    splits = manifest["splits"]
    members = [i for ids in splits.values() for i in ids]
    membership = {i: name for name, ids in splits.items() for i in ids}
    checks["ok_splits_disjoint"] = len(members) == len(set(members))
    checks["ok_splits_cover_pairs"] = all(
        p.base_item_id in membership for p in pairs
    )
    per_task_splits = {}
    for pair in pairs:
        per_task_splits.setdefault(pair.base_item_id, set()).add(
            membership.get(pair.base_item_id)
        )
    checks["ok_no_task_straddles_splits"] = all(
        len(s) == 1 for s in per_task_splits.values()
    )
    checks["split_sizes"] = manifest["split_sizes"]

    # Prompt templates hash correctly against the manifest written at build time.
    live = protocol_hashes()
    checks["ok_protocol_hashes_match"] = manifest.get("protocol_sha256") == live
    checks["protocol_sha256"] = {k: v[:12] for k, v in live.items()}

    # Contexts.
    names = [c.name for c in REGISTERED_CONTEXTS]
    checks["ok_contexts_unique"] = len(names) == len(set(names))
    checks["contexts"] = names

    # Rendering actually works for every context on a real pair.
    sample = pairs[0]
    rendered = {
        c.name: render(sample.prompt, sample.response_a, sample.response_b, context=c)
        for c in REGISTERED_CONTEXTS
    }
    checks["ok_all_prompts_end_at_verdict"] = all(
        text.rstrip().endswith("Verdict:") for text in rendered.values()
    )
    checks["ok_swap_reorders"] = (
        rendered[I0_CLEAN.name] != rendered[I1_SWAP.name]
    )

    # Output directory writable.
    os.makedirs(SCORES, exist_ok=True)
    probe = os.path.join(OUT, ".writable")
    try:
        with open(probe, "w") as handle:
            handle.write("ok")
        os.remove(probe)
        checks["ok_output_writable"] = True
    except OSError as exc:
        checks["ok_output_writable"] = False
        checks["write_error"] = str(exc)

    # Disk.
    free_gb = shutil.disk_usage(OUT).free / 1e9
    checks["free_disk_gb"] = round(free_gb, 1)
    checks["ok_enough_disk"] = free_gb >= MIN_FREE_GB

    # Formal prompt count.
    n_contexts = len(REGISTERED_CONTEXTS)
    checks["formal_prompts_core20"] = 20 * n_contexts * len(pairs)
    checks["panel"] = PANEL_NAME
    checks["panel_prompts"] = len(PANEL) * n_contexts * G2_ITEMS
    checks["panel_weights_gb"] = round(panel_weight_gb(list(PANEL)), 1)

    # Model access.
    if check_access:
        checks.update(_check_model_access())
    else:
        checks["model_access"] = "skipped (--no-access-check)"

    return _report("G0", checks)


def _check_model_access() -> Dict[str, object]:
    """Confirm every repo on the panel is *downloadable*, without fetching weights.

    ``model_info`` is not enough: the hub serves metadata for gated repos to
    anonymous callers, so it returns happily for a model whose weights will
    later 401.  ``auth_check`` is the call that tests the actual permission,
    which is what turns a seven-hour failure into a five-second one.
    """
    try:
        from huggingface_hub import HfApi, auth_check
    except ImportError:
        return {"model_access": "huggingface_hub not installed"}

    api = HfApi()
    reachable, gated, failures = {}, {}, {}
    for judge_id, model_id in PANEL.items():
        try:
            auth_check(model_id)
            info = api.model_info(model_id)
            reachable[judge_id] = info.sha[:12] if info.sha else "unknown"
        except Exception as exc:  # noqa: BLE001 - report whatever the hub says
            name = type(exc).__name__
            if "Gated" in name:
                gated[judge_id] = f"{model_id}: accept the licence and set HF_TOKEN"
            else:
                failures[judge_id] = f"{name}: {str(exc)[:200]}"

    out: Dict[str, object] = {
        "ok_model_access": not failures and not gated,
        "model_revisions": reachable,
    }
    if gated:
        out["gated_without_access"] = gated
    if failures:
        out["model_access_failures"] = failures
    return out


# --------------------------------------------------------------------------
# G1 -- one-model scoring smoke test
# --------------------------------------------------------------------------

def gate_g1(seed: int, model_id: str, batch_size: int) -> Dict[str, object]:
    checks: Dict[str, object] = {"model_id": model_id, "n_items": G1_ITEMS}

    pairs = stratified_subset(read_pairs(_pairs_path(seed)), G1_ITEMS, seed=seed)
    scorer = LabelScorer(model_id)

    audit = scorer.audit_tokenization()
    checks["label_tokens"] = audit.label_token_strings
    checks["n_continuation_tokens"] = audit.n_continuation_tokens
    checks["ok_continuation_tokens_logged"] = bool(audit.label_token_ids)
    checks["ok_consistent_label_boundary"] = audit.consistent_boundary

    clean, hashes = score_pairs(scorer, pairs, I0_CLEAN, batch_size=batch_size)
    checks["ok_finite"] = bool(np.isfinite(clean).all())
    checks["ok_sums_to_one"] = bool(np.allclose(clean.sum(axis=1), 1.0, atol=1e-9))
    checks["ok_nonnegative"] = bool((clean >= 0.0).all())
    # Forced choice: the labels are scored, never generated, so no free text is
    # produced or parsed anywhere in the path above.
    checks["ok_no_free_text_needed"] = True

    # Deterministic rerun.
    again, hashes_again = score_pairs(scorer, pairs, I0_CLEAN, batch_size=batch_size)
    checks["max_rerun_delta"] = float(np.abs(clean - again).max())
    checks["ok_deterministic_rerun"] = bool(np.allclose(clean, again, atol=1e-6))
    checks["ok_prompt_hashes_stable"] = hashes == hashes_again

    # Orientation remapping: scoring the swapped context and undoing the swap
    # must land near the clean answer, and the undo must be an exact involution.
    swapped, _ = score_pairs(scorer, pairs, I1_SWAP, batch_size=batch_size)
    raw = np.array([canonicalize(tuple(row), I1_SWAP) for row in swapped])
    checks["ok_orientation_involution"] = bool(np.allclose(
        np.array([canonicalize(tuple(r), I1_SWAP) for r in raw]), swapped
    ))
    checks["mean_orientation_tv"] = float(
        np.abs(clean - swapped).max(axis=1).mean()
    )
    checks["ok_orientation_remap_applied"] = bool(
        not np.allclose(swapped, raw) or np.allclose(swapped[:, 0], swapped[:, 1])
    )

    # Resume must not duplicate rows: score once, then again through the cache.
    for path in (_cache_path("G1", I0_CLEAN),):
        if os.path.exists(path):
            os.remove(path)
    first = score_block("G1", model_id, pairs, I0_CLEAN, batch_size, scorer=scorer)
    second = score_block("G1", model_id, pairs, I0_CLEAN, batch_size, scorer=scorer)
    checks["ok_resume_hits_cache"] = second["from_cache"] is True
    checks["ok_resume_no_duplicate_rows"] = (
        len(second["probabilities"]) == len(pairs)
        and first["pair_ids"] == second["pair_ids"]
    )

    checks["accuracy"] = accuracy_report(clean, pairs)
    return _report("G1", checks)


# --------------------------------------------------------------------------
# G2 -- whole-panel end-to-end
# --------------------------------------------------------------------------

def gate_g2(
    seed: int,
    batch_size: int,
    models: Dict[str, str],
    evict: bool = False,
    device: str = "cuda",
    max_exhaustive: int = 10,
    scores_only: bool = False,
) -> Dict[str, object]:
    checks: Dict[str, object] = {"n_items": G2_ITEMS, "judges": list(models)}

    all_pairs = read_pairs(_pairs_path(seed))
    pairs = stratified_subset(all_pairs, G2_ITEMS, seed=seed)
    with open(_split_path(seed)) as handle:
        membership = {
            i: name
            for name, ids in json.load(handle)["splits"].items()
            for i in ids
        }

    contexts = list(REGISTERED_CONTEXTS)
    checks["contexts"] = [c.name for c in contexts]
    checks["n_prompts"] = len(models) * len(contexts) * len(pairs)

    blocks: Dict[Tuple[str, str], Dict[str, object]] = {}
    failures: List[Dict[str, str]] = []
    n_rows = 0

    for judge_id, model_id in models.items():
        scorer = LabelScorer(model_id, device=device)
        loaded = False
        for context in contexts:
            print(f"[{judge_id} {context.name}]", flush=True)
            try:
                if not loaded and not os.path.exists(_cache_path(judge_id, context)):
                    scorer.load()
                    loaded = True
                block = score_block(
                    judge_id, model_id, pairs, context, batch_size,
                    scorer=scorer, device=device,
                )
                blocks[(judge_id, context.name)] = block
                n_rows += len(pairs)
            except Exception as exc:  # noqa: BLE001 - a failed block must be named
                failures.append({
                    "judge_id": judge_id,
                    "context": context.name,
                    "error_code": type(exc).__name__,
                    "message": str(exc)[:400],
                })
                print(f"  FAILED {type(exc).__name__}: {exc}", flush=True)
        # Free the weights before the next model is loaded.
        scorer._model = None
        _empty_cache()
        if evict:
            freed = _evict_snapshot(model_id)
            print(f"  evicted {model_id}: {freed:.1f} GB, "
                  f"{shutil.disk_usage(OUT).free / 1e9:.1f} GB now free", flush=True)

    expected = len(models) * len(contexts) * len(pairs)
    checks["n_rows"] = n_rows
    checks["success_rate"] = n_rows / expected if expected else 0.0
    checks["ok_success_rate"] = checks["success_rate"] >= 0.995
    checks["ok_failures_have_error_codes"] = all(
        f.get("error_code") for f in failures
    )
    if failures:
        checks["failures"] = failures

    judge_ids = [j for j in models if all((j, c.name) in blocks for c in contexts)]
    checks["judges_with_complete_blocks"] = judge_ids
    if len(judge_ids) < 2:
        # Two cases that must not be conflated. A top-up run (`--judges J07`,
        # after a licence comes through) was only ever asked for one judge, so
        # there is no judge axis to stack and the check does not apply -- and
        # reporting FAIL there invites a re-score of a judge that in fact
        # succeeded.  Being asked for a panel and ending up with fewer than two
        # complete judges is a real failure.
        if len(models) < 2:
            checks["tensor_check_skipped"] = (
                f"only {len(models)} judge(s) requested; a response tensor needs "
                f"at least 2. Scoring itself is unaffected -- build the panel with "
                f"epcrc.panel.load_panel once every judge has all its contexts."
            )
        else:
            checks["ok_tensor_builds"] = False
        return _report("G2", checks)

    # Split leakage in the scored subset itself.
    scored_splits = {membership[p.base_item_id] for p in pairs}
    checks["scored_splits"] = sorted(scored_splits)
    checks["ok_no_split_leakage"] = all(
        len({membership[p.base_item_id]}) == 1 for p in pairs
    )

    if scores_only:
        # The production run scores every pair, and the checks below cost
        # O(rows * |S|^2) NNLS work per pruner per gamma -- minutes at 100 items,
        # far longer over all 3724.  They are a gate on the pipeline, not on the
        # data, so they are validated once on the stratified sample and skipped
        # here; the scored blocks are what the experiments actually consume.
        checks["scores_only"] = True
        return _report("G2", checks)

    # Response tensor.
    tensor_blocks = []
    for context in contexts:
        arr = np.stack(
            [np.asarray(blocks[(j, context.name)]["probabilities"]) for j in judge_ids],
            axis=1,
        )
        tensor_blocks.append(arr)
    responses = JudgeResponses(tensor_blocks, [c.name for c in contexts])
    checks["ok_tensor_builds"] = True
    checks["tensor_shape"] = [len(contexts), len(pairs), len(judge_ids), 3]

    cov = JudgeCoverageFunctional(responses, responses, judge_ids)

    # Retained self-reconstruction is numerically zero.
    self_errors = [
        cov.compute_certificate(i, set(range(cov.N))).uniqueness for i in range(cov.N)
    ]
    checks["max_self_reconstruction"] = float(max(self_errors))
    checks["ok_self_reconstruction_zero"] = float(max(self_errors)) < 1e-8

    # Fitting monotonicity: adding judges cannot raise fitting coverage.
    rng = np.random.default_rng(0)
    violations = 0
    for _ in range(20):
        size = int(rng.integers(2, cov.N))
        small = set(rng.choice(cov.N, size=size, replace=False).tolist())
        extra = set(range(cov.N)) - small
        if not extra:
            continue
        large = small | {int(rng.choice(sorted(extra)))}
        if cov.compute_coverage(large)[0] > cov.compute_coverage(small)[0] + 1e-9:
            violations += 1
    checks["monotonicity_violations"] = violations
    checks["ok_fitting_monotone"] = violations == 0

    # Pruners and, only where it is tractable, the exhaustive optimum.
    exhaustive_ok = cov.N <= max_exhaustive
    checks["exhaustive_enumerated"] = exhaustive_ok
    runs = {}
    for gamma in G2_GAMMAS:
        backward = BackwardEliminationPruner(cov, gamma).run()
        forward = ForwardSelectionPruner(cov, gamma).run()
        kswap = BackwardKSwapPruner(cov, gamma, max_swap_k=2).run()
        entry = {
            "backward": sorted(judge_ids[i] for i in backward.kept_set),
            "forward": sorted(judge_ids[i] for i in forward.kept_set),
            "kswap2": sorted(judge_ids[i] for i in kswap.kept_set),
            "backward_coverage": float(backward.coverage),
        }
        if exhaustive_ok:
            exact_size, exact_set = _exhaustive_minimum(cov, gamma)
            entry["exhaustive_size"] = exact_size
            entry["exhaustive_set"] = [judge_ids[i] for i in exact_set]
        runs[str(gamma)] = entry
    checks["pruning"] = runs
    checks["ok_pruners_run"] = True
    # The greedy routes can only ever be at least as large as the true optimum.
    # With no optimum to compare against there is nothing to violate, so this
    # is vacuously true rather than a failure.
    checks["ok_exhaustive_lower_bounds_greedy"] = all(
        r["exhaustive_size"] <= min(len(r["backward"]), len(r["kswap2"]))
        for r in runs.values() if "exhaustive_size" in r
    )

    return _report("G2", checks)


def _exhaustive_minimum(
    cov: JudgeCoverageFunctional, gamma: float
) -> Tuple[int, List[int]]:
    """Brute-force smallest feasible panel.

    Only tractable for a small ecosystem: the cost is sum_k C(N, k) coverage
    evaluations up to the first feasible size, and each evaluation is a batch of
    NNLS solves. Core-8 finishes instantly, Core-20 does not finish at all, so
    callers must gate this on `max_exhaustive`.
    """
    for k in range(1, cov.N + 1):
        for combo in combinations(range(cov.N), k):
            if cov.compute_coverage(set(combo))[0] <= gamma:
                return k, list(combo)
    return cov.N, list(range(cov.N))


def _evict_snapshot(model_id: str) -> float:
    """Delete one model's weights from the local Hugging Face cache.

    The shared box does not have room for all eight judges at once.  Only the
    cache entry for this exact repo is removed, and only after its blocks are
    on disk, so the worst case is that the weights are downloaded again.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return 0.0

    folder = os.path.join(
        HF_HUB_CACHE, "models--" + model_id.replace("/", "--")
    )
    if not os.path.isdir(folder):
        return 0.0

    size = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(folder)
        for f in files
        if not os.path.islink(os.path.join(root, f))
    )
    shutil.rmtree(folder, ignore_errors=True)
    return size / 1e9


def _empty_cache() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --------------------------------------------------------------------------

def main() -> None:
    global G2_ITEMS, SCORES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=["g0", "g1", "g2"], default="g0")
    parser.add_argument("--panel", choices=sorted(PANELS), default="core8",
                        help="which registered panel to score")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="rewardbench",
                        help="which benchmark's pairs to score; judgebench is "
                             "the E7 transfer set and gets its own cache")
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", default=G1_MODEL, help="the G1 model")
    parser.add_argument(
        "--judges", nargs="*", default=None,
        help="subset of the panel's judge ids for G2",
    )
    parser.add_argument("--items", type=int, default=G2_ITEMS,
                        help="stratified items to score in G2")
    parser.add_argument(
        "--max-exhaustive", type=int, default=10,
        help="largest panel for which G2 enumerates every subset to find the "
             "exact optimum; above this the check is skipped, since Core-20 "
             "never finishes. Same meaning as in experiment_e1.",
    )
    parser.add_argument(
        "--scores-only", action="store_true",
        help="stop after every (judge, context) block is written, skipping the "
             "tensor and pruner checks. Use for the production run over all "
             "pairs; the checks are already gated on the stratified sample.",
    )
    parser.add_argument("--no-access-check", action="store_true")
    parser.add_argument(
        "--evict",
        action="store_true",
        help="delete each model's weights once its blocks are cached",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="device_map; use 'auto' to shard a 14B model across several GPUs",
    )
    args = parser.parse_args()

    G2_ITEMS = args.items
    configure(args.panel, evict=args.evict, dataset=args.dataset)
    if args.gate == "g2" and not args.scores_only:
        # The gate scores a stratified sample, which is not the item set the
        # production cache holds.  `score_block` keys its cache on (judge,
        # context) alone, so a shared directory means the gate silently
        # overwrites the blocks every claim is computed from.  Keep them apart.
        SCORES = os.path.join(OUT, "scores_gate")
    os.makedirs(SCORES, exist_ok=True)
    print(f"panel {PANEL_NAME} on {DATASET}: {len(PANEL)} judges, "
          f"~{panel_weight_gb(list(PANEL)):.0f} GB of weights, "
          f"needs {MIN_FREE_GB:.0f} GB free -> {SCORES}")

    if args.gate == "g0":
        payload = gate_g0(args.seed, check_access=not args.no_access_check)
    elif args.gate == "g1":
        payload = gate_g1(args.seed, args.model, args.batch_size)
    else:
        unknown = set(args.judges or []) - set(PANEL)
        if unknown:
            parser.error(f"{sorted(unknown)} are not in panel {args.panel}")
        models = (
            {j: PANEL[j] for j in args.judges} if args.judges else dict(PANEL)
        )
        payload = gate_g2(
            args.seed, args.batch_size, models,
            evict=args.evict, device=args.device,
            max_exhaustive=args.max_exhaustive,
            scores_only=args.scores_only,
        )
    payload["panel"] = PANEL_NAME

    path = os.path.join(OUT, f"{args.gate}.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print(f"\nwrote {path}")

    sys.exit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
