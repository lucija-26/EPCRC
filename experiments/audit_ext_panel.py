"""Pre-flight audit of candidate judges for a second, independent panel (no GPU).

For each model, downloads only the tokenizer and config and checks:
  * the chat template renders;
  * the labels " A", " B", " C" encode as single tokens (the fast scoring path);
  * the context window (max_position_embeddings);
  * the token length of every rendered prompt (all 3,724 pairs x 7 contexts,
    chat template applied), against the scorer's max_prompt_tokens = 8192 and
    against the model's own window.

Truncation in LabelScorer is right-sided, so a prompt longer than the limit
would lose its ending ("Verdict:" and the generation marker).  Any model with
prompts over min(window, 8192) is flagged.

Also audits a few ungated Core-20 tokenizers, to check whether the existing
Core-20 scores were ever affected by truncation.

Writes results/ext20_audit.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from epcrc.panel import pairs_path  # noqa: E402
from epcrc.prompts import LABELS, REGISTERED_CONTEXTS, render  # noqa: E402
from epcrc.rewardbench import PRIMARY_SEED, read_pairs  # noqa: E402

CANDIDATES = [
    "HuggingFaceTB/SmolLM2-1.7B-Instruct", "allenai/OLMo-2-0425-1B-Instruct",
    "allenai/OLMo-2-1124-13B-Instruct", "allenai/Llama-3.1-Tulu-3-8B",
    "ibm-granite/granite-3.3-2b-instruct", "ibm-granite/granite-3.1-8b-instruct",
    "tiiuae/Falcon3-3B-Instruct", "tiiuae/Falcon3-1B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-medium-4k-instruct",
    "01-ai/Yi-1.5-6B-Chat", "01-ai/Yi-1.5-9B-Chat",
    "openchat/openchat-3.5-0106", "HuggingFaceH4/zephyr-7b-beta",
    "teknium/OpenHermes-2.5-Mistral-7B", "NousResearch/Hermes-3-Llama-3.1-8B",
    "prometheus-eval/prometheus-7b-v2.0", "mistralai/Ministral-8B-Instruct-2410",
    "deepseek-ai/deepseek-llm-7b-chat", "upstage/SOLAR-10.7B-Instruct-v1.0",
    # alternates
    "Qwen/Qwen2.5-1.5B-Instruct", "stabilityai/stablelm-2-1_6b-chat",
    "microsoft/Phi-3-mini-128k-instruct", "allenai/OLMo-2-1124-7B-SFT",
]
CORE20_CHECK = ["Qwen/Qwen2.5-7B-Instruct", "microsoft/phi-4", "tiiuae/Falcon3-10B-Instruct",
                "ibm-granite/granite-3.3-8b-instruct", "allenai/OLMo-2-1124-7B-Instruct"]
SCORER_MAX = 8192


def audit(model_id, prompts):
    from transformers import AutoConfig, AutoTokenizer
    out = {"model": model_id}
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        cfg = AutoConfig.from_pretrained(model_id)
    except Exception as e:
        return dict(out, error=f"load: {type(e).__name__}: {str(e)[:200]}")
    window = getattr(cfg, "max_position_embeddings", None)
    out["window"] = window
    out["sliding_window"] = getattr(cfg, "sliding_window", None)
    enc = {l: tok.encode(" " + l, add_special_tokens=False) for l in LABELS}
    out["label_ids"] = enc
    out["single_token_labels"] = all(len(v) == 1 for v in enc.values())
    out["distinct_labels"] = len({tuple(v) for v in enc.values()}) == 3
    try:
        def chat(p):
            m = [{"role": "user", "content": p}]
            try:
                return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        sample = chat("Hello")
        out["chat_sample_tail"] = sample[-80:]
    except Exception as e:
        return dict(out, error=f"chat template: {type(e).__name__}: {str(e)[:200]}")
    lens = np.array([len(tok(chat(p), add_special_tokens=False)["input_ids"]) for p in prompts])
    limit = min(window or SCORER_MAX, SCORER_MAX)
    out.update(prompt_tokens_max=int(lens.max()), prompt_tokens_p99=float(np.percentile(lens, 99)),
               prompt_tokens_median=float(np.median(lens)),
               n_prompts=int(lens.size),
               n_over_4096=int((lens > 4096).sum()), n_over_8192=int((lens > 8192).sum()),
               n_over_effective_limit=int((lens > limit).sum()), effective_limit=limit)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="audit these instead of the built-in list")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "ext20_audit.json"))
    a = ap.parse_args()
    models = a.models if a.models else CANDIDATES + CORE20_CHECK
    pairs = read_pairs(pairs_path(PRIMARY_SEED))
    # exactly as score_pairs renders them (no reference is passed)
    prompts = [render(p.prompt, p.response_a, p.response_b, context=c)
               for c in REGISTERED_CONTEXTS for p in pairs]
    print(f"{len(prompts)} prompts", flush=True)
    rows = []
    for m in models:
        r = audit(m, prompts)
        r["group"] = "core20_check" if m in CORE20_CHECK else "candidate"
        rows.append(r)
        print(json.dumps({k: r.get(k) for k in ("model", "error", "window", "single_token_labels",
                                                   "prompt_tokens_max", "n_over_4096", "n_over_8192",
                                                   "n_over_effective_limit")}), flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
