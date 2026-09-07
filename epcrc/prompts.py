"""Frozen judge prompt protocols, interventions, and prompt rendering.

The retained set selects *models*; these protocols define *contexts*.  P1 is the
primary clean protocol, P2 and P3 are rubric interventions.  The text below is
copied verbatim from the execution plan and must not be reworded: the rendered
prompt is hashed and the hash goes into the response manifest, so any edit
invalidates an already-scored tensor.

A context is a (protocol, intervention) pair.  Each context is scored
separately and the coverage functional takes the maximum over them, so a judge
that is redundant on clean items but distinctive under an order swap is kept.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

P1 = """You are an impartial evaluator. Compare Response A and Response B for the user request.

Prioritize substantive quality:
1. correctness and factual accuracy;
2. satisfaction of the user's instruction;
3. relevance and completeness;
4. sound reasoning and appropriate evidence;
5. clarity, only after substantive quality.

Do not prefer a response merely because it is longer, more confident, more polished, or appears first.
Choose:
A — Response A is better.
B — Response B is better.
C — The responses are effectively tied in overall quality.

Return one label only.

User request:
{instruction}

Optional reference or task information:
{reference_block}

Response A:
{response_a}

Response B:
{response_b}

Verdict:"""

P2 = """You are a strict evaluator. Compare the two responses using the following priority order.

First decide whether either response contains a factual, logical, mathematical, coding, safety, or instruction-following failure. A response with a substantive failure must lose to a response without that failure, even if it is more detailed or polished.

Only when substantive correctness is comparable should you consider completeness, usefulness, and clarity. Ignore superficial style advantages.

Choose:
A — Response A is better.
B — Response B is better.
C — The responses are substantively tied.

Return one label only.

User request:
{instruction}

Optional reference or task information:
{reference_block}

Response A:
{response_a}

Response B:
{response_b}

Verdict:"""

P3 = """You are auditing response quality under possible presentation bias.

Evaluate only the substantive content. Explicitly ignore:
- which response appears first;
- response length by itself;
- formatting, headings, and rhetorical confidence;
- repeated conclusions;
- model or source names;
- disclaimers that do not change correctness;
- citations that do not support the claims they accompany.

Choose:
A — Response A is substantively better.
B — Response B is substantively better.
C — The responses are substantively tied.

Return one label only.

User request:
{instruction}

Optional reference or task information:
{reference_block}

Response A:
{response_a}

Response B:
{response_b}

Verdict:"""

PROTOCOLS: Dict[str, str] = {"P1": P1, "P2": P2, "P3": P3}

LABELS: Tuple[str, str, str] = ("A", "B", "C")

# Padding for the verbosity intervention I4: it must add length without adding
# any substance, so that a judge swayed by it is demonstrably length-biased.
VERBOSITY_PADDING = (
    "\n\nTo restate the above in different words: the essential content of this "
    "response is unchanged, and the following remarks add no new information. "
    "It is worth emphasising that the points already made stand on their own, "
    "and that this additional elaboration is included only for completeness."
)

SOURCE_TAGS = ("[Source: System Alpha]", "[Source: System Beta]")


@dataclass(frozen=True)
class Context:
    """One registered evaluation condition.

    `swap` exchanges the two responses, which requires the returned
    probabilities to be mapped back to canonical identity afterwards (see
    `canonicalize`).  `pad_response` inflates one side's length, `tag_sources`
    prepends neutral identity labels, `hide_reference` drops the reference
    block.
    """

    name: str
    protocol: str = "P1"
    swap: bool = False
    pad_response: Optional[str] = None
    tag_sources: bool = False
    hide_reference: bool = False


# The registered set. Deliberately not the full Cartesian product: each
# intervention is varied against the clean baseline one at a time, so an
# observed failure is attributable to a single cause.
I0_CLEAN = Context("I0_clean", protocol="P1")
I1_SWAP = Context("I1_order_swap", protocol="P1", swap=True)
I2_CORRECTNESS = Context("I2_correctness_rubric", protocol="P2")
I3_BIAS_RESISTANT = Context("I3_bias_resistant_rubric", protocol="P3")
I4_VERBOSITY = Context("I4_verbosity_pad_a", protocol="P1", pad_response="A")
I5_SOURCE_TAGS = Context("I5_source_tags", protocol="P1", tag_sources=True)
I6_REFERENCE_HIDDEN = Context("I6_reference_hidden", protocol="P1", hide_reference=True)

REGISTERED_CONTEXTS: Tuple[Context, ...] = (
    I0_CLEAN,
    I1_SWAP,
    I2_CORRECTNESS,
    I3_BIAS_RESISTANT,
    I4_VERBOSITY,
    I5_SOURCE_TAGS,
    I6_REFERENCE_HIDDEN,
)


def render(
    instruction: str,
    response_a: str,
    response_b: str,
    context: Context = I0_CLEAN,
    reference: Optional[str] = None,
) -> str:
    """Render one prompt under a context.

    Interventions are applied to the *content* before the protocol template is
    filled, so the protocol text itself is never modified.
    """
    if context.protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {context.protocol!r}")

    first, second = response_a, response_b

    if context.pad_response == "A":
        first = first + VERBOSITY_PADDING
    elif context.pad_response == "B":
        second = second + VERBOSITY_PADDING
    elif context.pad_response is not None:
        raise ValueError(f"pad_response must be 'A', 'B' or None, got {context.pad_response!r}")

    if context.swap:
        first, second = second, first

    if context.tag_sources:
        first = f"{SOURCE_TAGS[0]} {first}"
        second = f"{SOURCE_TAGS[1]} {second}"

    if context.hide_reference or not reference:
        reference_block = "None provided."
    else:
        reference_block = reference

    return PROTOCOLS[context.protocol].format(
        instruction=instruction,
        reference_block=reference_block,
        response_a=first,
        response_b=second,
    )


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def protocol_hashes() -> Dict[str, str]:
    """Hashes of the frozen templates, recorded once per run."""
    return {name: prompt_sha256(text) for name, text in sorted(PROTOCOLS.items())}


def canonicalize(probabilities, context: Context):
    """Map probabilities back to canonical response identity.

    Under an order swap the judge saw the responses reversed, so its "A" mass
    belongs to canonical response B and vice versa.  Tie mass is unaffected.
    Skipping this silently turns a well-behaved judge into an apparent outlier.
    """
    p_a, p_b, p_tie = probabilities
    if context.swap:
        return (p_b, p_a, p_tie)
    return (p_a, p_b, p_tie)


def canonical_gold_label(gold_label: str, context: Context) -> str:
    """The gold label as the judge sees it under a context."""
    if not context.swap or gold_label == "C":
        return gold_label
    return "B" if gold_label == "A" else "A"
