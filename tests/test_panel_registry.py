"""The panel registry must match plan section 10 exactly.

A typo in a judge id or a model repo is not caught by any other test: it would
survive G0's static checks, download a wrong-but-real model, and only surface
after hours of GPU time. These tests are the cheap version of that discovery.
"""

from __future__ import annotations

import pytest

from epcrc.panel import (
    CORE8,
    CORE8_IDS,
    CORE20,
    PANELS,
    PARAMS_B,
    panel_weight_gb,
    scores_dir,
)

# Transcribed from the section 10 table: judge id -> (repo, family, params B).
SECTION_10 = {
    "J01": ("Qwen/Qwen2.5-3B-Instruct", "Qwen 2.5", 3),
    "J02": ("Qwen/Qwen2.5-7B-Instruct", "Qwen 2.5", 7),
    "J03": ("Qwen/Qwen2.5-14B-Instruct", "Qwen 2.5", 14),
    "J04": ("Qwen/Qwen3-4B", "Qwen 3", 4),
    "J05": ("Qwen/Qwen3-8B", "Qwen 3", 8),
    "J06": ("Qwen/Qwen3-14B", "Qwen 3", 14),
    "J07": ("meta-llama/Llama-3.2-3B-Instruct", "Llama", 3),
    "J08": ("meta-llama/Llama-3.1-8B-Instruct", "Llama", 8),
    "J09": ("google/gemma-3-4b-it", "Gemma", 4),
    "J10": ("google/gemma-3-12b-it", "Gemma", 12),
    "J11": ("microsoft/Phi-4-mini-instruct", "Phi", 4),
    "J12": ("microsoft/phi-4", "Phi", 14),
    "J13": ("microsoft/Phi-3.5-mini-instruct", "Phi", 4),
    "J14": ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral", 7),
    "J15": ("mistralai/Mistral-Nemo-Instruct-2407", "Mistral", 12),
    "J16": ("ibm-granite/granite-3.3-8b-instruct", "Granite", 8),
    "J17": ("tiiuae/Falcon3-7B-Instruct", "Falcon", 7),
    "J18": ("tiiuae/Falcon3-10B-Instruct", "Falcon", 10),
    "J19": ("CohereLabs/aya-expanse-8b", "Aya/Command", 8),
    "J20": ("allenai/OLMo-2-1124-7B-Instruct", "OLMo", 7),
}

# Section 10.3, verbatim: "validate the entire pipeline using: J02, J05, J08,
# J10, J12, J14, J16, and J20".
SECTION_10_3 = ["J02", "J05", "J08", "J10", "J12", "J14", "J16", "J20"]


def test_core20_matches_the_plan_table():
    assert CORE20 == {j: repo for j, (repo, _, _) in SECTION_10.items()}


def test_core20_judge_ids_are_contiguous_and_ordered():
    assert list(CORE20) == [f"J{i:02d}" for i in range(1, 21)]


def test_core8_is_the_prescribed_smoke_panel():
    assert CORE8_IDS == SECTION_10_3
    assert CORE8 == {j: CORE20[j] for j in SECTION_10_3}


def test_core8_is_a_subset_of_core20():
    assert set(CORE8) <= set(CORE20)
    assert all(CORE8[j] == CORE20[j] for j in CORE8)


def test_core8_covers_eight_distinct_families():
    families = {SECTION_10[j][1] for j in CORE8_IDS}
    assert len(families) == 8, f"smoke panel should span 8 families, got {families}"


def test_core20_has_within_family_pairs():
    """C1 needs judges that could plausibly be redundant given a sibling.

    Core-8 deliberately has none, which is why C1 is untestable there; Core-20
    must have them or the claim cannot be posed at all.
    """
    counts: dict = {}
    for _, family, _ in SECTION_10.values():
        counts[family] = counts.get(family, 0) + 1
    assert sum(1 for n in counts.values() if n >= 2) >= 4


def test_no_duplicate_repos():
    assert len(set(CORE20.values())) == len(CORE20)


def test_param_counts_cover_the_panel_and_match_the_plan():
    assert set(PARAMS_B) == set(CORE20)
    for judge, (_, _, params) in SECTION_10.items():
        assert PARAMS_B[judge] == params


def test_panel_weight_is_the_bf16_total():
    assert panel_weight_gb(["J01"]) == pytest.approx(6.0)
    assert panel_weight_gb(list(CORE20)) == pytest.approx(
        2.0 * sum(p for _, _, p in SECTION_10.values())
    )
    assert panel_weight_gb(list(CORE8)) < panel_weight_gb(list(CORE20))


def test_registered_panels():
    assert PANELS == {"core8": CORE8, "core20": CORE20}


def test_panels_write_to_separate_directories():
    """A Core-20 run must not overwrite the Core-8 blocks that licensed it."""
    assert scores_dir("core8") != scores_dir("core20")
    assert scores_dir("core8").endswith("smoke_core8/scores")


def test_unknown_panel_is_rejected():
    with pytest.raises(ValueError):
        scores_dir("core12")
