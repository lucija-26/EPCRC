"""Provenance/completeness checks using tiny temporary, mocked real-run files."""
import csv
import json

import numpy as np
import pytest

from supplementary.common import load_npz, sha256_file
from supplementary.exp03_ecosystem_replication.summarize_runs import main, summarize, write_outputs


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def runs(tmp_path):
    gammas = [.12, .15, .20, .25, .30]
    entries, sources = [], []
    for index, name in enumerate(["original", "new_judges", "nli"]):
        run = tmp_path / name
        run.mkdir()
        paths = {key: run / filename for key, filename in {
            "input": "responses.npz", "analysis_result": "results.json", "locked_manifest": "locked.json",
            "pairs": "pairs.jsonl", "source_provenance": "provenance.json"}.items()}
        interface = "nli" if index == 2 else "judge"
        model_ids = [f"org/{name}-1", f"org/{name}-2"]
        ids = [f"{name}1", f"{name}2"]
        locked = {"name": name, "interface": interface,
                  "endpoints": [{"judge_id": j, "model_id": m, "revision": str(index + 1) * 40}
                                for j, m in zip(ids, model_ids)]}
        dump(paths["locked_manifest"], locked)
        rows = [{"pair_id": f"p{i}", "base_item_id": f"g{i // 2}", "text": "fixture"} for i in range(12)]
        paths["pairs"].write_text("".join(json.dumps(row) + "\n" for row in rows))
        dataset = {"dataset_id": "dataset/nli" if index == 2 else "dataset/judge", "config": None,
                   "split": "test", "revision": "a" * 40}
        labels = ["entailment", "neutral", "contradiction"] if interface == "nli" else ["A", "B", "C"]
        source = dict(dataset, file_sha256=sha256_file(paths["pairs"]), n_rows=len(rows), canonical_labels=labels)
        dump(paths["source_provenance"], source)
        metadata = {"synthetic": False, "manifest": locked, "model_ids": model_ids, "interface": interface,
                    "collection_protocol": {"manifest_sha256": sha256_file(paths["locked_manifest"]),
                                            "pairs_sha256": sha256_file(paths["pairs"]), "interface": interface},
                    "input_provenance": source, "canonical_labels": labels, "split_seed": 20260818}
        np.savez_compressed(paths["input"], probabilities=np.full((1, 12, 2, 3), 1 / 3),
                            pair_ids=np.array([r["pair_id"] for r in rows]),
                            groups=np.array([r["base_item_id"] for r in rows]),
                            judge_ids=np.array(ids), model_ids=np.array(model_ids), context_names=np.array(["clean"]),
                            split=np.array(["FIT"] * 4 + ["CERT"] * 4 + ["TEST"] * 4),
                            metadata_json=np.array(json.dumps(metadata)))
        entry = dict(name=name, relationship="original_panel" if index == 0 else "independent_ecosystem",
                     benchmark="exact version above", assembly_protocol="Tiny mocked fixture, not empirical evidence",
                     model_ids=model_ids, model_families=["shared_family", f"family_{index}"],
                     budgets=[1, 2], dataset=dataset,
                     **{key: str(value.relative_to(tmp_path)) for key, value in paths.items()})
        provenance = {key: entry[key] for key in ("name", "relationship", "benchmark", "assembly_protocol", "model_ids")}
        provenance.update(input=str(paths["input"]), input_sha256=sha256_file(paths["input"]))
        frontier = [dict(k=1, method="backward", kept_indices=[0], kept_ids=ids[:1], fit_tv=.4, cert_tv=.45, test_tv=.46),
                    dict(k=2, method="backward", kept_indices=[0, 1], kept_ids=ids, fit_tv=0., cert_tv=0., test_tv=0.)]
        audits = [dict(gamma=g, status="no_individually_removable_endpoints", removed_indices=[], removed_ids=[],
                       kept_indices=[0, 1], selection_split="CERT", loo_cert_errors=dict(zip(ids, [.45, .5])))
                  for g in gammas]
        report = {"experiment": "ecosystem_replication", "declared_manifest_complete": True,
                  "model_overlap_audit": [], "protocol": {"primary_gamma": .2, "primary_gamma_included": True},
                  "results": [{"provenance": provenance, "metadata": load_npz(paths["input"]).metadata,
                               "n_judges": 2, "judge_ids": ids, "frontier": frontier, "audits": audits,
                               "cert_selected_budget": [dict(gamma=g, smallest_tested_k_accepted_on_cert=2,
                                                             frozen_test_tv=0., is_global_minimum=False) for g in gammas]}]}
        dump(paths["analysis_result"], report)
        entries.append(entry)
        sources.append(paths)
    manifest = tmp_path / "ecosystems.json"
    dump(manifest, {"ecosystems": entries, "expected_ecosystem_count": 3, "gammas": gammas,
                    "primary_gamma": .2, "family_label_note": "Explicit fixture family labels"})
    return manifest, sources


def mutate_report(path, mutate):
    report = json.loads(path.read_text())
    mutate(report)
    dump(path, report)


def test_complete_summary_retains_all_gammas_budgets_and_cross_panel_audit(runs, tmp_path):
    manifest, sources = runs
    before = {path: sha256_file(path) for paths in sources for path in paths.values()}
    output = tmp_path / "summary.json"
    assert main(["--manifest", str(manifest), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "complete"
    assert report["verified_complete_ecosystem_count"] == 3
    assert len(report["model_overlap_audit"]) == 3
    assert all(r["model_overlap_count"] == 0 for r in report["model_overlap_audit"])
    assert all(r["shared_declared_families"] == ["shared_family"] for r in report["model_overlap_audit"])
    assert report["model_overlap_audit"][0]["same_declared_dataset"] is True
    for panel in report["ecosystems"]:
        assert [r["gamma"] for r in panel["audits"]] == [.12, .15, .2, .25, .3]
        assert [r["k"] for r in panel["frontier"]] == [1, 2]
        assert panel["split_audit"]["TEST"]["n_groups"] == 2
    with output.with_suffix(".csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 21
    assert sum(r["gamma_role"] == "primary" for r in rows) == 3
    assert sum(r["gamma_role"] == "secondary" for r in rows) == 12
    assert before == {path: sha256_file(path) for path in before}


def test_one_missing_panel_cannot_be_reported_complete(runs):
    manifest, sources = runs
    sources[0]["analysis_result"].unlink()
    report = summarize(manifest)
    assert report["status"] == "pending"
    assert report["verified_complete_ecosystem_count"] == 2
    assert report["ecosystems"][0]["pending_reason"] == "missing_artifacts"
    assert report["model_overlap_audit"][0]["both_results_verified"] is False


def test_incomplete_single_panel_report_is_pending(runs):
    manifest, sources = runs
    mutate_report(sources[1]["analysis_result"], lambda r: r.update(declared_manifest_complete=False))
    assert summarize(manifest)["verified_complete_ecosystem_count"] == 2


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["results"][0]["provenance"].update(input_sha256="0" * 64), "response SHA256"),
    (lambda r: r["results"][0]["metadata"]["manifest"]["endpoints"][0].update(revision="f" * 40), "pinned model IDs/revisions"),
    (lambda r: r["results"][0]["metadata"]["input_provenance"].update(revision="f" * 40), "dataset provenance"),
    (lambda r: r["results"][0]["frontier"].pop(), "Actual exp03 budgets"),
    (lambda r: r["results"][0]["audits"].pop(), "Actual audit gamma"),
    (lambda r: r["results"][0]["cert_selected_budget"][0].update(smallest_tested_k_accepted_on_cert=1), "CERT budget choice"),
    (lambda r: r["results"][0]["judge_ids"].reverse(), "endpoint IDs/order"),
])
def test_provenance_or_declared_analysis_mismatch_fails(runs, mutation, match):
    manifest, sources = runs
    mutate_report(sources[0]["analysis_result"], mutation)
    with pytest.raises(ValueError, match=match):
        summarize(manifest)


def test_source_text_change_fails_even_if_pair_ids_unchanged(runs):
    manifest, sources = runs
    path = sources[0]["pairs"]
    path.write_text(path.read_text().replace("fixture", "changed"))
    with pytest.raises(ValueError, match="source-pairs SHA256"):
        summarize(manifest)


@pytest.mark.parametrize("error", [.19, .20])
def test_removal_set_must_match_loo_cert_threshold_including_equality(runs, error):
    manifest, sources = runs
    def contradict_removal(report):
        row = report["results"][0]["audits"][2]
        assert row["gamma"] == .20 and row["removed_indices"] == []
        row["loo_cert_errors"]["original1"] = error
    mutate_report(sources[0]["analysis_result"], contradict_removal)
    with pytest.raises(ValueError, match="Removal set disagrees with LOO CERT errors <= gamma"):
        summarize(manifest)


def test_loo_cert_schema_requires_exact_endpoint_inventory(runs):
    manifest, sources = runs
    mutate_report(sources[0]["analysis_result"],
                  lambda report: report["results"][0]["audits"][0]["loo_cert_errors"].pop("original1"))
    with pytest.raises(ValueError, match="LOO CERT errors must name every endpoint"):
        summarize(manifest)


def test_unstored_split_uses_original_analysis_seed_and_checks_it(runs):
    manifest, sources = runs
    paths = sources[2]
    with np.load(paths["input"], allow_pickle=False) as stored:
        fields = {key: stored[key] for key in stored.files if key != "split"}
    np.savez_compressed(paths["input"], **fields)
    def update(report):
        row = report["results"][0]
        row["metadata"] = load_npz(paths["input"], seed=73).metadata
        row["provenance"]["input_sha256"] = sha256_file(paths["input"])
    mutate_report(paths["analysis_result"], update)
    report = summarize(manifest)
    assert report["ecosystems"][2]["split_seed"] == 73
    assert report["ecosystems"][2]["split_source"] == "grouped_random_50_25_25"
    mutate_report(paths["analysis_result"], lambda r: r["results"][0]["metadata"].update(requested_split_seed=74))
    with pytest.raises(ValueError, match="Split seed mismatch"):
        summarize(manifest)


def test_single_entry_registry_cannot_redefine_expected_completion(runs):
    manifest, _ = runs
    document = json.loads(manifest.read_text())
    document["ecosystems"] = document["ecosystems"][:1]
    document["expected_ecosystem_count"] = 1
    dump(manifest, document)
    with pytest.raises(ValueError, match="expected_ecosystem_count=3"):
        summarize(manifest)


def test_outputs_preserve_existing_files_and_pending_inputs(runs, tmp_path):
    manifest, sources = runs
    missing = sources[0]["analysis_result"]
    missing.unlink()
    report = summarize(manifest)
    with pytest.raises(ValueError, match="pending artifacts"):
        write_outputs(report, missing)
    assert not missing.exists()
    output = tmp_path / "summary.json"
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        write_outputs(report, output)
    assert output.read_text() == "existing"


def test_evaluated_secondary_case_and_interval_are_copied_without_pooling(runs, tmp_path):
    manifest, sources = runs
    def add_case(report):
        row = report["results"][0]["audits"][-1]
        row.update(status="evaluated", removed_indices=[0], removed_ids=["original1"], kept_indices=[1],
                   loo_cert_errors={"original1": .2, "original2": .5},
                   splits={s: {"individual_max_tv": .2, "joint_tv": .4} for s in ["FIT", "CERT", "TEST"]},
                   test_joint_minus_individual={"gap": .2, "paired_cluster_percentile_interval_95": [.1, .3],
                                                "bootstrap_repetitions": 2000, "base_item_groups": 2})
    mutate_report(sources[0]["analysis_result"], add_case)
    report = summarize(manifest)
    original = json.loads(sources[0]["analysis_result"].read_text())["results"][0]["audits"][-1]
    assert report["ecosystems"][0]["audits"][-1] == original
    output, csv_path = write_outputs(report, tmp_path / "summary.json")
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    row = next(r for r in rows if r["ecosystem"] == "original" and r["gamma"] == "0.3")
    assert row["gamma_role"] == "secondary"
    assert float(row["gap_ci_low"]) == .1
    assert float(row["gap_ci_high"]) == .3
