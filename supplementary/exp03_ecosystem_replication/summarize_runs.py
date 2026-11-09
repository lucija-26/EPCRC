"""Verify and summarize existing ecosystem reports, without fitting any model.

Missing reports or responses produce an explicit pending summary. Complete
means every declared ecosystem has passed verification, not merely that a
single-panel report declares itself complete. Existing outputs are refused.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from supplementary.common import load_npz, sha256_file
from supplementary.exp03_ecosystem_replication.run import validate_manifest

DATASET_KEYS = ("dataset_id", "config", "split", "revision")
SPLITS = ("FIT", "CERT", "TEST")
AUDIT_STATUSES = {"evaluated", "no_individually_removable_endpoints",
                  "all_endpoints_marked_removable_no_executable_nonempty_support"}
CSV_FIELDS = ["ecosystem", "status", "row_type", "n_models", "gamma", "gamma_role",
              "audit_status", "removed_ids", "individual_test_tv", "joint_test_tv",
              "joint_minus_individual", "gap_ci_low", "gap_ci_high",
              "cert_selected_k", "cert_selected_frozen_test_tv", "k", "method",
              "kept_ids", "fit_tv", "cert_tv", "test_tv", "pending_reason"]


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve(base, value):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def artifact(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def require_equal(actual, expected, message):
    if actual != expected:
        raise ValueError(message)


def check_tv(value, field):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not np.isfinite(value) or not 0 <= value <= 1 + 1e-8):
        raise ValueError(f"Invalid TV value: {field}")


def registry(manifest_path):
    path = Path(manifest_path).resolve()
    document = read_json(path)
    entries = validate_manifest(document["ecosystems"])
    # This deliverable covers the three predeclared panels, not an arbitrary
    # one-panel subset that could be relabelled as a completed comparison.
    require_equal(document.get("expected_ecosystem_count"), 3,
                  "Declare expected_ecosystem_count=3 for the three-panel summary")
    require_equal(len(entries), 3, "Exactly three ecosystem entries are required")
    gammas = document.get("gammas", [])
    if (not gammas or len(set(gammas)) != len(gammas)
            or any(not isinstance(g, (int, float)) or not 0 <= g <= 1 for g in gammas)
            or document.get("primary_gamma") not in gammas):
        raise ValueError("Declare unique gamma values including the primary gamma")
    prepared = []
    for entry in entries:
        paths = {key: resolve(path.parent, entry[key]) for key in
                 ("input", "analysis_result", "locked_manifest", "pairs", "source_provenance")}
        locked = read_json(paths["locked_manifest"])
        if locked.get("interface") not in {"judge", "nli"}:
            raise ValueError("Locked registry interface must be judge or nli")
        endpoints = locked.get("endpoints", [])
        ids = [e["judge_id"] for e in endpoints]
        models = [e["model_id"] for e in endpoints]
        require_equal(locked.get("name"), entry["name"], "Locked registry name mismatch")
        require_equal(models, entry["model_ids"], "Declared model IDs/order differ from locked registry")
        if (len(ids) != len(set(ids)) or any(not re.fullmatch(r"[0-9a-f]{40}", e.get("revision", ""))
                                          for e in endpoints)):
            raise ValueError("Locked registry needs unique endpoint IDs and full checkpoint revisions")
        families = entry.get("model_families", [])
        if len(families) != len(models) or any(not isinstance(f, str) or not f.strip() for f in families):
            raise ValueError("Provide one explicit family label per declared model")
        budgets = entry.get("budgets", [])
        if (not budgets or len(set(budgets)) != len(budgets)
                or any(type(k) is not int or not 1 <= k <= len(models) for k in budgets)):
            raise ValueError("Declared budgets must be distinct valid endpoint counts")
        if set(entry.get("dataset", {})) != set(DATASET_KEYS):
            raise ValueError("Declare exact dataset_id/config/split/revision provenance")
        prepared.append({"entry": entry, "paths": paths, "locked": locked,
                         "locked_artifact": artifact(paths["locked_manifest"])})
    return path, document, prepared


def verify_panel(prepared, gammas, primary_gamma):
    entry, paths, locked = (prepared[k] for k in ("entry", "paths", "locked"))
    result = {"name": entry["name"], "status": "pending", "relationship": entry["relationship"],
              "assembly_protocol": entry["assembly_protocol"], "dataset": entry["dataset"],
              "interface": locked["interface"], "model_ids": entry["model_ids"],
              "model_families": entry["model_families"], "n_models": len(entry["model_ids"]),
              "declared_budgets": entry["budgets"], "locked_manifest": prepared["locked_artifact"],
              "declared_artifacts": {key: str(path) for key, path in paths.items()}}
    missing = [key for key in ("input", "analysis_result") if not paths[key].is_file()]
    if missing:
        result["pending_reason"] = "missing_artifacts"
        result["missing_artifacts"] = {key: str(paths[key]) for key in missing}
        return result
    report = read_json(paths["analysis_result"])
    require_equal(report.get("experiment"), "ecosystem_replication", "Not an exp03 report")
    if report.get("declared_manifest_complete") is not True:
        result["pending_reason"] = "source_analysis_incomplete"
        return result
    require_equal(len(report.get("results", [])), 1, "Expected one ecosystem in each source report")
    original = report["results"][0]
    hashes = {key: artifact(path) for key, path in paths.items()}
    require_equal(hashes["locked_manifest"], prepared["locked_artifact"], "Locked registry changed during read")
    provenance, metadata = original["provenance"], original["metadata"]
    require_equal(provenance.get("name"), entry["name"], "Report ecosystem name mismatch")
    require_equal(provenance.get("relationship"), entry["relationship"], "Report relationship mismatch")
    require_equal(provenance.get("model_ids"), entry["model_ids"], "Report model order mismatch")
    require_equal(provenance.get("input_sha256"), hashes["input"]["sha256"], "Report response SHA256 mismatch")
    require_equal(metadata.get("input_sha256"), hashes["input"]["sha256"], "Metadata response SHA256 mismatch")
    seed = metadata.get("requested_split_seed")
    if type(seed) is not int:
        raise ValueError("Source report must record the actual requested split seed")
    data = load_npz(paths["input"], seed=seed)
    require_equal(original.get("judge_ids"), data.judge_ids, "Report endpoint IDs/order mismatch")
    require_equal(original.get("n_judges"), len(data.judge_ids), "Report endpoint count mismatch")
    require_equal(data.judge_ids, [e["judge_id"] for e in locked["endpoints"]], "NPZ endpoint order differs from registry")
    for label, meta in (("NPZ", data.metadata), ("report", metadata)):
        if meta.get("synthetic") is not False:
            raise ValueError(f"{label} must explicitly declare synthetic=false")
        require_equal(meta.get("manifest"), locked, f"{label} pinned model IDs/revisions differ from registry")
        require_equal(meta.get("model_ids"), entry["model_ids"], f"{label} model IDs differ from registry")
        require_equal(meta.get("interface"), locked["interface"], f"{label} interface mismatch")
        protocol = meta.get("collection_protocol", {})
        require_equal(protocol.get("manifest_sha256"), hashes["locked_manifest"]["sha256"],
                      f"{label} locked-manifest SHA256 mismatch")
        require_equal(protocol.get("pairs_sha256"), hashes["pairs"]["sha256"], f"{label} source-pairs SHA256 mismatch")
        require_equal(protocol.get("interface"), locked["interface"], f"{label} collected interface mismatch")
    require_equal(metadata.get("collection_protocol"), data.metadata.get("collection_protocol"),
                  "Report and NPZ collection protocols differ")
    require_equal(metadata.get("split_source"), data.metadata.get("split_source"), "Split source mismatch")
    require_equal(metadata.get("split_seed"), data.metadata.get("split_seed"), "Split seed mismatch")
    require_equal(metadata.get("split_manifest_sha256"), data.metadata.get("split_manifest_sha256"),
                  "Stored split-manifest SHA256 mismatch")
    source = read_json(paths["source_provenance"])
    require_equal({key: source.get(key) for key in DATASET_KEYS}, entry["dataset"], "Declared dataset provenance mismatch")
    require_equal(source.get("file_sha256"), hashes["pairs"]["sha256"], "Dataset sidecar source SHA256 mismatch")
    for label, meta in (("NPZ", data.metadata), ("report", metadata)):
        require_equal(meta.get("input_provenance"), source, f"{label} dataset provenance differs from source sidecar")
    expected_labels = ["A", "B", "C"] if locked["interface"] == "judge" else ["entailment", "neutral", "contradiction"]
    require_equal(data.metadata.get("canonical_labels"), expected_labels, "NPZ canonical label order mismatch")
    require_equal(metadata.get("canonical_labels"), expected_labels, "Report canonical label order mismatch")
    with paths["pairs"].open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    with np.load(paths["input"], allow_pickle=False) as stored:
        require_equal(stored["pair_ids"].astype(str).tolist(), [r["pair_id"] for r in rows], "Source pair IDs/order mismatch")
        require_equal(stored["groups"].astype(str).tolist(), [r["base_item_id"] for r in rows], "Source groups/order mismatch")
    require_equal(source.get("n_rows"), len(rows), "Dataset source row count mismatch")
    frontier = original.get("frontier", [])
    require_equal([r["k"] for r in frontier], entry["budgets"], "Actual exp03 budgets differ from declared budgets")
    for row in frontier:
        kept = row.get("kept_indices", [])
        if (len(kept) != row["k"] or len(set(kept)) != len(kept)
                or any(type(j) is not int or not 0 <= j < len(data.judge_ids) for j in kept)):
            raise ValueError("Invalid frozen frontier support")
        require_equal(row.get("kept_ids"), [data.judge_ids[j] for j in kept], "Frontier support IDs/order mismatch")
        for split in SPLITS:
            check_tv(row[f"{split.lower()}_tv"], f"frontier {split}")
    audits = original.get("audits", [])
    choices = original.get("cert_selected_budget", [])
    require_equal([row["gamma"] for row in audits], gammas, "Actual audit gamma values differ from registry")
    require_equal([row["gamma"] for row in choices], gammas, "Actual budget-choice gamma values differ from registry")
    require_equal(report.get("protocol", {}).get("primary_gamma"), primary_gamma, "Primary gamma mismatch")
    require_equal(report.get("protocol", {}).get("primary_gamma_included"), True, "Primary gamma is missing")
    for row in audits:
        if row.get("status") not in AUDIT_STATUSES:
            raise ValueError("Unknown audit status")
        require_equal(row.get("selection_split"), "CERT", "Removal set must be frozen on CERT")
        removed = row.get("removed_indices", [])
        if (len(removed) != len(set(removed))
                or any(type(j) is not int or not 0 <= j < len(data.judge_ids) for j in removed)):
            raise ValueError("Invalid frozen removal set")
        loo_cert = row.get("loo_cert_errors")
        if not isinstance(loo_cert, dict) or set(loo_cert) != set(data.judge_ids):
            raise ValueError("LOO CERT errors must name every endpoint exactly once")
        for judge_id, error in loo_cert.items():
            check_tv(error, f"LOO CERT/{judge_id}")
        expected_removed = [j for j, judge_id in enumerate(data.judge_ids)
                            if loo_cert[judge_id] <= row["gamma"]]
        require_equal(sorted(removed), expected_removed,
                      "Removal set disagrees with LOO CERT errors <= gamma")
        require_equal(row.get("removed_ids"), [data.judge_ids[j] for j in removed], "Removal IDs/order mismatch")
        require_equal(row.get("kept_indices"), sorted(set(range(len(data.judge_ids))) - set(removed)),
                      "Removal/retention sets do not partition the ecosystem")
        if row["status"] == "evaluated":
            if not removed or not row["kept_indices"]:
                raise ValueError("Evaluated removal requires nonempty removed and retained sets")
            for split in SPLITS:
                for key in ("joint_tv", "individual_max_tv"):
                    check_tv(row["splits"][split][key], f"audit {split}/{key}")
            gap = row["test_joint_minus_individual"]
            interval = gap["paired_cluster_percentile_interval_95"]
            expected_gap = row["splits"]["TEST"]["joint_tv"] - row["splits"]["TEST"]["individual_max_tv"]
            if (len(interval) != 2 or not np.isfinite(interval).all() or interval[0] > interval[1]
                    or not np.isclose(gap["gap"], expected_gap, rtol=1e-10, atol=1e-12)):
                raise ValueError("Frozen joint-gap summary is invalid or inconsistent")
            require_equal(gap.get("base_item_groups"), len(set(data.groups["TEST"])), "Gap bootstrap group count mismatch")
        elif row["status"] == "no_individually_removable_endpoints" and removed:
            raise ValueError("No-removal status contradicts the removal set")
        elif row["status"] == "all_endpoints_marked_removable_no_executable_nonempty_support" and row["kept_indices"]:
            raise ValueError("All-removable status contradicts the retained set")
    # Recompute only the documented lookup over the frozen frontier, not any
    # weights, selection, bootstrap, or risks. TEST never chooses the budget.
    for choice in choices:
        accepted = [r for r in frontier if r["cert_tv"] <= choice["gamma"]]
        selected = min(accepted, key=lambda r: r["k"]) if accepted else None
        require_equal(choice.get("smallest_tested_k_accepted_on_cert"), selected["k"] if selected else None,
                      "CERT budget choice disagrees with frozen frontier")
        require_equal(choice.get("frozen_test_tv"), selected["test_tv"] if selected else None,
                      "Frozen TEST risk disagrees with selected budget")
        require_equal(choice.get("is_global_minimum"), False, "Tested frontier is not a global-minimum claim")
    result.update(status="complete", source_artifacts=hashes,
                  input_provenance=source, canonical_labels=expected_labels,
                  context_names=data.context_names,
                  split_audit={split: {"n_pairs": len(data.groups[split]),
                                      "n_groups": len(set(data.groups[split])),
                                      "group_sha256": hashlib.sha256("\n".join(sorted(set(data.groups[split]))).encode()).hexdigest()}
                               for split in SPLITS},
                  split_source=data.metadata["split_source"], split_seed=data.metadata["split_seed"],
                  audits=copy.deepcopy(audits), frontier=copy.deepcopy(frontier),
                  cert_selected_budget=copy.deepcopy(choices), source_protocol=report["protocol"])
    for item in hashes.values():
        require_equal(sha256_file(item["path"]), item["sha256"], "Source artifact changed during verification")
    return result


def summarize(manifest_path):
    path, document, prepared = registry(manifest_path)
    source_manifest = artifact(path)
    panels = [verify_panel(p, document["gammas"], document["primary_gamma"]) for p in prepared]
    overlaps = []
    for i, first in enumerate(panels):
        for second in panels[i + 1:]:
            x, y = set(first["model_ids"]), set(second["model_ids"])
            common = sorted(x & y)
            overlaps.append({"first": first["name"], "second": second["name"],
                             "exact_model_ids_shared": common, "model_overlap_count": len(common),
                             "model_id_jaccard": len(common) / len(x | y),
                             "shared_declared_families": sorted(set(first["model_families"]) & set(second["model_families"])),
                             "same_declared_dataset": first["dataset"] == second["dataset"],
                             "same_interface": first["interface"] == second["interface"],
                             "both_results_verified": first["status"] == second["status"] == "complete"})
    require_equal(sha256_file(path), source_manifest["sha256"], "Aggregate manifest changed during verification")
    completed = sum(p["status"] == "complete" for p in panels)
    return {"experiment": "verified_cross_ecosystem_summary", "status": "complete" if completed == 3 else "pending",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "expected_ecosystem_count": 3, "verified_complete_ecosystem_count": completed,
            "source_manifest": source_manifest, "summary_script_sha256": sha256_file(__file__),
            "primary_gamma": document["primary_gamma"], "gammas": document["gammas"],
            "family_label_note": document.get("family_label_note"), "model_overlap_audit": overlaps,
            "ecosystems": panels,
            "interpretation": [
                "Existing frozen reports only; no inference, fitting, subset search, or new bootstrap was performed.",
                "All declared gamma values and actually tested budgets are retained; primary and secondary analyses remain distinct.",
                "Single-panel completion never implies completion of this three-panel deliverable.",
                "Exact model-ID disjointness does not establish independent model sampling, families, or training corpora.",
                "Core20 and judges8 reuse RewardBench2; NLI changes task/label semantics but all three use three-class probability outputs.",
                "No cross-ecosystem p-value pooling or uncertainty over ecosystem sampling is asserted.",
                "Frozen-witness observed risks and conditional bootstrap intervals are not global-optimality or population-risk certificates.",
            ]}


def csv_rows(summary):
    rows = []
    for panel in summary["ecosystems"]:
        base = {"ecosystem": panel["name"], "status": panel["status"], "n_models": panel["n_models"]}
        if panel["status"] != "complete":
            rows.append(dict(base, row_type="pending", pending_reason=panel["pending_reason"]))
            continue
        choices = {r["gamma"]: r for r in panel["cert_selected_budget"]}
        for audit in panel["audits"]:
            choice = choices[audit["gamma"]]
            row = dict(base, row_type="gamma_audit", gamma=audit["gamma"],
                       gamma_role="primary" if audit["gamma"] == summary["primary_gamma"] else "secondary",
                       audit_status=audit["status"], removed_ids="|".join(audit["removed_ids"]),
                       cert_selected_k=choice["smallest_tested_k_accepted_on_cert"],
                       cert_selected_frozen_test_tv=choice["frozen_test_tv"])
            if audit["status"] == "evaluated":
                metrics = audit["splits"]["TEST"]
                gap = audit["test_joint_minus_individual"]
                row.update(individual_test_tv=metrics["individual_max_tv"], joint_test_tv=metrics["joint_tv"],
                           joint_minus_individual=gap["gap"],
                           gap_ci_low=gap["paired_cluster_percentile_interval_95"][0],
                           gap_ci_high=gap["paired_cluster_percentile_interval_95"][1])
            rows.append(row)
        for point in panel["frontier"]:
            rows.append(dict(base, row_type="frontier", k=point["k"], method=point["method"],
                             kept_ids="|".join(point["kept_ids"]),
                             **{key: point[key] for key in ("fit_tv", "cert_tv", "test_tv")}))
    return rows


def write_outputs(summary, output, csv_output=None):
    output = Path(output).resolve()
    csv_output = Path(csv_output).resolve() if csv_output else output.with_suffix(".csv")
    if output == csv_output:
        raise ValueError("JSON and CSV destinations must differ")
    inputs = {Path(summary["source_manifest"]["path"]).resolve(), Path(__file__).resolve()}
    for panel in summary["ecosystems"]:
        inputs.update(Path(path).resolve() for path in panel["declared_artifacts"].values())
    if output in inputs or csv_output in inputs:
        raise ValueError("Output must be separate from every declared source, including pending artifacts")
    if output.exists() or csv_output.exists():
        raise FileExistsError("Refusing to overwrite an existing summary or CSV")
    rows = csv_rows(summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    with csv_output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output, csv_output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("ecosystems.json"))
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; existing files are refused")
    parser.add_argument("--csv", type=Path, help="New CSV file; default is the JSON path with .csv suffix")
    args = parser.parse_args(argv)
    summary = summarize(args.manifest)
    paths = write_outputs(summary, args.output, args.csv)
    print(f"{summary['status']}: {summary['verified_complete_ecosystem_count']}/3 verified; {paths[0]}; {paths[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
