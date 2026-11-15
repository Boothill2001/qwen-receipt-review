"""Compare three paired SROIE runs using the same NF4 recipe."""

import argparse
import json
from pathlib import Path

from benchmark_receipts import ARCHIVE_SHA, MODEL_SIZES, atomic_json
from compare_addresses import flags
from evaluation_metrics import GATES, metrics
from receipt_pipeline import FIELDS, PROMPT, PROMPT_VERSION, sha256_file

SIZES = ("3b", "7b", "32b")
COHORT_COUNTS = {"pilot": 20, "full": 337}
ALIGN = ("archive_sha256", "receipt_ids", "cohort", "split", "image_config", "generation_config",
         "ocr_config", "gate_versions", "code_sha256", "environment", "prompt_version", "prompt")
DECISION_GATE = GATES[2]


def check_reports(reports):
    baseline = reports["7b"]
    n = COHORT_COUNTS.get(baseline["cohort"])
    ids = baseline["receipt_ids"]
    if n is None or len(ids) != n or len(set(ids)) != n or baseline["split"] != "test":
        raise ValueError("Expected the fixed pilot or full SROIE test cohort")
    if baseline["archive_sha256"] != ARCHIVE_SHA:
        raise ValueError("Unexpected SROIE archive hash")
    if (baseline["prompt_version"], baseline["prompt"]) != (PROMPT_VERSION, PROMPT):
        raise ValueError("Expected the original extraction prompt")
    expected_quantization = {
        "library": "bitsandbytes", "weight_type": "nf4", "compute_dtype": "fp16", "double_quant": True,
    }
    if (baseline["generation_config"]["precision"] != "nf4"
            or baseline["generation_config"].get("quantization") != expected_quantization):
        raise ValueError("Expected the matched NF4 recipe")
    gold = [row["gold"] for row in baseline["receipts"]]
    for size in SIZES:
        report = reports[size]
        if ((report["model"], report["model_revision"]) != MODEL_SIZES[size]
                or report["method"] != "qwen" or report["runtime"] != "transformers-nf4"):
            raise ValueError(f"Unexpected checkpoint or runtime for {size}")
        mismatches = [key for key in ALIGN if report[key] != baseline[key]]
        if mismatches or [row["key"] for row in report["receipts"]] != ids:
            raise ValueError(f"{size} is not paired with 7b ({', '.join(mismatches) or 'row IDs'})")
        if [row["gold"] for row in report["receipts"]] != gold:
            raise ValueError(f"{size} uses different reference labels")
    return baseline


def peaks(report):
    peak = {}
    batches = [row.get("session_cuda_peaks", []) for row in report["receipts"]]
    batches += [session["load_peaks"] for session in report["sessions"].values()]
    for batch in batches:
        for row in batch:
            device = row["device"]
            entry = peak.setdefault(device, {"allocated_bytes": 0, "reserved_bytes": 0})
            entry["allocated_bytes"] = max(entry["allocated_bytes"], row["peak_allocated_bytes"])
            entry["reserved_bytes"] = max(entry["reserved_bytes"], row["peak_reserved_bytes"])
    return [{"device": device, **peak[device]} for device in sorted(peak)]


def transitions(ids, baseline, candidate):
    fields = (*FIELDS, "all_fields", "address_format_tolerant")
    return {
        field: {
            "fixed": [key for key in ids if not baseline[key][field] and candidate[key][field]],
            "regressed": [key for key in ids if baseline[key][field] and not candidate[key][field]],
        }
        for field in fields
    }


def summarize(report):
    rows = report["receipts"]
    measured = metrics(rows)
    matches = {row["key"]: flags(row) | {"all_fields": all(row["match"].values())} for row in rows}
    summary = {
        "documents": len(rows),
        "normalized_exact": measured["normalized_exact"],
        "address_format_tolerant": sum(row["address_format_tolerant"] for row in matches.values()),
        "inference_errors": measured["inference_errors"],
        "schema_valid": measured["schema_valid"],
        "gate_c": measured["gates"][DECISION_GATE],
        "end_to_end_c_ms": measured["latency_ms"]["end_to_end_C"],
        "peak_vram": peaks(report),
    }
    return summary, matches


def recommendation(summaries):
    baseline = summaries["7b"]
    rows = baseline["documents"]
    required_gain = 17
    if rows != COHORT_COUNTS["full"]:
        return {"status": "pilot", "eligible": []}
    def correct(model, field):
        return model["normalized_exact"][field]["count"]

    def latency(model):
        return model["end_to_end_c_ms"]["p95"]

    def memory(model):
        return max((entry["allocated_bytes"] for entry in model["peak_vram"]), default=None)
    if any(latency(row) is None or memory(row) is None for row in summaries.values()):
        return {"status": "incomplete_resource_measurement", "eligible": []}
    eligible = []
    for size in ("3b", "32b"):
        candidate = summaries[size]
        if (correct(candidate, "total") < correct(baseline, "total")
                or candidate["gate_c"]["false_pass_count"] > baseline["gate_c"]["false_pass_count"]):
            continue
        gain = correct(candidate, "all_fields") - correct(baseline, "all_fields")
        if size == "3b" and gain >= 0 and (latency(candidate) < latency(baseline)
                                           or memory(candidate) < memory(baseline)):
            eligible.append(size)
        if size == "32b" and gain >= required_gain and latency(candidate) <= 2 * latency(baseline):
            eligible.append(size)
    return {"status": "full", "eligible": eligible or ["7b"], "32b_min_net_all_fields": required_gain,
            "32b_max_p95_multiple": 2}


def compare(reports):
    baseline = check_reports(reports)
    summaries, flags_by_model = {}, {}
    for size in SIZES:
        summaries[size], flags_by_model[size] = summarize(reports[size])
    ids = baseline["receipt_ids"]
    return {
        "cohort": baseline["cohort"], "receipt_ids": ids,
        "archive_sha256": baseline["archive_sha256"], "prompt_version": baseline["prompt_version"],
        "image_config": baseline["image_config"], "generation_config": baseline["generation_config"],
        "gate_versions": baseline["gate_versions"], "environment": baseline["environment"],
        "models": {size: {"model": reports[size]["model"], "revision": reports[size]["model_revision"],
                          **summaries[size]} for size in SIZES},
        "paired_against_7b": {size: transitions(ids, flags_by_model["7b"], flags_by_model[size])
                             for size in ("3b", "32b")},
        "recommendation": recommendation(summaries),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for size in SIZES:
        parser.add_argument(f"--report-{size}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    paths = {size: args[f"report_{size}"] for size in SIZES}
    if output.resolve() in [path.resolve() for path in paths.values()] or output.suffix != ".json":
        parser.error("Use a separate .json output path")
    result = compare({size: json.loads(path.read_text(encoding="utf-8")) for size, path in paths.items()})
    result["source_reports_sha256"] = {size: sha256_file(path) for size, path in paths.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)


if __name__ == "__main__":
    main()
