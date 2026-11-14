"""Compare two aligned Qwen runs and record the address experiment decision."""

import argparse
import json
import re
from pathlib import Path

from address_protocol import ADDRESS_CASES, ADDRESS_PROMPT_VERSION, PROMPTS
from benchmark_receipts import ARCHIVE_SHA, COHORTS, REVISION, atomic_json
from receipt_pipeline import FIELDS, MODEL, PROMPT_VERSION, normalize, sha256_file

PAIR_KEYS = ("method", "model", "model_revision", "archive_sha256", "receipt_ids", "cohort", "split",
             "image_config", "generation_config", "environment", "ocr_config", "code_sha256")
COUNTS = {"address-dev10": len(ADDRESS_CASES), "dev": 50, "full": 337}


def address_format_key(value):
    # Treat selected punctuation as word boundaries. Preserve meaningful slashes and hyphens.
    return " ".join(re.sub(r"[.,;:]+", " ", normalize("address", value) or "").split())


def flags(row):
    values = {field: row["schema_valid"] and normalize(field, row["fields"][field]) is not None
              and normalize(field, row["fields"][field]) == normalize(field, row["gold"][field])
              for field in FIELDS}
    predicted = address_format_key(row["fields"]["address"])
    values["address_format_tolerant"] = bool(
        row["schema_valid"] and predicted and predicted == address_format_key(row["gold"]["address"])
    )
    return values


def compare(baseline, candidate, review_ids=()):
    different = [key for key in PAIR_KEYS if baseline[key] != candidate[key]]
    if different:
        raise ValueError(f"Reports differ in paired settings: {', '.join(different)}")
    for report, prompt_version in zip((baseline, candidate), (PROMPT_VERSION, ADDRESS_PROMPT_VERSION)):
        if report["prompt_version"] != prompt_version or report["prompt"] != PROMPTS[prompt_version]:
            raise ValueError(f"Expected the registered prompt {prompt_version}")
    if (baseline["method"], baseline["model"], baseline["model_revision"], baseline["archive_sha256"]) != (
        "qwen", MODEL, REVISION, ARCHIVE_SHA
    ):
        raise ValueError("Expected the pinned Qwen model and SROIE archive")
    cohort = baseline["cohort"]
    ids = baseline["receipt_ids"]
    if (cohort not in COUNTS or len(ids) != COUNTS[cohort] or len(set(ids)) != len(ids)
            or baseline["split"] != COHORTS[cohort][0]):
        raise ValueError("Unexpected cohort or receipt denominator")
    if cohort == "address-dev10" and ids != list(ADDRESS_CASES):
        raise ValueError("Unexpected address-dev10 IDs or order")
    if not set(review_ids) <= set(ids):
        raise ValueError("Label review IDs must belong to the comparison cohort")
    for report in (baseline, candidate):
        if [row["key"] for row in report["receipts"]] != ids:
            raise ValueError("Receipt rows do not match the ordered cohort")

    cases = []
    for old, new in zip(baseline["receipts"], candidate["receipts"]):
        key = old["key"]
        if old["gold"] != new["gold"]:
            raise ValueError(f"Labels differ for {key}")
        before, after = flags(old), flags(new)
        cases.append({
            "key": key, "case_group": ADDRESS_CASES.get(key), "baseline": old["fields"],
            "candidate": new["fields"], "gold": old["gold"], "before": before, "after": after,
            "address_conclusion": ("fixed" if not before["address"] else "still_correct") if after["address"]
            else ("regressed" if before["address"] else
                  "format_difference" if after["address_format_tolerant"] else "still_incorrect"),
            "inspect_label_against_scan": key in review_ids,
        })
    comparisons = {}
    for field in (*FIELDS, "address_format_tolerant"):
        comparisons[field] = {
            "baseline_correct": sum(row["before"][field] for row in cases),
            "candidate_correct": sum(row["after"][field] for row in cases),
            "fixed": sum(not row["before"][field] and row["after"][field] for row in cases),
            "regressed": sum(row["before"][field] and not row["after"][field] for row in cases),
        }
    valid = [sum(row["schema_valid"] for row in report["receipts"]) for report in (baseline, candidate)]
    criteria = {}
    if cohort == "address-dev10":
        criteria = {
            "both_runs_schema_valid": valid == [10, 10],
            "content_error_fixed": any(row["case_group"] == "content" and
                                       not row["before"]["address_format_tolerant"] and
                                       row["after"]["address_format_tolerant"] for row in cases),
            "controls_still_match": all(all(row["after"][field] for field in FIELDS)
                                        for row in cases if row["case_group"] == "control"),
            "no_format_tolerant_regressions": comparisons["address_format_tolerant"]["regressed"] == 0,
            "no_other_field_regressions": all(comparisons[f]["regressed"] == 0 for f in FIELDS if f != "address"),
        }
    if cohort == "dev":
        criteria = {"exact_address_improves": comparisons["address"]["fixed"] > comparisons["address"]["regressed"],
                    "other_scores_do_not_drop": all(value["fixed"] >= value["regressed"]
                                                   for field, value in comparisons.items() if field != "address")}
    return {
        "cohort": cohort, "documents": len(ids), "receipt_ids": ids,
        "baseline_prompt_version": PROMPT_VERSION, "candidate_prompt_version": ADDRESS_PROMPT_VERSION,
        "archive_sha256": ARCHIVE_SHA, "model_revision": REVISION, "schema_valid": valid,
        "metrics": comparisons, "criteria": criteria,
        "advance": all(criteria.values()) if criteria else None, "receipts": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inspect-label", action="append", default=[], metavar="RECEIPT_ID")
    args = parser.parse_args()
    if args.output.resolve() in {args.baseline.resolve(), args.candidate.resolve()}:
        parser.error("Comparison output must be separate from the input reports")
    result = compare(*(json.loads(path.read_text(encoding="utf-8")) for path in (args.baseline, args.candidate)),
                     review_ids=args.inspect_label)
    result["report_sha256"] = {name: sha256_file(path) for name, path in
                               (("baseline", args.baseline), ("candidate", args.candidate))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "receipts"}, indent=2))


if __name__ == "__main__":
    main()
