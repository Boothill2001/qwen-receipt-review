"""Replay a field-level review policy on completed benchmark reports."""

import argparse
import json
from pathlib import Path

from evaluation_metrics import GATES, rate
from receipt_pipeline import FIELDS

INVALID_PREDICTION = frozenset({"INFERENCE_ERROR", "INVALID_OUTPUT", "INVALID_IMAGE"})


def total_review_reasons(row):
    """Require contextual OCR and only the format checks relevant to total."""
    base = row["gate_reasons"][GATES[0]]
    reasons = [reason for reason in base if reason in INVALID_PREDICTION
               or reason in {"MISSING_TOTAL", "TOTAL_FORMAT"}]
    evidence = row["contextual_evidence"]["reason"]
    return reasons + ([evidence] if evidence else [])


def compare_review_workloads(rows):
    legacy = [row for row in rows if row["gate_reasons"][GATES[1]]]
    contextual = [row for row in rows if row["gate_reasons"][GATES[2]]]
    field_level = [row for row in rows if total_review_reasons(row)]
    wrong = sum(not row["match"]["total"] for row in rows)
    policies = {}
    for name, review in (("legacy_record_gate", legacy), ("contextual_record_gate", contextual),
                         ("contextual_total_only", field_level)):
        reviewed_ids = {row["key"] for row in review}
        passed = [row for row in rows if row["key"] not in reviewed_ids]
        wrong_passed = sum(not row["match"]["total"] for row in passed)
        policies[name] = {
            "total_needs_review": rate(len(review), len(rows)),
            "total_exact_among_passed": rate(len(passed) - wrong_passed, len(passed)),
            "wrong_total_sent_to_review": rate(wrong - wrong_passed, wrong),
            "wrong_total_passed": wrong_passed,
            "full_record_errors_among_total_passes": sum(not all(row["match"].values()) for row in passed),
        }
    # Illustrative workload proxy if a reviewer must verify every unapproved field once.
    policies["contextual_total_only"]["fields_to_verify_when_all_required"] = (
        len(rows) * (len(FIELDS) - 1) + len(field_level)
    )
    return policies


def evaluate_report(report):
    rows = report["receipts"]
    ids = report["receipt_ids"]
    if not ids or len(set(ids)) != len(ids) or [row["key"] for row in rows] != ids:
        raise ValueError("Expected a complete report with unique ordered receipt IDs")
    return {
        "method": report["method"], "archive_sha256": report["archive_sha256"],
        "model_revision": report["model_revision"], "cohort": report["cohort"],
        "receipt_ids": ids, "policies": compare_review_workloads(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate_report(json.loads(args.report.read_text(encoding="utf-8"))), indent=2))


if __name__ == "__main__":
    main()
