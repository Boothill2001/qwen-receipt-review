"""Paired evaluation of extraction quality and review policy."""

import math
import re
import statistics

from receipt_pipeline import FIELDS, MONEY_TEXT, TOTAL_LINE, normalize

CONTEXT_GATE_VERSION = "contextual-total-v1"
MIN_LINE_CONFIDENCE = 60
EXPLICIT_TOTAL = re.compile(
    r"\b(?:grand total|nett? total|total payable|amount due|balance due|"
    r"total sales\s*\(?\s*(?:inclusive of|incl\.?)\s*(?:gst|vat|tax))\b", re.I,
)
NON_PAYABLE = re.compile(
    r"\b(?:sub\s*-?\s*total|tax|gst|vat|excl\w*|before|pre[- ]?tax|saving\w*|"
    r"discount|change|cash|tendered|items?|qty|quantity|rounding)\b", re.I,
)
OTHER_AMOUNT_CONTEXT = re.compile(r"\b(?:excl\w*|before|pre[- ]?tax|"
                                  r"discount|change|cash|tendered|qty|quantity|rounding)\b", re.I)
GATES = ("A_format_only", "B_legacy_ocr", "C_contextual_ocr")


def contextual_evidence(lines, predicted_total):
    candidates = []
    for line in lines:
        label = " ".join(line["text"].split())
        if OTHER_AMOUNT_CONTEXT.search(label):
            continue
        if EXPLICIT_TOTAL.search(label):
            tier = "payable"
        elif TOTAL_LINE.search(label) and not NON_PAYABLE.search(label):
            tier = "generic"
        else:
            continue
        for amount in MONEY_TEXT.findall(label):
            candidates.append({**line, "amount": normalize("total", amount), "tier": tier})
    payable = [row for row in candidates if row["tier"] == "payable"]
    selected = payable or candidates
    amounts = {row["amount"] for row in selected}
    if len(amounts) > 1:
        reason = "TOTAL_EVIDENCE_CONFLICT"
    elif not selected or any(row["confidence"] < MIN_LINE_CONFIDENCE for row in selected):
        reason = "TOTAL_EVIDENCE_UNCLEAR"
    elif normalize("total", predicted_total) not in amounts:
        reason = "TOTAL_EVIDENCE_MISMATCH"
    else:
        reason = None
    return {"reason": reason, "candidates": candidates}


def rate(numerator, denominator):
    if not denominator:
        return {"count": numerator, "n": denominator, "rate": None, "wilson_95": None}
    p = numerator / denominator
    z = 1.95996398454
    divisor = 1 + z**2 / denominator
    center = (p + z**2 / (2 * denominator)) / divisor
    margin = z * math.sqrt(p * (1 - p) / denominator + z**2 / (4 * denominator**2)) / divisor
    return {"count": numerator, "n": denominator, "rate": p,
            "wilson_95": [max(0, center - margin), min(1, center + margin)]}


def distribution(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"n": 0, "p50": None, "p95": None}
    return {"n": len(values), "p50": round(statistics.median(values), 3),
            "p95": values[math.ceil(0.95 * len(values)) - 1]}


def gate_metrics(rows, gate):
    passed = [row for row in rows if not row["gate_reasons"][gate]]
    review = [row for row in rows if row["gate_reasons"][gate]]
    wrong_totals = [row for row in rows if not row["match"]["total"]
                    and normalize("total", row["fields"]["total"]) is not None]
    flagged = sum(bool(row["gate_reasons"][gate]) for row in wrong_totals)
    incorrect_passed = sum(not all(row["match"].values()) for row in passed)
    correct_flagged = sum(all(row["match"].values()) for row in review)
    return {
        "coverage": rate(len(passed), len(rows)),
        "selective_risk": rate(incorrect_passed, len(passed)),
        "review_workload": rate(len(review), len(rows)),
        "wrong_total_detected": rate(flagged, len(wrong_totals)),
        "wrong_total_stopped_after_format": sum(not row["gate_reasons"][GATES[0]]
                                                and bool(row["gate_reasons"][gate]) for row in wrong_totals),
        "missing_total": sum(normalize("total", row["fields"]["total"]) is None for row in rows),
        "false_pass_count": incorrect_passed,
        "correct_flagged_count": correct_flagged,
    }


def metrics(rows):
    def exact(field):
        return rate(sum(row["schema_valid"] and row["gold"][field] is not None
                        and row["fields"][field] == row["gold"][field] for row in rows), len(rows))

    stages = ("preprocess", "inference", "ocr", "validation", "legacy_evidence",
              "context_evidence", "end_to_end_A", "end_to_end_B", "end_to_end_C", "evaluation_wall")
    return {
        "documents": len(rows),
        "schema_valid": rate(sum(row["schema_valid"] for row in rows), len(rows)),
        "raw_exact": {**{key: exact(key) for key in FIELDS},
                      "all_fields": rate(sum(row["schema_valid"] and all(
                          row["gold"][key] is not None and row["fields"][key] == row["gold"][key]
                          for key in FIELDS) for row in rows), len(rows))},
        "normalized_exact": {**{key: rate(sum(row["match"][key] for row in rows), len(rows)) for key in FIELDS},
                             "all_fields": rate(sum(all(row["match"].values()) for row in rows), len(rows))},
        "gates": {gate: gate_metrics(rows, gate) for gate in GATES},
        "latency_ms": {stage: distribution(row["timings_ms"].get(stage) for row in rows) for stage in stages},
        "generation": {key: distribution(row["generation"].get(key) for row in rows)
                       for key in ("generation_ms", "input_tokens", "visual_tokens", "output_tokens")},
        "inference_errors": sum("INFERENCE_ERROR" in row["gate_reasons"][GATES[0]] for row in rows),
        "invalid_images": sum("INVALID_IMAGE" in row["gate_reasons"][GATES[0]] for row in rows),
        "ocr_errors": sum(bool(row["ocr_error"]) for row in rows),
    }
