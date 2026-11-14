import json
from pathlib import Path

from PIL import Image

from evaluation_metrics import GATES
from field_review import compare_review_workloads, total_review_reasons
from receipt_pipeline import parse_output

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "replay"


def test_real_cases_cover_correct_hold_wrong_total_and_address_escape():
    bundle = json.loads((ASSETS / "cases.json").read_text(encoding="utf-8"))
    assert bundle["provenance"]["documents"] == 337
    correct, wrong_total, wrong_address = [case["row"] for case in bundle["cases"]]
    assert all(correct["match"].values()) and correct["gate_reasons"][GATES[1]]
    assert not wrong_total["match"]["total"] and total_review_reasons(wrong_total) == ["TOTAL_EVIDENCE_MISMATCH"]
    assert wrong_address["match"]["total"] and not wrong_address["match"]["address"]
    assert wrong_address["gate_reasons"][GATES[1]] and not wrong_address["gate_reasons"][GATES[2]]
    assert not total_review_reasons(wrong_address)
    for row in (correct, wrong_total, wrong_address):
        assert parse_output(row["raw"]) == row["fields"]
        with Image.open(ASSETS / f"{row['key']}.webp") as image:
            for evidence in (row["evidence"], row["contextual_evidence"]):
                for candidate in evidence["candidates"]:
                    coordinates = candidate["polygon"]
                    assert all(0 <= x <= image.width for x in coordinates[::2])
                    assert all(0 <= y <= image.height for y in coordinates[1::2])


def test_total_policy_blocks_invalid_predictions_and_ignores_date_format():
    row = {
        "key": "sample", "gate_reasons": {gate: ["DATE_FORMAT"] for gate in GATES},
        "contextual_evidence": {"reason": None},
        "match": {"total": True, "date": False, "company": True, "address": True},
    }
    result = compare_review_workloads([row])["contextual_total_only"]
    assert result["total_needs_review"]["count"] == 0
    assert result["total_exact_among_passed"]["count"] == 1
    assert result["full_record_errors_among_total_passes"] == 1
    assert result["fields_to_verify_when_all_required"] == 3
    row["gate_reasons"][GATES[0]] = ["INVALID_OUTPUT"]
    assert total_review_reasons(row) == ["INVALID_OUTPUT"]
