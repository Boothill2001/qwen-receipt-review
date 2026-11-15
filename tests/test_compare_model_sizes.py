from copy import deepcopy

import pytest

from benchmark_receipts import ARCHIVE_SHA, MODEL_SIZES
from compare_model_sizes import SIZES, check_reports, recommendation, transitions
from receipt_pipeline import PROMPT, PROMPT_VERSION


def pilot_reports():
    ids = [f"receipt-{index}" for index in range(20)]
    common = {
        "method": "qwen", "runtime": "transformers-nf4", "archive_sha256": ARCHIVE_SHA,
        "receipt_ids": ids, "split": "test", "cohort": "pilot",
        "generation_config": {"precision": "nf4", "quantization": {
            "library": "bitsandbytes", "weight_type": "nf4", "compute_dtype": "fp16", "double_quant": True,
        }},
        "prompt_version": PROMPT_VERSION, "prompt": PROMPT,
        "image_config": {}, "ocr_config": {}, "gate_versions": {}, "code_sha256": {}, "environment": {},
        "receipts": [{"key": key, "gold": {"address": "9 Street"}} for key in ids],
    }
    return {size: {**deepcopy(common), "model": MODEL_SIZES[size][0], "model_revision": MODEL_SIZES[size][1]}
            for size in SIZES}


def test_rejects_unpaired_order_and_labels():
    reports = pilot_reports()
    assert check_reports(reports) is reports["7b"]
    reports["32b"]["receipts"].reverse()
    with pytest.raises(ValueError, match="row IDs"):
        check_reports(reports)
    reports["32b"]["receipts"].reverse()
    reports["32b"]["receipts"][0]["gold"]["address"] = "95 Street"
    with pytest.raises(ValueError, match="reference labels"):
        check_reports(reports)


def test_paired_fixes_and_regressions_are_kept_separate():
    fields = ("company", "date", "total", "address", "all_fields", "address_format_tolerant")
    hits = dict.fromkeys(fields, True)
    misses = dict.fromkeys(fields, False)
    result = transitions(["a", "b"], {"a": hits, "b": misses}, {"a": misses, "b": hits})
    assert result["address"] == {"fixed": ["b"], "regressed": ["a"]}


def result(all_fields, p95, vram, total=320, false_passes=2, documents=337):
    return {
        "documents": documents, "normalized_exact": {"all_fields": {"count": all_fields}, "total": {"count": total}},
        "gate_c": {"false_pass_count": false_passes}, "end_to_end_c_ms": {"p95": p95},
        "peak_vram": [{"allocated_bytes": vram}],
    }


def test_decision_requires_net_quality_gain_and_budget():
    summaries = {"7b": result(180, 10, 7), "3b": result(180, 8, 4), "32b": result(197, 20, 14)}
    assert recommendation(summaries)["eligible"] == ["3b", "32b"]
    summaries.update({"3b": result(180, 8, 4, false_passes=3), "32b": result(196, 20, 14)})
    assert recommendation(summaries)["eligible"] == ["7b"]
    summaries["32b"] = result(197, 21, 14)
    assert recommendation(summaries)["eligible"] == ["7b"]
    summaries["7b"]["documents"] = 20
    assert recommendation(summaries) == {"status": "pilot", "eligible": []}
