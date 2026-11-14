from copy import deepcopy

import pytest

from address_protocol import ADDRESS_CASES, ADDRESS_PROMPT_VERSION, PROMPTS
from benchmark_receipts import ARCHIVE_SHA, REVISION
from compare_addresses import PAIR_KEYS, address_format_key, compare
from receipt_pipeline import MODEL, PROMPT_VERSION


def pair():
    gold = {"company": "A", "date": "01/01/2020", "address": "NO. 12, ROAD 23/1", "total": "10.00"}
    baseline = dict.fromkeys(PAIR_KEYS)
    baseline.update(method="qwen", model=MODEL, model_revision=REVISION, archive_sha256=ARCHIVE_SHA,
                    cohort="address-dev10", split="train", receipt_ids=list(ADDRESS_CASES),
                    prompt_version=PROMPT_VERSION, prompt=PROMPTS[PROMPT_VERSION],
                    receipts=[{"key": key, "schema_valid": True, "gold": gold.copy(),
                               "fields": {**gold, "address": "NO 12 ROAD 23/4" if group == "content"
                                          else "NO 12 ROAD 23/1" if group == "format" else gold["address"]}}
                              for key, group in ADDRESS_CASES.items()])
    candidate = deepcopy(baseline)
    candidate.update(prompt_version=ADDRESS_PROMPT_VERSION, prompt=PROMPTS[ADDRESS_PROMPT_VERSION])
    candidate["receipts"][3]["fields"]["address"] = gold["address"]
    return baseline, candidate


def test_format_preserves_address_tokens():
    assert address_format_key("NO.32, ROAD 23/1") == address_format_key("no 32 road 23/1")
    assert address_format_key("23/1") != address_format_key("23/4")
    assert address_format_key("7-A") != address_format_key("7 A")


def test_stage_gate_and_label_inspection():
    baseline, candidate = pair()
    result = compare(baseline, candidate, review_ids=[baseline["receipt_ids"][3]])
    assert result["advance"] is True
    assert result["metrics"]["address"]["fixed"] == 1
    assert result["receipts"][3]["inspect_label_against_scan"]
    candidate["receipts"][0]["fields"]["total"] = "11.00"
    assert not compare(baseline, candidate)["criteria"]["no_other_field_regressions"]
    candidate["receipts"][0]["schema_valid"] = False
    assert not compare(baseline, candidate)["criteria"]["both_runs_schema_valid"]


def test_reject_different_reports():
    baseline, candidate = pair()
    candidate["environment"] = {"device": "another GPU"}
    with pytest.raises(ValueError, match="environment"):
        compare(baseline, candidate)
    baseline, candidate = pair()
    candidate["receipts"][0]["gold"]["address"] = "Altered label"
    with pytest.raises(ValueError, match="Labels differ"):
        compare(baseline, candidate)
