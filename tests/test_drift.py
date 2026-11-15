from io import BytesIO
import json

import pytest
from PIL import Image

from drift_data import is_duplicate, reading_order
from benchmark_receipts import REVISION
from drift_experiment import CROP_FRACTION, CROP_PROMPT, audited_rows, decision, holdout_allowed, score, stats
from receipt_pipeline import MAX_PIXELS, PROMPT


def test_reading_order_and_duplicate_filter():
    def box(x, y, text, label=3):
        return {"box": [x, y, x+10, y, x+10, y+10, x, y+10], "text": text, "label": label}

    rows = [box(0, 30, "PALO ALTO"), box(30, 11, "MAIN ST"), box(5, 10, "90"), box(0, 0, "STORE", 1)]
    assert reading_order(rows, 3) == "90 MAIN ST PALO ALTO"
    assert is_duplicate(("a", 1, 0.5), ("b", 2, 0.51))
    assert not is_duplicate(("a", 1, 0.5), ("b", 2, 1))
    assert not is_duplicate(("a", 1, 0.5), ("b", 255, 0.5))


def test_crop_preserves_fields_and_falls_back():
    raw = BytesIO()
    Image.new("RGB", (100, 200)).save(raw, format="PNG")
    baseline = {"company": "A", "address": "23/4", "date": "01/01/2020", "total": "2.00"}

    def infer(png, prompt):
        with Image.open(BytesIO(png)) as image:
            assert image.height == (90 if prompt == CROP_PROMPT else 200)
        return json.dumps({**baseline, "total": "999.00", "address": "23/1"} if prompt == CROP_PROMPT else baseline)

    row = score(raw.getvalue(), "23/1", infer)
    assert row["candidate"] == {**baseline, "address": "23/1"}
    assert stats([row])["exact"]["fixed"] == 1
    failed = score(raw.getvalue(), "23/1", lambda png, prompt: "invalid" if prompt == CROP_PROMPT else infer(png, prompt))
    assert failed["candidate"] == baseline and failed["fallback"]
    assert stats([failed])["format_tolerant"]["candidate"] == 0


def test_audit_does_not_silently_score_unverified_addresses():
    manifest = {"cohorts": {"dev": {"wildreceipt": [{"id": "external"}], "sroie": []}}}
    with pytest.raises(ValueError, match="Missing verified address"):
        audited_rows(manifest, {}, "dev")


def test_holdout_gate_requires_matching_passing_development_evidence():
    manifest = {"archive_sha256": {"wildreceipt": "digest"},
                "cohorts": {"dev": {"wildreceipt": [{"id": "a"}], "sroie": [{"id": "b"}]}}}
    dev = {"stage": "dev", "advance": True, "receipts": [{"key": "a"}, {"key": "b"}],
           "provenance": {"archive_sha256": manifest["archive_sha256"], "model_revision": REVISION,
                          "crop_prompt": CROP_PROMPT, "baseline_prompt_text": PROMPT,
                          "crop_fraction": CROP_FRACTION, "max_pixels": MAX_PIXELS, "cases": {"a": {}, "b": {}}}}
    assert holdout_allowed(dev, manifest)
    assert not holdout_allowed({**dev, "advance": False}, manifest)
    dev["provenance"]["crop_fraction"] = 0.5
    assert not holdout_allowed(dev, manifest)


def test_development_rejects_a_regressed_control_even_when_external_improves():
    external = {"exact": {"baseline": 4, "candidate": 5}, "format_tolerant": {"fixed": 1}}
    controls = {"documents": 3, "exact": {"baseline": 3, "regressed": 0}}
    assert decision("dev", external, controls)
    controls["exact"]["regressed"] = 1
    assert not decision("dev", external, controls)
