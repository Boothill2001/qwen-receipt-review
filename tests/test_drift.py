from io import BytesIO
import json

import pytest
from PIL import Image

from drift_data import is_duplicate, reading_order
from drift_experiment import CROP_PROMPT, audited_rows, score, stats


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
