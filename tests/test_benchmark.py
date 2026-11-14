import hashlib
import json
from contextlib import closing
from io import BytesIO
from zipfile import ZipFile

import pytest
from PIL import Image, UnidentifiedImageError

import benchmark_receipts as runner
from evaluation_metrics import GATES, contextual_evidence, metrics
from receipt_pipeline import clean_test, inventory, normalize, parse_output, prepare_image

GOLD = {"company": "EXAMPLE", "date": "27/01/2018", "address": "1 MAIN ROAD", "total": "22.26"}


def image_bytes(color="white"):
    file = BytesIO()
    Image.new("RGB", (20, 100), color).save(file, "PNG")
    return file.getvalue()


def line(text, confidence=90, y=0):
    return {"text": text, "confidence": confidence, "polygon": [0, y, 20, y, 20, y + 5, 0, y + 5]}


def test_image_preparation():
    file = BytesIO()
    image = Image.new("RGB", (2200, 1100), "white")
    exif = image.getexif()
    exif[274] = 6
    image.save(file, "JPEG", exif=exif)
    with Image.open(BytesIO(prepare_image(file.getvalue(), uploads_only=True))) as prepared:
        assert (prepared.format, prepared.mode, prepared.size) == ("PNG", "RGB", (1024, 2048))
    file = BytesIO()
    image.save(file, "GIF")
    with pytest.raises(ValueError, match="JPEG or PNG"):
        prepare_image(file.getvalue(), uploads_only=True)
    with pytest.raises(UnidentifiedImageError):
        prepare_image(b"invalid")


@pytest.mark.parametrize("raw", [
    "bad JSON", '{"company":"A"}', '{"company":"A","company":"B"}',
    '{"company":"A","date":"1","address":"B","total":1}', '[]',
])
def test_invalid_schema(raw):
    with pytest.raises(ValueError):
        parse_output(raw)


def test_duplicate_key_in_otherwise_valid_output():
    raw = json.dumps(GOLD).replace('{"company"', '{"company":"duplicate","company"', 1)
    with pytest.raises(ValueError, match="Duplicate JSON key: company"):
        parse_output(raw)


def test_normalization():
    assert parse_output("```json\n" + json.dumps(GOLD) + "\n```") == GOLD
    assert normalize("total", "RM 1,234.5") == "1234.50"
    assert normalize("date", "27.01.18") == "2018-01-27"
    assert normalize("company", "Ａ  B") == "a b"
    assert normalize("address", "  ") is None


def test_context_uses_payable_label_and_retains_conflicts():
    lines = [line("Total Qty 30 18.00"), line("Total Sales (Excluding GST) 21.00"),
             line("Total GST 1.26"), line("Total Sales (Inclusive of GST) 22.26", y=10)]
    evidence = contextual_evidence(lines, "22.26")
    assert evidence["reason"] is None
    assert len(evidence["candidates"]) == 1
    assert evidence["candidates"][0]["polygon"][1] == 10
    assert contextual_evidence(lines, "1.26")["reason"] == "TOTAL_EVIDENCE_MISMATCH"
    assert contextual_evidence(lines + [line("Amount Due 24.00")], "22.26")["reason"] == "TOTAL_EVIDENCE_CONFLICT"
    assert contextual_evidence([line("Grand Total 22.26", 50)], "22.26")["reason"] == "TOTAL_EVIDENCE_UNCLEAR"


def test_fixed_denominators_and_failure_modes():
    image = image_bytes()
    lines = [line("Grand Total 22.26")]

    def fail(_):
        raise TimeoutError("connection timeout")

    def score(values, ocr=lambda _: lines):
        return runner.score(lambda: image, GOLD, "qwen", lambda _: json.dumps(values), ocr)

    exact = score({**GOLD, "date": "2018-01-27"})
    missing = score({**GOLD, "total": None})
    wrong = score({**GOLD, "total": "50.00"})
    malformed = runner.score(lambda: image, GOLD, "qwen", lambda _: "extra text", fail)
    failure = runner.score(lambda: image, GOLD, "qwen", fail, lambda _: lines)
    invalid_image = runner.score(lambda: b"not an image", GOLD, "qwen", fail, fail)
    unavailable = score(GOLD, fail)
    report = metrics([exact, missing, wrong, malformed, failure, invalid_image, unavailable])

    assert malformed["raw"] == "extra text" and malformed["schema_valid"] is False
    assert missing["schema_valid"] and "MISSING_TOTAL" in missing["gate_reasons"][GATES[0]]
    assert failure["raw"] is None and "TimeoutError" in failure["error_detail"]
    assert unavailable["gate_reasons"][GATES[1]] == ["TOTAL_EVIDENCE_UNAVAILABLE"]
    assert report["schema_valid"]["count"] == 4
    assert report["raw_exact"]["all_fields"]["count"] == 1
    assert report["normalized_exact"]["all_fields"]["count"] == 2
    assert report["normalized_exact"]["all_fields"]["n"] == 7
    assert report["inference_errors"] == report["invalid_images"] == 1
    assert report["latency_ms"]["ocr"]["n"] == 6
    assert report["latency_ms"]["end_to_end_C"]["n"] == 7
    assert report["gates"][GATES[0]]["false_pass_count"] == 1
    contextual = report["gates"][GATES[2]]
    assert contextual["false_pass_count"] == 0
    assert contextual["wrong_total_detected"]["count"] == contextual["wrong_total_detected"]["n"] == 1
    assert contextual["correct_flagged_count"] == 1
    assert metrics([failure])["gates"][GATES[1]]["selective_risk"]["rate"] is None


def fixture_archive(path):
    a, b, c = map(image_bytes, ("white", "black", "red"))
    with ZipFile(path, "w") as z:
        for split, key, image in (("train", "train", a), ("test", "leak", a), ("test", "unique", b),
                                   ("test", "zzdup", b), ("test", "unseen", c)):
            z.writestr(f"{split}/img/{key}.png", image)
            label = GOLD if key != "unseen" else {**GOLD, "company": "UNSEEN"}
            z.writestr(f"{split}/entities/{key}.txt", json.dumps(label))


def test_deterministic_leakage_audit(tmp_path):
    path = tmp_path / "archive.zip"
    fixture_archive(path)
    with ZipFile(path) as z:
        kept, excluded = clean_test(z, inventory(z))
        assert excluded == {"overlap_with_train": ["leak"], "duplicate_in_test": ["zzdup"]}
        assert {row.key for row in kept} == {"train", "unique", "unseen"}
        assert [row.key for row in runner.dataset(z, "pilot")[0]] == ["unseen", "unique"]


def test_checkpoint_resumes_after_interruption(tmp_path):
    path = tmp_path / "report.sqlite3"
    manifest = {"receipt_ids": ["a", "b"], "prompt": "original"}
    calls = []

    def interrupted(key):
        if key == "b":
            raise KeyboardInterrupt()
        return {"key": key, "status": "RULES_PASSED", "raw": "original"}

    with closing(runner.open_checkpoint(path, manifest)) as db, pytest.raises(KeyboardInterrupt):
        runner.collect(db, manifest["receipt_ids"], interrupted, lambda: {"worker": 1})
    with pytest.raises(ValueError, match="prompt"):
        runner.open_checkpoint(path, {**manifest, "prompt": "changed"})

    def finish(key):
        calls.append(key)
        return {"key": key, "status": "NEEDS_REVIEW", "raw": None}

    with closing(runner.open_checkpoint(path, manifest)) as db:
        rows, sessions = runner.collect(db, manifest["receipt_ids"], finish, lambda: {"worker": 2})
        assert [row["raw"] for row in rows] == ["original", None]
        assert [row["session_id"] for row in rows] == [1, 2]
        assert sessions == {1: {"worker": 1}, 2: {"worker": 2}}
        assert runner.collect(db, manifest["receipt_ids"], finish, lambda: pytest.fail("loaded"))[0] == rows
    assert calls == ["b"]


def test_report_is_workbench_compatible_and_reproducible(tmp_path, monkeypatch):
    class FakeOCR:
        version = "fixture"

        def __call__(self, _):
            return [line("EXAMPLE"), line("1 MAIN ROAD"), line("27/01/2018"), line("Total 22.26")]

    archive = tmp_path / "archive.zip"
    fixture_archive(archive)
    monkeypatch.setattr(runner, "ARCHIVE_SHA", hashlib.sha256(archive.read_bytes()).hexdigest())
    monkeypatch.setattr(runner, "TesseractBackend", FakeOCR)
    output = tmp_path / "test.json"
    report = runner.run(archive, output, "tesseract", "pilot")
    assert report == runner.run(archive, output, "tesseract", "pilot")
    assert output.exists() and output.with_suffix(".sqlite3").exists()
    assert report["summary"]["documents"] == 2
    assert [row["key"] for row in report["receipts"]] == report["receipt_ids"]
    assert sum(group["documents"] for group in report["merchant_groups"].values()) == 2
    archive.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256"):
        runner.run(archive, output, "tesseract", "pilot")
