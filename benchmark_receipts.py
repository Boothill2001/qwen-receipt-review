"""Resumable, pinned SROIE evaluation. Run as a CLI on CPU or CUDA."""

import argparse
import json
import os
import platform
import sqlite3
import tempfile
import time
from contextlib import closing
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

from address_protocol import ADDRESS_CASES, PROMPTS
from evaluation_metrics import (
    CONTEXT_GATE_VERSION, GATES, MIN_LINE_CONFIDENCE, contextual_evidence, metrics,
)
from receipt_pipeline import (
    FIELDS, IMAGE_MAX_EDGE, IMAGE_MAX_PIXELS, MAX_NEW_TOKENS, MAX_PIXELS, MIN_PIXELS,
    MODEL, PROMPT_VERSION, SOURCE, TOTAL_EVIDENCE_VERSION,
    QwenBackend, TesseractBackend, annotated_ocr_baseline, clean_test, inventory,
    merchant_key, normalize, prepare_image, process, read_label, sample, sha256_file,
    summarize, total_evidence,
)

REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MODEL_SIZES = {
    "3b": ("Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3"),
    "7b": (MODEL, REVISION),
    "32b": ("Qwen/Qwen2.5-VL-32B-Instruct", "7cfb30d71a1f4f49a57592323337a4a4727301da"),
}
PRECISIONS = ("fp16", "nf4")
ARCHIVE_SHA = "e4502ab884e5fe49e138673f16f6fa72ae0d5fa1d59a2b59a860a2d75e534844"
SEED = 42
COHORTS = {"smoke": ("train", 1), "dev": ("train", 50), "pilot": ("test", 20), "full": ("test", 0)}
COHORTS["address-dev10"] = ("train", len(ADDRESS_CASES))


def elapsed(start):
    return round((time.perf_counter() - start) * 1000, 3)


def dataset(archive, cohort):
    receipts, exclusions = clean_test(archive, inventory(archive))
    if cohort == "address-dev10":
        dev = {item.key: item for item in sample(receipts, "train", 50, SEED)}
        if not ADDRESS_CASES.keys() <= dev.keys():
            raise ValueError("The fixed address cases must belong to the 50-receipt train dev cohort")
        selected = [dev[key] for key in ADDRESS_CASES]
    else:
        split, limit = COHORTS[cohort]
        selected = sample(receipts, split, limit, SEED)
    if cohort == "full" and len(selected) != 337:
        raise ValueError(f"Expected 337 clean test receipts, got {len(selected)}")
    gold = {item.key: read_label(archive, item) for item in selected}
    invalid = [key for key, label in gold.items() if any(
        not isinstance(label[field], str) or normalize(field, label[field]) is None for field in FIELDS
    )]
    # Missing gold can exist in train. Keep the fixed development cohort explicit and fail visibly.
    if invalid:
        raise ValueError(f"Missing or invalid gold fields in selected receipts: {invalid}")
    merchants = {merchant_key(read_label(archive, item)) for item in receipts if item.split == "train"}
    return selected, gold, merchants, exclusions


def score(load_bytes, expected, method, backend, ocr):
    document_start = time.perf_counter()
    started = time.perf_counter()
    try:
        png = prepare_image(load_bytes())
    except Exception as error:
        row = process(lambda: None, expected)
        row.update(reasons=["INVALID_IMAGE"], error_detail=f"{type(error).__name__}: {str(error)[:280]}")
        return evaluate(row, [], None, {"preprocess": elapsed(started),
                                        "evaluation_wall": elapsed(document_start)}, {})
    timings = {"preprocess": elapsed(started)}
    started = time.perf_counter()
    lines, ocr_error = [], None
    try:
        lines = ocr(png)
    except Exception as error:
        ocr_error = f"{type(error).__name__}: {str(error)[:220]}"
    timings["ocr"] = elapsed(started)

    def infer():
        started = time.perf_counter()
        try:
            if method == "qwen":
                return backend(png)
            if ocr_error:
                raise RuntimeError(ocr_error)
            return annotated_ocr_baseline(lines)
        finally:
            timings["inference"] = elapsed(started)

    started = time.perf_counter()
    row = process(infer, expected)
    timings["validation"] = max(0, round(elapsed(started) - timings["inference"], 3))
    generation = getattr(backend, "last_generation", None) or {}
    if "INFERENCE_ERROR" in row["reasons"]:
        generation = {}
    row = evaluate(row, lines, ocr_error, timings, generation, uses_ocr=method == "tesseract")
    timings["evaluation_wall"] = elapsed(document_start)
    return row


def evaluate(row, lines, ocr_error, timings, generation, uses_ocr=False):
    base = row["reasons"].copy()
    evidences = []
    for name, check in (("legacy_evidence", total_evidence), ("context_evidence", contextual_evidence)):
        started = time.perf_counter()
        evidence = check(lines, row["fields"]["total"])
        if ocr_error or "INVALID_IMAGE" in base:
            evidence["reason"] = "TOTAL_EVIDENCE_UNAVAILABLE"
        timings[name] = elapsed(started) if "ocr" in timings else None
        evidences.append(evidence)
    reasons = {GATES[0]: base}
    for gate, evidence in zip(GATES[1:], evidences):
        reasons[gate] = base + ([evidence["reason"]] if evidence["reason"] else [])

    if "inference" in timings:
        base_time = sum(timings[key] for key in ("preprocess", "inference", "validation"))
        timings.update(end_to_end_A=round(base_time + (timings["ocr"] if uses_ocr else 0), 3),
                       end_to_end_B=round(base_time + timings["ocr"] + timings["legacy_evidence"], 3),
                       end_to_end_C=round(base_time + timings["ocr"] + timings["context_evidence"], 3))
    else:
        timings.update(dict.fromkeys(("end_to_end_A", "end_to_end_B", "end_to_end_C"), timings["preprocess"]))
    # Keep the legacy fields readable by the workbench.
    row.update(
        gate_reasons=reasons, evidence=evidences[0], contextual_evidence=evidences[1],
        ocr_lines=lines, ocr_error=ocr_error, format_status=row["status"],
        reasons=reasons[GATES[1]], status="NEEDS_REVIEW" if reasons[GATES[1]] else "RULES_PASSED",
        latency_ms=timings.get("end_to_end_B", timings["preprocess"]), timings_ms=timings,
        generation=generation,
    )
    return row


def open_checkpoint(path, config):
    db = sqlite3.connect(path, timeout=2, isolation_level=None)
    try:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("BEGIN IMMEDIATE")
        db.execute("CREATE TABLE IF NOT EXISTS manifest (id INTEGER PRIMARY KEY CHECK(id = 1), json TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS receipts (id TEXT PRIMARY KEY, json TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, json TEXT NOT NULL)")
        config_json = json.dumps(config, sort_keys=True)
        db.execute("INSERT OR IGNORE INTO manifest VALUES (1, ?)", (config_json,))
        saved, = db.execute("SELECT json FROM manifest WHERE id = 1").fetchone()
        if saved != config_json:
            previous = json.loads(saved)
            changed = sorted(key for key in config.keys() | previous.keys() if config.get(key) != previous.get(key))
            raise ValueError(f"Checkpoint manifest mismatch ({', '.join(changed)}). Use a new output path.")
        db.commit()
        return db
    except BaseException:
        db.close()
        raise


def collect(db, ids, score_id, initialize):
    session_id = None
    for index, key in enumerate(ids, 1):
        db.execute("BEGIN IMMEDIATE")
        try:
            exists = db.execute("SELECT 1 FROM receipts WHERE id = ?", (key,)).fetchone()
            if not exists:
                if session_id is None:
                    session_id = db.execute("INSERT INTO sessions (json) VALUES (?)",
                                            (json.dumps(initialize()),)).lastrowid
                row = score_id(key)
                row["session_id"] = session_id
                db.execute("INSERT INTO receipts VALUES (?, ?)", (key, json.dumps(row, ensure_ascii=False)))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        if not exists:
            print(f"{index}/{len(ids)} {key} {row['status']}", flush=True)
    rows = {key: json.loads(raw) for key, raw in db.execute("SELECT id, json FROM receipts")}
    if set(rows) != set(ids) or len(ids) != len(rows):
        raise ValueError("Checkpoint IDs do not exactly match the selected cohort")
    sessions = {key: json.loads(raw) for key, raw in db.execute("SELECT id, json FROM sessions")}
    return [rows[key] for key in ids], sessions


def cuda_runtime():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This command requires a CUDA-enabled PyTorch runtime")
    return {
        "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": version("transformers"),
        "accelerate": version("accelerate"), "torchvision": version("torchvision"),
        "devices": [{"name": device.name, "total_vram_bytes": device.total_memory,
                     "compute_capability": [device.major, device.minor]}
                    for device in map(torch.cuda.get_device_properties, range(torch.cuda.device_count()))],
    }


def load_qwen(png, max_pixels, prompt, model_id=MODEL, revision=REVISION, precision="fp16"):
    from torch import cuda

    for device in range(cuda.device_count()):
        cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    backend = QwenBackend(model_id=model_id, revision=revision, max_pixels=max_pixels, profile=True,
                          prompt=prompt, precision=precision)
    backend.synchronize()
    if backend.revision != revision:
        raise RuntimeError(f"Model resolved unexpected revision {backend.revision}")
    placement = getattr(backend.model, "hf_device_map", None)
    if placement:
        device_map = {name: str(device) for name, device in placement.items()}
    else:
        devices = {str(parameter.device) for parameter in backend.model.parameters()}
        if len(devices) != 1:
            raise RuntimeError("The checkpoint has no device map or single-device parameter placement.")
        device_map = {"": devices.pop()}
    if any(not (device.isdecimal() or device.startswith("cuda")) for device in device_map.values()):
        raise RuntimeError("The checkpoint has offloaded weights. Use a CUDA runtime with more available VRAM.")
    if precision == "nf4" and not getattr(backend.model, "is_loaded_in_4bit", False):
        raise RuntimeError("The checkpoint did not load with 4-bit quantization.")
    load_ms = elapsed(started)

    def peaks():
        return [{"device": device, "peak_allocated_bytes": cuda.max_memory_allocated(device),
                 "peak_reserved_bytes": cuda.max_memory_reserved(device)} for device in range(cuda.device_count())]

    load_peaks = peaks()
    started = time.perf_counter()
    backend(png)
    warmup_ms = elapsed(started)
    for device in range(cuda.device_count()):
        cuda.reset_peak_memory_stats(device)
    return backend, {"load_ms": load_ms, "warmup_ms": warmup_ms, "load_peaks": load_peaks,
                     "device_map": device_map, "warmup_generation": backend.last_generation}, peaks


def atomic_json(path, report):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            file.close()
            temporary.unlink()
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(archive_path, output, method, cohort, max_pixels=MAX_PIXELS, prompt_version=PROMPT_VERSION,
        model_size="7b", precision="fp16"):
    archive_path, output = Path(archive_path), Path(output)
    archive_sha = sha256_file(archive_path)
    if archive_sha != ARCHIVE_SHA:
        raise ValueError(f"Unexpected SROIE ZIP SHA-256: {archive_sha}")
    if method == "qwen" and cohort in {"pilot", "full"} and max_pixels != MAX_PIXELS:
        raise ValueError("Compare alternative visual budgets on dev before freezing a holdout protocol.")
    if prompt_version not in PROMPTS or (method != "qwen" and prompt_version != PROMPT_VERSION):
        raise ValueError("Alternative prompt versions require the Qwen backend")
    if cohort == "address-dev10" and (method != "qwen" or max_pixels != MAX_PIXELS):
        raise ValueError("Address dev10 requires Qwen with the original image budget")
    if method != "qwen" and (model_size, precision) != ("7b", "fp16"):
        raise ValueError("Model size and precision options require the Qwen backend")
    model_id, revision = MODEL_SIZES[model_size]
    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision: {precision}")
    if output.suffix.lower() != ".json":
        raise ValueError("Use a .json path for the final report")
    checkpoint = output.with_suffix(".sqlite3")
    if output.exists() and not checkpoint.exists():
        raise FileExistsError(f"Output exists without its matching checkpoint: {output}")

    with ZipFile(archive_path) as archive:
        selected, gold, merchants, exclusions = dataset(archive, cohort)
        by_id = {item.key: item for item in selected}
        ocr = TesseractBackend()
        config = {
            "source": SOURCE, "archive_sha256": archive_sha, "cohort": cohort, "split": selected[0].split,
            "seed": SEED, "receipt_ids": list(by_id), "test_exclusions": exclusions, "method": method,
            "code_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in
                            ("receipt_pipeline.py", "benchmark_receipts.py", "evaluation_metrics.py",
                             "address_protocol.py")},
            "model": model_id if method == "qwen" else None,
            "model_revision": revision if method == "qwen" else None,
            "prompt_version": prompt_version if method == "qwen" else None,
            "prompt": PROMPTS[prompt_version] if method == "qwen" else None,
            "image_config": {"max_edge": IMAGE_MAX_EDGE, "source_max_pixels": IMAGE_MAX_PIXELS,
                             "min_pixels": MIN_PIXELS if method == "qwen" else None,
                             "max_pixels": max_pixels if method == "qwen" else None},
            "generation_config": {"precision": precision, "do_sample": False, "max_new_tokens": MAX_NEW_TOKENS,
                                  "attention": "sdpa"} if method == "qwen" else None,
            "ocr_config": {"version": ocr.version, "language": "eng", "psm": 6},
            "gate_versions": {"legacy": TOTAL_EVIDENCE_VERSION, "contextual": CONTEXT_GATE_VERSION,
                              "contextual_min_line_confidence": MIN_LINE_CONFIDENCE},
            "environment": {"python": platform.python_version(), "os": platform.platform(),
                            "pillow": version("pillow"), "gpu": cuda_runtime() if method == "qwen" else None},
        }
        if precision == "nf4":
            config["generation_config"]["quantization"] = {
                "library": "bitsandbytes", "weight_type": "nf4", "compute_dtype": "fp16", "double_quant": True,
            }
            config["environment"]["gpu"]["bitsandbytes"] = version("bitsandbytes")
        backend = peaks = None

        def initialize():
            nonlocal backend, peaks
            if method == "qwen":
                # Fixed train image warms the runtime without consuming or inspecting test labels.
                warmup = sample(inventory(archive), "train", 1, SEED)[0]
                backend, profile, peaks = load_qwen(
                    prepare_image(archive.read(warmup.image)), max_pixels, PROMPTS[prompt_version],
                    model_id, revision, precision,
                )
                return {"warmup_id": warmup.key, **profile}
            return {}

        def score_id(key):
            row = score(lambda: archive.read(by_id[key].image), gold[key], method, backend, ocr)
            row.update(key=key, merchant_group="seen" if merchant_key(gold[key]) in merchants else "unseen")
            if peaks:
                row["session_cuda_peaks"] = peaks()
            return row

        output.parent.mkdir(parents=True, exist_ok=True)
        with closing(open_checkpoint(checkpoint, config)) as db:
            rows, sessions = collect(db, config["receipt_ids"], score_id, initialize)
        duration_ms = sum(row["timings_ms"]["evaluation_wall"] for row in rows)
        report = {
            **config, "runtime": f"transformers-{precision}" if method == "qwen" else "tesseract",
            "summary": summarize(rows), "metrics": metrics(rows), "sessions": sessions, "receipts": rows,
            "evaluation_documents_per_second": round(1000 * len(rows) / duration_ms, 3) if duration_ms else None,
            "merchant_groups": {group: summarize([row for row in rows if row["merchant_group"] == group])
                                for group in ("seen", "unseen")},
            "merchant_group_metrics": {group: metrics([row for row in rows if row["merchant_group"] == group])
                                       for group in ("seen", "unseen")},
        }
        atomic_json(output, report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("qwen", "tesseract"), required=True)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--max-pixels", default=MAX_PIXELS, type=int)
    parser.add_argument("--prompt-version", choices=PROMPTS, default=PROMPT_VERSION)
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="7b")
    parser.add_argument("--precision", choices=PRECISIONS, default="fp16")
    args = parser.parse_args()
    if args.max_pixels < MIN_PIXELS or (args.method != "qwen" and args.max_pixels != MAX_PIXELS):
        parser.error("--max-pixels is a Qwen option and must be at least the processor minimum")
    run(args.archive, args.output, args.method, args.cohort, args.max_pixels, args.prompt_version,
        args.model_size, args.precision)


if __name__ == "__main__":
    main()
