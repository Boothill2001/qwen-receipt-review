"""Profile matching 3B and 7B NF4 requests on one Tesla T4."""

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path
from threading import Event, Thread
from zipfile import ZipFile

from benchmark_receipts import ARCHIVE_SHA, MODEL_SIZES, atomic_json, cuda_runtime, dataset, load_qwen
from evaluation_metrics import distribution
from receipt_pipeline import MAX_PIXELS, PROMPT, PROMPT_VERSION, parse_output, prepare_image, sha256_file

RECEIPTS = (
    "X51005677336", "X51005757308", "X51005719898",
    "X51006327960", "X51007433809", "X51007846310",
)
REPEATS = 2


def sample_gpu(handle, stop, readings):
    from pynvml import nvmlDeviceGetUtilizationRates

    while not stop.wait(0.5):
        sample = nvmlDeviceGetUtilizationRates(handle)
        readings.append({"gpu_busy_percent": sample.gpu, "memory_busy_percent": sample.memory})


def measure(backend, image, handle, peaks, key, repeat):
    from torch import cuda

    cuda.reset_peak_memory_stats(0)
    stop, samples = Event(), []
    sampler = Thread(target=sample_gpu, args=(handle, stop, samples), daemon=True)
    sampler.start()
    try:
        output = backend(image, token_timing=True)
    finally:
        stop.set()
        sampler.join()
    parse_output(output)
    if backend.last_generation["decode_ms_per_token"] is None:
        raise ValueError(f"No usable token split for {key}")
    return {
        "receipt_id": key, "repeat": repeat, **backend.last_generation,
        "vram": peaks()[0], "utilization_samples": samples,
    }


def run(archive_path, output, size):
    from pynvml import nvmlDeviceGetHandleByIndex, nvmlInit, nvmlShutdown
    from torch import cuda

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or cuda.device_count() != 1:
        raise ValueError("Set CUDA_VISIBLE_DEVICES=0 to expose only the first physical GPU.")
    if "T4" not in cuda.get_device_name(0):
        raise ValueError("This run requires a T4 GPU.")
    if output.suffix != ".json" or output.resolve() == archive_path.resolve():
        raise ValueError("Use a separate JSON output path.")
    if sha256_file(archive_path) != ARCHIVE_SHA:
        raise ValueError("Unexpected SROIE archive hash.")
    model_id, revision = MODEL_SIZES[size]
    with ZipFile(archive_path) as archive:
        selected, *_ = dataset(archive, "pilot")
        paths = {row.key: row.image for row in selected}
        images = {key: prepare_image(archive.read(paths[key])) for key in RECEIPTS}
        warmup = next(row for row in selected if row.key not in RECEIPTS)
        backend, session, peaks = load_qwen(
            prepare_image(archive.read(warmup.image)), MAX_PIXELS, PROMPT, model_id, revision, "nf4",
        )
    runtime = cuda_runtime()
    runtime.update({"bitsandbytes": version("bitsandbytes"), "nvidia-ml-py": version("nvidia-ml-py")})
    nvmlInit()
    try:
        handle = nvmlDeviceGetHandleByIndex(0)
        rows = []
        for repeat in range(1, REPEATS + 1):
            order = RECEIPTS if repeat == 1 else RECEIPTS[::-1]
            for key in order:
                row = measure(backend, images[key], handle, peaks, key, repeat)
                rows.append(row)
                print(size, repeat, key, row["ttft_ms"], row["decode_ms_per_token"], flush=True)
    finally:
        nvmlShutdown()

    summary = {
        key: distribution(row[key] for row in rows)
        for key in ("ttft_ms", "decode_ms_per_token", "generation_ms", "output_tokens")
    }
    for key in ("gpu_busy_percent", "memory_busy_percent"):
        summary[key] = distribution(item[key] for row in rows for item in row["utilization_samples"])
    summary["peak_allocated_bytes"] = max(row["vram"]["peak_allocated_bytes"] for row in rows)
    summary["peak_reserved_bytes"] = max(row["vram"]["peak_reserved_bytes"] for row in rows)
    report = {
        "model": model_id, "model_revision": revision, "runtime": runtime, "precision": "nf4",
        "quantization": "bitsandbytes NF4 + double quantization, FP16 compute",
        "archive_sha256": ARCHIVE_SHA, "prompt_version": PROMPT_VERSION, "max_pixels": MAX_PIXELS,
        "receipt_ids": RECEIPTS, "repeats": REPEATS, "session": session, "warmup_id": warmup.key,
        "utilization_sample_interval_s": 0.5, "trials": rows, "summary": summary,
        "code_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in
                        ("profile_qwen_tokens.py", "receipt_pipeline.py", "benchmark_receipts.py")},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print("TOKEN PROFILE", size, json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-size", choices=("3b", "7b"), required=True)
    args = parser.parse_args()
    run(args.archive, args.output, args.model_size)


if __name__ == "__main__":
    main()
