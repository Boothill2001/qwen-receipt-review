# Qwen model size comparison

Can a smaller VLM preserve receipt quality at lower inference cost, or does a larger VLM repair enough errors to justify the resources? The established 7B FP16 SROIE numbers are a separate reference.

## Run on Kaggle

Use one Kaggle GPU T4 ×2 runtime with Internet and the SROIE v4 ZIP from the [evaluation guide](evaluation.md#reproduce). Install the repo's GPU requirements plus bitsandbytes 0.50.2. Install Tesseract English for review-gate metrics.

```bash
python -m pip install -r requirements-gpu.txt bitsandbytes==0.50.2
python benchmark_receipts.py --archive work/sroie-kaggle.zip --output work/scale/3b-pilot.json --method qwen --cohort pilot --model-size 3b --precision nf4
python benchmark_receipts.py --archive work/sroie-kaggle.zip --output work/scale/7b-pilot.json --method qwen --cohort pilot --model-size 7b --precision nf4
python benchmark_receipts.py --archive work/sroie-kaggle.zip --output work/scale/32b-pilot.json --method qwen --cohort pilot --model-size 32b --precision nf4
python compare_model_sizes.py --report-3b work/scale/3b-pilot.json --report-7b work/scale/7b-pilot.json --report-32b work/scale/32b-pilot.json --output work/scale/pilot-summary.json
```

Run each model in a separate process on the same machine. The runner verifies the data hash, pins each official Qwen revision, and warms up on a fixed train receipt. All three models use bitsandbytes NF4, nested quantization, FP16 compute and greedy decoding at batch size 1. The original prompt, preprocessing, pixel budget, scoring and contextual OCR gate are shared across models. Allow enough disk for the original 32B checkpoint before quantization.

Advance once all three pilot reports have 20 aligned rows, each model loads its four-bit weights entirely on CUDA, and all reports contain valid parsed examples and measured latency. Retry a load or infrastructure failure once in a clean session on the same accelerator class. Preserve completed reports and record the blocked model if the retry fails.

For the full run, repeat with `--cohort full` and new report paths ending in `-full.json`. Keep each SQLite checkpoint beside its JSON report to resume interrupted inference with the exact same configuration. Environment or code changes require a new checkpoint path.

## Interpret the comparison

The comparator rejects differences in receipt order, labels, data hash, prompt, precision, preprocessing, OCR, hardware, gate settings and code hashes. It reports normalized exact scores, format-tolerant address matches, fixes and regressions relative to 7B NF4, reviews and incorrect auto-passes, latency percentiles, and peak allocated and reserved GPU memory. Inference errors stay in the denominator. Gate C end-to-end latency covers preprocessing, inference, OCR, validation and contextual evidence. Load and warm-up are measured separately.

Full-run portfolio thresholds:

- **3B** qualifies when the final amount and all-field exact counts match or exceed 7B, incorrect auto-passes do not increase, and p95 or peak allocated memory on the most loaded GPU falls.
- **32B** qualifies when at least 17 more receipts match all four fields, the final amount count holds, incorrect auto-passes do not increase, and p95 is at most twice 7B's.
- The full report lists both models if both qualify and keeps 7B if neither qualifies. The 20-receipt pilot assigns no winner.

Read paired address disagreements against the source scan before describing what improved. The format-tolerant score preserves word order, slashes and hyphens. Exact and tolerant scores stay separate. Potentially ambiguous source labels stay unchanged for the benchmark.
