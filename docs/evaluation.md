# SROIE evaluation

## Audit and implementation order

The project uses `prepare_image` for orientation and size normalization, `QwenBackend` for the official checkpoint, `process` and `parse_output` for JSON and field validation, and `total_evidence` for the legacy OCR check. `review_app.py` renders the evidence and preserves original and approved fields. Those boundaries let the new runner reuse the extraction logic and evaluate three gates without calling the model three times.

| Priority | Gap and objective | Files | Verification and learning outcome |
| --- | --- | --- | --- |
| P0 | The original `benchmark` writes its report at the end. Persist raw predictions and failures per receipt, lock the configuration and resume safely. | `benchmark_receipts.py` | Interrupt during a receipt, rerun, verify committed rows remain and any configuration drift aborts. Learn the difference between rerunning inference and committing a result. |
| P0 | The original `summarize` mixes normalization and one review policy. Measure extraction and gate performance separately on fixed IDs. | `evaluation_metrics.py`, `review_app.py` | Fixture tests cover empty pass groups and failure denominators. Run on train development data, freeze the rules, then compare policies on the same test predictions. Learn why review load and risk move independently. |
| P1 | Separate generation, OCR and preprocessing time. Keep the evaluation runnable without a GPU in CI. | `receipt_pipeline.py`, `tests/`, `.github/workflows/checks.yml` | Run CPU tests and lint. On CUDA, verify warm-up, per-device peaks and token budgets before committing to a full run. |
| P2 | Bound serving load once the benchmark supplies latency and memory distributions. | Serving proposal below | Load-test one worker before choosing queue or worker capacity. |

The historical notebook embeds a snapshot of the original code and reproduces the pilot. This CLI is the reference for the new protocol. Use the same source checkout for every resume.

## Reproduce

Run from the repository root. All commands take the Kaggle SROIE v2 ZIP as downloaded, without unpacking or repacking it. Copy the exact bytes between environments. See the expected SHA-256 and excluded IDs in [data-audit.json](../data-audit.json).

On Ubuntu with a suitable CUDA-enabled PyTorch runtime, Python 3.11 or newer, access to Hugging Face and write access to a persistent local disk:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
python -m pip install -r requirements-gpu.txt
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'
mkdir -p work/receipt-v2
curl -fL https://www.kaggle.com/api/v1/datasets/download/urbikn/sroie-datasetv2 -o work/sroie-kaggle.zip
python benchmark_receipts.py --archive work/sroie-kaggle.zip --method qwen --cohort smoke --output work/receipt-v2/qwen-smoke-v2.json
python benchmark_receipts.py --archive work/sroie-kaggle.zip --method qwen --cohort dev --output work/receipt-v2/qwen-dev-v2.json
python benchmark_receipts.py --archive work/sroie-kaggle.zip --method tesseract --cohort dev --output work/receipt-v2/tesseract-dev-v2.json
```

In Kaggle, enable an accelerator and Internet, use the repository files and submit each `python` command from that checkout. Use the notebook image's preinstalled CUDA stack where possible. Retain the checkpoint file between sessions. If a notebook job is the only copy of the database, copy it out with the runner stopped and restore it before restarting. A terminated ephemeral session without a retained checkpoint loses its data.

Review extraction quality on the 50 train dev images and freeze the model and gate settings before running the test commands. The smoke image is a train example used to warm up and test inference. Use a different report path for each resolution experiment. For instance, add `--max-pixels 501760` to the dev command and compare normalized exact match, generation p95, visual tokens and memory with the default `1003520`.

```bash
python benchmark_receipts.py --archive work/sroie-kaggle.zip --method qwen --cohort full --output work/receipt-v2/qwen-full-v2.json
python benchmark_receipts.py --archive work/sroie-kaggle.zip --method tesseract --cohort full --output work/receipt-v2/tesseract-full-v2.json
python -c 'import json; from pathlib import Path; a,b=[json.loads(Path("work/receipt-v2", name).read_text(encoding="utf8")) for name in ("qwen-full-v2.json","tesseract-full-v2.json")]; assert a["archive_sha256"]==b["archive_sha256"] and a["receipt_ids"]==b["receipt_ids"] and len(a["receipt_ids"])==337; print("Aligned: 337 receipts")'
```

After an interruption run the identical command. The `.sqlite3` checkpoint next to the requested JSON holds committed rows. A crash during an image can cause that image to be inferred again. The receipt ID is the primary key and the row is committed in a transaction, so the report can contain only one row per ID. A recorded inference error keeps its place in the denominator. For a fresh run use a new output name. The manifest rejects changed code, package or GPU configuration and changed dataset, cohort, prompt or model revision. The final JSON is written atomically after every selected ID has a committed row. Keep the full row-level results and checkpoint out of Git.

## Read the report

- `raw_exact` compares the parsed values verbatim. `normalized_exact` applies existing date, amount and text normalization. A row with failed inference, invalid schema, or a missing field takes a miss for the affected fields. The full clean cohort has 337 receipts, including the 20 historical pilot IDs. It is a held-out SROIE split for this evaluation, with previously inspected pilot cases inside it. Use a newly collected blind cohort for an independent external claim.
- A applies the schema and field-format checks. B adds the existing OCR gate. C prefers explicit payable labels and uses generic totals as fallback. Conflicts among selected amounts, insufficient OCR confidence and mismatches go to review. The runtime Tesseract evidence is independent of Qwen's output, although both can mistake the same source text. The Tesseract baseline uses the same OCR for extraction and checking, so its check is correlated by construction.
- Coverage is passed / all, selective risk is incorrect / passed, and review workload is flagged / all. Wrong-total detection uses wrong nonempty total predictions as its denominator and reports missing totals separately. `wrong_total_stopped_after_format` isolates the additional detections beyond A. A zero denominator produces `null`. Wilson 95% intervals are descriptive document-level intervals. Repeated store templates can correlate errors and make intervals optimistic.
- The checkout logs each pipeline stage, then sums the applicable stages for sequential A/B/C latency estimates. `evaluation_wall` is the actual elapsed time for running all three comparisons together, including the ZIP image read. Throughput divides documents by summed evaluation wall time at concurrency one and excludes model load, warm-up and checkpoint I/O. Inference timing includes Qwen image processing and decoding, while `generation_ms` measures synchronized wall time around `generate`. It includes host overhead and GPU work. CUDA kernel timing would need a dedicated trace or events. The per-device memory peaks measure allocations and reservations reported by PyTorch's allocator, with a reset after warm-up. [PyTorch CUDA timing and memory semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- `merchant_group_metrics` splits by normalized company and address from the training data. Byte hashes catch byte-identical copies. Recompressed scans, near duplicates, known receipt layouts and spelling variations can still influence the benchmark.

The workbench accepts both report versions. On a new report, choose a gate to update the case list, status and evidence polygon. These report settings are for evaluation. The live review path still uses the legacy gate.

## Measured Tesseract gate comparison

The new CLI processed 50 SROIE train development receipts and all 337 clean test receipts with Tesseract 5.5.0. On test, Tesseract extracted 135/337 normalized totals and 3/337 complete records. Merchant groups account for all 337 rows (207 seen, 130 unseen). The same 337 predictions feed all three gates below.

| Gate | Auto-pass | Wrong full records among passed | Needs review | Wrong nonempty totals sent to review | Correct full records sent to review |
| --- | ---: | ---: | ---: | ---: | ---: |
| A · Field validation | 173/337 | 170/173 | 164 | 33/105 | 0 |
| B · Legacy OCR | 52/337 | 52/52 | 285 | 96/105 | 3 |
| C · Contextual OCR | 48/337 | 48/48 | 289 | 103/105 | 3 |

C sent a net seven additional wrong nonempty totals to review relative to B. Every record it passed still had an error in at least one of the four fields. This result argues for using total evidence as a field-level signal and measuring any record-level policy on all fields. The test run measured Tesseract only. The Qwen pilot remains the sole measured Qwen cohort. [Aggregate JSON and run manifests](tesseract-gates-summary.json)

## Engineering decisions for an interview

| Decision | Trade-off and failure mode | How to defend it |
| --- | --- | --- |
| Persist per receipt in SQLite | A write per image adds I/O. A crash before commit can repeat a model call. Hash and config mismatches require a fresh run. | **Can the model run twice on one image after a crash?** Yes. The transaction guarantees one committed benchmark result per ID; it cannot establish exactly-once GPU execution. |
| Replay A/B/C on identical fields and OCR | Easier to attribute decision changes to the gate. OCR and VLM errors can be correlated, and B/C verify just one field. | **Why might a better total gate increase the number of bad records passed?** It admits more receipts with correct totals; an incorrect address still makes the full record wrong. Report false-pass count, coverage and selective risk jointly. |
| Pin input, checkpoint and software; profile stages | Strict manifests require a new artifact after runtime drift. More synchronization changes production timing. | **Why can 10 s per receipt hide the bottleneck?** It may include CPU PNG preparation, OCR and transfers in addition to inference. Stage and CUDA metrics answer different capacity questions. |

**STAR.** A GPU pilot demonstrated extraction, and the review policy still allowed an incorrect full record through. The task was to scale evaluation without losing progress and measure review decisions separately. The implementation pins the cohort, persists one result per receipt, and evaluates three policies against the same output. The measured results in this upgrade cover the CPU test suite and Tesseract. The historical Qwen result remains 12/20 normalized four-field exact matches. A new full Qwen result requires a CUDA run.

## Serving proposal

Start with FastAPI, durable jobs in one database and a single GPU worker with measured concurrency and queue capacity. Accept a valid enqueued job with HTTP 202 and a status URL. Use 429 with a retry hint when an admission limit is reached and 503 when the worker service is unavailable. Store idempotency key, input hash, and model, prompt, preprocessing and gate versions. Reusing an idempotency key with different input is a conflict.

Transition jobs from queued to running to completed, needs-review, failed or cancelled. Use a worker lease and heartbeat to reclaim interrupted work, and fence stale workers from committing results after a lease expires. Remove or mark queued jobs on cancellation and treat a running GPU call as cooperatively cancellable before its result can be published. Retry transient failures with capped exponential backoff and jitter. Bound image pixels and generation length, validate device headroom before accepting more work, and surface OOM failures without a retry loop on the same oversized job. Observe queue wait, per-stage p95, OOM and retry counts, result state and review coverage before scaling out.
