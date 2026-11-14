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

Run from the repository root. All commands take the original ZIP from Kaggle dataset `urbikn/sroie-datasetv2`, version **4**. The dataset name contains “v2” and Kaggle versions its uploads separately. Copy the archive bytes between environments. See the expected SHA-256 and excluded IDs in [data-audit.json](../data-audit.json).

On Ubuntu with a suitable CUDA-enabled PyTorch runtime, Python 3.11 or newer, access to Hugging Face and write access to a persistent local disk:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
python -m pip install -r requirements-gpu.txt
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'
mkdir -p work/receipt-v2
curl -fL "https://api.kaggle.com/v1/datasets/download/urbikn/sroie-datasetv2?datasetVersionNumber=4" -o work/sroie-kaggle.zip
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

C sent a net seven additional wrong nonempty totals to review relative to B. Every record it passed still had an error in at least one of the four fields. The figures above come from the local Tesseract 5.5.0 run. The Kaggle Tesseract 5.3.4 run used the same OCR config, ZIP hash and ordered IDs as the Kaggle Qwen run. It matched 137/337 totals and 4/337 complete records. [Local Tesseract report](tesseract-gates-summary.json) · [Paired Kaggle Tesseract report](tesseract-kaggle-summary.json)

## Measured Qwen gate comparison

Qwen processed all 337 test receipts on Tesla T4 ×2 and returned 337 valid JSON objects with zero inference errors. The model matched 328/337 totals, 305/337 companies, 295/337 dates, 230/337 addresses and 185/337 complete records. Its synchronized generation median was 9.41 s and p95 was 10.95 s after warm-up. [Aggregate JSON and run manifest](qwen-full-summary.json)

| Gate | Needs review | Correct records sent to review | Incorrect among passed | Wrong totals sent to review |
| --- | ---: | ---: | ---: | ---: |
| A · Field validation | 54/337 | 12 | 110/283 | 1/9 |
| B · Legacy OCR | 288/337 | 158 | 22/49 | 8/9 |
| C · Contextual OCR | 261/337 | 137 | 28/76 | 8/9 |

For B, 199 receipts had unclear OCR evidence, 62 had conflicting candidates and 21 had OCR totals that disagreed with Qwen. Format reasons can overlap these counts. C reduced review by 27 receipts net and admitted six more incorrect complete records net. Address mismatches accounted for 18 of the 22 incorrect passes under B and 25 of the 28 under C. Review policy should use evidence at the field level and measure each downstream action on its own required fields.

## Field-level review

The total-only policy keeps the contextual OCR check and total format checks. It requires review of company, date and address separately. It produces the following retrospective results on the stored Qwen predictions.

| Cohort | Legacy gate sends total to review | Contextual record gate sends total to review | Total-only policy sends total to review | Wrong totals among supported totals |
| --- | ---: | ---: | ---: | ---: |
| Train dev | 40/50 | 33/50 | 33/50 | 0/17 |
| Clean test | 288/337 | 261/337 | 255/337 | 1/82 |

On test, the six extra supported totals relative to C came from receipts with a date-format flag. The policy supports the amount on 82 receipts, while 33 of those receipts still differ from the benchmark label in at least one field. If all four fields are required, the proposed workflow still has a human task for every receipt. It asks for 1,266 field checks out of 1,348 possible checks, assuming each unverified field is checked once. This counts fields, with no measurement of reviewer time. The full test report was available before this analysis, so use a fresh holdout to validate an operating policy.

```bash
python field_review.py work/receipt-v2/qwen-full-v2.json
```

[Dev and full test policy results](field-review-results.json)

## Address prompt experiment

The paired run isolates the address instruction, with the official FP16 checkpoint, original image budget and generation settings. The ten fixed train-development receipts cover formatting differences, missing or extra content, reading errors and three clean controls. This diagnostic subset is intentionally enriched for errors.

```bash
cohort=address-dev10
dir="work/address/$cohort"
python benchmark_receipts.py --archive work/sroie-kaggle.zip --method qwen --cohort "$cohort" --prompt-version receipt-fields-v1 --output "$dir/baseline.json"
python benchmark_receipts.py --archive work/sroie-kaggle.zip --method qwen --cohort "$cohort" --prompt-version receipt-address-v2 --output "$dir/candidate.json"
python compare_addresses.py "$dir/baseline.json" "$dir/candidate.json" --output "$dir/comparison.json"
```

Run both prompts in the same GPU session. Each output gets a separate checkpoint. The comparator checks the ordered receipt IDs, labels, model revision, archive hash, runtime and inference settings. The primary score stays unchanged. The secondary address score treats periods, commas, semicolons and colons as word boundaries and keeps slashes, hyphens and word order. Use `--inspect-label RECEIPT_ID` to flag a suspected label issue without editing the label. Store per-receipt reports outside Git.

Run the same commands with `cohort=dev` only when the ten-receipt comparison reports `advance: true`. Run with `cohort=full` only after the 50-receipt dev comparison also advances. Reports for the existing benchmark remain the historical baseline.

## Visual token budget

The pinned Qwen processor ran on the same 50 train dev receipts at two image budgets. At the original pixel cap, its per-receipt counts matched every recorded GPU dev count. Reducing the cap halved the median number of visual tokens from 1,241 to 620. This measures input size. The lower-budget checkpoint still needs matched GPU inference runs to measure extraction accuracy, memory, and latency. [Per-receipt token counts](visual-token-profile.json)

| Image pixel cap | Dev receipts | Visual tokens p50 | Visual tokens p95 |
| --- | ---: | ---: | ---: |
| 1,003,520 | 50 | 1,241 | 1,272 |
| 501,760 | 50 | 620 | 630 |

Run the lower cap on the same dev cohort with the same CUDA and OCR versions before choosing a deployment budget.

```bash
python benchmark_receipts.py --archive work/sroie-kaggle.zip --method qwen --cohort dev --max-pixels 501760 --output work/receipt-v2/qwen-dev-low-budget.json
```

## Engineering decisions for an interview

| Decision | Trade-off and failure mode | How to defend it |
| --- | --- | --- |
| Persist per receipt in SQLite | A write per image adds I/O. A crash before commit can repeat a model call. Hash and config mismatches require a fresh run. | **Can the model run twice on one image after a crash?** Yes. The transaction guarantees one committed benchmark result per ID; it cannot establish exactly-once GPU execution. |
| Replay A/B/C on identical fields and OCR | Easier to attribute decision changes to the gate. OCR and VLM errors can be correlated, and B/C verify just one field. | **Why might a better total gate increase the number of bad records passed?** It admits more receipts with correct totals; an incorrect address still makes the full record wrong. Report false-pass count, coverage and selective risk jointly. |
| Pin input, checkpoint and software; profile stages | Strict manifests require a new artifact after runtime drift. More synchronization changes production timing. | **Why can 10 s per receipt hide the bottleneck?** It may include CPU PNG preparation, OCR and transfers in addition to inference. Stage and CUDA metrics answer different capacity questions. |

**STAR.** A GPU pilot demonstrated extraction, and the review policy still allowed an incorrect full record through. The task was to scale evaluation without losing progress and measure review decisions separately. The implementation pins the cohort, persists one result per receipt, and evaluates three policies against the same output. On 337 receipts, Qwen matched 185 complete records. The contextual gate reduced review by 27 receipts net and increased incorrect passes by six, which helps explain why a total-only check cannot clear an entire record.

## Serving proposal

Start with FastAPI, durable jobs in one database and a single GPU worker with measured concurrency and queue capacity. Accept a valid enqueued job with HTTP 202 and a status URL. Use 429 with a retry hint when an admission limit is reached and 503 when the worker service is unavailable. Store idempotency key, input hash, and model, prompt, preprocessing and gate versions. Reusing an idempotency key with different input is a conflict.

Transition jobs from queued to running to completed, needs-review, failed or cancelled. Use a worker lease and heartbeat to reclaim interrupted work, and fence stale workers from committing results after a lease expires. Remove or mark queued jobs on cancellation and treat a running GPU call as cooperatively cancellable before its result can be published. Retry transient failures with capped exponential backoff and jitter. Bound image pixels and generation length, validate device headroom before accepting more work, and surface OOM failures without a retry loop on the same oversized job. Observe queue wait, per-stage p95, OOM and retry counts, result state and review coverage before scaling out.

Start the load test at worker concurrency one. Use 12.19 s, the measured extraction p95 on two T4s, as an input for an initial queue budget. For a 60 s queue-wait target, provision at most four waiting slots initially and verify that bound with a burst test. The estimate omits upload, job storage and dispatch time. Record p95 queue wait and API response time separately. Test a worker exit after inference and before commit, repeated submission with the same idempotency key, and a full queue. Keep human approval outside the GPU worker and persist the raw inference and reviewer edits with their respective version and actor.
