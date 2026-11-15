# Cross-dataset address evaluation

This experiment checks an address crop on receipts from a new source. The Streamlit incident timeline is a simulated operations scenario. The SROIE case it opens illustrates a known address error from a measured run. WildReceipt is evaluated separately.

## Prepare the cohorts

Download the [original WildReceipt archive](https://mmocr.readthedocs.io/en/latest/user_guides/data_prepare/kie.html) and the [SROIE v4 ZIP](evaluation.md#reproduce) into `work/`. The runner verifies both archive hashes. It reads the original class mapping, groups address boxes into lines, and sorts within lines from left to right. Selection uses the SHA-256 order of split-qualified IDs and skips repeated normalized merchant names, identical image bytes, and close 64-bit difference hashes at similar aspect ratios. This heuristic can miss merchant aliases and recompressed duplicates.

```bash
python drift_experiment.py --wild work/wildreceipt.tar --sroie work/sroie-kaggle.zip --manifest work/drift/manifest.json --stage prepare
```

The manifest locks 10 WildReceipt train receipts and three correct SROIE controls for development, followed by 50 WildReceipt test receipts and 20 SROIE regression samples. Inspect the address spans and related lines in the original scans before running inference. Supply a local `dev-audit.json` mapping each WildReceipt dev ID to its transcribed, verified merchant address. Use printed word boundaries, exclude a standalone branch or legal heading, and keep street numbers, slashes and postal codes. The original annotation remains in the manifest for provenance. Audit all selected holdout addresses before opening their predictions. A missing audit entry aborts evaluation.

## Run on CUDA

Use the pinned GPU requirements and the official Qwen2.5-VL checkpoint on a CUDA runtime that can hold FP16 weights. Keep the SQLite checkpoint and rerun the exact command if interrupted.

```bash
python drift_experiment.py --wild work/wildreceipt.tar --sroie work/sroie-kaggle.zip --manifest work/drift/manifest.json --audit work/drift/dev-audit.json --stage dev --output work/drift/dev.json
```

The full-image prompt produces all four fields. The second pass reads the top 45% of the original scan at its native resolution before resizing to the configured maximum. The crop uses image dimensions. An unusable crop output falls back to the full-image address. Company, date and total remain the full-image prediction. Metrics report strict and format-tolerant address matches, fixes, regressions, fallback count and added latency. The benchmark targets address only.

Advance to `--stage holdout --dev-report work/drift/dev.json` with audited holdout labels and a fresh report path only if dev fixes at least one format-tolerant error, preserves the strict score, and keeps all three SROIE controls correct. Propose shadow evaluation only after a strict improvement on the new domain and no aggregate regression on SROIE, with measured latency in the decision. The selected SROIE controls check for regressions on known cases. Their score does not estimate the size of the difference between domains.

## Measured result

Qwen2.5-VL-7B-Instruct ran in FP16 on two Kaggle Tesla T4 GPUs. The candidate includes fallback to the full-image result whenever the crop provides no usable address.

| Cohort | Exact, full → candidate | Format-tolerant, full → candidate | Fallbacks | Crop p95 added |
| --- | ---: | ---: | ---: | ---: |
| WildReceipt dev | 8/10 → 8/10 | 10/10 → 10/10 | 10/10 | 4.56 s |
| SROIE controls | 3/3 → 2/3 | 3/3 → 2/3 | 2/3 | 5.39 s |

All 12 fallbacks were missing crop addresses. The one crop address accepted by the fallback rule was a regression. On SROIE `X51006388065`, it stopped at `TAMAN BERKELEY` and dropped the postcode and remaining locality text. The result was valid JSON with an incomplete address. The two WildReceipt exact misses both matched after formatting normalization.

**Decision: keep full-image extraction.** The fixed crop repaired zero addresses, regressed on one control and added inference time. The 50 WildReceipt holdout receipts and 20 SROIE regression receipts remained unevaluated, and the candidate did not proceed to shadow. Cohort IDs, runtime, model revision and source hashes are in the [measured run summary](drift-dev-summary.json).

## Interpret the change

The domains can differ in layout, language, scan quality and annotation conventions. A score difference alone does not identify a cause. Diagnose individual errors against the image. If the business changes the address definition to require the branch header, version the label policy and re-score under that policy. That is a change in what counts as a correct output.
