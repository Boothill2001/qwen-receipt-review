"""Streamlit workbench for extracting and reviewing receipts."""
import hashlib
import json
import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import streamlit as st
from PIL import Image, ImageDraw

from evaluation_metrics import GATES, gate_metrics
from field_review import total_review_reasons
from monitoring import INCIDENT_DAY, INCIDENT_RECEIPT, daily_operations, verification_text
from receipt_pipeline import (
    FIELDS, OLLAMA_MODEL, OllamaBackend, TesseractBackend, extract_image,
    inventory, prepare_image, review_reasons, sha256_file, summarize,
)

PROJECT = Path(__file__).resolve().parent
WORK = Path(os.environ.get("RECEIPT_REVIEW_WORK", PROJECT / "work")).expanduser().resolve()
WORKSPACE_WORK = PROJECT.parent.parent / "work"
if "RECEIPT_REVIEW_WORK" not in os.environ and not WORK.is_dir():
    if (WORKSPACE_WORK / "sroie-kaggle.zip").is_file():
        WORK = WORKSPACE_WORK
LABELS = {"company": "Merchant", "date": "Date", "address": "Address", "total": "Total"}
METHODS = {"annotated-ocr": "Annotated OCR + rules", "tesseract": "Tesseract + rules",
           "qwen": "Qwen · checkpoint FP16", "ollama": "Qwen · Ollama quantized"}
GATE_LABELS = dict(zip(GATES, ("A · Format only", "B · Legacy OCR", "C · Contextual OCR")))
REPLAY_DIR = PROJECT / "assets" / "replay"
DRIFT_REPORT = PROJECT / "docs" / "drift-dev-summary.json"


def apply_gate(row, gate):
    reasons = row["gate_reasons"][gate]
    return {
        **row, "reasons": reasons, "status": "NEEDS_REVIEW" if reasons else "RULES_PASSED",
        "evidence": ({}, row["evidence"], row["contextual_evidence"])[GATES.index(gate)],
    }


@st.cache_data
def archive_index(path, modified, size):
    with ZipFile(path) as archive:
        return {(r.split, r.key): r.image for r in inventory(archive)}, sha256_file(path)


def percent(num, den):
    return f"{num}/{den} · {num / den:.1%}" if den else "—"


def draw_receipt(image_bytes, evidence, mode):
    with Image.open(BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    if mode == "Original scan":
        return image
    drawing = ImageDraw.Draw(image)
    color = "#087466" if not evidence.get("reason") else "#ab661b"
    for row in evidence.get("candidates", []):
        polygon = list(zip(row["polygon"][::2], row["polygon"][1::2]))
        drawing.line(polygon + polygon[:1], fill=color, width=max(3, image.width // 180))
    return image


def evidence_region(image, candidates):
    if candidates:
        coords = [item["polygon"] for item in candidates]
        xs = [x for polygon in coords for x in polygon[::2]]
        ys = [y for polygon in coords for y in polygon[1::2]]
        region = (max(0, min(xs) - 24), max(0, min(ys) - 24),
                  min(image.width, max(xs) + 24), min(image.height, max(ys) + 24))
        if region[2] > region[0] and region[3] > region[1]:
            return image.crop(region)
    return None


def review_workspace(png, row, key, metadata):
    evidence = row.get("evidence", {})
    candidates = evidence.get("candidates", [])
    overlay_key = f"overlay:{key}"
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Receipt scan")
            mode = st.radio("Image layer", ["Original scan", "Total evidence"], horizontal=True,
                            key=overlay_key, label_visibility="collapsed")
            zoom = st.slider("Zoom", 50, 180, 100, 10, key=f"zoom:{key}", format="%d%%")
            image = draw_receipt(png, evidence, mode)
            if mode == "Total evidence" and not candidates:
                st.caption("OCR found no total region for this receipt.")
            if mode == "Total evidence":
                crop = evidence_region(image, candidates)
                if crop is not None:
                    st.image(crop, caption="OCR evidence crop", width="stretch")
            st.image(image, width=int(450 * zoom / 100))
    with right:
        with st.container(border=True):
            st.subheader("Verify and approve")
            if row["status"] == "NEEDS_REVIEW":
                st.warning("Needs review · " + ", ".join(row["reasons"]))
            else:
                st.success("Auto-pass")
            st.caption(f"{metadata['runtime']} · {metadata.get('model') or metadata['method']} · {row['latency_ms']:.0f} ms")
            if row.get("error_detail") or row.get("ocr_error"):
                st.caption(row.get("error_detail") or row.get("ocr_error"))
            if candidates:
                st.caption("OCR amount evidence")
                st.button("Zoom to total", key=f"focus:{key}",
                          on_click=lambda: st.session_state.update({overlay_key: "Total evidence"}))
                for candidate in candidates:
                    st.text(f"{candidate['text']}  ·  OCR {candidate['confidence']:.0f}/100")
            if "gold" in row:
                with st.expander("Compare with SROIE labels"):
                    st.dataframe([{"Field": LABELS[f], "Prediction": row["fields"][f], "Label": row["gold"][f],
                                   "Match": row["match"][f]} for f in FIELDS], hide_index=True)
            state_key = f"approved:{key}"
            with st.form(f"review:{key}"):
                reviewed = {field: st.text_input(LABELS[field], row["fields"][field] or "",
                                                 max_chars=1000, key=f"value:{key}:{field}")
                            for field in FIELDS}
                confirmed = st.checkbox("I checked the original scan")
                submitted = st.form_submit_button("Save approval", type="primary")
            if submitted:
                st.session_state.pop(state_key, None)
                reasons = review_reasons(reviewed)
                if reasons or not confirmed:
                    st.error(", ".join(reasons) if reasons else "Confirm that you checked the original scan.")
                else:
                    st.session_state[state_key] = {
                        "receipt_id": metadata["receipt_id"], "runtime": metadata["runtime"],
                        "model": metadata.get("model"), "model_revision": metadata.get("model_revision"),
                        "status": "HUMAN_REVIEWED", "reviewed_at": datetime.now(timezone.utc).isoformat(),
                        "original": row["fields"], "reviewed": reviewed, "reasons_before_review": row["reasons"],
                    }
            if state_key in st.session_state:
                st.download_button("Export approved JSON", json.dumps(st.session_state[state_key], ensure_ascii=False, indent=2),
                                   "receipt-reviewed.json", "application/json", on_click="ignore", key=f"download:{key}")
            with st.expander("JSON and decision trace"):
                st.json(row["fields"])
                st.code(row.get("raw") or "No model output")
                if row.get("ocr_lines"):
                    st.json(row["ocr_lines"], expanded=False)


def case_replay():
    bundle = json.loads((REPLAY_DIR / "cases.json").read_text(encoding="utf-8"))
    source = bundle["provenance"]
    cases = {case["title"]: case for case in bundle["cases"]}
    st.caption(f"{source['model']} · Saved GPU predictions from {source['documents']} SROIE receipts")
    incident_case = st.session_state.pop("open_incident_receipt", False)
    if incident_case:
        st.session_state.replay_case = next(name for name, case in cases.items()
                                           if case["row"]["key"] == INCIDENT_RECEIPT)
        st.session_state.replay_gate = GATES[2]
    st.button("Investigate the review spike", on_click=navigate, args=("Monitoring",))
    choice = st.selectbox("Case", list(cases), key="replay_case")
    case = cases[choice]
    row = case["row"]
    st.write(case["lesson"])
    st.caption(f"SROIE receipt {row['key']}")
    st.session_state.setdefault("replay_gate", GATES[1])
    gate = st.radio("Decision policy", GATES, index=None, format_func=GATE_LABELS.get,
                    horizontal=True, key="replay_gate")
    selected = apply_gate(row, gate)
    candidates = selected["evidence"].get("candidates", [])
    with st.container(border=True):
        left, right = st.columns([1.08, 1], gap="large")
        with left:
            overlay = st.checkbox("Show OCR evidence", value=True, key=f"replay-overlay:{row['key']}")
            image = draw_receipt((REPLAY_DIR / f"{row['key']}.webp").read_bytes(),
                                 selected["evidence"], "Total evidence" if overlay else "Original scan")
            crop = evidence_region(image, candidates) if overlay else None
            if crop is not None:
                st.image(crop, caption="OCR region · click to enlarge", width="stretch")
            if overlay and not candidates:
                st.caption("This policy has no OCR region to show.")
            with st.expander("Full receipt", expanded=crop is None):
                st.image(image, width=420)
        with right:
            if selected["reasons"]:
                st.warning("Needs review")
                st.code("\n".join(selected["reasons"]), language=None)
            else:
                st.success("Auto-pass")
            st.caption(f"{sum(row['match'].values())}/4 fields match the dataset")
            total_reasons = total_review_reasons(row)
            if total_reasons:
                st.caption("Total-only policy · Verify total")
            else:
                st.caption("Total-only policy · Total supported by OCR. Company, date and address still require verification.")
            focus = case["focus"]
            st.markdown(f"**{LABELS[focus]}** · " + ("Matches label" if row["match"][focus] else "Differs from label"))
            st.caption("Qwen prediction")
            st.text(row["fields"][focus], width="stretch")
            st.caption("SROIE label")
            st.text(row["gold"][focus], width="stretch")
            with st.expander("Compare all four fields"):
                st.table([
                    {"Field": LABELS[f], "Qwen": row["fields"][f], "SROIE": row["gold"][f],
                     "Match": "Yes" if row["match"][f] else "No"}
                    for f in FIELDS
                ])
            if candidates:
                st.markdown("**OCR amount candidates**")
                for item in candidates:
                    st.text(f"{item['text']} · OCR {item['confidence']:.0f}/100")
            with st.expander("Raw model response and decision trace"):
                st.code(row["raw"], language="json")
                st.json({"gate": gate, "reasons": selected["reasons"],
                         "evidence": selected["evidence"], "model_revision": source["model_revision"],
                         "archive_sha256": source["archive_sha256"], "timings_ms": row["timings_ms"]})
                st.download_button("Download recorded case", json.dumps(row, ensure_ascii=False, indent=2),
                                   f"{row['key']}.json", "application/json", on_click="ignore")
    if row["key"] == INCIDENT_RECEIPT:
        prompt_decision()


def navigate(view, incident_receipt=False):
    st.session_state.workspace = view
    if incident_receipt:
        st.session_state.open_incident_receipt = True


def prompt_decision():
    result = json.loads((PROJECT / "docs" / "address-dev10-result.json").read_text(encoding="utf-8"))
    exact = result["metrics"]["address"]
    st.subheader("Why we kept the original prompt")
    st.caption(f"Measured Qwen A/B test · {result['documents']} diagnostic SROIE train receipts")
    st.write(f"Address exact match stayed at {exact['baseline_correct']}/{result['documents']}. "
             f"The new prompt fixed {exact['fixed']} address and regressed on {exact['regressed']}.")
    st.table([
        {"Receipt": "X51006008086", "Change": "Removed the extra AEON SHAH ALAM line",
         "Result": "Fixed"},
        {"Receipt": "X51006441474", "Change": "Included the printed Bentong-2 branch header",
         "Result": "Regression"},
    ])
    st.write("The prompt candidate failed the clean-control check. A dedicated address crop was the next experiment.")
    if DRIFT_REPORT.exists():
        measured = json.loads(DRIFT_REPORT.read_text(encoding="utf-8"))
        if measured["advance"]:
            st.caption("WildReceipt crop follow-up · Passed development. Hold rollout until the external holdout is scored.")
        else:
            st.caption("WildReceipt crop follow-up · Hold rollout. The paired crop did not pass development.")


def monitoring_view():
    days = daily_operations()
    by_day = {row["date"]: row for row in days}
    st.subheader("Review operations")
    st.caption("Simulated operations · 14 days. The linked receipt comes from the measured SROIE benchmark.")
    selected = st.selectbox("Investigation date", list(by_day), index=list(by_day).index(INCIDENT_DAY),
                            format_func=lambda day: day.strftime("%b %d, %Y"), key="monitor_day")
    row = by_day[selected]
    if selected == INCIDENT_DAY:
        st.warning("Review spike · A new receipt batch arrived and address checks drove most of the new review work.")
        st.button("Open recorded receipt", type="primary", on_click=navigate, args=("Case replay", True),
                  help="Recorded SROIE benchmark example linked to illustrate the simulated incident.")
    ops = st.columns(3)
    ops[0].metric("Receipts", row["received"])
    ops[1].metric("Needs review", f"{row['review_rate']:.1f}%")
    ops[2].metric("End-of-day backlog", row["backlog"])
    st.caption(f"Inference p95 {row['inference_p95_seconds']:.1f} s · "
               f"{row['inference_errors']} inference errors · {row['completed']} reviews completed")
    trend, detail = st.columns([1.12, 1], gap="large")
    with trend:
        st.markdown("**Daily review rate (%)**")
        st.line_chart([{"Date": item["date"].isoformat(), "Needs review (%)": item["review_rate"]}
                       for item in days], x="Date", y="Needs review (%)", height=220)
    with detail:
        st.markdown("**Selected day · Review reasons**")
        st.table([{"Primary reason": reason, "Receipts": count} for reason, count in row["reasons"].items()])
        st.markdown("**Verification**")
        st.write(f"Address corrections · {verification_text(*row['address_edits'])}")
        st.write(f"Total corrections · {verification_text(*row['total_edits'])}")
        st.write(f"Errors in sampled auto-passes · {verification_text(*row['auto_pass_audit'])}")
        st.caption("Corrections use verified fields as denominators. Auto-pass errors use audited receipts.")
    if selected == INCIDENT_DAY:
        drift_evaluation()


def drift_evaluation():
    st.subheader("Measured evaluation · SROIE and WildReceipt")
    st.caption("Offline address experiment on real receipts. The incident timeline above is simulated.")
    if not DRIFT_REPORT.exists():
        st.info("The datasets and labels are prepared. Paired GPU results are pending.")
        return
    report = json.loads(DRIFT_REPORT.read_text(encoding="utf-8"))
    comparison = []
    for cohort, label in (("wildreceipt", "WildReceipt dev"), ("sroie", "SROIE controls")):
        summary = report["metrics"][cohort]
        exact = summary["exact"]
        tolerant = summary["format_tolerant"]
        n = summary["documents"]
        comparison.append({
            "Cohort": label, "Exact · Full → Crop": f"{exact['baseline']}/{n} → {exact['candidate']}/{n}",
            "Format-tolerant · Full → Crop": f"{tolerant['baseline']}/{n} → {tolerant['candidate']}/{n}",
            "Fixed / regressed (exact)": f"{exact['fixed']} / {exact['regressed']}",
            "Fallbacks": f"{summary['fallbacks']}/{n}",
            "Crop p95 added": f"{summary['added_ms']['p95'] / 1000:.2f} s",
        })
    st.table(comparison)
    if report["advance"]:
        st.info("Advance to holdout. Keep the current pipeline while the sealed test runs.")
    else:
        st.warning("Keep full-image extraction. The crop fixed no WildReceipt addresses and broke one SROIE control.")
        st.write(f"In `{report['regression_id']}`, the second pass dropped the postcode and the rest of the address. "
                 "Its non-empty JSON passed the fallback check.")
    st.caption("The SROIE controls were selected for previously correct addresses. The cohort scores describe these samples.")


def new_receipt():
    st.subheader("Live extraction")
    uploaded = st.file_uploader("Upload a PNG or JPEG receipt", type=["png", "jpg", "jpeg"], key="image")
    if uploaded is None:
        st.info("Choose a receipt to start. Live extraction uses Ollama at 127.0.0.1.")
        return
    raw = uploaded.getvalue()
    fingerprint = hashlib.sha256(raw).hexdigest()
    try:
        png = prepare_image(raw, uploads_only=True)
    except (ValueError, OSError, Image.DecompressionBombError) as error:
        st.error(f"Could not read the image. {error}")
        return
    stored = st.session_state.get("live")
    if stored is None or stored["image_hash"] != fingerprint:
        st.session_state.pop("live", None)
        stored = None
    if st.button("Retry Qwen" if stored else "Run Qwen", type="primary"):
        with st.status("Connecting to Ollama and Tesseract", expanded=True) as progress:
            try:
                backend = OllamaBackend()
                try:
                    ocr = TesseractBackend()
                except (RuntimeError, OSError) as error:
                    reason = str(error)
                    def ocr(_):
                        raise RuntimeError(reason)
                progress.write("Extracting receipt fields and checking total evidence.")
                row = extract_image(png, "ollama", backend, ocr)
                stored = {"image_hash": fingerprint, "row": row, "revision": backend.revision}
                st.session_state["live"] = stored
                failed = "INFERENCE_ERROR" in row["reasons"]
                progress.update(label="Qwen failed" if failed else "Extraction complete",
                                state="error" if failed else "complete", expanded=False)
            except Exception as error:
                progress.update(label="Runtime unavailable", state="error", expanded=True)
                st.error(f"Check Ollama and model {OLLAMA_MODEL}, then retry. {error}")
    if stored:
        review_key = fingerprint + hashlib.sha256(repr(stored["row"]).encode()).hexdigest()[:12]
        review_workspace(png, stored["row"], review_key, {
            "receipt_id": fingerprint[:12], "method": "ollama", "runtime": "Ollama · quantized",
            "model": OLLAMA_MODEL, "model_revision": stored["revision"],
        })
    else:
        st.image(png, width=450, caption=uploaded.name)


def load_benchmark(path, uploads):
    info = path.stat()
    index, sha = archive_index(str(path), info.st_mtime_ns, info.st_size)
    report_paths = sorted((WORK / "receipt-v2").glob("*.json"))
    if not report_paths and (WORK / "qwen-receipt-clean-baseline.json").is_file():
        report_paths = [WORK / "qwen-receipt-clean-baseline.json"]
    reports = {}
    data = [(p.name, p.read_text(encoding="utf-8-sig")) for p in report_paths]
    data.extend((upload.name, upload.getvalue()) for upload in uploads)
    for name, raw in data:
        try:
            report = json.loads(raw)
            if (report.get("archive_sha256") != sha or not isinstance(report.get("receipts"), list)
                    or not report["receipts"] or not {"method", "split"} <= set(report)):
                raise ValueError("Empty report or mismatched ZIP hash.")
            ids = []
            for row in report["receipts"]:
                required = {"key", "fields", "gold", "match", "status", "latency_ms", "schema_valid"}
                if not isinstance(row, dict) or not required <= row.keys():
                    raise ValueError("Report is missing required fields.")
                if any(not isinstance(row[field], dict) or not set(FIELDS) <= row[field].keys()
                       for field in ("fields", "gold", "match")):
                    raise ValueError("Report contains incomplete receipt fields.")
                if not isinstance(row["key"], str) or not isinstance(row["latency_ms"], (int, float)):
                    raise ValueError("Invalid receipt ID or latency.")
                ids.append(row["key"])
            if len(ids) != len(set(ids)):
                raise ValueError("Report contains duplicate receipt IDs.")
            reports[name] = report
        except (AttributeError, ValueError, TypeError) as error:
            st.warning(f"{name} · {error}")
    return reports, index


def benchmark_view():
    st.subheader("Benchmark")
    with st.expander("Data sources"):
        archive_path = st.text_input("ZIP SROIE", str(WORK / "sroie-kaggle.zip"))
        uploads = st.file_uploader("Add benchmark JSON reports", type="json", accept_multiple_files=True)
    path = Path(archive_path)
    try:
        reports, index = load_benchmark(path, uploads or [])
    except (OSError, BadZipFile, ValueError) as error:
        st.info(f"Open the dataset ZIP to view benchmarks. {error}")
        return
    if not reports:
        st.info("No compatible reports found.")
        return
    groups = {}
    for name, report in reports.items():
        ids = tuple(sorted(row["key"] for row in report["receipts"]))
        groups.setdefault((report["split"], ids), []).append(name)
    options = list(groups)
    cohort = st.selectbox("Receipt cohort", options, format_func=lambda item: (
        f"{item[0]} · {len(item[1])} receipts · {len(groups[item])} report · "
        f"{hashlib.sha256('|'.join(item[1]).encode()).hexdigest()[:6]}"
    ))
    names = groups[cohort]
    merchant = st.selectbox("Merchant group", ["All", "Seen", "Unseen"])
    group = {"Seen": "seen", "Unseen": "unseen"}.get(merchant)
    filtered = {name: [row for row in reports[name]["receipts"] if not group or row.get("merchant_group") == group]
                for name in names}
    stats = []
    for name in names:
        s = summarize(filtered[name])
        n = s["documents"]
        stats.append({"Report": name, "Pipeline": METHODS.get(reports[name]["method"], reports[name]["method"]),
                      "Receipts": n, "Valid JSON": percent(s["schema_valid"], n),
                      "All four fields": percent(s["exact_all_fields"], n),
                      **{f: percent(s["field_exact"][f], n) for f in FIELDS},
                      "Auto-pass": percent(s["rules_passed"], n),
                      "Needs review": percent(s["needs_review"], n),
                      "Errors among passes": percent(s["incorrect_among_rules_passed"], s["rules_passed"]),
                      "Wrong totals flagged": percent(s["wrong_total_flagged"], s["wrong_total"]),
                      "Additional wrong totals flagged by OCR": percent(s["wrong_total_caught_by_evidence"], s["wrong_total"]),
                      "p50 ms": s["latency_ms"]["median"], "p95 ms": s["latency_ms"]["p95_nearest_rank"]})
    st.dataframe(stats, hide_index=True)
    st.caption("Reports share the same split and receipt IDs. Failures count in the denominator. "
               "The full runner includes image preparation in latency.")
    name = st.selectbox("Report to inspect", names)
    rows = filtered[name]
    if "metrics" in reports[name] and all("gate_reasons" in row for row in rows):
        with st.expander("Compare three gates on the same predictions", expanded=True):
            gate_rows = []
            for gate, label in GATE_LABELS.items():
                g = gate_metrics(rows, gate)
                gate_rows.append({
                    "Gate": label, "Passed": percent(g["coverage"]["count"], g["coverage"]["n"]),
                    "Errors among passes": percent(g["selective_risk"]["count"], g["selective_risk"]["n"]),
                    "Needs review": g["review_workload"]["count"], "Incorrect passes": g["false_pass_count"],
                    "Correct records flagged": g["correct_flagged_count"],
                    "Wrong totals flagged": percent(g["wrong_total_detected"]["count"], g["wrong_total_detected"]["n"]),
                })
            st.dataframe(gate_rows, hide_index=True)
        gate = st.selectbox("Show evidence and status for gate", GATES, index=1,
                            format_func=GATE_LABELS.get, key=f"gate:{name}")
        rows = [apply_gate(row, gate) for row in rows]
    case_filter = st.radio("Case filter", ["All", "Needs review", "Incorrect auto-pass"], horizontal=True)
    cases = [row for row in rows if case_filter == "All"
             or (case_filter == "Needs review" and row["status"] == "NEEDS_REVIEW")
             or (case_filter == "Incorrect auto-pass" and row["status"] == "RULES_PASSED" and not all(row["match"].values()))]
    if not cases:
        st.info("No receipts match this filter.")
        return
    key = st.selectbox("Receipt", [row["key"] for row in cases])
    row = next(item for item in cases if item["key"] == key)
    report = reports[name]
    try:
        with ZipFile(path) as archive:
            png = prepare_image(archive.read(index[(report["split"], key)]))
    except (OSError, KeyError, ValueError) as error:
        st.error(f"Could not open the image. {error}")
        return
    run_id = hashlib.sha256(repr((name, report.get("model_revision"), row)).encode()).hexdigest()[:16]
    review_workspace(png, row, f"{key}:{run_id}", {
        "receipt_id": key, "method": report["method"], "runtime": report.get("runtime", report["method"]),
        "model": report.get("model"), "model_revision": report.get("model_revision"),
    })


st.set_page_config(page_title="Receipt Review", layout="wide", initial_sidebar_state="collapsed")
st.title("Receipt Review")
st.caption("Receipt extraction, OCR evidence and human review")
st.session_state.setdefault("workspace", "Case replay")
view = st.segmented_control("Workspace", ["Case replay", "Monitoring", "Live extraction", "Benchmark"],
                            selection_mode="single", key="workspace")
if view == "Case replay":
    case_replay()
elif view == "Monitoring":
    monitoring_view()
elif view == "Benchmark":
    benchmark_view()
else:
    new_receipt()
