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
LABELS = {"company": "Cửa hàng", "date": "Ngày", "address": "Địa chỉ", "total": "Tổng tiền"}
METHODS = {"annotated-ocr": "OCR chú thích + rule", "tesseract": "Tesseract + rule",
           "qwen": "Qwen · checkpoint FP16", "ollama": "Qwen · Ollama lượng tử hóa"}


@st.cache_data
def archive_index(path, modified, size):
    with ZipFile(path) as archive:
        return {(r.split, r.key): r.image for r in inventory(archive)}, sha256_file(path)


def percent(num, den):
    return f"{num}/{den} · {num / den:.1%}" if den else "—"


def draw_receipt(png, evidence, mode):
    image = Image.open(BytesIO(png)).convert("RGB")
    if mode == "Ảnh gốc":
        return image
    drawing = ImageDraw.Draw(image)
    color = "#087466" if not evidence.get("reason") else "#ab661b"
    for row in evidence.get("candidates", []):
        polygon = list(zip(row["polygon"][::2], row["polygon"][1::2]))
        drawing.line(polygon + polygon[:1], fill=color, width=max(3, image.width // 180))
    return image


def review_workspace(png, row, key, metadata):
    evidence = row.get("evidence", {})
    candidates = evidence.get("candidates", [])
    overlay_key = f"overlay:{key}"
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Ảnh biên lai")
            mode = st.radio("Lớp hiển thị", ["Ảnh gốc", "Vùng tổng tiền"], horizontal=True,
                            key=overlay_key, label_visibility="collapsed")
            zoom = st.slider("Zoom ảnh", 50, 180, 100, 10, key=f"zoom:{key}", format="%d%%")
            image = draw_receipt(png, evidence, mode)
            if mode == "Vùng tổng tiền" and not candidates:
                st.caption("Ca này chưa có vùng tổng tiền từ OCR runtime.")
            if mode == "Vùng tổng tiền" and candidates:
                coords = [item["polygon"] for item in candidates]
                xs = [x for polygon in coords for x in polygon[::2]]
                ys = [y for polygon in coords for y in polygon[1::2]]
                region = (max(0, min(xs) - 24), max(0, min(ys) - 24),
                          min(image.width, max(xs) + 24), min(image.height, max(ys) + 24))
                if region[2] > region[0] and region[3] > region[1]:
                    st.image(image.crop(region), caption="Vùng OCR được phóng", width="stretch")
            st.image(image, width=int(450 * zoom / 100))
    with right:
        with st.container(border=True):
            st.subheader("Đối chiếu và duyệt")
            if row["status"] == "NEEDS_REVIEW":
                st.warning("Cần duyệt · " + ", ".join(row["reasons"]))
            else:
                st.success("Qua kiểm tra tự động")
            st.caption(f"{metadata['runtime']} · {metadata.get('model') or metadata['method']} · {row['latency_ms']:.0f} ms")
            if row.get("error_detail") or row.get("ocr_error"):
                st.caption(row.get("error_detail") or row.get("ocr_error"))
            if candidates:
                st.caption("Bằng chứng tổng tiền từ OCR")
                st.button("Xem vùng Tổng tiền", key=f"focus:{key}",
                          on_click=lambda: st.session_state.update({overlay_key: "Vùng tổng tiền"}))
                for candidate in candidates:
                    st.text(f"{candidate['text']}  ·  OCR {candidate['confidence']:.0f}/100")
            if "gold" in row:
                with st.expander("So với nhãn SROIE"):
                    st.dataframe([{"Trường": LABELS[f], "Dự đoán": row["fields"][f], "Nhãn": row["gold"][f],
                                   "Khớp": row["match"][f]} for f in FIELDS], hide_index=True)
            state_key = f"approved:{key}"
            with st.form(f"review:{key}"):
                reviewed = {field: st.text_input(LABELS[field], row["fields"][field] or "",
                                                 max_chars=1000, key=f"value:{key}:{field}")
                            for field in FIELDS}
                confirmed = st.checkbox("Đã đối chiếu với ảnh gốc")
                submitted = st.form_submit_button("Lưu bản duyệt", type="primary")
            if submitted:
                st.session_state.pop(state_key, None)
                reasons = review_reasons(reviewed)
                if reasons or not confirmed:
                    st.error(", ".join(reasons) if reasons else "Hãy xác nhận đã đối chiếu ảnh.")
                else:
                    st.session_state[state_key] = {
                        "receipt_id": metadata["receipt_id"], "runtime": metadata["runtime"],
                        "model": metadata.get("model"), "model_revision": metadata.get("model_revision"),
                        "status": "HUMAN_REVIEWED", "reviewed_at": datetime.now(timezone.utc).isoformat(),
                        "original": row["fields"], "reviewed": reviewed, "reasons_before_review": row["reasons"],
                    }
            if state_key in st.session_state:
                st.download_button("Xuất JSON đã duyệt", json.dumps(st.session_state[state_key], ensure_ascii=False, indent=2),
                                   "receipt-reviewed.json", "application/json", on_click="ignore", key=f"download:{key}")
            with st.expander("JSON và dấu vết xử lý"):
                st.json(row["fields"])
                st.code(row.get("raw") or "Chưa có đầu ra")
                if row.get("ocr_lines"):
                    st.json(row["ocr_lines"], expanded=False)


def new_receipt():
    st.subheader("Đọc ảnh mới")
    uploaded = st.file_uploader("Tải ảnh biên lai PNG hoặc JPEG", type=["png", "jpg", "jpeg"], key="image")
    if uploaded is None:
        st.info("Chọn một ảnh để mở workspace. App gọi Ollama qua 127.0.0.1.")
        return
    raw = uploaded.getvalue()
    fingerprint = hashlib.sha256(raw).hexdigest()
    try:
        png = prepare_image(raw, uploads_only=True)
    except (ValueError, OSError, Image.DecompressionBombError) as error:
        st.error(f"Ảnh không đọc được. {error}")
        return
    stored = st.session_state.get("live")
    if stored is None or stored["image_hash"] != fingerprint:
        st.session_state.pop("live", None)
        stored = None
    if st.button("Chạy lại Qwen" if stored else "Chạy Qwen", type="primary"):
        with st.status("Đang kết nối Ollama và Tesseract", expanded=True) as progress:
            try:
                backend = OllamaBackend()
                try:
                    ocr = TesseractBackend()
                except (RuntimeError, OSError) as error:
                    reason = str(error)
                    def ocr(_):
                        raise RuntimeError(reason)
                progress.write("Đang đọc ảnh và đối chiếu tổng tiền.")
                row = extract_image(png, "ollama", backend, ocr)
                stored = {"image_hash": fingerprint, "row": row, "revision": backend.revision}
                st.session_state["live"] = stored
                failed = "INFERENCE_ERROR" in row["reasons"]
                progress.update(label="Qwen gặp lỗi" if failed else "Đã xử lý ảnh",
                                state="error" if failed else "complete", expanded=False)
            except Exception as error:
                progress.update(label="Chưa kết nối được runtime", state="error", expanded=True)
                st.error(f"Kiểm tra Ollama và model {OLLAMA_MODEL}, sau đó thử lại. {error}")
    if stored:
        review_key = fingerprint + hashlib.sha256(repr(stored["row"]).encode()).hexdigest()[:12]
        review_workspace(png, stored["row"], review_key, {
            "receipt_id": fingerprint[:12], "method": "ollama", "runtime": "Ollama · lượng tử hóa",
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
                raise ValueError("Report trống hoặc hash ZIP khác.")
            ids = []
            for row in report["receipts"]:
                required = {"key", "fields", "gold", "match", "status", "latency_ms", "schema_valid"}
                if not isinstance(row, dict) or not required <= row.keys():
                    raise ValueError("Report thiếu trường dữ liệu.")
                if any(not isinstance(row[field], dict) or not set(FIELDS) <= row[field].keys()
                       for field in ("fields", "gold", "match")):
                    raise ValueError("Các trường trích xuất trong report không đầy đủ.")
                if not isinstance(row["key"], str) or not isinstance(row["latency_ms"], (int, float)):
                    raise ValueError("ID biên lai hoặc latency không hợp lệ.")
                ids.append(row["key"])
            if len(ids) != len(set(ids)):
                raise ValueError("Report chứa ID biên lai lặp.")
            reports[name] = report
        except (AttributeError, ValueError, TypeError) as error:
            st.warning(f"{name} · {error}")
    return reports, index


def benchmark_view():
    st.subheader("Benchmark")
    with st.expander("Nguồn dữ liệu"):
        archive_path = st.text_input("ZIP SROIE", str(WORK / "sroie-kaggle.zip"))
        uploads = st.file_uploader("Thêm report JSON từ notebook", type="json", accept_multiple_files=True)
    path = Path(archive_path)
    try:
        reports, index = load_benchmark(path, uploads or [])
    except (OSError, BadZipFile, ValueError) as error:
        st.info(f"Mở ZIP để xem benchmark. {error}")
        return
    if not reports:
        st.info("Chưa có report phù hợp.")
        return
    groups = {}
    for name, report in reports.items():
        ids = tuple(sorted(row["key"] for row in report["receipts"]))
        groups.setdefault((report["split"], ids), []).append(name)
    options = list(groups)
    cohort = st.selectbox("Tập biên lai", options, format_func=lambda item: (
        f"{item[0]} · {len(item[1])} biên lai · {len(groups[item])} report · "
        f"{hashlib.sha256('|'.join(item[1]).encode()).hexdigest()[:6]}"
    ))
    names = groups[cohort]
    merchant = st.selectbox("Nhóm cửa hàng", ["Tất cả", "Đã thấy", "Chưa thấy"])
    group = {"Đã thấy": "seen", "Chưa thấy": "unseen"}.get(merchant)
    filtered = {name: [row for row in reports[name]["receipts"] if not group or row.get("merchant_group") == group]
                for name in names}
    stats = []
    for name in names:
        s = summarize(filtered[name])
        n = s["documents"]
        stats.append({"Report": name, "Pipeline": METHODS.get(reports[name]["method"], reports[name]["method"]),
                      "Mẫu": n, "JSON hợp lệ": percent(s["schema_valid"], n),
                      "Đủ 4 trường": percent(s["exact_all_fields"], n),
                      **{f: percent(s["field_exact"][f], n) for f in FIELDS},
                      "Qua kiểm tra": percent(s["rules_passed"], n),
                      "Cần duyệt": percent(s["needs_review"], n),
                      "Sai trong nhóm qua": percent(s["incorrect_among_rules_passed"], s["rules_passed"]),
                      "Sai total được đưa duyệt": percent(s["wrong_total_flagged"], s["wrong_total"]),
                      "Sai total do bằng chứng chặn thêm": percent(s["wrong_total_caught_by_evidence"], s["wrong_total"]),
                      "p50 ms": s["latency_ms"]["median"], "p95 ms": s["latency_ms"]["p95_nearest_rank"]})
    st.dataframe(stats, hide_index=True)
    st.caption("Bảng so sánh các report có cùng split và cùng danh sách ID. Latency gồm OCR, trích xuất và kiểm tra sau khi nạp model. Ca hỏng vẫn nằm trong mẫu số.")
    name = st.selectbox("Report để xem ca", names)
    case_filter = st.radio("Nhóm ca", ["Tất cả", "Cần duyệt", "Qua kiểm tra nhưng sai"], horizontal=True)
    cases = [row for row in filtered[name] if case_filter == "Tất cả"
             or (case_filter == "Cần duyệt" and row["status"] == "NEEDS_REVIEW")
             or (case_filter == "Qua kiểm tra nhưng sai" and row["status"] == "RULES_PASSED" and not all(row["match"].values()))]
    if not cases:
        st.info("Chưa có ca trong bộ lọc.")
        return
    key = st.selectbox("Biên lai", [row["key"] for row in cases])
    row = next(item for item in cases if item["key"] == key)
    report = reports[name]
    try:
        with ZipFile(path) as archive:
            png = prepare_image(archive.read(index[(report["split"], key)]))
    except (OSError, KeyError, ValueError) as error:
        st.error(f"Không mở được ảnh. {error}")
        return
    run_id = hashlib.sha256(repr((name, report.get("model_revision"), row)).encode()).hexdigest()[:16]
    review_workspace(png, row, f"{key}:{run_id}", {
        "receipt_id": key, "method": report["method"], "runtime": report.get("runtime", report["method"]),
        "model": report.get("model"), "model_revision": report.get("model_revision"),
    })


st.set_page_config(page_title="Receipt Review", layout="wide", initial_sidebar_state="collapsed")
st.title("Receipt Review")
st.caption("Ảnh → trích xuất → đối chiếu vùng tổng tiền → duyệt → JSON")
view = st.segmented_control("Workspace", ["Đọc ảnh mới", "Benchmark"], default="Đọc ảnh mới",
                            selection_mode="single")
if view == "Benchmark":
    benchmark_view()
else:
    new_receipt()
