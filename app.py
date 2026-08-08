"""
Sparky OCR — Streamlit web UI.

Features:
  - PDF upload into ./uploads
  - Format selector (Markdown / DOCX)
  - Custom output filename
  - Conversion trigger (Docling → pdfplumber fallback)
  - File explorer for uploads/ and converted/ (list / preview / delete)

Binds to 0.0.0.0:8501 via Streamlit CLI flags in docker-compose.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import streamlit as st

import config
from converter import (
    convert_pdf,
    sanitize_filename,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sparky_ocr.app")

config.ensure_directories()

# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sparky OCR",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.5rem; max-width: 1100px; }
    h1.app-title {
        font-family: "Segoe UI", "Helvetica Neue", sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }
    .app-sub {
        color: #5c6570;
        margin-bottom: 1.25rem;
    }
    .file-meta { color: #6b7280; font-size: 0.85rem; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def _list_files(directory: Path, extensions: set[str] | None = None) -> list[Path]:
    if not directory.exists():
        return []
    files = [p for p in directory.iterdir() if p.is_file() and not p.name.startswith(".")]
    if extensions:
        files = [p for p in files if p.suffix.lower() in extensions]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _list_converted() -> list[Path]:
    results: list[Path] = []
    for sub in config.FORMAT_DIRS.values():
        results.extend(_list_files(sub, {".md", ".docx"}))
    return sorted(results, key=lambda p: p.stat().st_mtime, reverse=True)


def _save_upload(uploaded_file) -> Path:
    safe_stem = sanitize_filename(Path(uploaded_file.name).stem)
    dest = config.UPLOADS_DIR / f"{safe_stem}.pdf"
    # Avoid silent overwrite collisions
    if dest.exists():
        stamp = datetime.now().strftime("%H%M%S")
        dest = config.UPLOADS_DIR / f"{safe_stem}_{stamp}.pdf"
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


def _preview_text(path: Path, max_chars: int = 4000) -> str:
    try:
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8", errors="replace")
            return text if len(text) <= max_chars else text[:max_chars] + "\n\n… (truncated)"
        if path.suffix.lower() == ".docx":
            from docx import Document

            doc = Document(str(path))
            parts = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(parts)
            return text if len(text) <= max_chars else text[:max_chars] + "\n\n… (truncated)"
        if path.suffix.lower() == ".pdf":
            return f"(Binary PDF — {_human_size(path.stat().st_size)}. Convert to preview contents.)"
    except Exception as exc:
        return f"Preview unavailable: {exc}"
    return "(No preview for this file type.)"


def _delete_file(path: Path) -> None:
    # Only allow deletes inside our mounted data dirs
    resolved = path.resolve()
    allowed_roots = [
        config.UPLOADS_DIR.resolve(),
        config.CONVERTED_DIR.resolve(),
    ]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise PermissionError(f"Refusing to delete outside data directories: {resolved}")
    resolved.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Sidebar — system status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Sparky OCR")
    st.caption("DGX Spark · ARM64 · CUDA")
    st.divider()
    st.markdown("**Paths**")
    st.code(
        f"uploads  → {config.UPLOADS_DIR}\n"
        f"converted→ {config.CONVERTED_DIR}\n"
        f"models   → {config.MODELS_CACHE_DIR}",
        language=None,
    )
    st.markdown("**Engine**")
    st.write(f"OCR engine: `{config.OCR_ENGINE}`")
    st.write(f"Accelerator: `{config.ACCELERATOR_DEVICE}`")
    st.write(f"TZ: `{__import__('os').environ.get('TZ', 'unset')}`")

    cuda_ok = False
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        st.write(f"PyTorch CUDA: `{'available' if cuda_ok else 'not available'}`")
        if cuda_ok:
            st.write(f"Device: `{torch.cuda.get_device_name(0)}`")
    except Exception:
        st.write("PyTorch: `not importable`")

    if st.button("Refresh file lists", use_container_width=True):
        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<h1 class="app-title">Sparky OCR</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">Convert chat-export PDFs into clean Markdown or Word documents '
    "on your local Tailscale network.</p>",
    unsafe_allow_html=True,
)

tab_convert, tab_browser = st.tabs(["Convert", "File Explorer"])


# ---------------------------------------------------------------------------
# Tab 1 — Upload & Convert
# ---------------------------------------------------------------------------
with tab_convert:
    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        st.subheader("1. Upload PDF")
        uploaded = st.file_uploader(
            "Drop a PDF (Print-to-PDF chat exports work best)",
            type=["pdf"],
            accept_multiple_files=False,
            help=f"Files are stored under {config.UPLOADS_DIR}",
        )

        existing = _list_files(config.UPLOADS_DIR, {".pdf"})
        source_choice = None
        if existing:
            labels = {p.name: p for p in existing}
            pick = st.selectbox(
                "Or pick an already-uploaded PDF",
                options=["—"] + list(labels.keys()),
                index=0,
            )
            if pick != "—":
                source_choice = labels[pick]

        st.subheader("2. Output options")
        output_format = st.radio(
            "Target format",
            options=["markdown", "docx"],
            format_func=lambda x: "Markdown (.md)" if x == "markdown" else "Word Document (.docx)",
            horizontal=True,
        )

        default_name = ""
        if uploaded is not None:
            default_name = sanitize_filename(Path(uploaded.name).stem)
        elif source_choice is not None:
            default_name = sanitize_filename(source_choice.stem)

        custom_name = st.text_input(
            "Output filename (optional)",
            value=default_name,
            placeholder="e.g. grok_chat_conversion",
            help="Spaces become underscores; special characters are stripped. "
            "Date prefix YYYYMMDD is added automatically.",
        )

        force_chat = st.checkbox(
            "Force chat User / Assistant formatting",
            value=True,
            help="Detects prompts vs replies and inserts ### User / ### Assistant separators.",
        )

        engine_override = st.selectbox(
            "OCR engine",
            options=["auto", "docling", "pdfplumber"],
            index=0,
            help="auto tries Docling (GPU) then falls back to pdfplumber.",
        )

        convert_clicked = st.button("Convert", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Output path preview")
        stem_preview = sanitize_filename(custom_name) if custom_name.strip() else "filename"
        date_preview = datetime.now().strftime("%Y%m%d")
        ext = "md" if output_format == "markdown" else "docx"
        st.code(
            f"./converted/{output_format}/{date_preview}_{stem_preview}.{ext}",
            language=None,
        )
        st.caption("Timezone for the date prefix: Europe/Bratislava (via TZ).")

        if convert_clicked:
            pdf_path: Path | None = None
            try:
                if uploaded is not None:
                    with st.spinner("Saving upload…"):
                        pdf_path = _save_upload(uploaded)
                        st.success(f"Saved upload → `{pdf_path.name}`")
                elif source_choice is not None:
                    pdf_path = source_choice
                else:
                    st.error("Upload a PDF or select one from the list.")

                if pdf_path is not None:
                    with st.spinner(f"Converting with {engine_override}…"):
                        result = convert_pdf(
                            pdf_path,
                            output_format=output_format,  # type: ignore[arg-type]
                            output_filename=custom_name or None,
                            engine=engine_override,
                            force_chat_format=force_chat,
                        )
                    st.success("Conversion complete")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Engine", result.engine)
                    m2.metric("Chat formatted", "yes" if result.chat_formatted else "no")
                    m3.metric("Characters", f"{result.characters:,}")
                    st.code(str(result.output_path), language=None)

                    preview = _preview_text(result.output_path)
                    with st.expander("Preview output", expanded=True):
                        if result.format == "markdown":
                            st.markdown(preview)
                        else:
                            st.text(preview)
            except Exception as exc:
                logger.exception("Conversion failed")
                st.error(f"Conversion failed: {exc}")


# ---------------------------------------------------------------------------
# Tab 2 — File Explorer
# ---------------------------------------------------------------------------
with tab_browser:
    st.subheader("Uploads")
    uploads = _list_files(config.UPLOADS_DIR, {".pdf"})
    if not uploads:
        st.info("No uploaded PDFs yet.")
    else:
        for path in uploads:
            c1, c2, c3, c4 = st.columns([3.5, 1.2, 1.2, 1])
            with c1:
                st.markdown(f"**{path.name}**")
                st.caption(
                    f"{_human_size(path.stat().st_size)} · "
                    f"{datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}"
                )
            with c2:
                if st.button("Preview", key=f"prev_up_{path.name}"):
                    st.session_state["preview_path"] = str(path)
            with c3:
                st.download_button(
                    "Download",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/pdf",
                    key=f"dl_up_{path.name}",
                )
            with c4:
                if st.button("Delete", key=f"del_up_{path.name}"):
                    try:
                        _delete_file(path)
                        st.toast(f"Deleted {path.name}")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

    st.divider()
    st.subheader("Converted outputs")
    converted = _list_converted()
    if not converted:
        st.info("No converted files yet.")
    else:
        for path in converted:
            rel = path.relative_to(config.CONVERTED_DIR)
            c1, c2, c3, c4 = st.columns([3.5, 1.2, 1.2, 1])
            with c1:
                st.markdown(f"**{rel.as_posix()}**")
                st.caption(
                    f"{_human_size(path.stat().st_size)} · "
                    f"{datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}"
                )
            with c2:
                if st.button("Preview", key=f"prev_cv_{rel.as_posix()}"):
                    st.session_state["preview_path"] = str(path)
            with c3:
                mime = (
                    "text/markdown"
                    if path.suffix.lower() == ".md"
                    else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
                st.download_button(
                    "Download",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=mime,
                    key=f"dl_cv_{rel.as_posix()}",
                )
            with c4:
                if st.button("Delete", key=f"del_cv_{rel.as_posix()}"):
                    try:
                        _delete_file(path)
                        st.toast(f"Deleted {path.name}")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

    preview_path = st.session_state.get("preview_path")
    if preview_path:
        p = Path(preview_path)
        if p.exists():
            st.divider()
            st.subheader(f"Preview — {p.name}")
            content = _preview_text(p)
            if p.suffix.lower() == ".md":
                st.markdown(content)
            else:
                st.text(content)
        else:
            st.session_state.pop("preview_path", None)
