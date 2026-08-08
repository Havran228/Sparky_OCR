"""
Sparky OCR — document conversion engine.

Primary path: Docling (GPU-accelerated layout / OCR on ARM64 PyTorch).
Fallback: pdfplumber text extraction.

Includes specialized post-processing that detects AI chat "Print to PDF"
exports and reformats them as clean User / Assistant dialogue.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
from zoneinfo import ZoneInfo

from docx import Document
from docx.shared import Pt, RGBColor

import config

logger = logging.getLogger(__name__)

OutputFormat = Literal["markdown", "docx"]
EngineName = Literal["docling", "pdfplumber"]

LOCAL_TZ = ZoneInfo(__import__("os").environ.get("TZ", "Europe/Bratislava"))

# ---------------------------------------------------------------------------
# Chat-role detection patterns (Print-to-PDF from browsers / mobile apps)
# ---------------------------------------------------------------------------
# Each pattern captures an optional role label at the start of a line/block.
# Ordered from more specific → more generic.
ROLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # ChatGPT / OpenAI style
    (re.compile(r"^(?:You said|You)\s*[:：]\s*", re.IGNORECASE), "User"),
    (re.compile(r"^(?:ChatGPT(?:\s+said)?|GPT-\d+(?:\.\d+)?)\s*[:：]\s*", re.IGNORECASE), "Assistant"),
    # Claude / Anthropic
    (re.compile(r"^(?:Human|Claude User)\s*[:：]\s*", re.IGNORECASE), "User"),
    (re.compile(r"^(?:Claude(?:\s+\d+(?:\.\d+)?)?(?:\s+said)?|Anthropic)\s*[:：]\s*", re.IGNORECASE), "Assistant"),
    # Grok / xAI
    (re.compile(r"^(?:Grok(?:\s+\d+(?:\.\d+)?)?(?:\s+said)?|xAI)\s*[:：]\s*", re.IGNORECASE), "Assistant"),
    # Gemini / Google
    (re.compile(r"^(?:Gemini(?:\s+\w+)?(?:\s+said)?|Bard)\s*[:：]\s*", re.IGNORECASE), "Assistant"),
    # Generic role labels
    (re.compile(r"^(?:User|Prompt|Question|Me|I)\s*[:：]\s*", re.IGNORECASE), "User"),
    (re.compile(r"^(?:Assistant|AI|Bot|Response|Answer|Model|System)\s*[:：]\s*", re.IGNORECASE), "Assistant"),
    # Markdown-ish headings already present
    (re.compile(r"^#{1,3}\s*(?:User|Human|Prompt|You)\s*$", re.IGNORECASE), "User"),
    (re.compile(r"^#{1,3}\s*(?:Assistant|AI|ChatGPT|Claude|Grok|Gemini|Response)\s*$", re.IGNORECASE), "Assistant"),
]

# Noise often present in chat PDF exports
NOISE_LINE_RE = re.compile(
    r"^(?:"
    r"Skip to content|"
    r"Chat history|"
    r"New chat|"
    r"Regenerate|"
    r"Copy code|"
    r"Share|"
    r"Download|"
    r"Page\s+\d+(?:\s*/\s*\d+)?|"
    r"\d{1,2}:\d{2}\s*(?:AM|PM)?|"
    r"Today|Yesterday|"
    r"https?://\S+"
    r")\s*$",
    re.IGNORECASE,
)

MULTI_BLANK_RE = re.compile(r"\n{3,}")
TRAILING_WS_RE = re.compile(r"[ \t]+\n")


@dataclass
class ConversionResult:
    """Outcome of a single conversion job."""

    source_path: Path
    output_path: Path
    format: OutputFormat
    engine: EngineName
    chat_formatted: bool
    characters: int


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------
def sanitize_filename(name: str) -> str:
    """
    Normalize a user-provided or source filename stem:
    - Unicode normalize
    - spaces → underscores
    - strip characters that are not alphanumeric, underscore, hyphen, or period
    - collapse repeated underscores
    """
    if not name:
        return "untitled"

    # Drop any path components a caller might sneak in
    name = Path(name).name
    # Strip extension if present
    stem = Path(name).stem if Path(name).suffix else name

    stem = unicodedata.normalize("NFKD", stem)
    stem = stem.encode("ascii", "ignore").decode("ascii")
    stem = stem.replace(" ", "_")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")
    return stem or "untitled"


def dated_output_path(
    format_name: OutputFormat,
    filename_stem: str,
    when: Optional[datetime] = None,
) -> Path:
    """
    Build: ./converted/{FORMAT}/{YYYYMMDD}_{FILENAME}.{ext}
    """
    when = when or datetime.now(LOCAL_TZ)
    date_prefix = when.strftime("%Y%m%d")
    safe = sanitize_filename(filename_stem)
    ext = "md" if format_name == "markdown" else "docx"
    directory = config.FORMAT_DIRS[format_name]
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{date_prefix}_{safe}.{ext}"


# ---------------------------------------------------------------------------
# Text extraction engines
# ---------------------------------------------------------------------------
def _extract_with_docling(pdf_path: Path) -> str:
    """GPU-capable Docling extraction → markdown text."""
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    # Accelerator options (API differs slightly across Docling versions)
    try:
        from docling.datamodel.accelerator_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )

        device_map = {
            "CUDA": AcceleratorDevice.CUDA,
            "CPU": AcceleratorDevice.CPU,
            "AUTO": AcceleratorDevice.AUTO,
        }
        device = device_map.get(config.ACCELERATOR_DEVICE, AcceleratorDevice.AUTO)
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=device,
        )
    except Exception as exc:  # pragma: no cover - version skew
        logger.warning("Docling accelerator options unavailable: %s", exc)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


def _extract_with_pdfplumber(pdf_path: Path) -> str:
    """Lightweight text-layer extraction fallback."""
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            text = text.strip()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def extract_text(pdf_path: Path, engine: Optional[str] = None) -> tuple[str, EngineName]:
    """
    Extract text from a PDF using the configured engine strategy.

    Returns (text, engine_used).
    """
    preferred = (engine or config.OCR_ENGINE or "auto").lower()

    if preferred == "pdfplumber":
        return _extract_with_pdfplumber(pdf_path), "pdfplumber"

    if preferred == "docling":
        return _extract_with_docling(pdf_path), "docling"

    # auto: prefer Docling, fall back to pdfplumber
    try:
        return _extract_with_docling(pdf_path), "docling"
    except Exception as exc:
        logger.warning("Docling failed (%s); falling back to pdfplumber", exc)
        return _extract_with_pdfplumber(pdf_path), "pdfplumber"


# ---------------------------------------------------------------------------
# Chat log post-processing
# ---------------------------------------------------------------------------
def _detect_role(line: str) -> tuple[Optional[str], str]:
    """
    If `line` begins with a known chat role marker, return (role, remainder).
    Otherwise return (None, original_line).
    """
    stripped = line.strip()
    if not stripped:
        return None, line

    for pattern, role in ROLE_PATTERNS:
        match = pattern.match(stripped)
        if match:
            remainder = stripped[match.end() :].strip()
            # Heading-only lines (### User) leave an empty remainder — that's OK
            return role, remainder
    return None, line


def _looks_like_chat(text: str) -> bool:
    """Heuristic: true if we can find at least two role markers of either kind."""
    hits = 0
    for line in text.splitlines():
        role, _ = _detect_role(line)
        if role:
            hits += 1
            if hits >= 2:
                return True
    return False


def _strip_noise(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if NOISE_LINE_RE.match(line.strip()):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = TRAILING_WS_RE.sub("\n", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def format_chat_transcript(text: str, force: bool = False) -> tuple[str, bool]:
    """
    Identify User vs Assistant turns and rewrite with clear Markdown separators.

    Returns (formatted_markdown, was_chat_formatted).
    """
    text = _strip_noise(text)
    if not text:
        return "", False

    if not force and not _looks_like_chat(text):
        # Still normalize whitespace for non-chat PDFs
        return MULTI_BLANK_RE.sub("\n\n", text).strip() + "\n", False

    turns: list[tuple[str, list[str]]] = []
    current_role: Optional[str] = None
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_role, current_body
        if current_role is None:
            # Preamble before first role — keep as a note if non-empty
            body = "\n".join(current_body).strip()
            if body:
                turns.append(("Preamble", [body]))
        else:
            turns.append((current_role, current_body[:]))
        current_body = []

    for raw_line in text.splitlines():
        role, remainder = _detect_role(raw_line)
        if role is not None:
            # Starting a new turn
            if current_role is not None or current_body:
                flush()
            current_role = role
            if remainder:
                current_body.append(remainder)
            continue

        # Continuation of current turn (or preamble)
        if current_role is None and not current_body and not raw_line.strip():
            continue
        current_body.append(raw_line.rstrip())

    if current_role is not None or current_body:
        flush()

    if not turns:
        return text + "\n", False

    blocks: list[str] = []
    for role, body_lines in turns:
        body = "\n".join(body_lines).strip()
        body = MULTI_BLANK_RE.sub("\n\n", body)
        if role == "Preamble":
            blocks.append(body)
            continue
        if role == "User":
            header = "### User"
        else:
            header = "### Assistant"
        if body:
            blocks.append(f"{header}\n\n{body}")
        else:
            blocks.append(header)

    formatted = "\n\n---\n\n".join(blocks).strip() + "\n"
    return formatted, True


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_markdown(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def write_docx(text: str, output_path: Path) -> None:
    """
    Convert the (already chat-formatted) Markdown-ish text into a .docx.

    Recognizes `### User` / `### Assistant` headings and horizontal rules.
    """
    document = Document()

    # Basic styles
    styles = document.styles
    try:
        normal = styles["Normal"]
        normal.font.name = "Calibri"
        normal.font.size = Pt(11)
    except KeyError:
        pass

    for style_name, color in (("Heading 3", RGBColor(0x1A, 0x1A, 0x2E)),):
        try:
            style = styles[style_name]
            style.font.color.rgb = color
        except KeyError:
            pass

    paragraphs = text.split("\n")
    buffer: list[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        # Join soft-wrapped lines within a paragraph block
        para_text = "\n".join(buffer).strip()
        buffer = []
        if not para_text:
            return
        # Blockquote lines → italic indented paragraph
        if all(line.startswith(">") for line in para_text.splitlines()):
            cleaned = "\n".join(line.lstrip("> ").rstrip() for line in para_text.splitlines())
            p = document.add_paragraph(cleaned)
            for run in p.runs:
                run.italic = True
            return
        document.add_paragraph(para_text)

    for line in paragraphs:
        stripped = line.strip()
        if stripped in {"---", "***", "___"}:
            flush_buffer()
            document.add_paragraph("─" * 40)
            continue
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_buffer()
            level = min(len(heading_match.group(1)), 3)
            document.add_heading(heading_match.group(2).strip(), level=level)
            continue
        if stripped == "":
            flush_buffer()
            continue
        buffer.append(line.rstrip())

    flush_buffer()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(output_path))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def convert_pdf(
    pdf_path: Path | str,
    output_format: OutputFormat = "markdown",
    output_filename: Optional[str] = None,
    engine: Optional[str] = None,
    force_chat_format: bool = False,
) -> ConversionResult:
    """
    Convert a PDF into Markdown or DOCX under the dated naming convention.

    Parameters
    ----------
    pdf_path:
        Path to a PDF inside (or reachable from) the container.
    output_format:
        ``markdown`` or ``docx``.
    output_filename:
        Optional custom stem. Defaults to the source PDF stem.
    engine:
        ``docling``, ``pdfplumber``, or ``auto`` / None.
    force_chat_format:
        Always run chat role separation even if heuristics are weak.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF inputs are supported (got {pdf_path.suffix})")

    raw_text, engine_used = extract_text(pdf_path, engine=engine)
    formatted, chat_formatted = format_chat_transcript(
        raw_text, force=force_chat_format
    )

    stem = output_filename or pdf_path.stem
    out_path = dated_output_path(output_format, stem)

    if output_format == "markdown":
        write_markdown(formatted, out_path)
    elif output_format == "docx":
        write_docx(formatted, out_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    logger.info(
        "Converted %s → %s via %s (chat_formatted=%s, chars=%d)",
        pdf_path.name,
        out_path,
        engine_used,
        chat_formatted,
        len(formatted),
    )

    return ConversionResult(
        source_path=pdf_path,
        output_path=out_path,
        format=output_format,
        engine=engine_used,
        chat_formatted=chat_formatted,
        characters=len(formatted),
    )
