"""Unit tests for filename sanitization and chat transcript formatting.

These tests do not require GPU, Docling, or Streamlit — safe to run anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing project modules without installing a package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from converter import (  # noqa: E402
    format_chat_transcript,
    sanitize_filename,
)


def test_sanitize_replaces_spaces_and_strips_specials() -> None:
    assert sanitize_filename("grok chat conversion!") == "grok_chat_conversion"
    assert sanitize_filename("Sparky Specs (v2).pdf") == "Sparky_Specs_v2"
    assert sanitize_filename("hello world@@@") == "hello_world"
    assert sanitize_filename("") == "untitled"


def test_format_chatgpt_export() -> None:
    raw = """
Skip to content
You said:
What is the CUDA compute capability of GB10?
ChatGPT said:
GB10 targets sm_121 on the Grace Blackwell platform.
You said:
Thanks!
ChatGPT said:
You're welcome.
"""
    md, was_chat = format_chat_transcript(raw)
    assert was_chat is True
    assert "### User" in md
    assert "### Assistant" in md
    assert "sm_121" in md
    assert "Skip to content" not in md
    assert md.count("---") >= 1


def test_format_generic_roles() -> None:
    raw = """
User: Summarize Docling.
Assistant: Docling is a document conversion toolkit with GPU OCR.
User: And Marker?
Assistant: Marker is an alternative PDF-to-markdown engine.
"""
    md, was_chat = format_chat_transcript(raw)
    assert was_chat is True
    assert md.index("### User") < md.index("### Assistant")


def test_non_chat_passthrough() -> None:
    raw = "This is a plain technical PDF extract.\n\nIt has two paragraphs."
    md, was_chat = format_chat_transcript(raw)
    assert was_chat is False
    assert "plain technical PDF" in md
