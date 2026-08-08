"""
Sparky OCR — configuration, paths, and environment bootstrap.

All model caches are forced into the mounted models volume so weights never
land in the container root filesystem layer.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Timezone (docker-compose also sets TZ; keep a code-level fallback)
# ---------------------------------------------------------------------------
os.environ.setdefault("TZ", "Europe/Bratislava")

# ---------------------------------------------------------------------------
# Application paths
# ---------------------------------------------------------------------------
# Inside the container APP_ROOT is /app. For local unit tests / host runs,
# fall back to the repository root when /app is not writable.
_REPO_ROOT = Path(__file__).resolve().parent


def _default_app_root() -> Path:
    env = os.getenv("APP_ROOT")
    if env:
        return Path(env).resolve()
    container_root = Path("/app")
    if container_root.is_dir() and os.access(container_root, os.W_OK):
        return container_root.resolve()
    return _REPO_ROOT


APP_ROOT = _default_app_root()
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", str(APP_ROOT / "uploads"))).resolve()
CONVERTED_DIR = Path(os.getenv("CONVERTED_DIR", str(APP_ROOT / "converted"))).resolve()
MODELS_CACHE_DIR = Path(
    os.getenv("MODELS_CACHE_DIR", str(APP_ROOT / "models_cache"))
).resolve()

# Format subdirectories under ./converted/
FORMAT_DIRS = {
    "markdown": CONVERTED_DIR / "markdown",
    "docx": CONVERTED_DIR / "docx",
}

# ---------------------------------------------------------------------------
# Model / cache environment overrides (must point at the mounted volume)
# ---------------------------------------------------------------------------
def apply_model_cache_env() -> None:
    """Force Hugging Face / Torch / XDG caches onto the models volume."""
    cache = str(MODELS_CACHE_DIR)
    os.environ["HF_HOME"] = os.getenv("HF_HOME", cache)
    os.environ["TORCH_HOME"] = os.getenv("TORCH_HOME", cache)
    os.environ["XDG_CACHE_HOME"] = os.getenv("XDG_CACHE_HOME", cache)
    # Hugging Face transformers / hub extras
    os.environ.setdefault("TRANSFORMERS_CACHE", cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(cache) / "hub"))
    os.environ.setdefault("HF_HUB_CACHE", str(Path(cache) / "hub"))


apply_model_cache_env()

# ---------------------------------------------------------------------------
# Conversion / engine settings
# ---------------------------------------------------------------------------
# docling | pdfplumber | auto (try docling, fall back to pdfplumber)
OCR_ENGINE = os.getenv("OCR_ENGINE", "auto").strip().lower()

# CUDA | CPU | AUTO — passed through to Docling AcceleratorOptions when available
ACCELERATOR_DEVICE = os.getenv("ACCELERATOR_DEVICE", "AUTO").strip().upper()

# Max upload size hint for the UI (bytes). Streamlit also has its own limit.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))

# Allowed upload extensions
ALLOWED_EXTENSIONS = {".pdf"}

# Server bind (Streamlit CLI also uses these via compose command)
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8501"))


def ensure_directories() -> None:
    """Create required runtime directories if they do not exist."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for path in FORMAT_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


# Bootstrap on import so any entrypoint has a consistent filesystem layout.
ensure_directories()
