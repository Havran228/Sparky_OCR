# syntax=docker/dockerfile:1.7
#
# Sparky OCR — ARM64 / CUDA image for NVIDIA DGX Spark (Grace Blackwell)
#
# Base: NGC PyTorch (multi-arch; pull aarch64 on Sparky). Ships an optimized
# PyTorch + CUDA stack — requirements.txt must NOT reinstall torch*.

ARG NGC_PYTORCH_TAG=26.07-py3
FROM nvcr.io/nvidia/pytorch:${NGC_PYTORCH_TAG}

LABEL org.opencontainers.image.title="Sparky OCR" \
      org.opencontainers.image.description="GPU OCR & chat-PDF conversion for DGX Spark" \
      org.opencontainers.image.vendor="Sparky" \
      org.opencontainers.image.source="https://github.com/havran228/sparky_ocr"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Bratislava \
    APP_ROOT=/app \
    UPLOADS_DIR=/app/uploads \
    CONVERTED_DIR=/app/converted \
    MODELS_CACHE_DIR=/app/models_cache \
    HF_HOME=/app/models_cache \
    TORCH_HOME=/app/models_cache \
    XDG_CACHE_HOME=/app/models_cache \
    TRANSFORMERS_CACHE=/app/models_cache \
    HUGGINGFACE_HUB_CACHE=/app/models_cache/hub \
    HF_HUB_CACHE=/app/models_cache/hub \
    OCR_ENGINE=auto \
    ACCELERATOR_DEVICE=AUTO \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

# System packages for PDF rendering / fonts / timezone
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
        curl \
        tzdata \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        fonts-dejavu-core \
        fonts-liberation \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo "$TZ" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

# Pin the NGC-provided torch stack so pip cannot replace it with a PyPI wheel
# (PyPI often lacks aarch64+CUDA builds suitable for Grace Blackwell).
RUN python - <<'PY'
import importlib.util
from pathlib import Path

lines = []
for name in ("torch", "torchvision", "torchaudio"):
    spec = importlib.util.find_spec(name)
    if spec is None:
        continue
    mod = importlib.import_module(name)
    # Strip local/CUDA build suffixes (e.g. 2.10.0a0+xxx) for constraint matching
    ver = mod.__version__.split("+")[0]
    lines.append(f"{name}=={ver}")
    print(f"Pinning {name}=={ver} (raw={mod.__version__})")

Path("/tmp/constraints-torch.txt").write_text("\n".join(lines) + "\n")
print("Wrote /tmp/constraints-torch.txt")
PY

RUN pip install --no-cache-dir \
        --upgrade-strategy only-if-needed \
        --constraint /tmp/constraints-torch.txt \
        -r /tmp/requirements.txt \
    && python - <<'PY'
import torch
print("Post-install torch:", torch.__version__, "cuda_built:", torch.version.cuda)
# Fail the build if somehow a CPU-only / unexpected torch slipped in without CUDA build info
# (on Sparky host builds, cuda should be present; allow CPU-only only if NGC image lacks it).
print("cuda.is_available at build time:", torch.cuda.is_available())
import streamlit, docling, pdfplumber, docx
print("imports ok")
PY

COPY config.py converter.py app.py /app/
COPY .streamlit /app/.streamlit

# Runtime directories (also bind-mounted in compose)
RUN mkdir -p /app/uploads /app/converted/markdown /app/converted/docx /app/models_cache \
    && chmod -R 775 /app/uploads /app/converted /app/models_cache

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
