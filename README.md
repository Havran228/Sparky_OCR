# Sparky OCR

GPU-accelerated OCR and document conversion for the **NVIDIA DGX Spark** (“Sparky”) workstation. Convert Print-to-PDF AI chat exports into clean Markdown or Word documents, reachable over your local Tailscale network on port **8501**.

## Hardware target

| Component | Spec |
|-----------|------|
| Architecture | NVIDIA Grace Blackwell (ARM64 / aarch64) |
| GPU | Blackwell, CUDA, unified memory |
| OS | DGX OS (Ubuntu 24.04 LTS) |
| Network | Tailscale / LAN only — port `8501` |

## Project layout

```
Sparky_OCR/
├── app.py                 # Streamlit UI (upload, convert, file explorer)
├── converter.py           # Docling / pdfplumber engine + chat formatting
├── config.py              # Paths, env, model-cache overrides
├── requirements.txt       # App deps (does NOT install torch*)
├── Dockerfile             # nvcr.io/nvidia/pytorch ARM64/CUDA image
├── docker-compose.yml     # Host mounts, GPU, port 8501, TZ
├── .streamlit/config.toml # Binds 0.0.0.0:8501
├── uploads/               # Incoming PDFs
├── converted/
│   ├── markdown/
│   └── docx/
└── tests/
```

### Host → container mounts (strict isolation)

| Host | Container |
|------|-----------|
| `/home/havran/Projects/Sparky_OCR/uploads` | `/app/uploads` |
| `/home/havran/Projects/Sparky_OCR/converted` | `/app/converted` |
| `/home/havran/models/Sparky_OCR` | `/app/models_cache` |

Model caches are forced via `HF_HOME`, `TORCH_HOME`, and `XDG_CACHE_HOME` → `/app/models_cache`.

## Quick start (on Sparky)

```bash
# 1. Place this repo at:
#    /home/havran/Projects/Sparky_OCR
mkdir -p /home/havran/Projects/Sparky_OCR/{uploads,converted/markdown,converted/docx}
mkdir -p /home/havran/models/Sparky_OCR

# 2. Authenticate to NGC (once) if required for nvcr.io pulls
#    docker login nvcr.io

# 3. Build & run
cd /home/havran/Projects/Sparky_OCR
docker compose up -d --build

# 4. Open on any Tailscale device
#    http://<sparky-tailscale-ip>:8501
```

The Streamlit server binds to `0.0.0.0:8501` inside the container so it is reachable over Tailscale. Host port **8000 is not used**.

## Features

- **Upload** PDFs into `./uploads/`
- **Convert** with Docling (GPU) and automatic **pdfplumber** fallback
- **Chat log formatting** — detects User vs Assistant turns from ChatGPT / Claude / Grok / Gemini style Print-to-PDF exports and rewrites them with `### User` / `### Assistant` separators
- **Output formats** — Markdown (`.md`) or Word (`.docx`)
- **Naming** — `./converted/{FORMAT}/{YYYYMMDD}_{FILENAME}.{ext}` with spaces → `_` and special characters stripped (`TZ=Europe/Bratislava`)
- **File Explorer** — list, preview, download, and delete uploads and converted outputs

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TZ` | `Europe/Bratislava` | Date prefix for output filenames |
| `HF_HOME` / `TORCH_HOME` / `XDG_CACHE_HOME` | `/app/models_cache` | Model weight cache |
| `OCR_ENGINE` | `auto` | `auto` \| `docling` \| `pdfplumber` |
| `ACCELERATOR_DEVICE` | `AUTO` | `AUTO` \| `CUDA` \| `CPU` |

## PyTorch note

The NGC base image (`nvcr.io/nvidia/pytorch:26.07-py3`) already ships an ARM64/CUDA-optimized PyTorch. `requirements.txt` intentionally omits `torch`, `torchvision`, and `torchaudio`. The Dockerfile pins the preinstalled versions as pip constraints so Docling cannot replace them with a PyPI wheel.

## Local unit tests (no GPU required)

```bash
python -m pip install python-docx pdfplumber markdown pydantic
python -m pytest tests/ -q
```

## Security posture

- Bind-mounts limited to uploads, converted outputs, and model cache
- `no-new-privileges`, dropped capabilities
- Application listens only on port 8501
- Delete operations in the UI are confined to the uploads/converted trees
