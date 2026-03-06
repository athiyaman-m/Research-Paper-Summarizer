# Research Paper Summarizer

A professional scientific PDF summarization system with:

- Metadata and section extraction
- Structure-aware section summarization
- Factual consistency auditing
- Figure/table segmentation and media metrics
- Streamlit UI and reproducible experiment notebook

## Main Workflow

PDF Upload -> GROBID (metadata + sections) -> Figure/Table Segmentation -> Text Chunking -> LLM Summarization -> Section Summaries -> Final Summary + UI Output

## Run App

```bash
cd "/home/cdac/Office-Projects/Research-Paper-Summarizer"
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8513
```

Open: <http://127.0.0.1:8513>

## Research Notebook

`research_paper_novelty_experiments.ipynb` includes:

- Structure-aware and factual experiments
- Phase-2 media segmentation metrics
- Publication plots and ablation tables

## Repository Notes

- Local virtual environment and model binaries are excluded via `.gitignore`.
- Figures/tables under `outputs/` are included for publication readiness.


## Streamlit Cloud Deployment

This repo includes deployment files:

- `requirements.txt` (installs `streamlit` and `PyMuPDF`)
- `runtime.txt` (pins Python `3.12`)

Set app entrypoint to `app.py`.

This deployment is configured to require an active LLM backend (no fallback mode).

### Real-Time LLaMA (No Fallback)

This app is now **LLaMA-only**.

- If `OLLAMA_BASE_URL` is set, it uses **Ollama API** (recommended for deployed Streamlit).
- If `OLLAMA_BASE_URL` is empty, it uses **local GGUF** model path.

Set secrets in **Streamlit Cloud -> App settings -> Secrets** (or `.streamlit/secrets.toml` locally):

```toml
SUMMARIX_REQUIRE_LLM = "true"

# optional explicit backend: ollama or local
SUMMARIX_LLM_PROVIDER = "ollama"

OLLAMA_BASE_URL = "https://your-public-tunnel-url"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT_SEC = "120"
OLLAMA_NUM_CTX = "2048"
OLLAMA_MAX_INPUT_CHARS = "12000"
OLLAMA_MAX_RETRIES = "3"

SUMMARIX_MODEL_PATH = "models/llama-3.2-1b-instruct.Q4_K_M.gguf"
```

If Ollama is unreachable or model path is missing, app shows a clear LLaMA initialization error.
