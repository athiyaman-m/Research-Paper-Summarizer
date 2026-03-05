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

- `requirements.txt` (installs `streamlit`, `PyMuPDF`, and `openai`)
- `runtime.txt` (pins Python `3.12`)

Set app entrypoint to `app.py`.

This deployment is configured to require an active LLM backend (no fallback mode).

### Real-Time LLM (No Fallback)

The app accepts three providers: `openai`, `ollama`, and `local`.

Base secrets (required):

```toml
SUMMARIX_LLM_PROVIDER = "openai"
SUMMARIX_REQUIRE_LLM = "true"
```

Always set `SUMMARIX_LLM_PROVIDER` explicitly and reboot the Streamlit app after changing secrets.

#### Option A: OpenAI (Cloud)

```toml
SUMMARIX_LLM_PROVIDER = "openai"
OPENAI_API_KEY = "your_openai_api_key"
OPENAI_MODEL = "gpt-4o-mini"  # optional
```

#### Option B: Ollama (Free, using your local machine)

1. On your local machine, run Ollama and pull a model:

```bash
ollama serve
ollama pull llama3.2:3b-instruct
```

2. Expose local Ollama (`11434`) with a tunnel (Cloudflare or ngrok).
3. Put the tunnel URL in Streamlit secrets:

```toml
SUMMARIX_LLM_PROVIDER = "ollama"
OLLAMA_BASE_URL = "https://your-public-tunnel-url"
OLLAMA_MODEL = "llama3.2:3b-instruct"
OLLAMA_TIMEOUT_SEC = "120"
SUMMARIX_REQUIRE_LLM = "true"
```

#### Option C: Local GGUF (Run app locally)

```toml
SUMMARIX_LLM_PROVIDER = "local"
SUMMARIX_MODEL_PATH = "models/llama-3.2-1b-instruct.Q4_K_M.gguf"
SUMMARIX_REQUIRE_LLM = "true"
```

If provider settings are invalid, the app stops with a configuration error instead of fallback summarization.
