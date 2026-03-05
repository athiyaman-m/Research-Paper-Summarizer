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

The UI now lets users choose LLM provider directly: `LLaMA`, `Gemini`, or `Groq`.

- `LLaMA` uses `ollama` backend if `OLLAMA_BASE_URL` exists, otherwise `local` GGUF backend.
- `Gemini` uses `GEMINI_API_KEY`.
- `Groq` uses `GROQ_API_KEY`.

Place API keys in **Streamlit Cloud -> App settings -> Secrets** (or `.streamlit/secrets.toml` for local run).

Base flags:

```toml
SUMMARIX_REQUIRE_LLM = "true"
```

#### Option A: Gemini

```toml
GEMINI_API_KEY = "your_gemini_api_key"
GEMINI_MODEL = "gemini-2.0-flash"
```

#### Option B: Groq

```toml
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "llama-3.1-8b-instant"
```

#### Option C: LLaMA with Ollama (free via your local system + tunnel)

```toml
OLLAMA_BASE_URL = "https://your-public-tunnel-url"
OLLAMA_MODEL = "llama3.2:3b-instruct"
OLLAMA_TIMEOUT_SEC = "120"
```

#### Option D: LLaMA local GGUF (run app on same machine)

```toml
SUMMARIX_MODEL_PATH = "models/llama-3.2-1b-instruct.Q4_K_M.gguf"
```

If provider secrets are missing/invalid, the app shows a clear provider-specific initialization error.
