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

If local GGUF model files are not present in cloud, the app automatically runs with fallback summarization mode.
