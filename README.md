# Research Paper Summarizer

A research-grade scientific document summarization system built for rigorous evaluation and deployment.

## Project Outcome

This repository delivers a complete long-form research paper summarizer that converts academic PDFs into structured, evidence-aware summaries with section-level breakdowns, citation-aware media alignment, and fact consistency auditing.

Key outcomes:
- Structured extraction of paper metadata, sections, citations, figures, and tables
- Section-aware and graph-informed summarization to preserve document logic
- Factual consistency checks and summary revision to reduce hallucinations
- Media segmentation evaluation for figures and tables
- Proven experimental workflow and publication-ready metric outputs

## Architecture and Capabilities

The system is organized as a modular pipeline:

1. Document extraction
   - GROBID-style section parsing and metadata extraction
   - PDF text and media extraction with `PyMuPDF`
   - Section graph construction supporting contextual summarization
2. LLM-driven summarization
   - Section-level summarization using LLaMA through either a local GGUF model or Ollama API
   - Final summary composition from priority-ranked sections
   - Domain-specific adaptation for legal, medical, government, and general documents
3. Factual auditing and revision
   - Support scoring between summary sentences and source sentences
   - Contradiction detection based on negation and numeric alignment
   - Audit-driven summary revision to remove unsupported claims
4. Multi-document literature synthesis
   - Cross-paper highlight extraction
   - Combined trends, common findings and differences
5. Media metrics
   - Figure/table assignment coverage and alignment
   - Caption and preview quality assessment

## Performance and Metrics

The evaluation framework produces quantitative metrics for comparison between a baseline summarization pipeline and the proposed structure-aware approach.

Representative results from a single long-document experiment on an arXiv paper sample:

- ROUGE-1 F1: 0.1277 → 0.1346
- ROUGE-2 F1: 0.0483 → 0.0957
- ROUGE-L F1: 0.0747 → 0.0832
- Semantic proxy score: 0.4532 → 0.5650
- Factual consistency score: 0.3235 → 0.5022
- Section coverage: 0.60 → 0.80
- Structure coherence signal: 0.00 → 0.1663

These metrics demonstrate improved summary relevance, structure preservation, and evidence alignment when using section-aware selection and graph-context summarization.

## Research Contributions

This project includes research-ready components for evaluating summarization quality and media-aware document understanding:

- `research_paper_novelty_experiments.ipynb` for reproducible experimentation
- `research_experiment_framework.py` with evaluation, auditing, and multi-document summarization logic
- `run_research_experiments.py` for end-to-end experiment execution and metric output generation
- `outputs/tables/` containing publication-ready CSV and LaTeX tables for metrics and ablation analysis

## Technology Stack

- Python 3.12
- Streamlit for user-facing dashboard and interactive document exploration
- `PyMuPDF` for PDF parsing and figure cropping
- LLaMA model integration via Ollama API or local GGUF runtime
- Custom summarization and evaluation pipeline in Python

## Dataset and Sample Inputs

This repository is built around academic PDF summarization for long documents. Included sample content and experiment inputs include:

- `data/2004.05150v2.pdf` as a representative arXiv research paper
- `research_paper_novelty_experiments.ipynb` for evaluation workflows
- `research_experiment_results.json` recording experiment outputs and metric comparisons

## Output Artifacts

Primary deliverables in this repository:

- `Structured_Summary.txt` and `structured_summary_output.txt`
- `research_experiment_results.json` with baseline and structure-aware metrics
- `outputs/tables/` for publication-ready results and ablation tables
- `app.py` Streamlit interface for interactive paper summarization

## Why this matters

This project demonstrates an end-to-end, deployment-ready pipeline that bridges academic PDF parsing with modern LLM summarization while emphasizing structure, factual rigor, and research evaluation. It is intended for technical reviewers and recruiters who want to see a concrete engineering and research outcome rather than only run instructions.
