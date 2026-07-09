# APPENDIX A

# IMPORTANT SOURCE CODE MODULES

---

## A.1 Project Overview

The *Smart Research Paper Summarizer* is a multi-document research paper analysis and summarization system built using Python and Streamlit. It leverages the Groq API (with Llama 3.3 70B model) for LLM-powered summarization, comparative analysis, and survey synthesis. The system processes academic PDF documents to extract metadata, structural sections, citations, figures, and tables, providing an interactive web-based interface for researchers.

**Project Statistics:**

| Attribute | Value |
|---|---|
| Total files (source code) | 6 Python files |
| Main entry file | `app.py` |
| Core pipeline module | `pipeline.py` |
| Experiment framework | `research_experiment_framework.py` |
| External services | Groq API (primary), Ollama (secondary), Local GGUF (fallback) |
| Frontend | Streamlit |

**Architecture Overview:**

The system follows a modular architecture with three primary layers:

1. **Presentation Layer** (`app.py`): Streamlit-based web UI with tabs, sidebar, and rendering components.
2. **Core Processing Layer** (`pipeline.py`): Contains `DocumentExtractor` for PDF parsing and `LLMService` for AI-powered text generation.
3. **Evaluation Layer** (`research_experiment_framework.py`): Provides ROUGE evaluation, factual consistency checking, and multi-document comparative experiments.

Data flows unidirectionally: PDF upload $\rightarrow$ DocumentExtractor $\rightarrow$ structured document representation $\rightarrow$ LLMService $\rightarrow$ summarized output $\rightarrow$ Streamlit UI rendering.

---

## A.2 Application Entry Point

The application begins execution in `app.py`, which initializes the Streamlit page configuration, resolves runtime LLM configuration (Groq/Ollama/Local), and launches the tabbed user interface. The main navigation is driven by Streamlit's `st.tabs()` widget, presenting five functional tabs: Paper Analysis, Comparative Analysis, Survey Synthesis, Citations, and Figures & Tables.

**Code: Application Entry and Configuration**

```python
# app.py (lines 1-57)
import base64
import logging
import os
import tempfile
from html import escape

import streamlit as st

from pipeline import DocumentExtractor, LLMService, crop_figure

st.set_page_config(page_title="Research Paper Summarizer", layout="wide")

DEFAULT_LLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
APP_VERSION = "2.0.2"
services = {}


def resolve_runtime_config(model_name: str) -> dict:
    model = model_name.strip() or DEFAULT_LLAMA_MODEL
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        runtime_provider = "groq"
    elif os.getenv("OLLAMA_BASE_URL", "").strip():
        runtime_provider = "ollama"
    else:
        runtime_provider = "groq"
    return {"provider": runtime_provider, "model": model}


def llm_config_signature(config: dict) -> tuple:
    return (
        config.get("provider", ""),
        config.get("model", ""),
        os.getenv("GROQ_API_KEY", "").strip(),
        os.getenv("OLLAMA_BASE_URL", "").strip(),
        os.getenv("SUMMARIX_MODEL_PATH", ""),
        os.getenv("SUMMARIX_REQUIRE_LLM", "").strip().lower(),
        APP_VERSION,
    )


@st.cache_resource
def get_services(config_signature: tuple):
    provider = config_signature[0]
    llm_kwargs = {"provider": provider, "require_llm": False}
    if provider == "ollama":
        llm_kwargs["ollama_model"] = config_signature[1]
    return {
        "extractor": DocumentExtractor(),
        "llm": LLMService(**llm_kwargs),
    }
```

**Code: Main Application Loop and Tab Navigation**

```python
# app.py (lines 472-609)
def main():
    global services

    with st.sidebar:
        st.markdown("### Research Paper Summarizer")
        st.markdown("Upload PDFs to analyze, compare, and summarize.")

        is_dark = True
        render_styles(is_dark)
        st.markdown("---")

        model_options = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
        ]
        model_name = st.selectbox(
            "LLM Model",
            options=model_options,
            index=0,
            help="Select the Groq-hosted model for summarization.",
        )

        runtime_config = resolve_runtime_config(model_name)
        runtime_provider = runtime_config["provider"]
        uploaded_files = st.file_uploader(
            "Upload arXiv Papers (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
        )

        st.markdown("---")
        st.markdown(f"**Runtime Backend**: `{runtime_provider}`")

    os.environ["GROQ_MODEL"] = runtime_config.get("model", "llama-3.3-70b-versatile")
    config_signature = llm_config_signature(runtime_config)
    try:
        services = get_services(config_signature)
    except Exception as exc:
        st.error(llm_init_help(runtime_provider))
        st.exception(exc)
        return

    llm = services.get("llm")
    if llm and getattr(llm, "mode", "") == "fallback":
        st.warning(llm_init_help(runtime_provider), icon="warning")
        return

    if not uploaded_files:
        st.info("Upload one or more PDFs from the sidebar to begin.")
        return

    current_names = sorted(f.name for f in uploaded_files)
    prev_names = st.session_state.get("uploaded_names", [])
    files_changed = prev_names != current_names
    if files_changed:
        st.session_state["uploaded_names"] = current_names
        old_papers = st.session_state.get("papers", {})
        st.session_state["papers"] = {k: v for k, v in old_papers.items() if k in current_names}

    with st.spinner(f"Parsing {len(uploaded_files)} paper(s)..."):
        papers = parse_uploaded_files(uploaded_files)

    tab_labels = ["Paper Analysis", "Comparative", "Survey", "Citations", "Figures & Tables"]
    t1, t2, t3, t4, t5 = st.tabs(tab_labels)

    with t1:
        tab_paper_analysis(papers)
    with t2:
        tab_comparative(papers)
    with t3:
        tab_survey(papers)
    with t4:
        tab_citations(papers)
    with t5:
        tab_figures_tables(papers)


if __name__ == "__main__":
    main()
```

The application starts by configuring the Streamlit page, then resolves which LLM backend to use (Groq API by default). The sidebar provides model selection and PDF upload. Uploaded files are parsed via `DocumentExtractor.parse_document()` and stored in session state. The main content area presents five tabs for different analytical views, each backed by dedicated rendering functions.

---

## A.3 PDF Processing Module

The PDF processing module is implemented within the `DocumentExtractor` class in `pipeline.py`. It uses PyMuPDF (`fitz`) to open PDF files, extract line-level text with positional and font metadata, cleanly separate content from running headers and footers, and return a structured dictionary containing all extracted information. The `crop_figure` standalone function handles extraction of embedded images from PDF coordinates.

**Code: PDF Text Extraction and Line Item Processing**

```python
# pipeline.py (lines 53-132)
import pymupdf as fitz


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class DocumentExtractor:
    def __init__(self, chunk_size: int = 420, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_document(self, pdf_path: str, include_media: bool = True) -> Dict:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(pdf_path)

        lines = self._extract_line_items(pdf_path)
        metadata = self._extract_metadata(pdf_path, lines)
        sections = self._extract_sections(lines, metadata)
        figures: List[Dict] = []
        tables: List[Dict] = []
        if include_media:
            figures = self._extract_figures(pdf_path)
            tables = self._extract_tables(pdf_path)
        self._attach_media_to_sections(sections, figures, tables)
        citations = self.extract_citations(sections)

        return {
            "metadata": metadata,
            "sections": sections,
            "figures": figures,
            "tables": tables,
            "citations": citations,
            "pdf_path": pdf_path,
        }

    def _extract_line_items(self, pdf_path: str) -> List[Dict]:
        items: List[Dict] = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                page_number = page.number + 1
                page_height = page.rect.height
                blocks = page.get_text("dict").get("blocks", [])
                for block in blocks:
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        if not spans:
                            continue
                        text = "".join(span.get("text", "") for span in spans).strip()
                        if not text:
                            continue

                        y0 = min(span.get("bbox", [0, 0, 0, 0])[1] for span in spans)
                        y1 = max(span.get("bbox", [0, 0, 0, 0])[3] for span in spans)

                        if y0 < 24 or y1 > (page_height - 24):
                            continue

                        item = {
                            "page": page_number,
                            "text": text,
                            "size": max(span.get("size", 0) for span in spans),
                            "font": " ".join(span.get("font", "") for span in spans).lower(),
                            "y0": y0,
                            "y1": y1,
                            "page_height": page_height,
                        }
                        items.append(item)
        return self._remove_repeating_headers_footers(items)

    @staticmethod
    def _remove_repeating_headers_footers(items: List[Dict]) -> List[Dict]:
        if not items:
            return items
        text_to_pages: Dict[str, set] = {}
        text_to_y: Dict[str, List[float]] = {}
        for item in items:
            text = _normalize_text(item["text"])
            if len(text) > 40 or len(text.split()) > 6:
                continue
            text_to_pages.setdefault(text, set()).add(item["page"])
            text_to_y.setdefault(text, []).append(item["y0"])

        removable = set()
        for text, pages in text_to_pages.items():
            if len(pages) < 3:
                continue
            y_med = median(text_to_y[text])
            if y_med < 95:
                removable.add(text)

        filtered = []
        for item in items:
            text = _normalize_text(item["text"])
            if text in removable:
                continue
            filtered.append(item)
        return filtered
```

**Code: Figure Cropping Utility**

```python
# pipeline.py (lines 1104-1117)
def crop_figure(pdf_path: str, coords: Dict[str, float], output_path: str):
    with fitz.open(pdf_path) as doc:
        page_idx = int(coords["page"]) - 1
        if page_idx < 0 or page_idx >= len(doc):
            return
        page = doc[page_idx]
        rect = fitz.Rect(
            coords["x"],
            coords["y"],
            coords["x"] + coords["w"],
            coords["y"] + coords["h"],
        )
        pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
        pix.save(output_path)
```

The `parse_document` method orchestrates the entire extraction pipeline. It first extracts line items with positional metadata, removes repeating headers and footers (text appearing on 3+ pages at fixed Y-coordinates), then delegates to metadata extraction, section detection, figure/table extraction, and citation parsing.

---

## A.4 Metadata Extraction Module

Metadata extraction is handled by the `_extract_metadata` method and its helper functions in `DocumentExtractor`. The system first attempts to read metadata from the PDF's internal metadata fields (title, author, creation date). When these are missing or contain placeholder values like "Unknown" or "LaTeX", it falls back to heuristic inference from the document's first-page text content.

**Code: Metadata Extraction**

```python
# pipeline.py (lines 164-289)
def _extract_metadata(self, pdf_path: str, lines: List[Dict]) -> Dict:
    with fitz.open(pdf_path) as doc:
        raw_meta = doc.metadata or {}

    title = self._safe_meta_value(raw_meta.get("title"))
    authors = self._safe_meta_value(raw_meta.get("author"))
    year = self._extract_year_from_meta(raw_meta)

    if not title:
        title = self._infer_title_from_text(lines)
    if not authors:
        authors = self._infer_authors_from_text(lines, title)
    if not year:
        year = self._infer_year_from_text(lines)

    return {
        "title": title or os.path.splitext(os.path.basename(pdf_path))[0],
        "authors": authors or "Unknown Authors",
        "year": year,
    }

@staticmethod
def _safe_meta_value(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = _normalize_text(value)
    invalid_tokens = {
        "unknown", "untitled", "latex", "hyperref",
        "microsoft word", "adobe",
    }
    low = cleaned.lower()
    if any(token in low for token in invalid_tokens):
        return None
    if len(cleaned) <= 2:
        return None
    return cleaned

def _infer_title_from_text(self, lines: List[Dict]) -> Optional[str]:
    first_page = [line for line in lines if line["page"] == 1]
    if not first_page:
        return None
    top_lines = [line for line in first_page if line["y0"] < 240]
    if not top_lines:
        return None
    max_size = max(line["size"] for line in top_lines)
    title_lines = [
        line for line in top_lines
        if line["size"] >= (max_size - 0.6)
        and len(line["text"].split()) >= 3
        and not self._looks_like_heading(line["text"])
        and "@" not in line["text"]
    ]
    if not title_lines:
        candidates = sorted(top_lines, key=lambda x: (-x["size"], x["y0"]))
        return _normalize_text(candidates[0]["text"]) if candidates else None
    title_lines = sorted(title_lines, key=lambda x: x["y0"])
    title = " ".join(line["text"] for line in title_lines[:2])
    return _normalize_text(title)

@staticmethod
def _infer_year_from_text(lines: List[Dict]) -> Optional[str]:
    years = re.findall(r"\b((?:19|20)\d{2})\b", " ".join(line["text"] for line in lines[:240]))
    if not years:
        return None
    return sorted(years)[-1]
```

Title inference uses font size heuristics: the largest text on the first page above the abstract is likely the title. Author names are identified by position (between title and abstract), excluding affiliations, URLs, and email addresses. Year is extracted by scanning for 4-digit numbers in the range 1900-2099.

---

## A.5 Section Detection Module

Section detection employs a two-pronged strategy: (1) a set of common academic section titles (`COMMON_SECTION_TITLES`) and (2) a regular expression matching numbered headings (e.g., "1.", "1.1", "I.", "A."). The system filters out false positives using font size analysis, bold detection, and title-case heuristics. It also anchors section detection to avoid capturing front-matter noise before the abstract or introduction.

**Code: Section Detection and Heading Classification**

```python
# pipeline.py (lines 25-50, 291-402)
SECTION_HEADING_REGEX = re.compile(
    r"^((\d+(\.\d+)*)|([IVXLCM]+))[\).\s-]+[A-Z][A-Za-z0-9\-\s,():/]+$"
)

COMMON_SECTION_TITLES = {
    "abstract", "introduction", "background", "related work",
    "method", "methods", "methodology", "approach", "model",
    "experiments", "experimental setup", "results", "discussion",
    "conclusion", "conclusions", "future work", "limitations",
    "appendix", "references", "acknowledgments",
}


def _extract_sections(self, lines: List[Dict], metadata: Dict) -> Dict[str, Dict]:
    if not lines:
        return {}

    body_sizes = [line["size"] for line in lines if len(line["text"].split()) >= 4]
    body_size = median(body_sizes) if body_sizes else 10.5

    heading_positions = []
    for idx, line in enumerate(lines):
        if self._is_section_heading(line, body_size, metadata):
            heading_positions.append(idx)

    anchor_idx = self._find_heading_anchor(heading_positions, lines)
    if anchor_idx is not None:
        heading_positions = [idx for idx in heading_positions if idx >= anchor_idx]

    if not heading_positions:
        joined = "\n".join(line["text"] for line in lines)
        return {
            "Document Content": self._build_section(
                "Document Content", joined, {line["page"] for line in lines}
            )
        }

    sections: Dict[str, Dict] = {}
    for pos, start_idx in enumerate(heading_positions):
        end_idx = heading_positions[pos + 1] \
            if pos + 1 < len(heading_positions) else len(lines)
        heading_line = lines[start_idx]
        title = _normalize_text(heading_line["text"].rstrip(":"))
        body_lines = lines[start_idx + 1: end_idx]
        content_lines = [line["text"] for line in body_lines]
        pages = {heading_line["page"]}
        pages.update(line["page"] for line in body_lines)

        key = self._deduplicate_title(sections, title)
        sections[key] = self._build_section(key, "\n".join(content_lines), pages)

    return sections


def _is_section_heading(self, line: Dict, body_size: float, metadata: Dict) -> bool:
    text = _normalize_text(line["text"])
    if len(text) < 2 or len(text) > 120:
        return False
    if text.lower() == metadata.get("title", "").lower():
        return False
    if text.lower().startswith(("figure ", "fig.", "table ", "algorithm ")):
        return False
    if text.lower().startswith("arxiv:"):
        return False
    if "@" in text:
        return False

    lower = text.lower()
    words = text.split()

    if line["page"] == 1 and line["y0"] < 360:
        if lower not in {"abstract", "introduction"} \
                and not SECTION_HEADING_REGEX.match(text):
            return False

    if lower in COMMON_SECTION_TITLES and line["size"] >= body_size:
        return True
    if SECTION_HEADING_REGEX.match(text) and len(words) <= 14:
        return True

    looks_bold = "bold" in line["font"] or "semibold" in line["font"]
    if len(words) <= 12 and text[-1] not in ".,;":
        if line["size"] >= body_size + 1.0:
            return True
        if looks_bold and line["size"] >= body_size + 0.3 \
                and self._looks_like_title_case(text):
            return True

    return False
```

The section detection algorithm computes the median body text font size, then scans all lines for heading candidates. A line is classified as a heading if it matches a known section title, matches the numbering regex, or is sufficiently larger/bolder than body text. The `_find_heading_anchor` method locates the "Abstract" or "1." heading to establish the true start of the document body.

---

## A.6 Long Document Chunking Module

Documents are chunked into overlapping segments to manage LLM context window limitations. The `chunk_text` function splits text into word-based segments with configurable overlap. Each section is stored with its pre-computed chunks during the `_build_section` call.

**Code: Text Chunking**

```python
# pipeline.py (lines 57-67)
def chunk_text(text: str, max_words: int = 450, overlap: int = 80) -> List[str]:
    words = text.split()
    if not words:
        return []
    stride = max(max_words - overlap, 1)
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start: start + max_words]))
        start += stride
    return chunks
```

**Code: Section Builder with Chunking**

```python
# pipeline.py (lines 411-423)
def _build_section(self, title: str, content: str, pages: set) -> Dict:
    normalized_content = content.strip()
    chunks = chunk_text(normalized_content, self.chunk_size, self.chunk_overlap)
    return {
        "title": title,
        "content": normalized_content,
        "pages": sorted(pages),
        "chunks": chunks,
        "summary": "",
        "chunk_count": len(chunks),
        "figures": [],
        "tables": [],
    }
```

The chunking strategy uses a sliding window of 420 words (by default) with 80 words of overlap between consecutive chunks. This ensures smooth transitions across chunk boundaries when each chunk is independently summarized, preserving continuity of ideas.

---

## A.7 Groq API Integration

The Groq API integration is encapsulated in the `LLMService` class. It supports primary (Groq), secondary (Ollama), and fallback (local GGUF) providers. The Groq provider initializes a client using `groq.Groq(api_key=...)`, validates the key by listing available models, then handles summarization and generation requests with retry logic and context-length adaptation.

**Code: Groq Client Initialization**

```python
# pipeline.py (lines 556-650)
class LLMService:
    def __init__(
        self,
        model_path: Optional[str] = None,
        provider: Optional[str] = None,
        require_llm: Optional[bool] = None,
        ollama_model: Optional[str] = None,
    ):
        self.llm = None
        self.model_path = model_path or os.getenv(
            "SUMMARIX_MODEL_PATH", "models/llama-3.2-1b-instruct.Q4_K_M.gguf"
        )
        self.provider = (provider or os.getenv("SUMMARIX_LLM_PROVIDER", "")).strip().lower()
        self.require_llm = self._env_flag("SUMMARIX_REQUIRE_LLM", default=False) \
            if require_llm is None else bool(require_llm)
        self.mode = "fallback"

        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.groq_max_input_chars = self._env_int("GROQ_MAX_INPUT_CHARS", default=12000, minimum=2000)
        self.groq_max_retries = self._env_int("GROQ_MAX_RETRIES", default=3, minimum=1)
        self._groq_client = None

        if not self.provider:
            if self.groq_api_key:
                self.provider = "groq"
            elif self.ollama_base_url:
                self.provider = "ollama"
            else:
                self.provider = "local"

        init_error: Optional[Exception] = None
        try:
            if self.provider == "groq":
                self._init_groq()
            elif self.provider == "ollama":
                self._init_ollama()
            elif self.provider == "local":
                self._init_local()
        except Exception as exc:
            init_error = exc

        if init_error is not None:
            if self.require_llm:
                raise RuntimeError(
                    "LLM initialization failed. Set GROQ_API_KEY for Groq, "
                    "or OLLAMA_BASE_URL + OLLAMA_MODEL for Ollama, "
                    "or provide a valid local GGUF model via SUMMARIX_MODEL_PATH."
                ) from init_error
            logging.warning("LLM unavailable, fallback summarizer enabled: %s", init_error)
            self.mode = "fallback"

    def _init_groq(self):
        if not self.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing for provider='groq'.")
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=self.groq_api_key)
            self._groq_client.models.list()
        except ImportError as exc:
            raise RuntimeError(
                "groq package is not installed. Add 'groq>=0.11,<1' to requirements.txt."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Groq API initialization failed: {exc}") from exc
        self.mode = "groq"
```

**Code: Groq Summarization Request**

```python
# pipeline.py (lines 734-790)
def _summarize_groq(self, text: str, context: str = "") -> str:
    if self._groq_client is None:
        raise RuntimeError("Groq client is not initialized.")

    max_chars_seq = [
        self.groq_max_input_chars,
        int(self.groq_max_input_chars * 0.75),
        int(self.groq_max_input_chars * 0.5),
    ]
    last_error: Optional[Exception] = None
    for max_chars in max_chars_seq:
        clipped_text = self._truncate_for_context(text, max_chars)
        messages = [
            {
                "role": "system",
                "content": "You are an expert research assistant. "
                           "Write concise, accurate summaries of research paper sections.",
            },
            {
                "role": "user",
                "content": (
                    f"Context: {context}\n"
                    f"Section text:\n{clipped_text}\n\n"
                    "Write a concise summary of the above section:"
                ),
            },
        ]
        for attempt in range(1, self.groq_max_retries + 1):
            try:
                response = self._groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.2,
                )
                generated = response.choices[0].message.content.strip()
                if generated:
                    return generated
            except Exception as exc:
                last_error = exc
                msg = str(exc).lower()
                if "context" in msg and ("length" in msg or "window" in msg or "token" in msg):
                    break
                if "rate" in msg and attempt < self.groq_max_retries:
                    time.sleep(min(2.0 * attempt, 8.0))
                    continue
                if self.require_llm:
                    raise RuntimeError(f"Groq summarization failed: {exc}") from exc
                logging.warning("Groq summarization failed, using fallback: %s", exc)
                return self._fallback_summary(text)

    if self.require_llm and last_error:
        raise RuntimeError(f"Groq summarization failed after retries: {last_error}")
    logging.warning("Groq summarization failed after retries, using fallback summary")
    return self._fallback_summary(text)
```

**Code: Generic LLM Generation Dispatch**

```python
# pipeline.py (lines 1051-1085)
def _llm_generate(self, prompt: str, max_tokens: int = 600) -> str:
    if self.mode == "groq":
        return self._groq_generate(prompt, max_tokens)
    if self.mode == "ollama":
        return self._ollama_generate_text(prompt, max_tokens)
    if self.mode == "local-llama":
        return self._local_generate(prompt, max_tokens)
    return self._fallback_summary(prompt)


def _groq_generate(self, prompt: str, max_tokens: int = 600) -> str:
    if self._groq_client is None:
        raise RuntimeError("Groq client is not initialized.")
    messages = [
        {"role": "system", "content": "You are an expert research analyst."},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(1, self.groq_max_retries + 1):
        try:
            response = self._groq_client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            generated = response.choices[0].message.content.strip()
            if generated:
                return generated
        except Exception as exc:
            msg = str(exc).lower()
            if "rate" in msg and attempt < self.groq_max_retries:
                time.sleep(min(2.0 * attempt, 8.0))
                continue
            raise RuntimeError(f"Groq generation failed: {exc}") from exc
    return "Generation failed after retries."
```

The Groq integration employs a progressive input truncation strategy: if a request fails due to context length limits, the input is reduced to 75% and then 50% of the original size. Rate-limited requests are retried with exponential backoff (up to 8 seconds). The system defaults to the Llama 3.3 70B versatile model, which provides high-quality academic text summarization.

---

## A.8 Research Paper Summarization Module

Section-level and full-paper summarization is handled by `LLMService` methods. The `summarize` method routes to the appropriate backend, while the prompt templates are designed to extract concise, accurate summaries of research content. The UI layer in `app.py` provides caching of section summaries and a "Summarize All" bulk operation.

**Code: Core Summarization Dispatch**

```python
# pipeline.py (lines 713-732)
def summarize(self, text: str, context: str = "") -> str:
    if not text.strip():
        return "No content found for summarization."

    if self.mode == "groq":
        return self._summarize_groq(text, context)
    if self.mode == "ollama":
        return self._summarize_ollama(text, context)
    if self.mode == "local-llama":
        return self._summarize_local(text, context)
    if self.require_llm:
        raise RuntimeError(
            "LLM is required but unavailable. Set GROQ_API_KEY, "
            "OLLAMA_BASE_URL + OLLAMA_MODEL, "
            "or local GGUF via SUMMARIX_MODEL_PATH."
        )
    return self._fallback_summary(text)
```

**Code: Section Summary with Caching (app.py)**

```python
# app.py (lines 159-181)
def section_summary(title: str, section: dict, metadata: dict) -> str:
    cache = st.session_state.setdefault("section_summaries", {})
    doc_key = f"{metadata.get('title', '')}::{title}"
    if doc_key in cache:
        return cache[doc_key]

    text = " ".join(section.get("chunks", [])[:3]) or section.get("content", "")
    context = f"{metadata.get('title', 'Paper')} -> {title}"

    try:
        summary = services["llm"].summarize(text, context)
    except Exception as exc:
        msg = str(exc).strip() or "Unknown LLM failure."
        summary = f"LLM error: {msg}"

    cache[doc_key] = summary
    return summary


def summarize_all_sections(sections: dict, metadata: dict):
    for title, section in sections.items():
        section_summary(title, section, metadata)
```

The fallback summary mechanism (used when no LLM backend is available) applies extractive summarization by selecting the first 4 longest sentences from the input:

```python
# pipeline.py (lines 966-973)
@staticmethod
def _fallback_summary(text: str, sentences: int = 4) -> str:
    clean = _normalize_text(text)
    sentence_candidates = re.split(r"(?<=[.!?])\s+", clean)
    picked = [s for s in sentence_candidates if len(s) > 20][:sentences]
    if picked:
        return " ".join(picked)
    return clean[:700]
```

---

## A.9 Comparative Analysis Module

The comparative analysis module generates structured comparisons across multiple papers. The `compare_papers` method in `LLMService` extracts abstract and conclusion from each paper, then constructs a prompt requesting analysis across six dimensions: research objectives, methodologies, key findings, datasets, strengths/limitations, and agreements/contradictions.

**Code: Comparative Analysis Generation**

```python
# pipeline.py (lines 975-1014)
def compare_papers(self, papers: List[Dict]) -> str:
    if not papers or len(papers) < 2:
        return "At least two papers are required for comparative analysis."

    paper_briefs = []
    for i, paper in enumerate(papers, 1):
        meta = paper.get("metadata", {})
        sections = paper.get("sections", {})
        abstract = ""
        conclusion = ""
        for title, sec in sections.items():
            low = title.lower().strip()
            if low == "abstract" and not abstract:
                abstract = self._truncate_for_context(sec.get("content", ""), 800)
            if low in {"conclusion", "conclusions"} and not conclusion:
                conclusion = self._truncate_for_context(sec.get("content", ""), 600)
        brief = (
            f"Paper {i}: {meta.get('title', 'Untitled')}\n"
            f"Authors: {meta.get('authors', 'Unknown')}\n"
            f"Year: {meta.get('year', 'N/A')}\n"
            f"Abstract: {abstract or 'Not available'}\n"
            f"Conclusion: {conclusion or 'Not available'}\n"
        )
        paper_briefs.append(brief)

    combined = "\n---\n".join(paper_briefs)
    prompt_text = (
        f"Below are summaries of {len(papers)} research papers:\n\n"
        f"{combined}\n\n"
        "Provide a structured comparative analysis covering:\n"
        "1. **Research Objectives**\n"
        "2. **Methodologies**\n"
        "3. **Key Findings**\n"
        "4. **Datasets & Evaluation**\n"
        "5. **Strengths & Limitations**\n"
        "6. **Agreements & Contradictions**\n\n"
        "Format with clear headings and bullet points."
    )
    return self._llm_generate(prompt_text, max_tokens=1200)
```

**Code: Comparative Tab UI (app.py)**

```python
# app.py (lines 319-355)
def tab_comparative(papers: dict):
    names = list(papers.keys())
    if len(names) < 2:
        st.info("Upload at least **2 papers** to enable comparative analysis.")
        return

    st.subheader("Paper Overview")
    header = "<tr><th>#</th><th>Title</th><th>Authors</th><th>Year</th><th>Sections</th><th>Citations</th></tr>"
    rows = ""
    for i, (name, data) in enumerate(papers.items(), 1):
        m = data.get("metadata", {})
        rows += (
            f"<tr><td>{i}</td>"
            f"<td>{escape(m.get('title', 'Untitled'))}</td>"
            f"<td>{escape(m.get('authors', 'Unknown'))}</td>"
            f"<td>{escape(str(m.get('year', 'N/A')))}</td>"
            f"<td>{len(data.get('sections', {}))}</td>"
            f"<td>{len(data.get('citations', []))}</td></tr>"
        )
    st.markdown(f'<table class="cmp-table">{header}{rows}</table>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("AI Comparative Analysis")
    cache_key = "comparative_result"
    if st.button("Generate Comparative Analysis", key="cmp_btn", use_container_width=True):
        with st.spinner("Analyzing papers..."):
            try:
                result = services["llm"].compare_papers(list(papers.values()))
                st.session_state[cache_key] = result
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")

    result = st.session_state.get(cache_key, "")
    if result:
        st.markdown(result)
```

The comparative analysis first renders a metadata comparison table (title, authors, year, section count, citation count), then provides a button to generate AI-powered cross-paper analysis. The LLM prompt is designed to produce structured markdown output covering six analytical dimensions.

---

## A.10 Survey Synthesis Module

The survey synthesis module (`synthesize_survey`) generates a unified thematic survey across multiple papers. It extracts abstracts, constructs a prompt that asks the LLM to identify overarching themes, sub-themes, idea evolution, and future directions.

**Code: Survey Synthesis Generation**

```python
# pipeline.py (lines 1016-1049)
def synthesize_survey(self, papers: List[Dict]) -> str:
    if not papers:
        return "No papers provided for survey synthesis."

    paper_briefs = []
    for i, paper in enumerate(papers, 1):
        meta = paper.get("metadata", {})
        sections = paper.get("sections", {})
        abstract = ""
        for title, sec in sections.items():
            if title.lower().strip() == "abstract":
                abstract = self._truncate_for_context(sec.get("content", ""), 800)
                break
        brief = (
            f"Paper {i}: {meta.get('title', 'Untitled')} "
            f"({meta.get('authors', 'Unknown')}, {meta.get('year', 'N/A')})\n"
            f"Abstract: {abstract or 'Not available'}"
        )
        paper_briefs.append(brief)

    combined = "\n---\n".join(paper_briefs)
    prompt_text = (
        f"Below are abstracts from {len(papers)} research papers on related topics:\n\n"
        f"{combined}\n\n"
        "Write a unified survey-style summary that:\n"
        "1. Identifies the **overarching research theme**\n"
        "2. Groups papers by **sub-themes or approaches**\n"
        "3. Traces the **evolution of ideas** across papers\n"
        "4. Highlights **open problems and future directions**\n"
        "5. Provides **a concluding synthesis** of the field's state\n\n"
        "Write in a formal academic tone with clear paragraphs."
    )
    return self._llm_generate(prompt_text, max_tokens=1500)
```

**Code: Survey Tab UI (app.py)**

```python
# app.py (lines 360-380)
def tab_survey(papers: dict):
    names = list(papers.keys())
    if not names:
        st.info("Upload papers first.")
        return

    st.subheader("AI Survey Synthesis")
    st.caption(f"Synthesizing a unified survey across {len(names)} paper(s).")

    cache_key = "survey_result"
    if st.button("Generate Survey Summary", key="survey_btn", use_container_width=True):
        with st.spinner("Generating thematic survey synthesis..."):
            try:
                result = services["llm"].synthesize_survey(list(papers.values()))
                st.session_state[cache_key] = result
            except Exception as exc:
                st.error(f"Survey synthesis failed: {exc}")

    result = st.session_state.get(cache_key, "")
    if result:
        st.markdown(result)
```

---

## A.11 Citation Analysis Module

Citations are extracted from the References/Bibliography section using two parsing strategies. Strategy 1 handles numbered references (e.g., `[1]`, `1.`), while Strategy 2 splits on double-newlines or author-name patterns when numbering is inconsistent.

**Code: Citation Extraction**

```python
# pipeline.py (lines 503-553)
@staticmethod
def extract_citations(sections: Dict[str, Dict]) -> List[Dict]:
    ref_section = None
    for title, sec in sections.items():
        if title.lower().strip() in {"references", "bibliography", "works cited"}:
            ref_section = sec
            break
    if ref_section is None:
        return []

    raw = ref_section.get("content", "")
    if not raw.strip():
        return []

    citations: List[Dict] = []

    # Strategy 1: numbered references [1], [2], ... or 1. 2. ...
    numbered = re.split(r"\n?\[?(\d{1,3})\]?\.?\s+", raw)
    if len(numbered) >= 5:
        idx = 1
        while idx + 1 < len(numbered):
            num = numbered[idx].strip()
            text = _normalize_text(numbered[idx + 1])
            if len(text) > 15:
                citations.append({
                    "id": f"ref_{num}",
                    "number": int(num),
                    "text": text,
                    "pages": ref_section.get("pages", []),
                })
            idx += 2
        if citations:
            return citations

    # Strategy 2: split on double-newlines
    blocks = re.split(r"\n{2,}", raw)
    if len(blocks) < 3:
        blocks = re.split(r"\n(?=[A-Z][a-z]+,?\s+[A-Z])", raw)

    for i, block in enumerate(blocks, start=1):
        text = _normalize_text(block)
        if len(text) > 15:
            citations.append({
                "id": f"ref_{i}",
                "number": i,
                "text": text,
                "pages": ref_section.get("pages", []),
            })

    return citations
```

**Code: Citations Tab UI (app.py)**

```python
# app.py (lines 385-409)
def tab_citations(papers: dict):
    if not papers:
        st.info("Upload papers to extract citations.")
        return

    for name, data in papers.items():
        citations = data.get("citations", [])
        meta = data.get("metadata", {})
        title = meta.get("title", name)

        with st.expander(f"{title} - {len(citations)} reference(s)",
                         expanded=len(papers) == 1):
            if not citations:
                st.caption("No citations could be extracted from this paper.")
                continue

            html_items = ""
            for cite in citations:
                num = cite.get("number", "?")
                text = escape(cite.get("text", ""))
                html_items += f'<div class="cite-item">' \
                              f'<span class="cite-num">[{num}]</span>{text}</div>'

            st.markdown(
                f'<div class="panel" style="max-height: 500px; overflow-y: auto;">'
                f'{html_items}</div>',
                unsafe_allow_html=True,
            )
```

---

## A.12 Figures and Tables Extraction Module

Figures and tables are extracted directly from the PDF using PyMuPDF's built-in methods. Figures are identified via `page.get_images()`, which returns embedded image references with their bounding rectangles. Tables are detected using `page.find_tables()`, which identifies tabular structures in the PDF.

**Code: Figure Extraction**

```python
# pipeline.py (lines 425-453)
def _extract_figures(self, pdf_path: str) -> List[Dict]:
    figures: List[Dict] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_number = page.number + 1
            images = page.get_images(full=True)
            for img_index, image in enumerate(images, start=1):
                xref = image[0]
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                rect = rects[0]
                figures.append({
                    "id": f"figure_{page_number}_{img_index}",
                    "label": f"Figure {len(figures) + 1}",
                    "description": f"Image extracted from page {page_number}",
                    "page": page_number,
                    "coords": {
                        "page": page_number,
                        "x": rect.x0,
                        "y": rect.y0,
                        "w": max(rect.width, 10),
                        "h": max(rect.height, 10),
                    },
                    "summary": "",
                })
    return figures
```

**Code: Table Extraction**

```python
# pipeline.py (lines 455-494)
def _extract_tables(self, pdf_path: str) -> List[Dict]:
    tables: List[Dict] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_number = page.number + 1
            if not hasattr(page, "find_tables"):
                continue
            try:
                found = page.find_tables()
            except Exception as exc:
                logging.warning("Table detection failed on page %s: %s",
                                page_number, exc)
                continue
            for idx, table in enumerate(found.tables, start=1):
                rows = table.extract() if hasattr(table, "extract") else []
                preview = self._table_preview(rows)
                bbox = table.bbox if getattr(table, "bbox", None) else (0, 0, 0, 0)
                tables.append({
                    "id": f"table_{page_number}_{idx}",
                    "label": f"Table {len(tables) + 1}",
                    "page": page_number,
                    "bbox": bbox,
                    "preview": preview,
                    "summary": "",
                })
    return tables

@staticmethod
def _table_preview(rows: List) -> str:
    if not rows:
        return "Table detected, but cells could not be parsed."
    preview_lines = []
    for row in rows[:4]:
        if isinstance(row, dict):
            cells = list(row.values())
        else:
            cells = list(row)
        preview_lines.append(" | ".join(
            _normalize_text(str(cell)) for cell in cells))
    return "\n".join(preview_lines)
```

**Code: Media-to-Section Attachment**

```python
# pipeline.py (lines 496-501)
@staticmethod
def _attach_media_to_sections(sections: Dict[str, Dict],
                               figures: List[Dict],
                               tables: List[Dict]):
    for section in sections.values():
        pages = set(section.get("pages", []))
        section["figures"] = [fig["id"] for fig in figures if fig["page"] in pages]
        section["tables"] = [tab["id"] for tab in tables if tab["page"] in pages]
```

---

## A.13 User Interface Module

The user interface is built entirely with Streamlit components. The sidebar contains the application header, model selection dropdown, PDF file uploader, and runtime status. The main area dynamically renders content across five tabs. Custom CSS provides a professional dark-themed design with responsive layouts.

**Code: CSS Styling Engine**

```python
# app.py (lines 88-154)
def render_styles(is_dark: bool = False):
    st.markdown(
        f"""
        <style>
        @import url("https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;700&display=swap");
        @import url("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css");

        :root {{
            --bg-primary:    {"#0f1117" if is_dark else "#ffffff"};
            --bg-secondary:  {"#1a1d27" if is_dark else "#f8fbff"};
            --bg-card:       {"#1e2130" if is_dark else "#ffffff"};
            --bg-panel:      {"#252838" if is_dark else "#f4f8fe"};
            --border:        {"#2d3148" if is_dark else "#dbe4f0"};
            --text-primary:  {"#e2e8f0" if is_dark else "#0f172a"};
            --text-secondary:{"#94a3b8" if is_dark else "#334155"};
            --text-muted:    {"#64748b" if is_dark else "#475569"};
            --accent:        {"#60a5fa" if is_dark else "#3b82f6"};
        }}

        html, body, [class*="css"] {{ font-family: "Public Sans", sans-serif; }}

        .hero {{
            border: 1px solid var(--border);
            background: var(--hero-bg);
            border-radius: 16px; padding: 20px; margin-bottom: 18px;
        }}
        .hero h1 {{ margin: 0; color: var(--text-primary); font-size: 1.8rem; }}
        .hero p  {{ margin: 8px 0 0 0; color: var(--text-secondary); }}

        .meta-card {{
            border: 1px solid var(--border); border-radius: 14px;
            background: var(--bg-card); padding: 18px; margin-bottom: 16px;
        }}
        .meta-row {{ display: grid; grid-template-columns: 160px 1fr; gap: 10px;
                     margin-bottom: 10px; color: var(--text-primary); }}
        .meta-key {{ color: var(--text-muted); font-weight: 600; }}

        .panel {{ border: 1px solid var(--border); border-radius: 12px;
                  padding: 14px; background: var(--bg-card); }}
        .panel-title {{ font-weight: 700; margin-bottom: 10px;
                        color: var(--text-primary); }}

        .source-box {{ height: 440px; overflow-y: auto; line-height: 1.65;
                       color: var(--text-primary); font-size: 0.95rem;
                       white-space: normal; }}
        .summary-box {{ min-height: 220px; line-height: 1.7;
                        color: var(--text-primary); background: var(--bg-panel);
                        border: 1px solid var(--border); border-radius: 10px;
                        padding: 12px; margin-top: 12px; }}

        .cmp-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        .cmp-table th, .cmp-table td {{ border: 1px solid var(--border);
            padding: 10px 14px; text-align: left; font-size: 0.93rem; }}
        .cmp-table th {{ background: var(--table-header); font-weight: 700;
                         color: var(--text-primary); }}

        .fig-gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                        gap: 16px; margin-top: 12px; }}
        .fig-card {{ border: 1px solid var(--border); border-radius: 12px;
                     padding: 12px; background: var(--bg-card); text-align: center; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
```

**Code: Metadata Card and Section Viewer**

```python
# app.py (lines 211-294)
def render_metadata_card(metadata: dict, section_count: int,
                         citation_count: int, fig_count: int, table_count: int):
    title = escape(metadata.get("title") or "Untitled Document")
    authors = escape(metadata.get("authors") or "Unknown Authors")
    year = escape(str(metadata.get("year") or "N/A"))
    st.markdown(
        f"""
        <div class="meta-card">
            <div class="meta-row"><div class="meta-key">Title</div><div>{title}</div></div>
            <div class="meta-row"><div class="meta-key">Authors</div><div>{authors}</div></div>
            <div class="meta-row"><div class="meta-key">Year</div><div>{year}</div></div>
            <div class="meta-row"><div class="meta-key">Sections</div><div>{section_count}</div></div>
            <div class="meta-row"><div class="meta-key">Citations</div><div>{citation_count}</div></div>
            <div class="meta-row"><div class="meta-key">Figures / Tables</div>
                 <div>{fig_count} / {table_count}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sections(sections: dict, metadata: dict):
    titles = list(sections.keys())
    if not titles:
        st.warning("No sections detected in this document.")
        return

    if st.button("Summarize All Sections",
                 key=f"sumall_{metadata.get('title','')}",
                 use_container_width=True):
        with st.spinner("Generating summaries for all sections"):
            summarize_all_sections(sections, metadata)

    tabs = st.tabs(titles)
    for title, tab in zip(titles, tabs):
        section = sections[title]
        with tab:
            pages = ", ".join(str(p) for p in section.get("pages", [])) or "N/A"
            chunk_count = section.get("chunk_count", 0)
            word_count = len(section.get("content", "").split())

            st.markdown(
                f"""<div class="section-info">
                    <span><strong>Pages:</strong> {pages}</span>
                    <span><strong>Chunks:</strong> {chunk_count}</span>
                    <span><strong>Words:</strong> {word_count}</span>
                </div>""",
                unsafe_allow_html=True,
            )

            left, right = st.columns([1.2, 1])
            with left:
                source = escape(section.get("content") or "No text found.") \
                    .replace("\n", "<br/>")
                st.markdown(
                    f"""<div class="panel">
                        <div class="panel-title">Original Content</div>
                        <div class="source-box">{source}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with right:
                btn_key = f"sum_{metadata.get('title','')}_{title}"
                if st.button("Summarize This Section",
                             key=btn_key, use_container_width=True):
                    with st.spinner(f"Summarizing: {title}"):
                        section_summary(title, section, metadata)

                doc_key = f"{metadata.get('title', '')}::{title}"
                summary = st.session_state.get("section_summaries", {}).get(doc_key, "")
                if summary:
                    rendered = escape(summary)
                else:
                    rendered = "Click <strong>Summarize This Section</strong> " \
                               "to generate summary."

                st.markdown(
                    f"""<div class="panel">
                        <div class="panel-title">Section Summary</div>
                        <div class="summary-box">{rendered}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
```

The UI follows a consistent design pattern: metadata cards provide quick reference, section content is displayed in scrollable panels, and summary results appear in bordered containers. The "Summarize All Sections" button allows batch processing with progress feedback.

---

## A.14 Experimental Evaluation Module

The experimental evaluation framework (`research_experiment_framework.py`) provides ROUGE-N and ROUGE-L metrics, semantic similarity proxies via cosine similarity, factual consistency checking, multi-document summarization, domain-specific summarization, and an interactive research assistant.

**Code: ROUGE Metrics Implementation**

```python
# research_experiment_framework.py (lines 78-325)
from collections import Counter
import math


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower())
            if t and t not in STOPWORDS]


def cosine_sim(counter_a: Counter, counter_b: Counter) -> float:
    if not counter_a or not counter_b:
        return 0.0
    keys = set(counter_a) | set(counter_b)
    dot = sum(counter_a.get(k, 0) * counter_b.get(k, 0) for k in keys)
    norm_a = math.sqrt(sum(v * v for v in counter_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counter_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


class EvaluationFramework:
    @staticmethod
    def rouge_n_f1(candidate: str, reference: str, n: int = 1) -> float:
        def ngrams(tokens: List[str], nsize: int) -> List[Tuple[str, ...]]:
            return [tuple(tokens[i:i + nsize])
                    for i in range(max(0, len(tokens) - nsize + 1))]

        c_tokens = tokenize(candidate)
        r_tokens = tokenize(reference)
        c_ngrams = Counter(ngrams(c_tokens, n))
        r_ngrams = Counter(ngrams(r_tokens, n))
        if not c_ngrams or not r_ngrams:
            return 0.0
        overlap = sum((c_ngrams & r_ngrams).values())
        precision = overlap / max(1, sum(c_ngrams.values()))
        recall = overlap / max(1, sum(r_ngrams.values()))
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def rouge_l_f1(candidate: str, reference: str) -> float:
        c = tokenize(candidate)
        r = tokenize(reference)
        if not c or not r:
            return 0.0

        dp = [[0] * (len(r) + 1) for _ in range(len(c) + 1)]
        for i in range(1, len(c) + 1):
            for j in range(1, len(r) + 1):
                if c[i - 1] == r[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs = dp[-1][-1]
        precision = lcs / max(1, len(c))
        recall = lcs / max(1, len(r))
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def evaluate_summary(self, candidate: str, reference: str,
                         source_text: str) -> Dict:
        audit = self.fact_checker.audit(candidate, source_text)
        return {
            "rouge1_f1": round(self.rouge_n_f1(candidate, reference, 1), 4),
            "rouge2_f1": round(self.rouge_n_f1(candidate, reference, 2), 4),
            "rougeL_f1": round(self.rouge_l_f1(candidate, reference), 4),
            "semantic_f1_proxy": round(
                self.semantic_f1_proxy(candidate, reference), 4),
            "factual_score": audit.factual_score,
            "contradictions": audit.contradiction_count,
        }
```

**Code: Factual Consistency Checker**

```python
# research_experiment_framework.py (lines 194-273)
class FactualConsistencyChecker:
    def __init__(self):
        self.neg_terms = {"no", "not", "never", "none", "without",
                          "cannot", "can't", "won't"}

    def audit(self, summary: str, source_text: str,
              threshold: float = 0.17) -> SummaryAudit:
        source_sentences = split_sentences(source_text)
        sum_sentences = split_sentences(summary)

        if not sum_sentences:
            return SummaryAudit(summary=summary, factual_score=0.0,
                                contradiction_count=0, flagged_sentences=[])

        supports = []
        flagged = []
        contradiction_count = 0

        for s in sum_sentences:
            best_src, score = self._best_support_sentence(s, source_sentences)
            contradiction = self._is_contradiction(s, best_src)
            if contradiction:
                contradiction_count += 1
            supports.append(score)
            if score < threshold or contradiction:
                flagged.append({
                    "summary_sentence": s,
                    "best_source_sentence": best_src,
                    "support_score": round(score, 4),
                    "contradiction": contradiction,
                })

        base = sum(supports) / len(supports)
        penalty = contradiction_count * 0.08
        factual_score = max(0.0, min(1.0, base - penalty))

        return SummaryAudit(summary=summary,
                            factual_score=round(factual_score, 4),
                            contradiction_count=contradiction_count,
                            flagged_sentences=flagged)

    def _is_contradiction(self, summary_sentence: str,
                          support_sentence: str) -> bool:
        sum_low = summary_sentence.lower()
        src_low = support_sentence.lower()
        sum_neg = any(n in sum_low for n in self.neg_terms)
        src_neg = any(n in src_low for n in self.neg_terms)
        if sum_neg != src_neg and support_sentence:
            return True
        sum_nums = set(self._numbers(summary_sentence))
        src_nums = set(self._numbers(support_sentence))
        if sum_nums and src_nums and not (sum_nums & src_nums):
            return True
        return False
```

**Code: Experiment Runner**

```python
# run_research_experiments.py (lines 1-40)
import json
from pathlib import Path
from research_experiment_framework import ResearchPipeline, pretty_metric_table


def main():
    root = Path(__file__).resolve().parent
    pdf_paths = [str(root / "data" / "2004.05150v2.pdf")]

    pipeline = ResearchPipeline(use_local_model=True)
    if not pipeline.llm.has_local_model():
        raise RuntimeError("Local llama.cpp model failed to load")
    papers = pipeline.load_papers(pdf_paths, include_media=True)

    single = pipeline.run_single_paper_experiment(papers[0])
    phase2_media = pipeline.run_phase2_media_experiment(papers[0])
    multi = pipeline.run_multi_document_experiment(
        papers if len(papers) > 1 else papers * 3
    )

    payload = {
        "single": single,
        "phase2_media": phase2_media,
        "multi": multi,
        "metric_table": pretty_metric_table(single),
    }

    out_file = root / "research_experiment_results.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Saved:", out_file)
    print(payload["metric_table"])


if __name__ == "__main__":
    main()
```

---

## A.15 Error Handling and Validation

The system implements robust error handling at multiple levels. The `LLMService` handles API failures, rate limits, and context window overflows with retry logic and graceful degradation to fallback summarization. File validation ensures uploaded PDFs exist before processing. Input validation guards against empty text and invalid configurations.

**Code: API Error Handling and Retry Logic**

```python
# pipeline.py (lines 760-784, 1068-1084)
# Inside _summarize_groq:
for attempt in range(1, self.groq_max_retries + 1):
    try:
        response = self._groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            max_tokens=300,
            temperature=0.2,
        )
        generated = response.choices[0].message.content.strip()
        if generated:
            return generated
    except Exception as exc:
        last_error = exc
        msg = str(exc).lower()
        # Context-length errors: retry with smaller input
        if "context" in msg and ("length" in msg or "window" in msg or "token" in msg):
            break
        # Rate-limit: wait and retry
        if "rate" in msg and attempt < self.groq_max_retries:
            time.sleep(min(2.0 * attempt, 8.0))
            continue
        if self.require_llm:
            raise RuntimeError(f"Groq summarization failed: {exc}") from exc
        logging.warning("Groq summarization failed, using fallback: %s", exc)
        return self._fallback_summary(text)
```

**Code: Ollama Transient Error Handling**

```python
# pipeline.py (lines 838-858)
last_error: Optional[Exception] = None
for attempt in range(1, self.ollama_max_retries + 1):
    try:
        return self._ollama_post_json("/api/generate", payload)
    except Exception as exc:
        last_error = exc
        msg = str(exc).lower()
        transient = (
            "http error (500)" in msg
            or "http error (502)" in msg
            or "http error (503)" in msg
            or "http error (504)" in msg
            or "timed out" in msg
            or "temporary" in msg
            or "connection reset" in msg
        )
        if transient and attempt < self.ollama_max_retries:
            time.sleep(min(1.5 * attempt, 5.0))
            continue
        raise RuntimeError(f"Ollama generate failed: {exc}") from exc
```

**Code: File and Input Validation**

```python
# pipeline.py (lines 76-77) - File existence check
def parse_document(self, pdf_path: str, include_media: bool = True) -> Dict:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

# pipeline.py (lines 714-715) - Empty text guard
def summarize(self, text: str, context: str = "") -> str:
    if not text.strip():
        return "No content found for summarization."

# app.py (lines 185-206) - Upload parsing with error isolation
def parse_uploaded_files(uploaded_files):
    papers = st.session_state.get("papers", {})
    changed = False
    for uploaded in uploaded_files:
        if uploaded.name in papers:
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getvalue())
            pdf_path = tmp.name
        try:
            data = services["extractor"].parse_document(pdf_path, include_media=True)
            papers[uploaded.name] = data
            changed = True
        except Exception as exc:
            logging.error("Parsing failed for %s: %s", uploaded.name, exc, exc_info=True)
            st.error(f"Unable to process **{uploaded.name}**. Skipping.")
    if changed:
        st.session_state["papers"] = papers
    return papers
```

The error handling architecture ensures that a single malformed PDF or a transient API failure does not disrupt the entire session. Each file is processed independently with per-file exception isolation. LLM failures automatically degrade to extractive fallback summarization, preserving system availability.

---

## A.16 Project Directory Structure

The complete project directory structure is organized as follows:

```
Research-Paper-Summarizer/
│
├── app.py                          # Main Streamlit application entry point
├── pipeline.py                     # Core pipeline: DocumentExtractor & LLMService
├── research_experiment_framework.py# Evaluation framework with ROUGE metrics
├── run_research_experiments.py     # Experiment runner script
├── execute_notebook_simple.py      # Jupyter notebook executor
├── analyze_pdf.py                  # PDF analysis utility
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── docker-compose.yml              # Docker deployment configuration
├── runtime.txt                     # Python runtime specification
├── grobid_config.json              # GROBID configuration
│
├── data/                           # Sample PDF papers for testing
│   └── 2004.05150v2.pdf
│
├── models/                         # Local GGUF model files (gitignored)
│   └── llama-3.2-1b-instruct.Q4_K_M.gguf
│
├── outputs/                        # Generated outputs
│   ├── figures/                    # Evaluation result plots (PNG)
│   └── tables/                     # Evaluation result tables (CSV, TeX)
│
├── grobid_output/                  # GROBID processing outputs
│
├── static/                         # Static assets
│
├── third_party/                    # Third-party libraries (musl libc)
│   └── musl/lib/
│       ├── ld-musl-x86_64.so.1
│       └── libc.musl-x86_64.so.1
│
├── .streamlit/                     # Streamlit configuration
│   ├── secrets.toml                # API keys (gitignored)
│   └── secrets.toml.example        # Example secrets file
│
├── .devcontainer/                  # Dev container configuration
│   └── devcontainer.json
│
├── .venv/                          # Virtual environment (gitignored)
│
└── .vscode/                        # VS Code settings
    └── settings.json
```

---

## A.17 Summary

The Smart Research Paper Summarizer integrates multiple modules into a cohesive research analysis pipeline. The entry point (`app.py`) initializes the Streamlit web interface and orchestrates user interaction through a tabbed layout. PDF documents uploaded via the sidebar are processed by `DocumentExtractor` (from `pipeline.py`), which performs line-level text extraction, running header/footer removal, metadata inference, section detection using font-size and pattern heuristics, figure/table extraction via PyMuPDF, and citation parsing. The extracted structured representation is then passed to `LLMService`, which communicates with the Groq API (Llama 3.3 70B) to generate section-level summaries, comparative analyses across multiple papers, and thematic survey syntheses. All summarization results are cached in Streamlit session state for responsive interaction. The optional evaluation framework (`research_experiment_framework.py`) provides quantitative assessment via ROUGE-1, ROUGE-2, ROUGE-L metrics, cosine-based semantic similarity, and factual consistency audits using negation and numeric contradiction detection. The system architecture ensures graceful degradation: if the LLM backend is unavailable, an extractive fallback summarizer maintains basic functionality. Error handling is comprehensive, with per-file isolation during PDF parsing, retry logic with exponential backoff for API rate limits, and progressive input truncation for context window overflows. The combination of Streamlit's rapid UI development, PyMuPDF's robust PDF capabilities, and Groq's high-performance LLM inference creates a production-ready tool for academic research analysis.
