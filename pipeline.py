import ctypes
import json
import logging
import os
import re
import time
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional
from urllib import error as urlerror, request as urlrequest

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required but not installed. Add 'PyMuPDF' to requirements.txt "
            "for Streamlit deployment."
        ) from exc



SECTION_HEADING_REGEX = re.compile(
    r"^((\d+(\.\d+)*)|([IVXLCM]+))[\).\s-]+[A-Z][A-Za-z0-9\-\s,():/]+$"
)

COMMON_SECTION_TITLES = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "future work",
    "limitations",
    "appendix",
    "references",
    "acknowledgments",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, max_words: int = 450, overlap: int = 80) -> List[str]:
    words = text.split()
    if not words:
        return []
    stride = max(max_words - overlap, 1)
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + max_words]))
        start += stride
    return chunks


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

                        # Remove running headers/footers that pollute metadata/sections.
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
            # Running headers/footers appear on many pages at near-fixed y.
            if y_med < 95:
                removable.add(text)

        filtered = []
        for item in items:
            text = _normalize_text(item["text"])
            if text in removable:
                continue
            filtered.append(item)
        return filtered

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
            "unknown",
            "untitled",
            "latex",
            "hyperref",
            "microsoft word",
            "adobe",
        }
        low = cleaned.lower()
        if any(token in low for token in invalid_tokens):
            return None
        if len(cleaned) <= 2:
            return None
        return cleaned

    @staticmethod
    def _extract_year_from_meta(meta: Dict) -> Optional[str]:
        for key in ("creationDate", "modDate"):
            raw = meta.get(key)
            if not raw:
                continue
            match = re.search(r"(19|20)\d{2}", raw)
            if match:
                return match.group(0)
        return None

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

    def _infer_authors_from_text(self, lines: List[Dict], title: Optional[str]) -> Optional[str]:
        first_page = sorted([line for line in lines if line["page"] == 1], key=lambda x: x["y0"])
        if not first_page:
            return None

        abstract_y = None
        for line in first_page:
            if _normalize_text(line["text"]).lower() == "abstract":
                abstract_y = line["y0"]
                break

        candidates = []
        for line in first_page:
            text = _normalize_text(line["text"])
            low = text.lower()
            if title and text in title:
                continue
            if abstract_y is not None and line["y0"] >= abstract_y:
                continue
            if len(text) < 4 or len(text) > 120:
                continue
            if any(token in low for token in ("http", "www.", "@", "arxiv", "doi")):
                continue
            if any(token in low for token in ("university", "institute", "department", "school", "laboratory")):
                continue
            if not re.search(r"[A-Za-z]", text):
                continue
            if re.search(r"\d{4}", text):
                continue
            if re.fullmatch(r"[A-Z\s]+", text):
                continue
            candidates.append(text)

        if not candidates:
            return None

        # Keep compact name lines and strip affiliation-like residues.
        filtered = [c for c in candidates if len(c.split()) <= 12]
        final = ", ".join(filtered[:3]) if filtered else ", ".join(candidates[:2])
        return _normalize_text(final)

    @staticmethod
    def _infer_year_from_text(lines: List[Dict]) -> Optional[str]:
        years = re.findall(r"\b((?:19|20)\d{2})\b", " ".join(line["text"] for line in lines[:240]))
        if not years:
            return None
        # Prefer more recent valid publication-like year in text.
        return sorted(years)[-1]

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

        # Fallback if no headings are detected.
        if not heading_positions:
            joined = "\n".join(line["text"] for line in lines)
            return {
                "Document Content": self._build_section("Document Content", joined, {line["page"] for line in lines})
            }

        sections: Dict[str, Dict] = {}
        for pos, start_idx in enumerate(heading_positions):
            end_idx = heading_positions[pos + 1] if pos + 1 < len(heading_positions) else len(lines)
            heading_line = lines[start_idx]
            title = _normalize_text(heading_line["text"].rstrip(":"))
            body_lines = lines[start_idx + 1 : end_idx]
            content_lines = [line["text"] for line in body_lines]
            pages = {heading_line["page"]}
            pages.update(line["page"] for line in body_lines)

            key = self._deduplicate_title(sections, title)
            sections[key] = self._build_section(key, "\n".join(content_lines), pages)

        return sections

    @staticmethod
    def _find_heading_anchor(heading_positions: List[int], lines: List[Dict]) -> Optional[int]:
        for idx in heading_positions:
            text = _normalize_text(lines[idx]["text"]).lower()
            if text in {"abstract", "introduction"}:
                return idx
            if re.match(r"^1[\.\)\s-]+", text):
                return idx
        return heading_positions[0] if heading_positions else None

    @staticmethod
    def _deduplicate_title(existing: Dict[str, Dict], title: str) -> str:
        if title not in existing:
            return title
        idx = 2
        while f"{title} ({idx})" in existing:
            idx += 1
        return f"{title} ({idx})"

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
        if re.match(r"^\d{4}\)\.", text) or re.match(r"^\d{4}\)", text):
            return False
        if re.match(r"^(?:19|20)\d{2}[a-z]?\.", text):
            return False
        if "@" in text:
            return False

        lower = text.lower()
        words = text.split()

        # Ignore front-matter lines on page 1 unless they are canonical headings.
        if line["page"] == 1 and line["y0"] < 360:
            if lower not in {"abstract", "introduction"} and not SECTION_HEADING_REGEX.match(text):
                return False

        if lower in COMMON_SECTION_TITLES and line["size"] >= body_size:
            return True
        if SECTION_HEADING_REGEX.match(text) and len(words) <= 14:
            return True

        # Bold/larger-than-body short lines are likely headings.
        looks_bold = "bold" in line["font"] or "semibold" in line["font"]
        if len(words) <= 12 and text[-1] not in ".,;":
            if line["size"] >= body_size + 1.0:
                return True
            if looks_bold and line["size"] >= body_size + 0.3 and self._looks_like_title_case(text):
                return True

        return False

    @staticmethod
    def _looks_like_title_case(text: str) -> bool:
        words = [w for w in re.split(r"\s+", text) if w]
        if not words:
            return False
        good = 0
        for word in words:
            stripped = re.sub(r"^[\d\.\)\(]+", "", word)
            if stripped and stripped[0].isupper():
                good += 1
        ratio = good / len(words)
        if ratio >= 0.6:
            if not text.endswith(".") and not text.endswith(","):
                return True
        return False

    @staticmethod
    def _looks_like_heading(text: str) -> bool:
        t = _normalize_text(text)
        if t.lower() in COMMON_SECTION_TITLES:
            return True
        return bool(SECTION_HEADING_REGEX.match(t))

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
                    figures.append(
                        {
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
                        }
                    )
        return figures

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
                    logging.warning("Table detection failed on page %s: %s", page_number, exc)
                    continue
                for idx, table in enumerate(found.tables, start=1):
                    rows = table.extract() if hasattr(table, "extract") else []
                    preview = self._table_preview(rows)
                    bbox = table.bbox if getattr(table, "bbox", None) else (0, 0, 0, 0)
                    tables.append(
                        {
                            "id": f"table_{page_number}_{idx}",
                            "label": f"Table {len(tables) + 1}",
                            "page": page_number,
                            "bbox": bbox,
                            "preview": preview,
                            "summary": "",
                        }
                    )
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
            preview_lines.append(" | ".join(_normalize_text(str(cell)) for cell in cells))
        return "\n".join(preview_lines)

    @staticmethod
    def _attach_media_to_sections(sections: Dict[str, Dict], figures: List[Dict], tables: List[Dict]):
        for section in sections.values():
            pages = set(section.get("pages", []))
            section["figures"] = [fig["id"] for fig in figures if fig["page"] in pages]
            section["tables"] = [tab["id"] for tab in tables if tab["page"] in pages]

    @staticmethod
    def extract_citations(sections: Dict[str, Dict]) -> List[Dict]:
        """Parse the References / Bibliography section into individual citation entries."""
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

        # Strategy 1: numbered references like [1], [2], ... or 1. 2. ...
        numbered = re.split(r"\n?\[?(\d{1,3})\]?\.?\s+", raw)
        if len(numbered) >= 5:  # at least 2 references found (number+text pairs)
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

        # Strategy 2: split on double-newlines or lines that start with an author pattern
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
        self.require_llm = self._env_flag("SUMMARIX_REQUIRE_LLM", default=False) if require_llm is None else bool(require_llm)
        self.mode = "fallback"

        # Groq settings
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.2-3b-preview")
        self.groq_max_input_chars = self._env_int("GROQ_MAX_INPUT_CHARS", default=12000, minimum=2000)
        self.groq_max_retries = self._env_int("GROQ_MAX_RETRIES", default=3, minimum=1)
        self._groq_client = None

        # Ollama settings
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.ollama_timeout = self._env_int("OLLAMA_TIMEOUT_SEC", default=120, minimum=20)
        self.ollama_num_ctx = self._env_int("OLLAMA_NUM_CTX", default=2048, minimum=512)
        self.ollama_max_input_chars = self._env_int("OLLAMA_MAX_INPUT_CHARS", default=12000, minimum=2000)
        self.ollama_max_retries = self._env_int("OLLAMA_MAX_RETRIES", default=3, minimum=1)

        if not self.provider:
            # Auto mode: Groq > Ollama > local GGUF
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
            else:
                raise ValueError(
                    f"Unsupported SUMMARIX_LLM_PROVIDER='{self.provider}'. Use 'groq', 'ollama', or 'local'."
                )
        except Exception as exc:
            init_error = exc

        if init_error is not None:
            if self.require_llm:
                raise RuntimeError(
                    "LLM initialization failed. Set GROQ_API_KEY for Groq, or OLLAMA_BASE_URL + OLLAMA_MODEL "
                    "for Ollama, or provide a valid local GGUF model via SUMMARIX_MODEL_PATH."
                ) from init_error
            logging.warning("LLM unavailable, fallback summarizer enabled: %s", init_error)
            self.mode = "fallback"

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _env_int(name: str, default: int, minimum: int = 1) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = int(str(raw).strip())
        except ValueError:
            return default
        return max(value, minimum)

    def _init_groq(self):
        if not self.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing for provider='groq'.")
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=self.groq_api_key)
            # Quick validation: list models to confirm the key is valid
            self._groq_client.models.list()
        except ImportError as exc:
            raise RuntimeError("groq package is not installed. Add 'groq>=0.11,<1' to requirements.txt.") from exc
        except Exception as exc:
            raise RuntimeError(f"Groq API initialization failed: {exc}") from exc
        self.mode = "groq"

    def _init_ollama(self):
        if not self.ollama_base_url:
            raise RuntimeError("OLLAMA_BASE_URL is missing for provider='ollama'.")

        tags = self._ollama_get_json("/api/tags")
        model_names = {
            str(item.get("name", "")).strip()
            for item in tags.get("models", [])
            if isinstance(item, dict)
        }
        model_aliases = {name.split(":")[0] for name in model_names if name}
        if self.ollama_model not in model_names and self.ollama_model not in model_aliases:
            raise RuntimeError(
                f"Ollama model '{self.ollama_model}' is not available at {self.ollama_base_url}. "
                f"Available: {', '.join(sorted(model_names)) or 'none'}"
            )
        self.mode = "ollama"

    def _init_local(self):
        Llama = self._load_llama_class()
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Local model file not found: {self.model_path}")
        self.llm = Llama(model_path=self.model_path, n_ctx=2048, n_threads=4, verbose=False)
        self.mode = "local-llama"

    @staticmethod
    def _load_llama_class():
        try:
            from llama_cpp import Llama
            return Llama
        except Exception as exc:
            # Some linux wheels depend on musl; preload bundled musl libc and retry.
            if "libc.musl-x86_64.so.1" not in str(exc):
                raise

            musl_lib = Path(__file__).resolve().parent / "third_party" / "musl" / "lib" / "libc.musl-x86_64.so.1"
            if not musl_lib.exists():
                raise

            ctypes.CDLL(str(musl_lib), mode=ctypes.RTLD_GLOBAL)
            from llama_cpp import Llama
            return Llama

    def check_health(self) -> bool:
        return self.mode in {"local-llama", "ollama"}

    def has_local_model(self) -> bool:
        return self.mode == "local-llama"

    def has_active_llm(self) -> bool:
        return self.mode in {"local-llama", "ollama"}

    def status_label(self) -> str:
        if self.mode == "groq":
            return f"Groq API active ({self.groq_model})"
        if self.mode == "ollama":
            return f"Ollama model active ({self.ollama_model})"
        if self.mode == "local-llama":
            return "Local LLaMA model loaded"
        return "Fallback summarizer active"

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
                "LLM is required but unavailable. Set GROQ_API_KEY, OLLAMA_BASE_URL + OLLAMA_MODEL, "
                "or local GGUF via SUMMARIX_MODEL_PATH."
            )

        return self._fallback_summary(text)

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
                    "content": "You are an expert research assistant. Write concise, accurate summaries of research paper sections.",
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

        if self.require_llm and last_error:
            raise RuntimeError(f"Groq summarization failed after retries: {last_error}")

        logging.warning("Groq summarization failed after retries, using fallback summary")
        return self._fallback_summary(text)

    def _summarize_ollama(self, text: str, context: str = "") -> str:
        max_chars_seq = [
            self.ollama_max_input_chars,
            int(self.ollama_max_input_chars * 0.75),
            int(self.ollama_max_input_chars * 0.5),
        ]
        for max_chars in max_chars_seq:
            clipped_text = self._truncate_for_context(text, max_chars)
            prompt = (
                "You are an expert research assistant.\n"
                "Write a concise, accurate summary of the provided paper section.\n"
                f"Context: {context}\n"
                f"Section text:\n{clipped_text}\n\n"
                "Summary:"
            )
            try:
                response = self._generate_with_ollama(prompt)
                generated = str(response.get("response", "")).strip()
                if generated:
                    return generated
            except Exception as exc:
                msg = str(exc).lower()
                if "context" in msg and ("window" in msg or "length" in msg):
                    continue
                if self.require_llm:
                    raise
                logging.warning("Ollama summarization failed, using fallback summary: %s", exc)
                return self._fallback_summary(text)

        if self.require_llm:
            raise RuntimeError("Ollama summarization exceeded context window repeatedly.")

        logging.warning("Ollama summarization exceeded context window repeatedly, using fallback summary")
        return self._fallback_summary(text)

    def _generate_with_ollama(self, prompt: str) -> Dict:
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 220,
                "num_ctx": self.ollama_num_ctx,
            },
        }
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
        raise RuntimeError(f"Ollama generate failed: {last_error}")

    def _summarize_local(self, text: str, context: str = "") -> str:
        if self.llm is None:
            raise RuntimeError("Local LLM is not initialized.")

        # Retry with progressively smaller context windows to avoid token overflow.
        for max_chars in (5200, 3600, 2400):
            clipped_text = self._truncate_for_context(text, max_chars)
            prompt = (
                "You are an expert research assistant.\n"
                "Write a concise, accurate summary of the provided paper section.\n"
                f"Context: {context}\n"
                f"Section text:\n{clipped_text}\n\n"
                "Summary:"
            )
            try:
                response = self.llm(prompt, max_tokens=180, stop=["\n\n"])
                generated = response["choices"][0]["text"].strip()
                if generated:
                    return generated
            except Exception as exc:
                msg = str(exc).lower()
                if "exceed context window" in msg or "context" in msg and "window" in msg:
                    continue
                if self.require_llm:
                    raise
                logging.warning("LLM inference failed, using fallback summary: %s", exc)
                return self._fallback_summary(text)

        if self.require_llm:
            raise RuntimeError("Local LLM summarization exceeded context window repeatedly.")

        logging.warning("LLM inference exceeded context window repeatedly, using fallback summary")
        return self._fallback_summary(text)

    def _http_get_json(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict:
        merged_headers = {"Content-Type": "application/json"}
        if "ngrok" in url:
            # Avoid ngrok browser warning/interstitial on free tunnels for API calls.
            merged_headers["ngrok-skip-browser-warning"] = "1"
        if headers:
            merged_headers.update(headers)
        req = urlrequest.Request(url, method="GET", headers=merged_headers)
        try:
            with urlrequest.urlopen(req, timeout=self.ollama_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            details = ""
            try:
                details = exc.read().decode("utf-8").strip()
            except Exception:
                details = ""
            extra = f" - {details}" if details else ""
            raise RuntimeError(f"HTTP error ({exc.code}) at {url}{extra}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"Unable to reach endpoint {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Endpoint returned invalid JSON from {url}") from exc

    def _http_post_json(
        self,
        url: str,
        payload: Dict,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict:
        body = json.dumps(payload).encode("utf-8")
        merged_headers = {"Content-Type": "application/json"}
        if "ngrok" in url:
            # Avoid ngrok browser warning/interstitial on free tunnels for API calls.
            merged_headers["ngrok-skip-browser-warning"] = "1"
        if headers:
            merged_headers.update(headers)
        req = urlrequest.Request(url, data=body, method="POST", headers=merged_headers)
        try:
            with urlrequest.urlopen(req, timeout=self.ollama_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            details = ""
            try:
                details = exc.read().decode("utf-8").strip()
            except Exception:
                details = ""
            extra = f" - {details}" if details else ""
            raise RuntimeError(f"HTTP error ({exc.code}) at {url}{extra}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"Unable to reach endpoint {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Endpoint returned invalid JSON from {url}") from exc

    def _ollama_get_json(self, path: str) -> Dict:
        url = f"{self.ollama_base_url}{path}"
        return self._http_get_json(url)

    def _ollama_post_json(self, path: str, payload: Dict) -> Dict:
        url = f"{self.ollama_base_url}{path}"
        return self._http_post_json(url, payload)

    @staticmethod
    def _truncate_for_context(text: str, max_chars: int) -> str:
        clean = _normalize_text(text)
        if len(clean) <= max_chars:
            return clean

        head = int(max_chars * 0.7)
        tail = max_chars - head
        return f"{clean[:head]} ... {clean[-tail:]}"

    @staticmethod
    def _fallback_summary(text: str, sentences: int = 4) -> str:
        clean = _normalize_text(text)
        sentence_candidates = re.split(r"(?<=[.!?])\s+", clean)
        picked = [s for s in sentence_candidates if len(s) > 20][:sentences]
        if picked:
            return " ".join(picked)
        return clean[:700]

    def compare_papers(self, papers: List[Dict]) -> str:
        """Generate a structured comparative analysis across multiple papers."""
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
            "1. **Research Objectives** — What each paper aims to solve\n"
            "2. **Methodologies** — Approaches and techniques used\n"
            "3. **Key Findings** — Main results and contributions\n"
            "4. **Datasets & Evaluation** — Data sources and metrics used\n"
            "5. **Strengths & Limitations** — Advantages and gaps of each paper\n"
            "6. **Agreements & Contradictions** — Where papers align or disagree\n\n"
            "Format with clear headings and bullet points."
        )
        return self._llm_generate(prompt_text, max_tokens=1200)

    def synthesize_survey(self, papers: List[Dict]) -> str:
        """Generate a unified thematic survey summary across multiple papers."""
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

    def _llm_generate(self, prompt: str, max_tokens: int = 600) -> str:
        """Dispatch a generation request to whichever backend is active."""
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

    def _ollama_generate_text(self, prompt: str, max_tokens: int = 600) -> str:
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": max_tokens, "num_ctx": self.ollama_num_ctx},
        }
        result = self._ollama_post_json("/api/generate", payload)
        return str(result.get("response", "")).strip() or "No response from Ollama."

    def _local_generate(self, prompt: str, max_tokens: int = 600) -> str:
        if self.llm is None:
            raise RuntimeError("Local LLM is not initialized.")
        response = self.llm(prompt, max_tokens=max_tokens, stop=["\n\n\n"])
        return response["choices"][0]["text"].strip() or "No response from local LLM."


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
