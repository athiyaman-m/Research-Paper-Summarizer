import ctypes
import logging
import os
import re
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

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

    def parse_document(self, pdf_path: str, include_media: bool = False) -> Dict:
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

        return {
            "metadata": metadata,
            "sections": sections,
            "figures": figures,
            "tables": tables,
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


class LLMService:
    def __init__(self, model_path: Optional[str] = None):
        self.llm = None
        self.model_path = model_path or os.getenv(
            "SUMMARIX_MODEL_PATH", "models/llama-3.2-1b-instruct.Q4_K_M.gguf"
        )
        self.mode = "fallback"
        try:
            Llama = self._load_llama_class()

            if os.path.exists(self.model_path):
                self.llm = Llama(model_path=self.model_path, n_ctx=2048, n_threads=4, verbose=False)
                self.mode = "local-llama"
            else:
                logging.warning("Local model file not found: %s", self.model_path)
        except Exception as exc:
            logging.warning("Local LLM unavailable, fallback summarizer enabled: %s", exc)

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
        return True

    def has_local_model(self) -> bool:
        return self.llm is not None

    def status_label(self) -> str:
        if self.has_local_model():
            return "Local model loaded"
        return "Fallback summarizer active"

    def summarize(self, text: str, context: str = "") -> str:
        if not text.strip():
            return "No content found for summarization."

        if self.llm is None:
            return self._fallback_summary(text)

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
                logging.warning("LLM inference failed, using fallback summary: %s", exc)
                return self._fallback_summary(text)

        logging.warning("LLM inference exceeded context window repeatedly, using fallback summary")
        return self._fallback_summary(text)

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
