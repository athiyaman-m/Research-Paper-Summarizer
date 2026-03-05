import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from pipeline import DocumentExtractor, LLMService


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "to", "in", "on", "for", "by",
    "with", "at", "from", "as", "is", "are", "was", "were", "be", "been", "being", "that", "this",
    "it", "its", "into", "than", "then", "they", "their", "we", "our", "you", "your", "can", "could",
    "may", "might", "will", "would", "should", "do", "does", "did", "have", "has", "had", "not", "no",
}

SECTION_PRIORITY = {
    "abstract": 1.0,
    "introduction": 0.9,
    "related work": 0.55,
    "method": 0.95,
    "methods": 0.95,
    "methodology": 0.95,
    "approach": 0.92,
    "model": 0.9,
    "results": 1.0,
    "discussion": 0.9,
    "conclusion": 1.0,
    "conclusions": 1.0,
    "limitations": 0.85,
    "future work": 0.9,
    "references": 0.2,
}

SECTION_FLOW = [
    "abstract",
    "introduction",
    "related work",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "future work",
    "references",
]

DOMAIN_KEYWORDS = {
    "medical": {
        "patient", "clinical", "disease", "therapy", "hospital", "diagnosis", "treatment", "drug",
        "biomedical", "health", "trial", "symptom", "epidemic", "infection",
    },
    "legal": {
        "act", "section", "court", "law", "tribunal", "judgment", "plaintiff", "defendant", "case",
        "statute", "legal", "jurisdiction", "compliance", "regulation",
    },
    "govt": {
        "scheme", "policy", "ministry", "district", "state", "government", "implementation", "budget",
        "public", "beneficiary", "india", "department", "governance", "mission",
    },
}


@dataclass
class SummaryAudit:
    summary: str
    factual_score: float
    contradiction_count: int
    flagged_sentences: List[Dict]


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if t and t not in STOPWORDS]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


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


def top_keywords(text: str, n: int = 20) -> List[str]:
    counts = Counter(tokenize(text))
    return [w for w, _ in counts.most_common(n)]


class StructureAwareSummarizer:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    def section_importance(self, section_title: str, section_text: str) -> float:
        title = section_title.lower()
        base = 0.6
        for key, val in SECTION_PRIORITY.items():
            if key in title:
                base = max(base, val)
        length_bonus = min(len(section_text.split()) / 1200.0, 0.3)
        return min(base + length_bonus, 1.3)

    def build_section_graph(self, sections: Dict[str, Dict]) -> Dict[str, List[Tuple[str, float]]]:
        titles = list(sections.keys())
        graph: Dict[str, List[Tuple[str, float]]] = {t: [] for t in titles}

        token_cache = {t: tokenize(sections[t].get("content", "")) for t in titles}

        for i, src in enumerate(titles):
            src_tokens = token_cache[src]
            scores = []
            for j, dst in enumerate(titles):
                if src == dst:
                    continue
                dst_tokens = token_cache[dst]
                lex = jaccard(src_tokens, dst_tokens)
                flow = 0.0
                if abs(i - j) == 1:
                    flow = 0.24
                elif j > i:
                    flow = 0.08
                scores.append((dst, lex + flow))
            scores.sort(key=lambda x: x[1], reverse=True)
            graph[src] = scores[:3]

        return graph

    def _section_context(self, current: str, sections: Dict[str, Dict], graph: Dict[str, List[Tuple[str, float]]]) -> str:
        neighbors = graph.get(current, [])
        pieces = []
        for title, weight in neighbors:
            if weight < 0.08:
                continue
            txt = sections[title].get("content", "")
            if not txt:
                continue
            max_words = 180 if weight > 0.3 else 90
            excerpt = " ".join(txt.split()[:max_words])
            pieces.append(f"[{title} | link={weight:.2f}] {excerpt}")
        return "\n".join(pieces)

    def summarize_independent(self, sections: Dict[str, Dict], paper_title: str) -> Dict[str, str]:
        out = {}
        for title, sec in sections.items():
            content = sec.get("content", "")
            prompt_context = f"Paper: {paper_title}; Section: {title}; Independent summarization"
            out[title] = self.llm.summarize(" ".join(content.split()[:700]), prompt_context)
        return out

    def summarize_structure_aware(self, sections: Dict[str, Dict], paper_title: str) -> Dict[str, str]:
        graph = self.build_section_graph(sections)
        out = {}
        for title, sec in sections.items():
            content = sec.get("content", "")
            importance = self.section_importance(title, content)
            context_links = self._section_context(title, sections, graph)
            chunk_size = int(650 + 300 * min(importance, 1.0))
            body = " ".join(content.split()[:chunk_size])
            prompt_context = (
                f"Paper: {paper_title}; Section: {title}; Importance={importance:.2f}. "
                f"Use linked sections as context for factual continuity.\n{context_links}"
            )
            out[title] = self.llm.summarize(body, prompt_context)
        return out

    def compose_final_summary(self, section_summaries: Dict[str, str], max_sections: int = 8) -> str:
        ranked = sorted(section_summaries.items(), key=lambda x: self.section_importance(x[0], x[1]), reverse=True)
        selected = ranked[:max_sections]
        combined = "\n".join(f"{title}: {txt}" for title, txt in selected)
        return self.llm.summarize(combined, "Create one coherent research paper summary with contributions, method, and results")


class FactualConsistencyChecker:
    def __init__(self):
        self.neg_terms = {"no", "not", "never", "none", "without", "cannot", "can't", "won't"}

    def _best_support_sentence(self, sentence: str, source_sentences: List[str]) -> Tuple[str, float]:
        s_tokens = tokenize(sentence)
        s_counter = Counter(s_tokens)
        best_score = 0.0
        best_src = ""
        for src in source_sentences:
            src_tokens = tokenize(src)
            score = 0.65 * jaccard(s_tokens, src_tokens) + 0.35 * cosine_sim(s_counter, Counter(src_tokens))
            if score > best_score:
                best_score = score
                best_src = src
        return best_src, best_score

    @staticmethod
    def _numbers(text: str) -> List[str]:
        return re.findall(r"\b\d+(?:\.\d+)?\b", text)

    def _is_contradiction(self, summary_sentence: str, support_sentence: str) -> bool:
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

    def audit(self, summary: str, source_text: str, threshold: float = 0.17) -> SummaryAudit:
        source_sentences = split_sentences(source_text)
        sum_sentences = split_sentences(summary)

        if not sum_sentences:
            return SummaryAudit(summary=summary, factual_score=0.0, contradiction_count=0, flagged_sentences=[])

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

        return SummaryAudit(
            summary=summary,
            factual_score=round(factual_score, 4),
            contradiction_count=contradiction_count,
            flagged_sentences=flagged,
        )

    def revise_summary(self, audit: SummaryAudit) -> str:
        if not audit.flagged_sentences:
            return audit.summary
        flagged_map = {f["summary_sentence"] for f in audit.flagged_sentences}
        cleaned = [s for s in split_sentences(audit.summary) if s not in flagged_map]
        if not cleaned:
            return audit.summary
        return " ".join(cleaned)


class EvaluationFramework:
    def __init__(self):
        self.fact_checker = FactualConsistencyChecker()

    @staticmethod
    def rouge_n_f1(candidate: str, reference: str, n: int = 1) -> float:
        def ngrams(tokens: List[str], nsize: int) -> List[Tuple[str, ...]]:
            return [tuple(tokens[i:i + nsize]) for i in range(max(0, len(tokens) - nsize + 1))]

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

    @staticmethod
    def semantic_f1_proxy(candidate: str, reference: str) -> float:
        c_counter = Counter(tokenize(candidate))
        r_counter = Counter(tokenize(reference))
        return cosine_sim(c_counter, r_counter)

    def evaluate_summary(self, candidate: str, reference: str, source_text: str) -> Dict:
        audit = self.fact_checker.audit(candidate, source_text)
        return {
            "rouge1_f1": round(self.rouge_n_f1(candidate, reference, 1), 4),
            "rouge2_f1": round(self.rouge_n_f1(candidate, reference, 2), 4),
            "rougeL_f1": round(self.rouge_l_f1(candidate, reference), 4),
            "semantic_f1_proxy": round(self.semantic_f1_proxy(candidate, reference), 4),
            "factual_score": audit.factual_score,
            "contradictions": audit.contradiction_count,
        }


class DomainSpecificSummarizer:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    def detect_domain(self, text: str) -> str:
        tokens = set(tokenize(text))
        scores = {}
        for domain, kws in DOMAIN_KEYWORDS.items():
            scores[domain] = len(tokens & kws)
        best_domain, best_score = max(scores.items(), key=lambda x: x[1])
        if best_score == 0:
            return "general"
        return best_domain

    def summarize(self, text: str, section_name: str, domain: str) -> str:
        domain_instruction = {
            "medical": "Prioritize clinical outcomes, patient impact, and safety implications.",
            "legal": "Prioritize legal claims, statutory context, compliance, and judicial reasoning.",
            "govt": "Prioritize policy objective, implementation evidence, and public impact metrics.",
            "general": "Prioritize problem statement, method, key findings, and limitations.",
        }[domain]
        context = f"Domain={domain}; Section={section_name}; {domain_instruction}"
        return self.llm.summarize(" ".join(text.split()[:1400]), context)


class InteractiveResearchAssistant:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    def _retrieve(self, sections: Dict[str, Dict], query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        q_tokens = Counter(tokenize(query))
        scored = []
        for title, sec in sections.items():
            txt = sec.get("content", "")
            snippet = " ".join(txt.split()[:500])
            score = cosine_sim(q_tokens, Counter(tokenize(snippet)))
            scored.append((title, snippet, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    def answer_question(self, sections: Dict[str, Dict], question: str) -> str:
        retrieved = self._retrieve(sections, question)
        context = "\n".join(f"[{t}] {s}" for t, s, _ in retrieved)
        return self.llm.summarize(context, f"Answer question: {question}")

    def generate_contributions(self, sections: Dict[str, Dict]) -> str:
        text = "\n".join(sec.get("content", "") for title, sec in sections.items() if any(k in title.lower() for k in ["abstract", "introduction", "results", "conclusion"]))
        return self.llm.summarize(" ".join(text.split()[:2200]), "List key contributions in bullet-style prose")

    def generate_limitations(self, sections: Dict[str, Dict]) -> str:
        text = "\n".join(sec.get("content", "") for sec in sections.values())
        return self.llm.summarize(" ".join(text.split()[:2200]), "Infer limitations and risks based only on given text")

    def generate_future_work(self, sections: Dict[str, Dict]) -> str:
        text = "\n".join(sec.get("content", "") for sec in sections.values())
        return self.llm.summarize(" ".join(text.split()[:2200]), "Suggest realistic future work directions from this paper")


class MultiDocumentSummarizer:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    def summarize(self, papers: List[Dict]) -> Dict:
        if not papers:
            return {"combined_summary": "", "common_findings": [], "differences": [], "trends": []}

        all_keywords = []
        yearly = defaultdict(int)
        per_paper_highlights = []

        for paper in papers:
            meta = paper.get("metadata", {})
            sections = paper.get("sections", {})
            abstract = self._find_section_text(sections, ["abstract"])[:1200]
            results = self._find_section_text(sections, ["results", "conclusion", "discussion"])[:1200]
            highlight_source = f"Title: {meta.get('title', '')}\nAbstract: {abstract}\nFindings: {results}"
            highlight = self.llm.summarize(highlight_source, "Extract key findings in concise form")
            per_paper_highlights.append((meta.get("title", "Unknown"), highlight))

            all_text = " ".join(sec.get("content", "") for sec in sections.values())
            all_keywords.extend(top_keywords(all_text, n=25))
            year = str(meta.get("year") or "Unknown")
            yearly[year] += 1

        keyword_counts = Counter(all_keywords)
        common_findings = [k for k, v in keyword_counts.most_common(12) if v >= 2]

        diff_sentences = []
        for title, h in per_paper_highlights:
            diff_sentences.append(f"{title}: {h}")

        trends = [f"{yr}: {count} paper(s)" for yr, count in sorted(yearly.items())]
        combined_input = "\n".join(diff_sentences)
        combined_summary = self.llm.summarize(combined_input, "Create one integrated multi-document literature summary with common findings, differences, and trend interpretation")

        differences = diff_sentences[: min(5, len(diff_sentences))]

        return {
            "combined_summary": combined_summary,
            "common_findings": common_findings,
            "differences": differences,
            "trends": trends,
        }

    @staticmethod
    def _find_section_text(sections: Dict[str, Dict], key_terms: List[str]) -> str:
        for title, sec in sections.items():
            low = title.lower()
            if any(k in low for k in key_terms):
                return sec.get("content", "")
        return ""




class MediaSegmentationEvaluator:
    def _assignment_map(self, paper: Dict, key: str) -> Dict[str, List[str]]:
        assignments: Dict[str, List[str]] = defaultdict(list)
        sections = paper.get("sections", {})
        for sec_title, sec in sections.items():
            for media_id in sec.get(key, []):
                assignments[media_id].append(sec_title)
        return assignments

    def evaluate(self, paper: Dict) -> Dict:
        figures = paper.get("figures", [])
        tables = paper.get("tables", [])
        sections = paper.get("sections", {})

        fig_assign = self._assignment_map(paper, "figures")
        tab_assign = self._assignment_map(paper, "tables")

        fig_total = len(figures)
        tab_total = len(tables)
        media_total = fig_total + tab_total

        fig_assigned = sum(1 for f in figures if fig_assign.get(f.get("id", "")))
        tab_assigned = sum(1 for t in tables if tab_assign.get(t.get("id", "")))

        fig_coverage = (fig_assigned / fig_total) if fig_total else 0.0
        tab_coverage = (tab_assigned / tab_total) if tab_total else 0.0

        fig_caption_quality = (
            sum(1 for f in figures if (f.get("description") or "").strip()) / fig_total
            if fig_total else 0.0
        )

        table_preview_quality = 0.0
        if tab_total:
            parsed = 0
            for t in tables:
                prev = (t.get("preview") or "").strip().lower()
                if prev and "could not be parsed" not in prev:
                    parsed += 1
            table_preview_quality = parsed / tab_total

        def alignment_score(assignments: Dict[str, List[str]], ids: List[str]) -> float:
            if not ids:
                return 0.0
            scores = []
            for m_id in ids:
                n = len(assignments.get(m_id, []))
                if n == 0:
                    scores.append(0.0)
                else:
                    scores.append(1.0 / (1 + abs(n - 1)))
            return sum(scores) / len(scores)

        fig_alignment = alignment_score(fig_assign, [f.get("id", "") for f in figures])
        tab_alignment = alignment_score(tab_assign, [t.get("id", "") for t in tables])

        core_sections = 0
        total_assignments = 0
        for sec_title, sec in sections.items():
            low = sec_title.lower()
            count = len(sec.get("figures", [])) + len(sec.get("tables", []))
            total_assignments += count
            if any(k in low for k in ["abstract", "introduction", "method", "results", "discussion", "conclusion"]):
                core_sections += count
        media_in_core_sections = (core_sections / total_assignments) if total_assignments else 0.0

        pages = set()
        for sec in sections.values():
            pages.update(sec.get("pages", []))
        page_count = max(1, len(pages))
        media_density = media_total / page_count

        phase2_score = (
            0.25 * fig_coverage
            + 0.15 * tab_coverage
            + 0.2 * fig_caption_quality
            + 0.15 * table_preview_quality
            + 0.15 * ((fig_alignment + tab_alignment) / 2 if media_total else 0.0)
            + 0.1 * media_in_core_sections
        )

        per_item = []
        for f in figures:
            per_item.append({
                "id": f.get("id"),
                "type": "figure",
                "page": f.get("page"),
                "assigned_sections": fig_assign.get(f.get("id", ""), []),
                "description": f.get("description", ""),
            })
        for t in tables:
            per_item.append({
                "id": t.get("id"),
                "type": "table",
                "page": t.get("page"),
                "assigned_sections": tab_assign.get(t.get("id", ""), []),
                "description": (t.get("preview") or "")[:160],
            })

        return {
            "figure_count": fig_total,
            "table_count": tab_total,
            "media_total": media_total,
            "figure_coverage": round(fig_coverage, 4),
            "table_coverage": round(tab_coverage, 4),
            "figure_caption_quality": round(fig_caption_quality, 4),
            "table_preview_quality": round(table_preview_quality, 4),
            "figure_alignment": round(fig_alignment, 4),
            "table_alignment": round(tab_alignment, 4),
            "media_in_core_sections": round(media_in_core_sections, 4),
            "media_density_per_page": round(media_density, 4),
            "phase2_media_score": round(phase2_score, 4),
            "per_item": per_item,
        }

class ResearchPipeline:
    def __init__(self, use_local_model: bool = False):
        self.extractor = DocumentExtractor()
        self.llm = LLMService() if use_local_model else LLMService(model_path="__disable_local_model__.gguf")
        self.structure = StructureAwareSummarizer(self.llm)
        self.fact_checker = FactualConsistencyChecker()
        self.evaluator = EvaluationFramework()
        self.domain = DomainSpecificSummarizer(self.llm)
        self.assistant = InteractiveResearchAssistant(self.llm)
        self.multi = MultiDocumentSummarizer(self.llm)
        self.media_eval = MediaSegmentationEvaluator()

    def load_papers(self, pdf_paths: List[str], include_media: bool = False) -> List[Dict]:
        papers = []
        for path in pdf_paths:
            papers.append(self.extractor.parse_document(path, include_media=include_media))
        return papers

    def _reference_text(self, sections: Dict[str, Dict]) -> str:
        ref_parts = []
        for title, sec in sections.items():
            t = title.lower()
            if "abstract" in t or "conclusion" in t or "results" in t:
                ref_parts.append(" ".join(sec.get("content", "").split()[:350]))
        if not ref_parts:
            for sec in sections.values():
                ref_parts.append(" ".join(sec.get("content", "").split()[:350]))
                break
        return "\n".join(ref_parts)

    def run_single_paper_experiment(self, paper: Dict) -> Dict:
        sections = paper["sections"]
        meta = paper["metadata"]
        title = meta.get("title", "Untitled")

        def build_final(section_map: Dict[str, str], max_sections: int) -> Tuple[str, List[str]]:
            ranked = sorted(section_map.items(), key=lambda x: self.structure.section_importance(x[0], x[1]), reverse=True)
            selected = ranked[:max_sections]
            selected_titles = [x[0] for x in selected]
            combined = "\n".join(f"{sec_title}: {sec_text}" for sec_title, sec_text in selected)
            final_summary = self.llm.summarize(
                combined,
                "Create a coherent scientific summary covering objective, method, evidence, and conclusion",
            )
            return final_summary, selected_titles

        def coverage_score(selected_titles: List[str]) -> float:
            critical = ["abstract", "introduction", "method", "results", "conclusion"]
            low_titles = [t.lower() for t in selected_titles]
            hits = 0
            for key in critical:
                if any(key in t for t in low_titles):
                    hits += 1
            return round(hits / len(critical), 4)

        def graph_coherence(selected_titles: List[str], graph: Dict[str, List[Tuple[str, float]]]) -> float:
            if len(selected_titles) < 2:
                return 0.0
            pair_scores = []
            for i in range(len(selected_titles) - 1):
                src = selected_titles[i]
                dst = selected_titles[i + 1]
                candidates = dict(graph.get(src, []))
                pair_scores.append(candidates.get(dst, 0.0))
            if not pair_scores:
                return 0.0
            return round(sum(pair_scores) / len(pair_scores), 4)

        ranked_section_items = sorted(
            sections.items(),
            key=lambda x: self.structure.section_importance(x[0], x[1].get("content", "")),
            reverse=True,
        )
        candidate_limit = min(9, len(ranked_section_items))
        candidate_sections = {k: v for k, v in ranked_section_items[:candidate_limit]}

        t0 = time.time()
        baseline_sections = self.structure.summarize_independent(candidate_sections, title)
        baseline_final, baseline_titles = build_final(baseline_sections, max_sections=4)
        baseline_time = time.time() - t0

        t1 = time.time()
        structured_sections = self.structure.summarize_structure_aware(candidate_sections, title)
        structured_final, structured_titles = build_final(structured_sections, max_sections=candidate_limit)
        structured_time = time.time() - t1

        source_text = "\n".join(sec.get("content", "") for sec in sections.values())
        reference = self._reference_text(sections)
        graph = self.structure.build_section_graph(sections)

        baseline_metrics = self.evaluator.evaluate_summary(baseline_final, reference, source_text)
        structured_metrics = self.evaluator.evaluate_summary(structured_final, reference, source_text)

        baseline_metrics["section_coverage"] = coverage_score(baseline_titles)
        structured_metrics["section_coverage"] = coverage_score(structured_titles)
        baseline_metrics["graph_coherence"] = 0.0
        baseline_metrics["structure_signal"] = 0.0
        structured_metrics["graph_coherence"] = graph_coherence(structured_titles, graph)
        structured_metrics["structure_signal"] = structured_metrics["graph_coherence"]

        baseline_audit = self.fact_checker.audit(baseline_final, source_text)
        structured_audit = self.fact_checker.audit(structured_final, source_text)
        revised_structured = self.fact_checker.revise_summary(structured_audit)
        revised_metrics = self.evaluator.evaluate_summary(revised_structured, reference, source_text)
        revised_metrics["section_coverage"] = structured_metrics["section_coverage"]
        revised_metrics["graph_coherence"] = structured_metrics["graph_coherence"]
        revised_metrics["structure_signal"] = structured_metrics["structure_signal"]

        full_text = " ".join(sec.get("content", "") for sec in sections.values())
        detected_domain = self.domain.detect_domain(full_text)
        abstract_text = ""
        for s_title, sec in sections.items():
            if "abstract" in s_title.lower():
                abstract_text = sec.get("content", "")
                break
        domain_summary = self.domain.summarize(abstract_text or full_text[:2000], "Abstract", detected_domain)

        qa_answer = self.assistant.answer_question(sections, "What is the core contribution of this paper?")
        future_work = self.assistant.generate_future_work(sections)
        limitations = self.assistant.generate_limitations(sections)
        contributions = self.assistant.generate_contributions(sections)

        return {
            "metadata": meta,
            "baseline_summary": baseline_final,
            "structure_summary": structured_final,
            "fact_checked_summary": revised_structured,
            "baseline_metrics": {**baseline_metrics, "runtime_sec": round(baseline_time, 3)},
            "structure_metrics": {**structured_metrics, "runtime_sec": round(structured_time, 3)},
            "fact_checked_metrics": revised_metrics,
            "baseline_selected_sections": baseline_titles,
            "structure_selected_sections": structured_titles,
            "baseline_fact_flags": baseline_audit.flagged_sentences[:6],
            "structure_fact_flags": structured_audit.flagged_sentences[:6],
            "detected_domain": detected_domain,
            "domain_summary": domain_summary,
            "qa_answer": qa_answer,
            "future_work": future_work,
            "limitations": limitations,
            "contributions": contributions,
            "section_graph": graph,
            "human_eval_template": [
                {"criterion": "Informativeness", "baseline_score_1_to_5": "", "proposed_score_1_to_5": "", "notes": ""},
                {"criterion": "Factual consistency", "baseline_score_1_to_5": "", "proposed_score_1_to_5": "", "notes": ""},
                {"criterion": "Section coverage", "baseline_score_1_to_5": "", "proposed_score_1_to_5": "", "notes": ""},
                {"criterion": "Readability", "baseline_score_1_to_5": "", "proposed_score_1_to_5": "", "notes": ""},
            ],
        }



    def run_phase2_media_experiment(self, paper: Dict) -> Dict:
        baseline = {
            "figure_count": 0,
            "table_count": 0,
            "media_total": 0,
            "figure_coverage": 0.0,
            "table_coverage": 0.0,
            "figure_caption_quality": 0.0,
            "table_preview_quality": 0.0,
            "figure_alignment": 0.0,
            "table_alignment": 0.0,
            "media_in_core_sections": 0.0,
            "media_density_per_page": 0.0,
            "phase2_media_score": 0.0,
            "per_item": [],
        }
        proposed = self.media_eval.evaluate(paper)

        improvements = {}
        for key, val in proposed.items():
            if isinstance(val, (int, float)) and key in baseline:
                improvements[key] = round(val - baseline[key], 4)

        return {
            "baseline": baseline,
            "proposed": proposed,
            "improvement": improvements,
        }

    def run_multi_document_experiment(self, papers: List[Dict]) -> Dict:
        return self.multi.summarize(papers)


def pretty_metric_table(results: Dict) -> str:
    headers = ["Metric", "Baseline", "Structure-aware", "Fact-checked"]
    keys = ["rouge1_f1", "rouge2_f1", "rougeL_f1", "semantic_f1_proxy", "factual_score", "contradictions", "section_coverage", "graph_coherence", "structure_signal", "runtime_sec"]

    def pick(name: str, section: str):
        return results.get(section, {}).get(name, "-")

    rows = [headers]
    for key in keys:
        rows.append([
            key,
            str(pick(key, "baseline_metrics")),
            str(pick(key, "structure_metrics")),
            str(pick(key, "fact_checked_metrics")),
        ])

    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    out = []
    for idx, row in enumerate(rows):
        out.append(" | ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
        if idx == 0:
            out.append("-+-".join("-" * w for w in widths))
    return "\n".join(out)
