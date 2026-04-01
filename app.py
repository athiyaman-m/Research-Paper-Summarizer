import base64
import logging
import os
import tempfile
from html import escape

import streamlit as st

from pipeline import DocumentExtractor, LLMService, crop_figure


st.set_page_config(page_title="Research Paper Summarizer", layout="wide")


DEFAULT_LLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
APP_VERSION = "2.0.2"  # bump to bust @st.cache_resource when code changes
services = {}


# ── Config helpers ───────────────────────────────────────────────────────────

def resolve_runtime_config(model_name: str) -> dict:
    model = model_name.strip() or DEFAULT_LLAMA_MODEL
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        runtime_provider = "groq"
    elif os.getenv("OLLAMA_BASE_URL", "").strip():
        runtime_provider = "ollama"
    else:
        # Default to groq even if key is absent — LLMService will raise a clear error
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
    # require_llm=False so LLMService falls back gracefully; we surface errors in main()
    llm_kwargs = {"provider": provider, "require_llm": False}
    if provider == "ollama":
        llm_kwargs["ollama_model"] = config_signature[1]
    return {
        "extractor": DocumentExtractor(),
        "llm": LLMService(**llm_kwargs),
    }


def llm_init_help(provider: str) -> str:
    if provider == "groq":
        return (
            "⚠️ **Groq API key not found or invalid.** "
            "Go to your Streamlit Cloud app → **Settings → Secrets** and add:\n"
            '```toml\n[secrets]\nGROQ_API_KEY = "your-groq-api-key-here"\n```\n'
            "Get a free key at https://console.groq.com"
        )
    if provider == "ollama":
        return (
            "LLaMA initialization failed for Ollama mode. Set OLLAMA_BASE_URL + OLLAMA_MODEL in secrets, "
            "and keep local Ollama + tunnel running."
        )
    return "LLaMA initialization failed for local mode. Set a valid SUMMARIX_MODEL_PATH GGUF file."


def llm_status_text(llm) -> str:
    status_method = getattr(llm, "status_label", None)
    if callable(status_method):
        return status_method()
    check_health = getattr(llm, "check_health", None)
    if callable(check_health):
        return "Model loaded" if check_health() else "No active LLM backend"
    return "LLM status unavailable"


# ── CSS ──────────────────────────────────────────────────────────────────────

def render_styles():
    theme = st.session_state.get("app_theme", "light")
    is_dark = theme == "dark"
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
            --hero-bg:       {"linear-gradient(120deg, #1a1d27 0%, #252838 100%)" if is_dark else "linear-gradient(120deg, #f8fbff 0%, #eef4fb 100%)"};
            --table-even:    {"#1a1d27" if is_dark else "#f8fbff"};
            --table-header:  {"#252838" if is_dark else "#eef4fb"};
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
        .meta-row {{ display: grid; grid-template-columns: 160px 1fr; gap: 10px; margin-bottom: 10px; color: var(--text-primary); }}
        .meta-key {{ color: var(--text-muted); font-weight: 600; }}

        .section-info {{ display: flex; gap: 18px; color: var(--text-secondary); margin-bottom: 10px; font-size: 0.92rem; }}

        .panel {{ border: 1px solid var(--border); border-radius: 12px; padding: 14px; background: var(--bg-card); }}
        .panel-title {{ font-weight: 700; margin-bottom: 10px; color: var(--text-primary); }}

        .source-box {{ height: 440px; overflow-y: auto; line-height: 1.65; color: var(--text-primary); font-size: 0.95rem; white-space: normal; }}
        .summary-box {{ min-height: 220px; line-height: 1.7; color: var(--text-primary); background: var(--bg-panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px; margin-top: 12px; }}

        /* Comparison table */
        .cmp-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        .cmp-table th, .cmp-table td {{ border: 1px solid var(--border); padding: 10px 14px; text-align: left; font-size: 0.93rem; }}
        .cmp-table th {{ background: var(--table-header); font-weight: 700; color: var(--text-primary); }}
        .cmp-table td {{ color: var(--text-primary); }}
        .cmp-table tr:nth-child(even) td {{ background: var(--table-even); }}

        /* Citation list */
        .cite-item {{ padding: 10px 14px; border-bottom: 1px solid var(--border); line-height: 1.6; color: var(--text-primary); font-size: 0.93rem; }}
        .cite-num  {{ font-weight: 700; color: var(--accent); margin-right: 8px; }}

        /* Figure gallery */
        .fig-gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; margin-top: 12px; }}
        .fig-card {{ border: 1px solid var(--border); border-radius: 12px; padding: 12px; background: var(--bg-card); text-align: center; }}
        .fig-card img {{ max-width: 100%; border-radius: 8px; }}
        .fig-label {{ margin-top: 8px; font-weight: 600; color: var(--text-primary); font-size: 0.9rem; }}

        /* Theme toggle button */
        .theme-pill {{
            display: inline-flex; align-items: center; gap: 6px;
            padding: 6px 14px; border-radius: 20px; font-size: 0.82rem; font-weight: 600;
            background: var(--bg-panel); color: var(--text-secondary);
            border: 1px solid var(--border); cursor: pointer;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Section summary logic ───────────────────────────────────────────────────

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


# ── Parse / cache helpers ────────────────────────────────────────────────────

def parse_uploaded_files(uploaded_files):
    """Parse all uploaded PDFs and store results in session_state."""
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


# ── Render helpers ───────────────────────────────────────────────────────────

def render_metadata_card(metadata: dict, section_count: int, citation_count: int, fig_count: int, table_count: int):
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
            <div class="meta-row"><div class="meta-key">Figures / Tables</div><div>{fig_count} / {table_count}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sections(sections: dict, metadata: dict):
    titles = list(sections.keys())
    if not titles:
        st.warning("No sections detected in this document.")
        return

    if st.button("Summarize All Sections", key=f"sumall_{metadata.get('title','')}", use_container_width=True):
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
                f"""
                <div class="section-info">
                    <span><strong>Pages:</strong> {pages}</span>
                    <span><strong>Chunks:</strong> {chunk_count}</span>
                    <span><strong>Words:</strong> {word_count}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            left, right = st.columns([1.2, 1])

            with left:
                source = escape(section.get("content") or "No text found.").replace("\n", "<br/>")
                st.markdown(
                    f"""
                    <div class="panel">
                        <div class="panel-title"><i class="fa-regular fa-file-lines"></i> Original Content</div>
                        <div class="source-box">{source}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right:
                btn_key = f"sum_{metadata.get('title','')}_{title}"
                if st.button("Summarize This Section", key=btn_key, use_container_width=True):
                    with st.spinner(f"Summarizing: {title}"):
                        section_summary(title, section, metadata)

                doc_key = f"{metadata.get('title', '')}::{title}"
                summary = st.session_state.get("section_summaries", {}).get(doc_key, "")
                if summary:
                    rendered = escape(summary)
                else:
                    rendered = "Click <strong>Summarize This Section</strong> to generate summary."

                st.markdown(
                    f"""
                    <div class="panel">
                        <div class="panel-title"><i class="fa-solid fa-pen-ruler"></i> Section Summary</div>
                        <div class="summary-box">{rendered}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ── TAB: Paper Analysis ─────────────────────────────────────────────────────

def tab_paper_analysis(papers: dict):
    names = list(papers.keys())
    if not names:
        st.info("Upload one or more PDFs to begin.")
        return

    selected = st.selectbox("Select Paper", names, key="paper_selector")
    data = papers[selected]
    metadata = data.get("metadata", {})
    sections = data.get("sections", {})
    citations = data.get("citations", [])
    figures = data.get("figures", [])
    tables = data.get("tables", [])

    render_metadata_card(metadata, len(sections), len(citations), len(figures), len(tables))
    render_sections(sections, metadata)


# ── TAB: Comparative Analysis ────────────────────────────────────────────────

def tab_comparative(papers: dict):
    names = list(papers.keys())
    if len(names) < 2:
        st.info("Upload at least **2 papers** to enable comparative analysis.")
        return

    # Metadata comparison table
    st.subheader("📋 Paper Overview")
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
    st.markdown(f'<table class="cmp-table">{header}{rows}</table>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔬 AI Comparative Analysis")

    cache_key = "comparative_result"
    if st.button("Generate Comparative Analysis", key="cmp_btn", use_container_width=True):
        with st.spinner("Analyzing papers — this may take a moment..."):
            try:
                result = services["llm"].compare_papers(list(papers.values()))
                st.session_state[cache_key] = result
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")

    result = st.session_state.get(cache_key, "")
    if result:
        st.markdown(result)


# ── TAB: Survey Synthesis ────────────────────────────────────────────────────

def tab_survey(papers: dict):
    names = list(papers.keys())
    if not names:
        st.info("Upload papers first.")
        return

    st.subheader("📊 AI Survey Synthesis")
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


# ── TAB: Citations ───────────────────────────────────────────────────────────

def tab_citations(papers: dict):
    if not papers:
        st.info("Upload papers to extract citations.")
        return

    for name, data in papers.items():
        citations = data.get("citations", [])
        meta = data.get("metadata", {})
        title = meta.get("title", name)

        with st.expander(f"📚 {title} — {len(citations)} reference(s)", expanded=len(papers) == 1):
            if not citations:
                st.caption("No citations could be extracted from this paper.")
                continue

            html_items = ""
            for cite in citations:
                num = cite.get("number", "?")
                text = escape(cite.get("text", ""))
                html_items += f'<div class="cite-item"><span class="cite-num">[{num}]</span>{text}</div>'

            st.markdown(
                f'<div class="panel" style="max-height: 500px; overflow-y: auto;">{html_items}</div>',
                unsafe_allow_html=True,
            )


# ── TAB: Figures & Tables ───────────────────────────────────────────────────

def tab_figures_tables(papers: dict):
    if not papers:
        st.info("Upload papers to extract figures and tables.")
        return

    for name, data in papers.items():
        meta = data.get("metadata", {})
        title = meta.get("title", name)
        figures = data.get("figures", [])
        tables = data.get("tables", [])
        pdf_path = data.get("pdf_path", "")

        with st.expander(f"🖼️ {title} — {len(figures)} figure(s), {len(tables)} table(s)", expanded=len(papers) == 1):

            # Figures
            if figures:
                st.markdown("#### Figures")
                cols = st.columns(min(len(figures), 3))
                for i, fig in enumerate(figures):
                    with cols[i % len(cols)]:
                        coords = fig.get("coords")
                        if coords and pdf_path and os.path.exists(pdf_path):
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                    crop_figure(pdf_path, coords, tmp.name)
                                    with open(tmp.name, "rb") as f:
                                        img_bytes = f.read()
                                    if img_bytes:
                                        b64 = base64.b64encode(img_bytes).decode()
                                        st.markdown(
                                            f'<div class="fig-card">'
                                            f'<img src="data:image/png;base64,{b64}" alt="{fig.get("label", "Figure")}">'
                                            f'<div class="fig-label">{escape(fig.get("label", "Figure"))}</div>'
                                            f'<div style="font-size:0.82rem;color:#64748b;">{escape(fig.get("description", ""))}</div>'
                                            f'</div>',
                                            unsafe_allow_html=True,
                                        )
                            except Exception as exc:
                                st.caption(f"{fig.get('label', 'Figure')}: extraction failed — {exc}")
                        else:
                            st.caption(f"{fig.get('label', 'Figure')} — Page {fig.get('page', '?')}")
            else:
                st.caption("No figures detected.")

            # Tables
            if tables:
                st.markdown("#### Tables")
                for tab in tables:
                    preview = tab.get("preview", "No preview available.")
                    label = tab.get("label", "Table")
                    st.markdown(f"**{label}** (Page {tab.get('page', '?')})")
                    st.code(preview, language=None)
            else:
                st.caption("No tables detected.")


# ── Main app ─────────────────────────────────────────────────────────────────

def main():
    global services

    render_styles()

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📄 Research Paper Summarizer")
        st.markdown("Upload PDFs to analyze, compare, and summarize research papers.")

        # Theme toggle
        if "app_theme" not in st.session_state:
            st.session_state["app_theme"] = "light"
        theme_label = "🌙 Dark Mode" if st.session_state["app_theme"] == "light" else "☀️ Light Mode"
        if st.button(theme_label, key="theme_toggle", use_container_width=True):
            st.session_state["app_theme"] = "dark" if st.session_state["app_theme"] == "light" else "light"
            st.rerun()

        st.markdown("---")

        model_name = st.text_input(
            "LLM Model Name",
            value=DEFAULT_LLAMA_MODEL,
            help="Model tag for Ollama, or auto-detected when using Groq.",
        )

        runtime_config = resolve_runtime_config(model_name)
        runtime_provider = runtime_config["provider"]

        if runtime_provider == "groq":
            st.caption("Runtime: Groq API ☁️ (fast, no tunnel needed).")
        elif runtime_provider == "local":
            st.caption("Runtime: Local GGUF (OLLAMA_BASE_URL not set).")
        else:
            st.caption("Runtime: Ollama API (for deployed app).")

        uploaded_files = st.file_uploader(
            "Upload Papers (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
        )

        st.markdown("---")
        st.markdown(f"**Runtime Backend**: `{runtime_provider}`")

    # ── Services init ────────────────────────────────────────────────────
    config_signature = llm_config_signature(runtime_config)
    try:
        services = get_services(config_signature)
    except Exception as exc:
        st.error(llm_init_help(runtime_provider))
        st.exception(exc)
        return

    # Warn if LLM is in fallback mode
    llm = services.get("llm")
    if llm and getattr(llm, "mode", "") == "fallback":
        st.warning(llm_init_help(runtime_provider), icon="⚠️")
        return

    if st.session_state.get("llm_signature") != config_signature:
        st.session_state["llm_signature"] = config_signature

    with st.sidebar:
        st.markdown(f"**LLM**: {llm_status_text(services['llm'])}")
        st.markdown("**Section Parser**: Active")

    # ── Hero ─────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero">
            <h1>Multi-Document Research Paper Analyzer</h1>
            <p>Upload research papers to extract metadata, sections, citations, figures, and tables.
            Compare papers side-by-side and generate AI-powered survey syntheses.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not uploaded_files:
        st.info("Upload one or more PDFs from the sidebar to begin.")
        return

    # ── Parse all uploaded files ─────────────────────────────────────────
    # Only reset if the actual set of uploaded files has genuinely changed
    current_names = sorted(f.name for f in uploaded_files)
    prev_names = st.session_state.get("uploaded_names", [])
    files_changed = prev_names != current_names
    if files_changed:
        st.session_state["uploaded_names"] = current_names
        # Remove papers that are no longer in the uploaded set
        old_papers = st.session_state.get("papers", {})
        st.session_state["papers"] = {k: v for k, v in old_papers.items() if k in current_names}
        # Keep existing summaries and results — don't wipe them

    with st.spinner(f"Parsing {len(uploaded_files)} paper(s)..."):
        papers = parse_uploaded_files(uploaded_files)

    if not papers:
        st.error("No papers could be processed.")
        return

    # Update sidebar with paper count
    with st.sidebar:
        st.markdown(f"**Papers Loaded**: {len(papers)}")
        for name, data in papers.items():
            m = data.get("metadata", {})
            st.caption(f"• {m.get('title', name)[:50]}")

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_labels = ["📄 Paper Analysis", "🔬 Comparative", "📊 Survey", "📚 Citations", "🖼️ Figures & Tables"]
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
