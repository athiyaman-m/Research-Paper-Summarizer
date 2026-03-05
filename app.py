import logging
import os
import tempfile
from html import escape

import streamlit as st

from pipeline import DocumentExtractor, LLMService


st.set_page_config(page_title="Research Paper Summarizer", layout="wide")


LLM_PROVIDER_OPTIONS = ("LLaMA", "Gemini", "Groq")
LLM_DEFAULT_MODELS = {
    "LLaMA": os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct"),
    "Gemini": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    "Groq": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
}

services = {}


def resolve_provider_config(provider_label: str, model_name: str) -> dict:
    model = model_name.strip() or LLM_DEFAULT_MODELS.get(provider_label, "")
    if provider_label == "LLaMA":
        # Prefer Ollama when configured; otherwise use local GGUF runtime.
        runtime_provider = "ollama" if os.getenv("OLLAMA_BASE_URL", "").strip() else "local"
        return {"provider": runtime_provider, "label": provider_label, "model": model}
    if provider_label == "Gemini":
        return {"provider": "gemini", "label": provider_label, "model": model}
    if provider_label == "Groq":
        return {"provider": "groq", "label": provider_label, "model": model}
    return {"provider": "local", "label": "LLaMA", "model": LLM_DEFAULT_MODELS["LLaMA"]}


def llm_config_signature(config: dict) -> tuple:
    return (
        config.get("provider", ""),
        config.get("label", ""),
        config.get("model", ""),
        os.getenv("OLLAMA_BASE_URL", "").strip(),
        os.getenv("SUMMARIX_MODEL_PATH", ""),
        "1" if bool(os.getenv("OPENAI_API_KEY")) else "0",
        "1" if bool(os.getenv("GEMINI_API_KEY")) else "0",
        "1" if bool(os.getenv("GROQ_API_KEY")) else "0",
        os.getenv("SUMMARIX_REQUIRE_LLM", "").strip().lower(),
    )


@st.cache_resource
def get_services(config_signature: tuple):
    provider, _label, model = config_signature[:3]
    llm_kwargs = {"provider": provider, "require_llm": True}

    if provider == "ollama":
        llm_kwargs["ollama_model"] = model
    elif provider == "gemini":
        llm_kwargs["gemini_model"] = model
    elif provider == "groq":
        llm_kwargs["groq_model"] = model

    return {
        "extractor": DocumentExtractor(),
        "llm": LLMService(**llm_kwargs),
    }


def llm_init_help(provider: str) -> str:
    if provider == "ollama":
        return (
            "LLM initialization failed for LLaMA (Ollama). Set OLLAMA_BASE_URL + OLLAMA_MODEL "
            "in Streamlit secrets, and keep your Ollama+tunnel process running."
        )
    if provider == "local":
        return "LLM initialization failed for local LLaMA. Set a valid SUMMARIX_MODEL_PATH to a GGUF file."
    if provider == "gemini":
        return "LLM initialization failed for Gemini. Set GEMINI_API_KEY (and optional GEMINI_MODEL)."
    if provider == "groq":
        return "LLM initialization failed for Groq. Set GROQ_API_KEY (and optional GROQ_MODEL)."
    return "LLM initialization failed. Check your provider secrets and model settings."


def llm_status_text(llm) -> str:
    status_method = getattr(llm, "status_label", None)
    if callable(status_method):
        return status_method()

    has_local_model = getattr(llm, "has_local_model", None)
    if callable(has_local_model):
        return "Local model loaded" if has_local_model() else "No active LLM backend"

    check_health = getattr(llm, "check_health", None)
    if callable(check_health):
        return "Local model loaded" if check_health() else "No active LLM backend"

    return "LLM status unavailable"


def render_styles():
    st.markdown(
        """
        <style>
        @import url("https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;700&display=swap");
        @import url("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css");

        html, body, [class*="css"] {
            font-family: "Public Sans", sans-serif;
        }

        .hero {
            border: 1px solid #dbe4f0;
            background: linear-gradient(120deg, #f8fbff 0%, #eef4fb 100%);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 18px;
        }

        .hero h1 {
            margin: 0;
            color: #0f172a;
            font-size: 1.8rem;
        }

        .hero p {
            margin: 8px 0 0 0;
            color: #334155;
        }

        .meta-card {
            border: 1px solid #dbe4f0;
            border-radius: 14px;
            background: #ffffff;
            padding: 18px;
            margin-bottom: 16px;
        }

        .meta-row {
            display: grid;
            grid-template-columns: 160px 1fr;
            gap: 10px;
            margin-bottom: 10px;
            color: #0f172a;
        }

        .meta-key {
            color: #475569;
            font-weight: 600;
        }

        .section-info {
            display: flex;
            gap: 18px;
            color: #334155;
            margin-bottom: 10px;
            font-size: 0.92rem;
        }

        .panel {
            border: 1px solid #dbe4f0;
            border-radius: 12px;
            padding: 14px;
            background: #ffffff;
        }

        .panel-title {
            font-weight: 700;
            margin-bottom: 10px;
            color: #0f172a;
        }

        .source-box {
            height: 440px;
            overflow-y: auto;
            line-height: 1.65;
            color: #1e293b;
            font-size: 0.95rem;
            white-space: normal;
        }

        .summary-box {
            min-height: 220px;
            line-height: 1.7;
            color: #0f172a;
            background: #f4f8fe;
            border: 1px solid #dbe4f0;
            border-radius: 10px;
            padding: 12px;
            margin-top: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_summary(title: str, section: dict, metadata: dict) -> str:
    cache = st.session_state.setdefault("section_summaries", {})
    if title in cache:
        return cache[title]

    text = " ".join(section.get("chunks", [])[:3]) or section.get("content", "")
    context = f"{metadata.get('title', 'Paper')} -> {title}"

    try:
        summary = services["llm"].summarize(text, context)
    except Exception as exc:
        msg = str(exc)
        lowered = msg.lower()
        active_mode = getattr(services.get("llm"), "mode", "provider")
        if "429" in lowered or "quota" in lowered or "rate" in lowered:
            msg = f"{active_mode} quota/rate-limit reached. Choose another provider from the sidebar."
        summary = f"LLM error: {msg}"

    cache[title] = summary
    return summary


def summarize_all_sections(sections: dict, metadata: dict):
    for title, section in sections.items():
        section_summary(title, section, metadata)


def reset_state_for_new_file(file_name: str):
    if st.session_state.get("doc_file") != file_name:
        st.session_state["doc_file"] = file_name
        st.session_state["section_summaries"] = {}


def render_metadata(metadata: dict, section_count: int):
    title = escape(metadata.get("title") or "Untitled Document")
    authors = escape(metadata.get("authors") or "Unknown Authors")
    year = escape(str(metadata.get("year") or "Not available"))

    st.markdown(
        f"""
        <div class="meta-card">
            <div class="meta-row"><div class="meta-key">Title</div><div>{title}</div></div>
            <div class="meta-row"><div class="meta-key">Authors</div><div>{authors}</div></div>
            <div class="meta-row"><div class="meta-key">Publication Year</div><div>{year}</div></div>
            <div class="meta-row"><div class="meta-key">Sections Found</div><div>{section_count}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sections(sections: dict, metadata: dict):
    titles = list(sections.keys())
    if not titles:
        st.warning("No sections detected in this document.")
        return

    if st.button("Summarize All Sections", use_container_width=True):
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
                source = escape(section.get("content") or "No text found for this section.").replace("\n", "<br/>")
                st.markdown(
                    f"""
                    <div class="panel">
                        <div class="panel-title"><i class="fa-regular fa-file-lines"></i> Original Section Content</div>
                        <div class="source-box">{source}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right:
                if st.button("Summarize This Section", key=f"sum_{title}", use_container_width=True):
                    with st.spinner(f"Summarizing section: {title}"):
                        section_summary(title, section, metadata)

                summary = st.session_state.get("section_summaries", {}).get(title, "")
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


def main():
    global services

    render_styles()

    env_provider = os.getenv("SUMMARIX_LLM_PROVIDER", "").strip().lower()
    env_to_label = {
        "ollama": "LLaMA",
        "local": "LLaMA",
        "gemini": "Gemini",
        "groq": "Groq",
    }
    default_label = env_to_label.get(env_provider, "LLaMA")
    default_index = LLM_PROVIDER_OPTIONS.index(default_label)

    with st.sidebar:
        st.markdown("### Research Paper Summarizer")
        st.markdown("Upload a PDF, choose LLM provider, and generate section-wise summaries.")

        provider_label = st.selectbox("LLM Provider", LLM_PROVIDER_OPTIONS, index=default_index)

        model_name = st.text_input(
            "Model Name",
            value=LLM_DEFAULT_MODELS.get(provider_label, ""),
            help="Set exact model ID for the selected provider.",
        )

        provider_config = resolve_provider_config(provider_label, model_name)

        if provider_config["provider"] == "local":
            st.caption("LLaMA is using local GGUF backend because OLLAMA_BASE_URL is not configured.")
        elif provider_config["provider"] == "ollama":
            st.caption("LLaMA is using Ollama API backend.")

        uploaded_file = st.file_uploader("Upload Paper (PDF)", type=["pdf"])

        st.markdown("---")
        st.markdown(f"**Selected Provider**: `{provider_label}`")
        st.markdown(f"**Runtime Backend**: `{provider_config['provider']}`")

    config_signature = llm_config_signature(provider_config)
    try:
        services = get_services(config_signature)
    except Exception as exc:
        st.error(llm_init_help(provider_config["provider"]))
        st.exception(exc)
        return

    if st.session_state.get("llm_signature") != config_signature:
        st.session_state["llm_signature"] = config_signature
        st.session_state["section_summaries"] = {}

    with st.sidebar:
        st.markdown(f"**LLM**: {llm_status_text(services['llm'])}")
        st.markdown("**Section Parser**: Active")

    st.markdown(
        """
        <div class="hero">
            <h1>Section Aware Research Paper Summarization</h1>
            <p>Upload any research paper. The app identifies metadata and sections from Abstract to References,
            separates content by section, and lets you summarize each section on demand.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not uploaded_file:
        st.info("Upload a PDF to begin.")
        return

    reset_state_for_new_file(uploaded_file.name)

    with st.spinner("Extracting metadata and sections from PDF"):
        if st.session_state.get("parsed_file") != uploaded_file.name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                pdf_path = tmp.name

            try:
                data = services["extractor"].parse_document(pdf_path, include_media=False)
            except Exception as exc:
                logging.error("Document parsing failed: %s", exc, exc_info=True)
                st.error("Unable to process this PDF. Please try another file.")
                return

            st.session_state["parsed_file"] = uploaded_file.name
            st.session_state["doc_data"] = data

    data = st.session_state.get("doc_data")
    if not data:
        st.error("Parsing failed. Please re upload the file.")
        return

    metadata = data.get("metadata", {})
    sections = data.get("sections", {})

    render_metadata(metadata, len(sections))
    render_sections(sections, metadata)


if __name__ == "__main__":
    main()
