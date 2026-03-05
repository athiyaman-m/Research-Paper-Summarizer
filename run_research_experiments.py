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
    multi = pipeline.run_multi_document_experiment(papers if len(papers) > 1 else papers * 3)

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
    print("\nPhase 2 Media Metrics (proposed):")
    for key, val in phase2_media["proposed"].items():
        if key == "per_item":
            continue
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
