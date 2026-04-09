from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_RESULTS_PATH = Path("outputs/rigorous_smoke/aggregate_results.json")
DEFAULT_OUTPUT_DIR = Path("outputs/rigorous_smoke")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_results(path: Path) -> pd.DataFrame:
    rows = load_json(path)
    return pd.DataFrame(rows)


def get_cli_defaults() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args, _ = parser.parse_known_args()
    return Path(args.results_path), Path(args.output_dir)


def list_prediction_files(output_dir: Path, dataset_name: str) -> list[Path]:
    dataset_dir = output_dir / dataset_name
    if not dataset_dir.exists():
        return []
    return sorted(dataset_dir.glob("*_predictions.json"))


def render_architecture_tab() -> None:
    st.subheader("RAG Architecture Options")
    st.markdown(
        """
        **Traditional RAG**: retrieves chunks and inserts them into the prompt.

        **Advanced RAG**: retrieves, filters, reranks, compresses context, and answers more precisely.

        **Multi-Agent RAG**: distributes work across specialized agents, for example planner, retriever,
        synthesizer, and evaluator.

        **Agentic RAG**: a single agent decides the next step itself, calls tools, checks results, and
        replans without a separate supervisor.
        """
    )
    st.info(
        "For this project, the practical upgrade path is Advanced RAG first, then Multi-Agent RAG. "
        "Agentic RAG is intentionally out of scope because your current architecture targets Flowise-style "
        "pipelines rather than LangGraph-style autonomous agents."
    )
    st.markdown(
        """
        **Recommended stack**

        1. `SCC`: host `vLLM` with `Qwen/Qwen2.5-7B-Instruct`
        2. `OpenWebUI`: chat frontend for model testing and demos
        3. `Dashboard`: inspect retrieval quality, chunking tradeoffs, and example-level outputs
        4. `Next`: add query rewriting, reranking, context compression, citations, then multi-agent orchestration
        """
    )


def render_results_tab(results: pd.DataFrame) -> None:
    st.subheader("Aggregate Results")
    if results.empty:
        st.warning("No aggregate results loaded.")
        return

    datasets = sorted(results["dataset"].dropna().unique().tolist())
    dataset_name = st.selectbox("Dataset", datasets, key="dataset_filter")
    filtered = results[results["dataset"] == dataset_name].copy()

    if "retriever" in filtered.columns:
        retrievers = ["all"] + sorted(filtered["retriever"].dropna().unique().tolist())
        retriever_name = st.selectbox("Retriever", retrievers, key="retriever_filter")
        if retriever_name != "all":
            filtered = filtered[filtered["retriever"] == retriever_name]

    metric_candidates = [column for column in ("f1_mean", "exact_match_mean", "supporting_doc_coverage_mean", "recall_at_k_mean") if column in filtered.columns]
    metric_name = st.selectbox("Metric", metric_candidates, key="metric_filter")

    chart_frame = filtered.set_index("system")[[metric_name]].sort_values(metric_name, ascending=False)
    st.bar_chart(chart_frame)
    st.dataframe(filtered.sort_values(metric_name, ascending=False), use_container_width=True)


def render_predictions_tab(output_dir: Path) -> None:
    st.subheader("Prediction Browser")
    dataset_options = [dataset_dir.name for dataset_dir in output_dir.iterdir() if dataset_dir.is_dir() and dataset_dir.name != "figures"] if output_dir.exists() else []
    if not dataset_options:
        st.warning("No prediction directories found for this output directory.")
        return

    dataset_name = st.selectbox("Prediction dataset", sorted(dataset_options), key="pred_dataset")
    prediction_files = list_prediction_files(output_dir, dataset_name)
    if not prediction_files:
        st.warning("No prediction files found.")
        return

    system_options = [path.stem.replace("_predictions", "") for path in prediction_files]
    system_name = st.selectbox("System", system_options, key="pred_system")
    prediction_path = output_dir / dataset_name / f"{system_name}_predictions.json"
    rows = load_json(prediction_path)
    if not rows:
        st.warning("This prediction file is empty.")
        return

    questions = [f"{index}: {row['question']}" for index, row in enumerate(rows)]
    selection = st.selectbox("Example", questions, key="pred_example")
    index = int(selection.split(":", 1)[0])
    row = rows[index]

    st.markdown(f"**Question:** {row['question']}")
    st.markdown(f"**Prediction:** {row['prediction']}")
    st.markdown(f"**Gold Answers:** {', '.join(row.get('gold_answers', []))}")
    st.markdown(
        f"**EM / F1:** {row.get('exact_match', 0.0):.3f} / {row.get('f1', 0.0):.3f}"
    )
    if row.get("retrieved_titles"):
        st.markdown("**Retrieved Titles:**")
        for title in row["retrieved_titles"]:
            st.write(f"- {title}")


def render_figures_tab(output_dir: Path) -> None:
    st.subheader("Generated Figures")
    figure_dir = output_dir / "figures"
    if not figure_dir.exists():
        st.warning("No figures directory found.")
        return
    figure_paths = sorted(figure_dir.glob("*.png"))
    if not figure_paths:
        st.warning("No figure images found.")
        return

    for figure_path in figure_paths:
        st.markdown(f"**{figure_path.name}**")
        st.image(str(figure_path), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="ChunkRAG Demo Dashboard", layout="wide")
    st.title("ChunkRAG Demo Dashboard")
    st.caption("Visual analysis for chunking, retrieval, and deployment planning.")

    default_results_path, default_output_dir = get_cli_defaults()

    results_path_str = st.sidebar.text_input("Aggregate results path", str(default_results_path))
    output_dir_str = st.sidebar.text_input("Output directory", str(default_output_dir))

    results_path = Path(results_path_str)
    output_dir = Path(output_dir_str)

    if not results_path.exists():
        st.error(f"Results file not found: {results_path}")
        return

    results = load_results(results_path)
    architecture_tab, results_tab, predictions_tab, figures_tab = st.tabs(
        ["Architecture", "Results", "Predictions", "Figures"]
    )

    with architecture_tab:
        render_architecture_tab()
    with results_tab:
        render_results_tab(results)
    with predictions_tab:
        render_predictions_tab(output_dir)
    with figures_tab:
        render_figures_tab(output_dir)


if __name__ == "__main__":
    main()
