from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from chunkrag.evaluation import answer_metrics
from chunkrag.generation import ExtractiveFallbackGenerator, OpenAICompatibleGenerator, QAGenerator, resolve_device
from chunkrag.live_rag import UploadPayload, build_demo_index, load_documents_from_uploads, run_live_rag


DEFAULT_RESULTS_PATH = Path("outputs/rigorous_smoke/aggregate_results.json")
DEFAULT_OUTPUT_DIR = Path("outputs/rigorous_smoke")
DEFAULT_ENDPOINT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ENDPOINT_URL = "http://127.0.0.1:8000/v1"
DEFAULT_ENDPOINT_KEY = "chunkrag-demo-key"
BUILTIN_CORPORA = {
    "README": Path("README.md"),
    "RAG roadmap": Path("docs/rag_roadmap.md"),
    "OpenWebUI deployment guide": Path("docs/openwebui_scc_deployment.md"),
    "Final report": Path("reports/final_report.md"),
}


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


def build_builtin_payloads(labels: list[str]) -> list[UploadPayload]:
    payloads: list[UploadPayload] = []
    for label in labels:
        path = BUILTIN_CORPORA[label]
        if not path.exists():
            continue
        payloads.append(UploadPayload(name=path.name, data=path.read_bytes()))
    return payloads


def build_payload_signature(payloads: list[UploadPayload], settings: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for payload in sorted(payloads, key=lambda item: item.name):
        digest.update(payload.name.encode("utf-8"))
        digest.update(payload.data)
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


@st.cache_resource(show_spinner=False)
def get_local_generator(model_name: str, device: str, max_input_tokens: int, max_new_tokens: int):
    return QAGenerator(
        model_name=model_name,
        device=device,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
    )


@st.cache_resource(show_spinner=False)
def get_endpoint_generator(
    model_name: str,
    base_url: str,
    api_key: str,
    max_input_tokens: int,
    max_new_tokens: int,
):
    return OpenAICompatibleGenerator(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
    )


@st.cache_resource(show_spinner=False)
def get_fallback_generator():
    return ExtractiveFallbackGenerator()


def render_architecture_tab() -> None:
    st.subheader("RAG Architecture Upgrade Path")
    card_one, card_two, card_three = st.columns(3)
    with card_one:
        st.markdown(
            """
            ### Traditional RAG
            - retrieve chunks
            - pack them into the prompt
            - answer directly
            """
        )
    with card_two:
        st.markdown(
            """
            ### Advanced RAG
            - rewrite and expand the query
            - hybrid retrieve and rerank
            - compress context before generation
            """
        )
    with card_three:
        st.markdown(
            """
            ### Multi-Agent RAG
            - planner proposes subqueries
            - retriever gathers evidence
            - filter agent keeps diverse support
            - writer and evaluator finalize
            """
        )

    st.info(
        "This project intentionally stops before Agentic RAG. The current architecture is better matched to "
        "controllable Advanced and Multi-Agent pipelines than to a LangGraph-style autonomous agent."
    )
    st.markdown(
        """
        **What is improved now**

        - upload your own PDF, TXT, Markdown, CSV, TSV, or JSON knowledge base
        - compare `Traditional`, `Advanced`, and `Multi-Agent` flows on the same question
        - switch between an extractive fallback, a local Hugging Face model, or an SCC-hosted OpenAI-compatible model
        - inspect query rewrites, citations, compressed evidence, and grounding estimates
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

    metric_candidates = [
        column
        for column in ("f1_mean", "exact_match_mean", "supporting_doc_coverage_mean", "recall_at_k_mean")
        if column in filtered.columns
    ]
    metric_name = st.selectbox("Metric", metric_candidates, key="metric_filter")

    chart_frame = filtered.set_index("system")[[metric_name]].sort_values(metric_name, ascending=False)
    st.bar_chart(chart_frame)
    st.dataframe(filtered.sort_values(metric_name, ascending=False), use_container_width=True)


def render_predictions_tab(output_dir: Path) -> None:
    st.subheader("Prediction Browser")
    dataset_options = [
        dataset_dir.name
        for dataset_dir in output_dir.iterdir()
        if dataset_dir.is_dir() and dataset_dir.name != "figures"
    ] if output_dir.exists() else []
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
    st.markdown(f"**EM / F1:** {row.get('exact_match', 0.0):.3f} / {row.get('f1', 0.0):.3f}")
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


def _resolve_generator(
    backend: str,
    *,
    device: str,
    model_name: str,
    endpoint_url: str,
    endpoint_key: str,
    max_input_tokens: int,
    max_new_tokens: int,
):
    if backend == "extractive_fallback":
        return get_fallback_generator()
    if backend == "huggingface_local":
        return get_local_generator(model_name, device, max_input_tokens, max_new_tokens)
    return get_endpoint_generator(model_name, endpoint_url, endpoint_key, max_input_tokens, max_new_tokens)


def render_playground_tab() -> None:
    st.subheader("Live Playground")
    st.caption("Upload your own corpus, choose the stack, and compare grounded answers with traces.")

    config_col, run_col = st.columns([1.05, 1.6], gap="large")

    with config_col:
        st.markdown("### Corpus")
        uploaded_files = st.file_uploader(
            "Upload knowledge files",
            accept_multiple_files=True,
            type=["pdf", "txt", "md", "markdown", "rst", "csv", "tsv", "json"],
        )
        builtin_labels = st.multiselect(
            "Include built-in project documents",
            list(BUILTIN_CORPORA),
            default=["README", "RAG roadmap"],
        )
        pasted_corpus = st.text_area(
            "Or paste a small knowledge base",
            height=140,
            placeholder="Paste notes, abstracts, or system design text here.",
        )

        st.markdown("### Retrieval stack")
        embedding_model = st.text_input(
            "Embedding model",
            value="sentence-transformers/all-MiniLM-L6-v2",
        )
        chunker_type = st.selectbox("Chunker", ["recursive", "semantic", "sentence", "fixed"], index=0)
        chunk_size = st.slider("Chunk size", min_value=96, max_value=768, value=256, step=32)
        chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=128, value=48, step=8)
        similarity_threshold = st.slider(
            "Semantic similarity threshold",
            min_value=0.45,
            max_value=0.95,
            value=0.72,
            step=0.01,
            disabled=chunker_type != "semantic",
        )
        retriever_name = st.selectbox("Retriever", ["hybrid", "hybrid_rerank", "dense", "bm25"], index=0)
        device_choice = st.selectbox("Embedding / local model device", ["auto", "cpu", "mps", "cuda"], index=0)

        st.markdown("### Generation backend")
        backend = st.selectbox(
            "Generator",
            ["extractive_fallback", "huggingface_local", "openai_compatible"],
            format_func=lambda value: {
                "extractive_fallback": "Extractive fallback",
                "huggingface_local": "Local Hugging Face model",
                "openai_compatible": "OpenAI-compatible endpoint",
            }[value],
        )
        local_model_name = st.text_input("Generator model", value="google/flan-t5-base")
        endpoint_url = st.text_input("Endpoint URL", value=DEFAULT_ENDPOINT_URL, disabled=backend != "openai_compatible")
        endpoint_key = st.text_input(
            "Endpoint API key",
            value=DEFAULT_ENDPOINT_KEY,
            type="password",
            disabled=backend != "openai_compatible",
        )
        endpoint_model = st.text_input(
            "Endpoint model ID",
            value=DEFAULT_ENDPOINT_MODEL,
            disabled=backend != "openai_compatible",
        )
        max_input_tokens = st.slider("Max input tokens", min_value=256, max_value=2048, value=768, step=64)
        max_new_tokens = st.slider("Max new tokens", min_value=16, max_value=256, value=96, step=16)

        payloads = [UploadPayload(name=file.name, data=file.getvalue()) for file in uploaded_files]
        payloads.extend(build_builtin_payloads(builtin_labels))
        if pasted_corpus.strip():
            payloads.append(UploadPayload(name="pasted_notes.txt", data=pasted_corpus.encode("utf-8")))

        index_settings = {
            "embedding_model": embedding_model,
            "chunker_type": chunker_type,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "similarity_threshold": round(similarity_threshold, 3),
            "retriever_name": retriever_name,
            "device_choice": device_choice,
        }
        signature = build_payload_signature(payloads, index_settings) if payloads else None
        stored_signature = st.session_state.get("playground_signature")
        if signature and stored_signature and signature != stored_signature:
            st.info("The corpus or retrieval settings changed. Rebuild the playground index to apply them.")

        if st.button("Build or rebuild playground index", use_container_width=True):
            if not payloads:
                st.error("Add at least one uploaded file, built-in document, or pasted corpus.")
            else:
                try:
                    documents = load_documents_from_uploads(payloads)
                    if not documents:
                        st.error("The uploaded sources did not produce any readable documents.")
                    else:
                        with st.spinner("Building chunks and retrieval index..."):
                            index = build_demo_index(
                                documents,
                                embedding_model=embedding_model,
                                chunker_type=chunker_type,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                similarity_threshold=similarity_threshold,
                                retriever_name=retriever_name,
                                retrieval_top_k=4,
                                device=resolve_device(device_choice),
                            )
                        st.session_state["playground_index"] = index
                        st.session_state["playground_signature"] = signature
                        st.session_state["playground_documents"] = documents
                        st.success("Playground index is ready.")
                except Exception as exc:  # pragma: no cover - UI guardrail
                    st.error(f"Could not build the playground index: {exc}")

    with run_col:
        index = st.session_state.get("playground_index")
        documents = st.session_state.get("playground_documents", [])

        if index is None:
            st.warning("Build the playground index on the left to start asking questions.")
        else:
            metric_one, metric_two, metric_three, metric_four = st.columns(4)
            metric_one.metric("Documents", len(index.documents))
            metric_two.metric("Chunks", len(index.chunks))
            metric_three.metric("Chunker", index.chunker_spec["type"])
            metric_four.metric("Retriever", index.retriever_spec["name"])

            with st.expander("Corpus preview", expanded=False):
                preview_rows = [
                    {
                        "title": document.title,
                        "doc_id": document.doc_id,
                        "characters": len(document.text),
                    }
                    for document in documents[:24]
                ]
                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

        question = st.text_area(
            "Question",
            height=110,
            placeholder="Ask a question about the uploaded corpus.",
        )
        reference_answer = st.text_input(
            "Optional gold/reference answer",
            placeholder="If you provide this, the app will compute EM and F1 for the live answer.",
        )
        modes = st.multiselect(
            "Modes to compare",
            ["traditional", "advanced", "multi_agent"],
            default=["traditional", "advanced", "multi_agent"],
            format_func=lambda value: {
                "traditional": "Traditional RAG",
                "advanced": "Advanced RAG",
                "multi_agent": "Multi-Agent RAG",
            }[value],
        )
        top_k = st.slider("Top-k evidence chunks", min_value=2, max_value=8, value=4, step=1)
        per_query_k = st.slider("Per-query candidate depth", min_value=4, max_value=16, value=8, step=2)
        compression_token_budget = st.slider(
            "Compression token budget",
            min_value=128,
            max_value=1024,
            value=384,
            step=64,
        )

        if st.button("Run comparison", type="primary", use_container_width=True):
            if index is None:
                st.error("Build the playground index first.")
            elif not question.strip():
                st.error("Enter a question first.")
            elif not modes:
                st.error("Select at least one mode to compare.")
            else:
                generator_model_name = endpoint_model if backend == "openai_compatible" else local_model_name
                try:
                    with st.spinner("Loading generator backend..."):
                        generator = _resolve_generator(
                            backend,
                            device=resolve_device(device_choice),
                            model_name=generator_model_name,
                            endpoint_url=endpoint_url,
                            endpoint_key=endpoint_key,
                            max_input_tokens=max_input_tokens,
                            max_new_tokens=max_new_tokens,
                        )
                    run_results = {}
                    for mode in modes:
                        with st.spinner(f"Running {mode}..."):
                            run_result = run_live_rag(
                                index,
                                question=question,
                                generator=generator,
                                mode=mode,
                                top_k=top_k,
                                per_query_k=per_query_k,
                                compression_token_budget=compression_token_budget,
                            )
                        if reference_answer.strip():
                            run_result["live_metrics"] = answer_metrics(run_result["answer"], [reference_answer.strip()])
                        run_results[mode] = run_result
                    st.session_state["playground_runs"] = run_results
                except Exception as exc:  # pragma: no cover - UI guardrail
                    st.error(f"Could not run the comparison: {exc}")

        run_results = st.session_state.get("playground_runs", {})
        if run_results:
            mode_tabs = st.tabs(
                [
                    {
                        "traditional": "Traditional",
                        "advanced": "Advanced",
                        "multi_agent": "Multi-Agent",
                    }[mode]
                    for mode in run_results
                ]
            )
            for tab, (mode, result) in zip(mode_tabs, run_results.items(), strict=True):
                with tab:
                    summary_left, summary_right, summary_third = st.columns(3)
                    summary_left.metric("Grounding support", result["support_label"].title())
                    summary_right.metric("Elapsed", f"{result['elapsed_seconds']:.2f}s")
                    summary_third.metric("Evidence docs", len({chunk['doc_id'] for chunk in result["selected_chunks"]}))

                    st.markdown("### Answer")
                    st.write(result["answer"])

                    if "live_metrics" in result:
                        metric_frame = pd.DataFrame([result["live_metrics"]])
                        st.markdown("### Live evaluation")
                        st.dataframe(metric_frame, use_container_width=True)

                    info_one, info_two = st.columns(2)
                    with info_one:
                        st.markdown("### Query flow")
                        st.write({"rewrites": result["query_variants"], "subqueries": result["subqueries"]})
                    with info_two:
                        st.markdown("### Citations")
                        st.dataframe(pd.DataFrame(result["citations"]), use_container_width=True)

                    st.markdown("### Trace")
                    trace_frame = pd.DataFrame(result["trace"])
                    st.dataframe(trace_frame, use_container_width=True)
                    if not trace_frame.empty:
                        st.bar_chart(trace_frame.set_index("stage")[["seconds"]])

                    st.markdown("### Evidence")
                    for index_number, chunk in enumerate(result["selected_chunks"], start=1):
                        label = f"[{index_number}] {chunk['title']} | score={chunk['score']:.3f}"
                        with st.expander(label, expanded=False):
                            st.markdown(f"**Matched queries:** {', '.join(chunk['matched_queries'])}")
                            st.markdown(f"**Support score:** {chunk['support_score']:.3f}")
                            if chunk["compressed_text"]:
                                st.markdown("**Compressed evidence**")
                                st.write(chunk["compressed_text"])
                            st.markdown("**Full chunk**")
                            st.write(chunk["text"])

                    with st.expander("Prompt context", expanded=False):
                        st.code(result["context"])


def render_deployment_tab() -> None:
    st.subheader("Deployment Notes")
    st.markdown(
        """
        **Recommended deployment**

        1. Run `vLLM` with `Qwen/Qwen2.5-7B-Instruct` on SCC.
        2. Tunnel the SCC endpoint to `http://127.0.0.1:8000/v1`.
        3. Connect both `OpenWebUI` and this dashboard to the same OpenAI-compatible backend.

        **The upgraded dashboard now supports**

        - uploaded corpora for live testing
        - side-by-side Traditional vs Advanced vs Multi-Agent comparisons
        - PDF, text, Markdown, CSV, TSV, and JSON ingestion
        - local or SCC-hosted generation backends
        """
    )
    st.code(
        "\n".join(
            [
                "bash scripts/submit_scc_vllm.sh /projectnb/cs505am/projects/kuromiqo_chunkrag_project/outputs/openwebui_vllm",
                "bash scripts/tunnel_scc_vllm.sh /projectnb/cs505am/projects/kuromiqo_chunkrag_project",
                "bash scripts/setup_openwebui_local.sh",
                "bash scripts/run_openwebui_local.sh",
                "bash scripts/setup_demo_dashboard.sh",
                "bash scripts/run_demo_dashboard.sh",
            ]
        ),
        language="bash",
    )
    st.markdown(
        """
        **OpenAI-compatible defaults**

        - Base URL: `http://127.0.0.1:8000/v1`
        - API key: `chunkrag-demo-key`
        - Model ID: `Qwen/Qwen2.5-7B-Instruct`
        """
    )


def main() -> None:
    st.set_page_config(page_title="ChunkRAG Demo Dashboard", layout="wide")
    st.title("ChunkRAG Demo Dashboard")
    st.caption("Interactive visual analysis for chunking, retrieval, deployment, and live RAG demos.")

    default_results_path, default_output_dir = get_cli_defaults()

    results_path_str = st.sidebar.text_input("Aggregate results path", str(default_results_path))
    output_dir_str = st.sidebar.text_input("Output directory", str(default_output_dir))

    results_path = Path(results_path_str)
    output_dir = Path(output_dir_str)

    if not results_path.exists():
        st.error(f"Results file not found: {results_path}")
        return

    results = load_results(results_path)
    architecture_tab, results_tab, playground_tab, predictions_tab, figures_tab, deployment_tab = st.tabs(
        ["Architecture", "Results", "Playground", "Predictions", "Figures", "Deployment"]
    )

    with architecture_tab:
        render_architecture_tab()
    with results_tab:
        render_results_tab(results)
    with playground_tab:
        render_playground_tab()
    with predictions_tab:
        render_predictions_tab(output_dir)
    with figures_tab:
        render_figures_tab(output_dir)
    with deployment_tab:
        render_deployment_tab()


if __name__ == "__main__":
    main()
