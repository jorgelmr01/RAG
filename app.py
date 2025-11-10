"""Gradio application for the RAG document assistant."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, List, Optional

import gradio as gr

from src.config import AppConfig
from src.pipeline import DocumentIngestionError, RAGPipeline

INSUFFICIENT_QUOTA_HINT = (
    "Your OpenAI account reports `insufficient_quota`. Add billing credit or increase "
    "the project budget on https://platform.openai.com/account/billing to continue."
)


def _friendly_error_message(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if "insufficient_quota" in lowered or "exceeded your current quota" in lowered:
        return f"⚠️ {INSUFFICIENT_QUOTA_HINT}"
    if "rate limit" in lowered:
        return (
            "⚠️ OpenAI rate-limited the request. Wait a few seconds and try again, or reduce "
            "concurrent uploads."
        )
    return f"⚠️ Unexpected error: {text}"


CONFIG = AppConfig()


def _ensure_pipeline(state: Optional[RAGPipeline], config: AppConfig = CONFIG) -> RAGPipeline:
    return state or RAGPipeline(config)


def _extract_paths(files: Optional[Iterable[Any]]) -> List[str]:
    paths: List[str] = []
    if not files:
        return paths
    for item in files:
        if item is None:
            continue
        if isinstance(item, (str, Path)):
            paths.append(str(item))
            continue
        path = getattr(item, "name", None) or getattr(item, "tmp_path", None)
        if path:
            paths.append(str(path))
    return paths


def _format_indexed_documents(raw: str) -> str:
    cleaned = (raw or "").strip()
    if not cleaned or cleaned == "No documents indexed yet.":
        return "**Indexed documents**\n\nNo documents indexed yet."
    return f"**Indexed documents**\n\n{cleaned}"


CUSTOM_CSS = """
.source-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 0.9rem;
    background: var(--block-background-fill);
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
}

#chat-column {
    min-height: 520px;
    gap: 0.75rem;
}

#chat-column textarea {
    min-height: 120px !important;
}

#input-controls {
    align-items: stretch;
    gap: 0.75rem;
}

#input-controls .gradio-column {
    gap: 0.5rem;
}

#control-row {
    gap: 0.75rem;
}
"""


def set_api_key(api_key: str, state: Optional[RAGPipeline]):
    pipeline = _ensure_pipeline(state)
    key = (api_key or "").strip()
    if key:
        os.environ["OPENAI_API_KEY"] = key
    try:
        pipeline.configure_api_key(key if key else None)
    except ValueError as exc:
        return pipeline, gr.update(value=f"⚠️ {exc}")
    return pipeline, gr.update(value="✅ API key configured. You can now ingest documents.")


def ingest_documents(files, append, state: Optional[RAGPipeline]):
    pipeline = _ensure_pipeline(state)
    file_paths = _extract_paths(files)
    if not file_paths:
        return (
            pipeline,
            gr.update(value="Upload at least one document to build the knowledge base."),
            gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
        )

    try:
        stats = pipeline.ingest(file_paths, append=bool(append))
    except ValueError as exc:
        return (
            pipeline,
            gr.update(value=_friendly_error_message(exc)),
            gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
        )
    except DocumentIngestionError as exc:
        return (
            pipeline,
            gr.update(value=f"⚠️ {exc}"),
            gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
        )
    except Exception as exc:  # pragma: no cover - defensive
        return (
            pipeline,
            gr.update(value=_friendly_error_message(exc)),
            gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
        )

    message = (
        f"Indexed {stats.chunks} chunks from {stats.documents} document pieces."
        if stats.chunks
        else "No text content detected in the uploaded files."
    )
    return (
        pipeline,
        gr.update(value=f"✅ {message}"),
        gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
    )


def respond(message, chat_history, state: Optional[RAGPipeline]):
    pipeline = _ensure_pipeline(state)
    indexed_docs = _format_indexed_documents(pipeline.render_loaded_sources())
    if not message:
        yield (
            chat_history,
            pipeline,
            pipeline.render_sources(),
            indexed_docs,
            gr.update(value=""),
        )
        return

    try:
        docs = pipeline.retrieve(message)
    except Exception as exc:
        chat_history = chat_history + [(message, _friendly_error_message(exc))]
        yield (
            chat_history,
            pipeline,
            pipeline.render_sources(),
            indexed_docs,
            gr.update(value=""),
        )
        return

    chat_history = chat_history + [(message, "")]
    answer = ""
    sources_markdown = pipeline.render_sources(docs)
    indexed_docs = _format_indexed_documents(pipeline.render_loaded_sources())
    for chunk in pipeline.stream_answer(message, docs, chat_history[:-1]):
        answer += chunk
        chat_history[-1] = (message, answer)
        yield (
            chat_history,
            pipeline,
            sources_markdown,
            indexed_docs,
            gr.update(value=""),
        )


def clear_chat(state: Optional[RAGPipeline]):
    pipeline = _ensure_pipeline(state)
    return (
        [],
        pipeline,
        pipeline.render_sources(),
        _format_indexed_documents(pipeline.render_loaded_sources()),
        gr.update(value=""),
    )


def reset_knowledge(state: Optional[RAGPipeline]):
    pipeline = _ensure_pipeline(state)
    pipeline.reset_knowledge()
    return (
        [],
        pipeline,
        gr.update(value="Knowledge base cleared."),
        gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
        gr.update(value=""),
    )


def build_interface(config: AppConfig = CONFIG) -> gr.Blocks:
    with gr.Blocks(title="Document RAG Assistant", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Document RAG Assistant

            Upload documents, then ask grounded questions. The assistant responds using the
            indexed context and includes inline references.
            """
        )

        state = gr.State(RAGPipeline(config))

        with gr.Accordion("Configuration", open=False):
            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="sk-... (stored for this session only)",
            )
            api_key_status = gr.Markdown(
                "Environment variables will be used unless you supply a key here."
            )
            api_key_button = gr.Button("Set API Key", variant="primary")
            api_key_button.click(
                set_api_key,
                inputs=[api_key_input, state],
                outputs=[state, api_key_status],
            )

        with gr.Row(elem_id="control-row"):
            file_input = gr.File(
                label="Upload documents",
                file_count="multiple",
                type="filepath",
                scale=3,
            )
            with gr.Column(scale=1, min_width=200):
                append_checkbox = gr.Checkbox(
                    label="Append to existing knowledge base",
                    value=False,
                )
                ingest_button = gr.Button("Process Documents", variant="primary")
                clear_kb = gr.Button("Clear Knowledge Base", variant="secondary")

        with gr.Row(equal_height=True, elem_id="chat-row"):
            with gr.Column(scale=3, elem_id="chat-column"):
                chatbot = gr.Chatbot(height=450, type="tuples")
                with gr.Row(elem_id="input-controls"):
                    user_input = gr.Textbox(
                        placeholder="Ask a question about your documents...",
                        lines=4,
                        scale=4,
                    )
                    with gr.Column(scale=1, min_width=150):
                        submit = gr.Button("Send", variant="primary")
                        clear_chat_button = gr.Button("Clear Conversation", variant="secondary")

            with gr.Column(scale=2, min_width=320):
                ingest_feedback = gr.Markdown(
                    "Awaiting document upload.",
                    elem_classes=["source-card"],
                )
                sources_panel = gr.Markdown(
                    "No responses yet.",
                    elem_classes=["source-card"],
                )
                source_overview = gr.Markdown(
                    _format_indexed_documents(""),
                    elem_classes=["source-card"],
                )

        ingest_button.click(
            ingest_documents,
            inputs=[file_input, append_checkbox, state],
            outputs=[state, ingest_feedback, source_overview],
        )

        submit.click(
            respond,
            inputs=[user_input, chatbot, state],
            outputs=[chatbot, state, sources_panel, source_overview, user_input],
            queue=True,
        )
        user_input.submit(
            respond,
            inputs=[user_input, chatbot, state],
            outputs=[chatbot, state, sources_panel, source_overview, user_input],
            queue=True,
        )

        clear_chat_button.click(
            clear_chat,
            inputs=[state],
            outputs=[chatbot, state, sources_panel, source_overview, user_input],
            queue=False,
        )

        clear_kb.click(
            reset_knowledge,
            inputs=[state],
            outputs=[chatbot, state, ingest_feedback, source_overview, user_input],
            queue=False,
        )

    return demo


def launch_app(
    *,
    inbrowser: bool = True,
    share: bool = False,
    server_name: str | None = None,
    server_port: int | None = None,
) -> gr.Blocks:
    demo = build_interface()
    demo.queue().launch(
        inbrowser=inbrowser,
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_api=False,
        show_error=True,
        quiet=True,
    )
    return demo


if __name__ == "__main__":
    launch_app()

