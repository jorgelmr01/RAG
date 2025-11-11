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
    if "invalid_api_key" in lowered or "incorrect api key" in lowered or "401" in text:
        return (
            "⚠️ Invalid API key. Please check that you've entered the correct key in the "
            "Configuration section. Your key should start with 'sk-' and be 51 characters long. "
            "Get your key at https://platform.openai.com/account/api-keys"
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


def _project_status_text(pipeline: RAGPipeline) -> str:
    return pipeline.project_status()


def _get_project_dropdown_update(pipeline: RAGPipeline) -> gr.update:
    """Get the project dropdown update with current choices and value."""
    choices = pipeline.project_options()
    value = pipeline.current_project.name if pipeline.current_project else None
    return gr.update(choices=choices, value=value)


def refresh_project_list(state: Optional[RAGPipeline]):
    pipeline = _ensure_pipeline(state)
    pipeline.ensure_project_selected()
    return (
        pipeline,
        _get_project_dropdown_update(pipeline),
        gr.update(value=_project_status_text(pipeline)),
        gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
    )


def create_project(project_name: str, state: Optional[RAGPipeline]):
    pipeline = _ensure_pipeline(state)
    name = (project_name or "").strip()
    if not name:
        msg = "⚠️ Enter a project name to create."
        return (
            pipeline,
            _get_project_dropdown_update(pipeline),
            gr.update(value=_project_status_text(pipeline)),
            gr.update(value=msg),
            gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
            [],
            gr.update(value="No responses yet."),
        )
    try:
        info = pipeline.create_project(name)
        msg = f"✅ Project `{info.display_name}` created. Upload documents to build its knowledge base."
    except ValueError as exc:
        msg = f"⚠️ {exc}"
    return (
        pipeline,
        _get_project_dropdown_update(pipeline),
        gr.update(value=_project_status_text(pipeline)),
        gr.update(value=msg),
        gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
        [],
        gr.update(value="No responses yet."),
    )


def load_project(selected: Optional[str], state: Optional[RAGPipeline]):
    pipeline = _ensure_pipeline(state)
    if not selected:
        return (
            pipeline,
            _get_project_dropdown_update(pipeline),
            gr.update(value=_project_status_text(pipeline)),
            gr.update(value="⚠️ Select a project to load."),
            gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
            [],
            gr.update(value="No responses yet."),
        )
    try:
        info = pipeline.load_project(selected)
        msg = f"✅ Loaded project `{info.display_name}`."
    except ValueError as exc:
        msg = f"⚠️ {exc}"
    return (
        pipeline,
        _get_project_dropdown_update(pipeline),
        gr.update(value=_project_status_text(pipeline)),
        gr.update(value=msg),
        gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
        [],
        gr.update(value="No responses yet."),
    )


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


def set_api_key(api_key: str, embedding_model: str, state: Optional[RAGPipeline]):
    pipeline = _ensure_pipeline(state)
    # Clean the key: strip whitespace and newlines from all sides
    if api_key:
        key = api_key.strip().replace("\n", "").replace("\r", "").replace(" ", "")
    else:
        key = ""
    
    # Validate the key format
    if not key:
        return pipeline, gr.update(value="⚠️ Please enter an API key.")
    
    if not key.startswith("sk-"):
        return pipeline, gr.update(
            value="⚠️ API key should start with 'sk-'. Check that you copied the full key. "
            f"Received key starts with: '{key[:5]}...' (length: {len(key)})"
        )
    
    if len(key) < 20:
        return pipeline, gr.update(
            value=f"⚠️ API key appears to be truncated. Received length: {len(key)} characters. "
            "OpenAI keys are typically 51 characters long. Please copy the complete key."
        )
    
    # Store in environment
    os.environ["OPENAI_API_KEY"] = key
    
    try:
        pipeline.configure_api_key(key, embedding_model=embedding_model)
    except ValueError as exc:
        return pipeline, gr.update(value=f"⚠️ {exc}")
    except Exception as exc:
        error_msg = str(exc)
        if "invalid_api_key" in error_msg.lower() or "incorrect api key" in error_msg.lower() or "401" in error_msg:
            return pipeline, gr.update(
                value="⚠️ Invalid API key. Please verify:\n"
                f"- The key you entered is {len(key)} characters long (should be ~51)\n"
                "- The key starts with 'sk-' and is complete\n"
                "- You copied it from https://platform.openai.com/account/api-keys\n"
                "- The key hasn't been revoked or expired\n"
                "- Try copying and pasting the key again"
            )
        return pipeline, gr.update(value=f"⚠️ Error configuring API key: {error_msg}")
    return pipeline, gr.update(value="✅ API key configured. You can now ingest documents.")


def ingest_documents(files, append, chunk_size, chunk_overlap, embedding_model, state: Optional[RAGPipeline]):
    pipeline = _ensure_pipeline(state)
    pipeline.ensure_project_selected()
    file_paths = _extract_paths(files)
    if not file_paths:
        return (
            pipeline,
            gr.update(value="Upload at least one document to build the knowledge base."),
            gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
            gr.update(value=_project_status_text(pipeline)),
        )

    # Update embedding model if changed
    if embedding_model and embedding_model != pipeline.config.embedding_model:
        try:
            pipeline.configure_api_key(embedding_model=embedding_model)
        except Exception as exc:
            return (
                pipeline,
                gr.update(value=f"⚠️ Error updating embedding model: {exc}"),
                gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
                gr.update(value=_project_status_text(pipeline)),
            )

    # Validate and convert chunk settings
    try:
        chunk_size_int = int(chunk_size) if chunk_size is not None else None
        chunk_overlap_int = int(chunk_overlap) if chunk_overlap is not None else None
        
        if chunk_size_int is not None and chunk_size_int <= 0:
            return (
                pipeline,
                gr.update(value="⚠️ Chunk size must be greater than 0."),
                gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
                gr.update(value=_project_status_text(pipeline)),
            )
        if chunk_overlap_int is not None and chunk_overlap_int < 0:
            return (
                pipeline,
                gr.update(value="⚠️ Chunk overlap cannot be negative."),
                gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
                gr.update(value=_project_status_text(pipeline)),
            )
        if chunk_size_int is not None and chunk_overlap_int is not None and chunk_overlap_int >= chunk_size_int:
            return (
                pipeline,
                gr.update(value="⚠️ Chunk overlap must be less than chunk size."),
                gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
                gr.update(value=_project_status_text(pipeline)),
            )
    except (ValueError, TypeError):
        return (
            pipeline,
            gr.update(value="⚠️ Invalid chunk settings. Please enter valid numbers."),
            gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
            gr.update(value=_project_status_text(pipeline)),
        )

    try:
        stats = pipeline.ingest(
            file_paths, 
            append=bool(append),
            chunk_size=chunk_size_int,
            chunk_overlap=chunk_overlap_int,
        )
    except ValueError as exc:
        return (
            pipeline,
            gr.update(value=_friendly_error_message(exc)),
            gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
            gr.update(value=_project_status_text(pipeline)),
        )
    except DocumentIngestionError as exc:
        return (
            pipeline,
            gr.update(value=f"⚠️ {exc}"),
            gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
            gr.update(value=_project_status_text(pipeline)),
        )
    except Exception as exc:  # pragma: no cover - defensive
        return (
            pipeline,
            gr.update(value=_friendly_error_message(exc)),
            gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
            gr.update(value=_project_status_text(pipeline)),
        )

    project_display = pipeline.current_project.display_name if pipeline.current_project else "current project"
    total_files = len(stats.files)
    if stats.chunks:
        message = (
            f"✅ Project `{project_display}` now holds {stats.chunks} chunks across {total_files} source files."
        )
    else:
        message = (
            f"⚠️ No text content detected in the uploaded files. Project `{project_display}` remains unchanged."
        )

    return (
        pipeline,
        gr.update(value=message),
        gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
        gr.update(value=_project_status_text(pipeline)),
    )


def respond(message, chat_history, state: Optional[RAGPipeline]):
    pipeline = _ensure_pipeline(state)
    pipeline.ensure_project_selected()
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
    try:
        pipeline.reset_current_project()
        message = "Knowledge base cleared for the current project."
    except ValueError as exc:
        message = f"⚠️ {exc}"
    return (
        [],
        pipeline,
        gr.update(value=message),
        gr.update(value=_format_indexed_documents(pipeline.render_loaded_sources())),
        gr.update(value=_project_status_text(pipeline)),
        gr.update(value=""),
    )


def build_interface(config: AppConfig = CONFIG) -> gr.Blocks:
    with gr.Blocks(title="Document RAG Assistant", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📚 Document RAG Assistant

            **Instructions:**
            1. Create or select a project above
            2. Upload your documents below
            3. Click 'Process Documents' to add them to the knowledge base
            4. Then ask questions in the chat!
            """
        )

        state = gr.State(RAGPipeline(config))

        with gr.Accordion("Projects", open=True):
            with gr.Row():
                project_dropdown = gr.Dropdown(
                    label="Saved projects",
                    choices=[],
                    value=None,
                    allow_custom_value=False,
                    scale=3,
                )
                load_project_btn = gr.Button("Load Selected", variant="secondary", scale=1)
                refresh_projects_btn = gr.Button("Refresh", variant="secondary", scale=1)
            with gr.Row():
                new_project_name = gr.Textbox(
                    label="Create new project",
                    placeholder="e.g. client-a",
                    scale=3,
                )
                create_project_btn = gr.Button("Create & Switch", variant="primary", scale=1)
            project_status = gr.Markdown("No project selected.")

        with gr.Accordion("API Key", open=False):
            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="sk-... (stored for this session only)",
            )
            api_key_status = gr.Markdown(
                "**Get your API key:** https://platform.openai.com/account/api-keys\n\n"
                "Your key is stored only for this session and never shared. "
                "You can also create a `.env` file in this folder with `OPENAI_API_KEY=sk-...`"
            )
            api_key_button = gr.Button("Set API Key", variant="primary")
        
        with gr.Accordion("Advanced Settings", open=False):
            embedding_model_input = gr.Dropdown(
                label="Embedding Model",
                choices=[
                    ("text-embedding-3-large (Recommended - Best Quality)", "text-embedding-3-large"),
                    ("text-embedding-3-small (Faster & Cheaper)", "text-embedding-3-small"),
                ],
                value=CONFIG.embedding_model,
                info="Large: Best accuracy, higher cost. Small: 6.5x cheaper, ~95% performance.",
            )
            
            gr.Markdown("### Chunking Settings")
            gr.Markdown(
                "Adjust how documents are split into chunks. "
                "**Recommended:** Chunk size 1000-2000, Overlap 200-400. "
                "Smaller chunks = more precise retrieval, larger chunks = more context."
            )
            chunk_size_input = gr.Number(
                label="Chunk Size (characters)",
                value=CONFIG.chunk_size,
                minimum=100,
                maximum=10000,
                step=100,
                info=f"Recommended: {CONFIG.chunk_size} (current default)",
            )
            chunk_overlap_input = gr.Number(
                label="Chunk Overlap (characters)",
                value=CONFIG.chunk_overlap,
                minimum=0,
                maximum=5000,
                step=50,
                info=f"Recommended: {CONFIG.chunk_overlap} (current default). Should be < chunk size.",
            )

        # Set up API key button handler (after embedding_model_input is defined)
        api_key_button.click(
            set_api_key,
            inputs=[api_key_input, embedding_model_input, state],
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
                clear_kb = gr.Button("Clear Project Knowledge", variant="secondary")

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
                    "Select or create a project, then upload documents.",
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

        demo.load(
            refresh_project_list,
            inputs=[state],
            outputs=[state, project_dropdown, project_status, source_overview],
        )

        ingest_button.click(
            ingest_documents,
            inputs=[file_input, append_checkbox, chunk_size_input, chunk_overlap_input, embedding_model_input, state],
            outputs=[state, ingest_feedback, source_overview, project_status],
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
            outputs=[chatbot, state, ingest_feedback, source_overview, project_status, user_input],
            queue=False,
        )

        create_project_btn.click(
            create_project,
            inputs=[new_project_name, state],
            outputs=[
                state,
                project_dropdown,
                project_status,
                ingest_feedback,
                source_overview,
                chatbot,
                sources_panel,
            ],
            queue=False,
        )

        load_project_btn.click(
            load_project,
            inputs=[project_dropdown, state],
            outputs=[
                state,
                project_dropdown,
                project_status,
                ingest_feedback,
                source_overview,
                chatbot,
                sources_panel,
            ],
            queue=False,
        )

        refresh_projects_btn.click(
            refresh_project_list,
            inputs=[state],
            outputs=[state, project_dropdown, project_status, source_overview],
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

