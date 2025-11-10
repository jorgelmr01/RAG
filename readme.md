# Document RAG Assistant

An end-to-end Retrieval Augmented Generation (RAG) web app that can ingest multiple
documents of different formats, index them efficiently and answer questions with
grounded, cited responses.

## Features

- Upload any number of documents in one step and append more later.
- Automatic loader selection for PDF, TXT/MD/RTF, CSV/TSV and DOCX files.
- Adaptive text chunking with configurable size and overlap.
- Fast in-memory Chroma vector store and OpenAI embedding pipeline.
- Streaming answers from OpenAI chat models with inline `[n]` citations.
- Visual source panel and knowledge-base summaries in the UI.
- Built-in controls to clear the chat or rebuild the knowledge base.

## Quick Start

1. **Clone & enter the project**

   ```bash
   git clone <repo-url>
   cd Chatbot-with-RAG-and-LangChain
   ```

2. **Create & activate a virtual environment** (Python 3.11+ recommended)

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate  # macOS / Linux
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure the OpenAI API key**

   - Create a `.env` file with your key, **or**
   - Provide the key in the app UI under the *Configuration* accordion.

   Example `.env`:

   ```ini
   OPENAI_API_KEY=sk-REPLACE_ME
   # Optional overrides
   # OPENAI_CHAT_MODEL=gpt-4o-mini
   # OPENAI_EMBEDDING_MODEL=text-embedding-3-large
   # CHUNK_SIZE=1200
   # CHUNK_OVERLAP=200
   # TOP_K=6
   # SCORE_THRESHOLD=0.35
   # MAX_CONTEXT_SECTIONS=6
   ```

5. **Launch the web app**

   - **Easiest**: double-click `start_app.bat` (uses the virtual environment automatically).
   - **Alternative**: double-click `start_app.pyw` (writes any errors to `last_error.log`).
   - **Developers**: run `python app.py` from a terminal.

   In each case, the assistant opens in your default browser.

## Using the App

1. (Optional) Enter an API key in the *Configuration* accordion if you did not
   set one in `.env`.
2. Drag-and-drop or browse for one or more documents and click **Process
   Documents**.
3. Ask questions in the chat box. Answers stream in real time with citations.
4. Review the supporting sources in the right-hand panel.
5. Use **Append to existing knowledge base** to keep indexing more files, or
   **Clear Knowledge Base** to start fresh.

## Supported File Types

| Extension | Loader |
| --------- | ------ |
| `.pdf`    | `PyPDFLoader`
| `.txt`, `.md`, `.rtf` | `TextLoader`
| `.csv`, `.tsv` | `CSVLoader`
| `.docx`   | `Docx2txtLoader`

Unrecognised formats fall back to plain-text parsing.

## Project Layout

```
app.py                     # Gradio web server
src/
  config.py               # App configuration helpers
  document_loaders.py     # File-type aware loaders
  pipeline.py             # RAG ingestion + retrieval pipeline
requirements.txt
.env.example
```

## Customisation

- Modify chunking, retrieval depth or model defaults via environment variables.
- Extend `SUPPORTED_EXTENSIONS` in `src/document_loaders.py` to add new loaders.
- Swap out OpenAI models by changing `OPENAI_CHAT_MODEL` / `OPENAI_EMBEDDING_MODEL`.
- Integrate telemetry or logging by adjusting the `RAGPipeline`.

## Troubleshooting

- **Missing API key** – provide one in `.env` or through the UI.
- **No content indexed** – confirm the documents contain text and are not
  password protected.
- **Large documents** – raise `CHUNK_SIZE` or reduce `TOP_K` to balance context
  size and latency.
