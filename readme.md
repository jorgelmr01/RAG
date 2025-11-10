# Document RAG Assistant

A turnkey Retrieval Augmented Generation (RAG) web app that ingests any mix of
documents, indexes them with LangChain/Chroma, and streams grounded answers with
citations.

## Highlights

- Multi-file upload with automatic loaders for PDF, DOCX, TXT/MD/RTF, CSV and TSV.
- Smart chunking, similarity filtering and inline `[n]` references.
- Clean Gradio interface with source cards and knowledge-base summary.
- One-click setup on Windows: the launcher creates the virtual environment,
  installs dependencies when they change, and starts the app automatically.

---

## Quick start (Windows)

1. **Unzip or clone** the folder.
2. **Double-click `start_app.bat`.**
   - First run: creates `.venv`, upgrades `pip`, installs `requirements.txt`.
   - Subsequent runs: skips the install step unless `requirements.txt` changes.
3. When the browser opens, enter your OpenAI API key in the Configuration panel
   (or create a `.env` file beforehand—see below).

That’s it—no manual terminal steps required.

## Quick start (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

The Gradio UI appears in your browser; use it exactly as on Windows.

## Configure your OpenAI credentials

The app needs an OpenAI API key for embeddings and chat completions.

- **Option 1:** Create a `.env` file next to `app.py`.
- **Option 2:** Paste the key in the Configuration accordion inside the UI.

Example `.env` file:

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

## Using the app

1. Provide your API key if you haven’t already.
2. Drag-and-drop documents or browse for them, then click **Process documents**.
3. Ask questions; responses stream with inline citations.
4. Review supporting sources and the indexed document list in the right column.
5. Append more files or clear the knowledge base as needed.

## Supported file types

| Extension           | Loader              |
| ------------------- | ------------------- |
| `.pdf`              | `PyPDFLoader`       |
| `.txt`, `.md`, `.rtf` | `TextLoader`     |
| `.csv`, `.tsv`      | `CSVLoader`         |
| `.docx`             | `Docx2txtLoader`    |

Unknown extensions fall back to plain-text parsing.

## Project layout

```
app.py                # Gradio web server
requirements.txt      # Runtime dependencies
start_app.bat         # Windows launcher (auto-setup)
start_app.pyw         # Console-free entry point
src/
  __init__.py
  config.py           # Runtime configuration helpers
  document_loaders.py # File-type aware ingestion
  pipeline.py         # RAG ingestion & streaming pipeline
```

Runtime artefacts such as `.venv/`, `last_error.log`, and `chroma_db/` are
ignored automatically and will be re-generated locally as needed.

## Troubleshooting

- **Python not found:** Install Python 3.11+ from
  [python.org](https://www.python.org/downloads/) and re-run the launcher.
- **Quota / billing errors:** The UI shows friendly messages when OpenAI returns
  `insufficient_quota` or rate limits—add credit or adjust your plan.
- **No relevant context:** Upload more documents or rephrase the question; the
  retriever filters out low-similarity matches by default.

## Sharing the project

Zip the folder (excluding `.venv/` if present) or share the Git repo. Recipients
can double-click `start_app.bat` (Windows) or follow the macOS/Linux steps above
to get started immediately.
