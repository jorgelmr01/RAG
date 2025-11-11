# Document RAG Assistant

A simple document assistant that lets you upload files and ask questions about them. Perfect for people with little technical background!

## ✨ What This Does

Upload your documents (PDFs, Word files, text files) and ask questions. The assistant reads your documents and answers using information from them.

## 🚀 Quick Start (Super Simple!)

### For Windows Users:
1. **Make sure Python is installed** - Download from https://www.python.org/downloads/ (check "Add Python to PATH" during installation)
2. **Double-click `start_app.bat`**
3. **Wait for browser to open** (first time takes 1-2 minutes)
4. **Enter your OpenAI API key** in the Configuration section
5. **Upload documents and ask questions!**

### For Mac Users:
1. **Make sure Python is installed** - Download from https://www.python.org/downloads/mac-osx/
2. **Double-click `start_app.command`** (you may need to right-click → Open the first time)
3. **Wait for browser to open** (first time takes 1-2 minutes)
4. **Enter your OpenAI API key** in the Configuration section
5. **Upload documents and ask questions!**

> 📖 **New to this?** See `GETTING_STARTED.md` for a detailed step-by-step guide with screenshots and troubleshooting.

---

## 🎯 Features

- **Easy to use** - Just double-click and go!
- **Works with many file types** - PDF, Word (.docx), Text files, CSV, and more
- **Save your work** - Create projects to organize different document sets
- **Smart answers** - Gets answers from your documents, not the internet
- **Shows sources** - See exactly where the answer came from

---

## 📋 Requirements

- **Python 3.11 or newer** - Download from https://www.python.org/downloads/
- **OpenAI API key** - Get one free at https://platform.openai.com/signup
- **Internet connection** - Needed to process documents and get answers

---

## 🔧 Advanced Setup (Optional)

If the simple launcher doesn't work, you can run it manually:

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

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
# CHUNK_SIZE=1500              # Increased for better context in large docs
# CHUNK_OVERLAP=300             # Increased overlap for continuity
# TOP_K=8                      # Increased for better recall
# SCORE_THRESHOLD=0.5           # More lenient threshold for better recall
# MAX_CONTEXT_SECTIONS=12       # Increased for better recall of specific details
# MAX_CONTEXT_CHARS=2000        # Increased for large documents
# USE_MMR=false                 # Disabled by default for better relevance (enable for diversity)
# MMR_DIVERSITY=0.7             # More relevance-focused when MMR is used
# ENABLE_QUERY_EXPANSION=true   # Enable query expansion for better matching
# PROJECTS_PATH=projects
```

## Using the app

1. Provide your API key if you haven’t already.
2. Choose an existing project or create a new one in the *Projects* accordion.
3. Drag-and-drop documents or browse for them, then click **Process documents**.
4. Ask questions; responses stream with inline citations.
5. Review supporting sources and the indexed document list in the right column.
6. Append more files, switch projects, or clear the current project’s knowledge as needed.

## Managing projects

- **Create & switch:** enter a descriptive name (e.g. `client-a`) and click **Create & Switch** to start a fresh vector store.
- **Load existing:** pick a project from the dropdown and click **Load Selected** to reuse its embedded sources instantly.
- **Clear project knowledge:** wipes the current project’s embeddings while keeping the project entry for new uploads.
- All project data lives under `projects/` by default (override with `PROJECTS_PATH`).

## Supported file types

| Extension           | Loader              | Status |
| ------------------- | ------------------- | ------ |
| `.pdf`              | `PyPDFLoader`       | ✅ Built-in |
| `.txt`, `.md`, `.rtf` | `TextLoader`     | ✅ Built-in |
| `.csv`, `.tsv`      | `CSVLoader`         | ✅ Built-in |
| `.docx`             | `Docx2txtLoader`    | ✅ Built-in |
| `.xlsx`, `.xls`     | `UnstructuredExcelLoader` | ⚠️ Optional (requires `openpyxl`) |
| `.pptx`, `.ppt`     | `UnstructuredPowerPointLoader` | ⚠️ Optional (requires `python-pptx`) |
| `.html`, `.htm`     | `BSHTMLLoader`      | ⚠️ Optional (falls back to text) |
| `.json`             | Text parsing        | ⚠️ Basic support |

**Unknown extensions** fall back to plain-text parsing.

### Adding Excel/PowerPoint Support

To enable Excel and PowerPoint support, install additional dependencies:
```bash
pip install openpyxl python-pptx unstructured[excel] unstructured[ppt]
```

> 💡 **Note:** The app works without these - Excel/PowerPoint files will be skipped with a warning if dependencies are missing.

## Project layout

```
app.py                # Gradio web server
requirements.txt      # Runtime dependencies
start_app.py          # Cross-platform launcher logic (console)
start_app.bat         # Windows double-click entry point
start_app.command     # macOS launcher (run with double-click)
src/
  __init__.py
  config.py           # Runtime configuration helpers
  document_loaders.py # File-type aware ingestion
  pipeline.py         # RAG ingestion & streaming pipeline
  project_store.py    # Persisted project metadata helpers
```

Runtime artefacts such as `.venv/`, `projects/`, `last_error.log`, and `chroma_db/` are
ignored automatically and will be re-generated locally as needed.

## Troubleshooting

- **Python not found:** Install Python 3.11+ from
  [python.org](https://www.python.org/downloads/) and re-run the launcher.
- **Quota / billing errors:** The UI shows friendly messages when OpenAI returns
  `insufficient_quota` or rate limits—add credit or adjust your plan.
- **No relevant context:** Upload more documents or rephrase the question; the
  retriever filters out low-similarity matches by default.

## 📦 Sharing the Project

### Easy Way (Recommended):

**Windows:**
- Double-click `CREATE_ZIP.bat` - it will create a clean ZIP file automatically!

**Mac/Linux:**
- Run `chmod +x CREATE_ZIP.sh` once
- Then double-click `CREATE_ZIP.sh` or run `./CREATE_ZIP.sh`

### Manual Way:

**Windows:**
1. Right-click the project folder
2. Select "Send to" → "Compressed (zipped) folder"
3. **IMPORTANT:** Before sharing, make sure the ZIP doesn't include:
   - `.venv` folder (if it exists)
   - `projects` folder (contains your personal data)
   - `.env` file (contains your API key - keep it private!)

**Mac:**
1. Right-click the project folder
2. Select "Compress [folder name]"
3. **IMPORTANT:** Same exclusions as Windows above

### What to Include in the ZIP:
✅ `app.py`  
✅ `start_app.bat` and `start_app.command`  
✅ `start_app.py`  
✅ `requirements.txt`  
✅ `readme.md` and `GETTING_STARTED.md`  
✅ `src/` folder (all files inside)  
✅ `.gitignore`  

### What NOT to Include:
❌ `.venv/` folder (too large, will be recreated)  
❌ `projects/` folder (personal data)  
❌ `.env` file (your API key - keep it secret!)  
❌ `__pycache__/` folders  

**Recipients can then:**
1. Unzip the folder
2. Double-click `start_app.bat` (Windows) or `start_app.command` (Mac)
3. Follow the same setup steps
