# HTML Version - Branch Information

## Overview

This branch (`html-version`) contains a **pure HTML/JavaScript implementation** of the Document RAG Assistant that requires **zero installation** and runs entirely in the web browser.

## What's Different?

### Python Version (main branch)
- Requires Python 3.11+
- Requires pip and package installation
- Uses Gradio for UI
- Uses ChromaDB for vector storage
- May trigger security blocks on corporate computers

### HTML Version (this branch)
- **No installation required** - just open `Start_App.html`
- Runs entirely in the browser
- Uses IndexedDB for local storage
- Uses CDN libraries (PDF.js, Mammoth.js)
- **No security blocks** - just a single HTML file
- Perfect for sharing with coworkers

## Files in This Branch

### Core Files
- **`Start_App.html`** - Complete standalone application (all-in-one file)
- **`README_HTML_VERSION.md`** - Detailed documentation for HTML version
- **`QUICK_START_HTML.txt`** - Quick reference guide

### Original Files (Unchanged)
- All original Python files remain in the repository
- You can switch back to the Python version anytime

## How to Use

1. **Open `Start_App.html`** in any modern web browser
2. **Set your OpenAI API key** in the UI
3. **Create a project** and upload documents
4. **Start asking questions!**

See `README_HTML_VERSION.md` for complete instructions.

## Sharing

To share with coworkers:
- Just send them the `Start_App.html` file
- They can open it immediately - no installation needed
- All data stays in their browser (private and secure)

## Technical Details

### Libraries Used (via CDN)
- **PDF.js** (v3.11.174) - For PDF text extraction
- **Mammoth.js** (v1.6.0) - For DOCX text extraction

### Storage
- **IndexedDB** - For projects and document chunks
- **localStorage** - For API key and settings

### API Integration
- Direct calls to OpenAI API from browser
- Same API endpoints as Python version
- Streaming responses for chat

## Switching Between Versions

To switch back to Python version:
```bash
git checkout main
```

To switch to HTML version:
```bash
git checkout html-version
```

## Benefits of HTML Version

✅ **Zero Installation** - No Python, no pip, no dependencies  
✅ **No Security Blocks** - Just an HTML file  
✅ **Easy Sharing** - Send one file to coworkers  
✅ **Same Functionality** - All features from Python version  
✅ **Privacy** - Everything runs locally in browser  
✅ **Lightweight** - Single ~50KB file (plus CDN libraries)  

## Limitations

⚠️ **Internet Required** - Needs connection for OpenAI API and CDN libraries  
⚠️ **Browser Memory** - Very large document sets may use significant memory  
⚠️ **File Size** - Processing very large files (>10MB) may be slower  
⚠️ **Browser Compatibility** - Requires modern browser with IndexedDB support  

## Maintenance

- The HTML version is self-contained
- Updates: Just replace `Start_App.html` with new version
- User data persists in browser (IndexedDB)
- No server or backend required

---

**Created**: HTML version branch for easy sharing without installation requirements.

