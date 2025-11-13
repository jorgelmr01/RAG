# Document RAG Assistant - HTML Version

## 🎉 No Installation Required!

This is a **pure HTML/JavaScript version** of the Document RAG Assistant that runs entirely in your web browser. No Python, no pip, no security blocks - just open the file and start using it!

## ✨ Features

- **Zero Installation** - Just open `index.html` in any modern web browser
- **No Server Required** - Everything runs in your browser
- **Same Functionality** - All features from the Python version:
  - Upload PDF, TXT, MD, DOCX, CSV files
  - Create and manage multiple projects
  - Ask questions about your documents
  - Get answers with source citations
  - Adjustable settings for chunking, retrieval, and responses
- **Privacy First** - All data stored locally in your browser (IndexedDB)
- **Lightweight** - Single HTML file (~50KB) + external CDN libraries

## 🚀 Quick Start

### Step 1: Open the File

Simply double-click `index.html` or:
- **Windows**: Right-click → "Open with" → Choose your browser (Chrome, Firefox, Edge, etc.)
- **Mac**: Right-click → "Open with" → Choose your browser
- **Linux**: Double-click or open from terminal: `xdg-open index.html`

### Step 2: Set Your API Key

1. Get your OpenAI API key from: https://platform.openai.com/account/api-keys
2. Click on the **🔑 API Key** section
3. Paste your API key (starts with `sk-`)
4. Click "Set API Key"
5. Your key is stored only in your browser - never shared!

### Step 3: Create a Project

1. Click on the **📁 Projects** section
2. Enter a project name (e.g., "client-a")
3. Click "Create & Switch"

### Step 4: Upload Documents

1. Click "Choose Files" and select your documents (PDF, TXT, DOCX, etc.)
2. Optionally check "Append to existing" to add to current knowledge base
3. Click "Process Documents"
4. Wait for processing to complete (this may take a minute for large files)

### Step 5: Ask Questions!

1. Type your question in the chat box
2. Press Enter or click "Send"
3. Get answers with source citations!

## 📋 Supported File Types

- **PDF** (.pdf) - Using PDF.js
- **Word Documents** (.docx) - Using Mammoth.js
- **Text Files** (.txt, .md) - Direct text extraction
- **CSV Files** (.csv) - Text extraction

## ⚙️ Advanced Settings

All settings from the Python version are available:

### Chunking Settings
- **Chunk Size**: How many characters per chunk (default: 1500)
- **Chunk Overlap**: Overlap between chunks for context (default: 300)

### Retrieval Settings
- **Initial Chunks Retrieved**: How many chunks to search (default: 8)
- **Max Context Sections**: Maximum chunks to use in answer (default: 12)
- **Similarity Score Threshold**: Minimum similarity to include (default: 0.5)

### Response Settings
- **Temperature**: Controls creativity vs accuracy (default: 0.2)
- **Embedding Model**: Choose between large (better) or small (faster/cheaper)

## 💾 Data Storage

All your data is stored locally in your browser using IndexedDB:
- **Projects**: Saved automatically
- **Document Chunks**: Stored with embeddings
- **Settings**: Saved in localStorage
- **API Key**: Stored in localStorage (browser only, never shared)

**Note**: If you clear your browser data, you'll lose your projects. Consider exporting important data.

## 🔒 Privacy & Security

- **No Server**: Everything runs in your browser
- **No Data Collection**: We don't collect any data
- **API Key**: Stored only in your browser's localStorage
- **OpenAI API**: Your API key is sent directly to OpenAI (same as Python version)
- **Local Storage**: All documents and embeddings stay in your browser

## 🌐 Browser Compatibility

Works in all modern browsers:
- ✅ Chrome/Edge (Chromium) - Recommended
- ✅ Firefox
- ✅ Safari
- ✅ Opera

**Minimum Requirements**:
- JavaScript enabled
- IndexedDB support (all modern browsers)
- Internet connection (for OpenAI API and CDN libraries)

## 📦 Sharing the HTML Version

To share with coworkers:

1. **Just send the `index.html` file** - That's it!
2. They can open it in any browser
3. No installation, no security blocks, no Python needed

**Optional**: You can also host it on a simple web server or share via:
- Email attachment
- Cloud storage (Google Drive, Dropbox, etc.)
- Company intranet
- USB drive

## 🆚 HTML Version vs Python Version

| Feature | HTML Version | Python Version |
|---------|-------------|----------------|
| Installation | None needed | Python + pip required |
| File Size | ~50KB | ~50MB+ (with dependencies) |
| Security Blocks | None | May trigger antivirus |
| Performance | Good for most use cases | Slightly faster for very large files |
| Offline | Partially (needs internet for API) | Partially (needs internet for API) |
| Sharing | Just send one file | Need to share entire project |
| Updates | Replace one file | Update dependencies |

## 🐛 Troubleshooting

### "Failed to extract text from PDF"
- Make sure the PDF is not password-protected
- Try a different PDF file
- Some scanned PDFs (images) won't work - need OCR

### "No text content found"
- The file might be empty or corrupted
- Try opening the file in another application first
- Some file types might not be fully supported

### "API key error"
- Make sure your API key starts with `sk-`
- Check that you copied the complete key (51 characters)
- Verify your OpenAI account has credits
- Try generating a new API key

### "No relevant context found"
- Upload more documents
- Try rephrasing your question
- Lower the similarity threshold in Advanced Settings
- Increase "Max Context Sections"

### Browser crashes or slow performance
- Close other browser tabs
- Process fewer files at once
- Use smaller chunk sizes
- Clear old browser data if IndexedDB is full

## 💡 Tips

1. **Start Small**: Test with a few small documents first
2. **Chunk Size**: For technical documents, use 1000-1500. For narrative text, 1500-2000 works well
3. **Temperature**: Keep it low (0.1-0.3) for factual answers, higher (0.5-0.7) for creative responses
4. **Projects**: Create separate projects for different document sets
5. **Backup**: If you have important projects, you can export the IndexedDB data (browser DevTools)

## 📝 Notes

- **Internet Required**: You need internet connection for OpenAI API calls
- **API Costs**: Same as Python version - you pay for OpenAI API usage
- **File Size Limits**: Very large files (>10MB) may take longer to process
- **Browser Memory**: Processing many large documents may use significant browser memory

## 🎯 Use Cases

Perfect for:
- ✅ Quick document Q&A without installation
- ✅ Sharing with non-technical users
- ✅ Corporate environments with strict security
- ✅ One-time document analysis
- ✅ Demonstrations and presentations

## 📞 Support

If you encounter issues:
1. Check the browser console (F12) for error messages
2. Try a different browser
3. Clear browser cache and try again
4. Make sure JavaScript is enabled

## 🔄 Updating

To update to a newer version:
1. Download the new `index.html`
2. Replace the old file
3. Your existing projects and data will remain (stored in browser)

---

**Enjoy your hassle-free document assistant!** 🚀

No Python, no pip, no problems - just open and go!

