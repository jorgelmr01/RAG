# Embedding Model Information

## Current Model

**Your app is using:** `text-embedding-3-large`

**Location:** `src/config.py` (default) or set via `OPENAI_EMBEDDING_MODEL` environment variable

## Is It The Best Choice?

### `text-embedding-3-large` (Current - Recommended)
- ✅ **Best accuracy** - Highest quality embeddings
- ✅ **3072 dimensions** - More detailed representations
- ✅ **Best for production** - Optimal for RAG applications
- ⚠️ **Higher cost** - ~$0.13 per 1M tokens
- ⚠️ **Slightly slower** - More processing time

### `text-embedding-3-small` (Alternative)
- ✅ **Lower cost** - ~$0.02 per 1M tokens (6.5x cheaper!)
- ✅ **Faster** - Less processing time
- ✅ **1536 dimensions** - Still very good quality
- ⚠️ **Slightly lower accuracy** - May miss some nuanced matches

### Recommendation

**For most users:** `text-embedding-3-large` is the best choice because:
- Better retrieval accuracy = better answers
- Cost difference is minimal for typical usage
- You're already using it, so no change needed

**Consider `text-embedding-3-small` if:**
- You process very large volumes of documents
- Cost is a primary concern
- You find the current model is "too good" (over-retrieving)

## How to Change It

Add to your `.env` file:
```ini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Or change the default in `src/config.py` line 50.

## Performance Comparison

Based on OpenAI's benchmarks:
- **text-embedding-3-large**: Best overall performance, especially for complex queries
- **text-embedding-3-small**: 95%+ of large model performance at 1/6th the cost

For RAG applications, the large model is generally recommended for better retrieval quality.

