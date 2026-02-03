# Quick Start Guide

Get your SFDA Cosmetics Chatbot running in 5 minutes!

## Prerequisites

- Python 3.9+
- OpenRouter API key ([Get one here](https://openrouter.ai/keys))

## Installation Steps

### 1. Install Dependencies (1 min)

```bash
# Activate virtual environment (if not already activated)
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 2. Configure Environment (1 min)

```bash
# Copy the template
copy .env.example .env  # Windows
# or
cp .env.example .env  # Linux/Mac
```

Edit `.env` and add your API key:
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 3. Prepare Knowledge Base (2 min)

Ensure you have files in the `knowledge/` directory:
- `sfda_articles.json` - SFDA regulations
- `banned_list.json` - Banned substances (optional)
- `*.xlsx` - Excel files (optional)

### 4. Build Vector Database (1 min)

```bash
# Use the improved version with better logging
python ingest_database_improved.py
```

Expected output:
```
INFO - Initializing SFDA Chatbot...
INFO - 📊 Loading Excel documents...
INFO - 🧩 Loading JSON/JSONL documents...
INFO - ✅ Loaded 250 regulation docs from JSON.
INFO - 💾 Building Chroma vector store...
INFO - ✅ Ingestion complete.
```

### 5. Run the Application (30 sec)

```bash
# Use the improved version
python app_gradio_improved.py
```

Expected output:
```
INFO - Initializing SFDA Chatbot...
INFO - Loading embedding model...
INFO - Initializing LLM...
INFO - Loading vector store...
INFO - ✅ Chatbot initialized successfully
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live
```

### 6. Open in Browser

Click the public URL or open http://127.0.0.1:7860

## Test Queries

Try these example queries to verify everything works:

### Test 1: Specific Article
**Source:** لوائح التجميل (PDF)
**Query:** ما هي المادة الرابعة؟

**Expected:** Should return the full text of Article 4 from the regulations.

### Test 2: General Regulation Query
**Source:** لوائح التجميل (PDF)
**Query:** اذكر التزامات المُدرج في النظام

**Expected:** Should list the obligations from the regulations with sources.

### Test 3: Banned Substance
**Source:** محظورات التجميل
**Query:** هل Mercury محظور في التجميل؟

**Expected:** Should confirm Mercury is banned with details.

### Test 4: List Banned Substances
**Source:** محظورات التجميل
**Query:** اذكر لي 5 مواد محظورة تبدأ بحرف M

**Expected:** Should list 5 banned substances starting with M.

## Troubleshooting

### Problem: "OPENROUTER_API_KEY not found"
**Solution:**
1. Ensure `.env` file exists in the project root
2. Check that `OPENROUTER_API_KEY=your-key` is set
3. No spaces around the `=` sign

### Problem: "Chroma database not found"
**Solution:**
```bash
python ingest_database_improved.py
```

### Problem: "No module named 'config'"
**Solution:**
```bash
# Ensure you're in the correct directory
cd smart_chatbot-main
python app_gradio_improved.py
```

### Problem: Empty responses
**Solution:**
1. Check if `knowledge/` directory has files
2. Verify ingestion completed successfully
3. Check logs for errors

### Problem: Slow responses
**Solution:**
1. First query is slower (model loading)
2. Subsequent queries should be faster
3. Consider using a smaller embedding model

## Next Steps

- Read [README.md](README.md) for detailed documentation
- See [IMPROVEMENTS.md](IMPROVEMENTS.md) for code enhancements
- Customize settings in `.env` file
- Add your own knowledge files to `knowledge/` directory

## Common Configuration Changes

### Use Different LLM Model
Edit `.env`:
```env
LLM_MODEL=gpt-4o-mini  # Faster but more expensive
```

### Use Smaller Embedding Model (Faster)
Edit `.env`:
```env
EMBEDDING_MODEL=intfloat/multilingual-e5-base
```

### Adjust Retrieval Count
Edit `.env`:
```env
RETRIEVAL_K=5  # Return fewer documents (faster)
```

### Enable Debug Mode
Edit `.env`:
```env
DEBUG=True  # See detailed logs
```

## Getting Help

1. Check the logs in the terminal
2. Review [README.md](README.md) troubleshooting section
3. Verify all prerequisites are met
4. Ensure Python version is 3.9+

## Success Checklist

- [x] Python 3.9+ installed
- [x] Virtual environment activated
- [x] Dependencies installed
- [x] `.env` file created with API key
- [x] Knowledge files in `knowledge/` directory
- [x] ChromaDB built successfully
- [x] Application running
- [x] Browser opened to Gradio interface
- [x] Test queries working

## Tips for Best Results

1. **Be specific** - Clear questions get better answers
2. **Use Arabic** - The system is optimized for Arabic
3. **Check sources** - Verify which documents were used
4. **Try different modes** - Switch between regulations and banned substances
5. **Rephrase if needed** - Try different wordings for better results

## Performance Tips

- First query takes longer (model loading)
- Subsequent queries are much faster
- Streaming responses appear gradually
- Use "لوائح التجميل" for regulation queries
- Use "محظورات التجميل" for banned substance queries

Enjoy using your SFDA Cosmetics Chatbot! 🤖
