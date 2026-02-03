# Deployment Guide - SFDA Inspector Assistant

This guide covers deploying your SFDA chatbot to the cloud for public access.

## Deployment Options

### Option 1: Hugging Face Spaces (Recommended - Free)
### Option 2: Streamlit Cloud (Alternative - Free)
### Option 3: Render (Full Control - Paid)

---

## Option 1: Hugging Face Spaces (Recommended)

### Why Hugging Face Spaces?
- ✅ **FREE** for public apps
- ✅ Built for Gradio apps
- ✅ Easy deployment (just push to Git)
- ✅ Persistent storage for ChromaDB
- ✅ Good performance
- ✅ Custom domain support

### Prerequisites
1. Hugging Face account ([signup here](https://huggingface.co/join))
2. Git installed
3. Your project ready

### Step-by-Step Deployment

#### 1. Create Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Configure:
   - **Space name**: `sfda-inspector-assistant`
   - **License**: Apache 2.0
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free) or upgrade if needed
   - **Visibility**: Public or Private

#### 2. Prepare Your Repository

Create these files in your project root:

**`requirements.txt`** (already created ✅)

**`app.py`** (create this):
```python
"""
Main entry point for Hugging Face Spaces deployment.
"""
from app_gradio_improved import main

if __name__ == "__main__":
    main()
```

**`README.md`** (update with Space info):
```markdown
---
title: SFDA Inspector Assistant
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.39.0
app_file: app.py
pinned: false
license: apache-2.0
---

# SFDA Inspector Assistant

AI-powered assistant for SFDA field inspectors to query cosmetics regulations and banned substances in Arabic.

## Features
- Arabic language support
- Regulation article lookup
- Banned substance queries
- Source citations
```

#### 3. Push to Hugging Face

```bash
# Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/sfda-inspector-assistant
cd sfda-inspector-assistant

# Copy your files
cp -r /path/to/your/project/* .

# Important: Add your .env secrets via HF Spaces UI, not git!
# Don't commit .env file to git

# Add files
git add .
git commit -m "Initial deployment"

# Push to Hugging Face
git push
```

#### 4. Configure Secrets

1. Go to your Space settings
2. Navigate to "Repository secrets"
3. Add your secrets:
   - `OPENROUTER_API_KEY`: Your OpenRouter API key

#### 5. Build ChromaDB

**Option A: Upload Pre-built DB**
```bash
# After running ingest locally
git add chroma_db/
git lfs track "chroma_db/**/*"
git add .gitattributes
git commit -m "Add ChromaDB"
git push
```

**Option B: Build During Startup** (slower first load)
Create `on_startup.py`:
```python
import os
from pathlib import Path

# Check if ChromaDB exists
if not Path("chroma_db").exists():
    print("ChromaDB not found. Building...")
    from ingest_database_improved import main
    main()
    print("ChromaDB built successfully!")
```

Update `app.py`:
```python
import on_startup  # Run before main app
from app_gradio_improved import main

if __name__ == "__main__":
    main()
```

#### 6. Monitor Deployment

- Check build logs in Space UI
- First build takes 5-10 minutes
- Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/sfda-inspector-assistant`

---

## Option 2: Streamlit Cloud

### Why Streamlit Cloud?
- ✅ FREE for public apps
- ✅ Easy GitHub integration
- ✅ Auto-deploys on commit
- ⚠️ Designed for Streamlit (need to convert from Gradio)

### Streamlit Conversion (Optional)

If you want to use Streamlit instead of Gradio:

**Create `streamlit_app.py`**:
```python
import streamlit as st
from app_gradio_improved import SFDAChatbot

st.set_page_config(
    page_title="SFDA Inspector Assistant",
    page_icon="🔍",
    layout="wide"
)

# Initialize chatbot
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = SFDAChatbot()

st.title("المساعد الذكي للمفتشين")

# Source selection
source_choice = st.radio(
    "مصدر البحث",
    ["لوائح التجميل (PDF)", "محظورات التجميل", "الكل"]
)

# Chat interface
if prompt := st.chat_input("اكتبي سؤالك..."):
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in st.session_state.chatbot.stream_response(prompt, [], source_choice):
            full_response = chunk
            response_placeholder.markdown(full_response)
```

### Deployment Steps

1. Push your code to GitHub
2. Go to https://streamlit.io/cloud
3. Sign in with GitHub
4. Click "New app"
5. Select your repository
6. Set main file: `streamlit_app.py`
7. Add secrets in "Advanced settings"
8. Click "Deploy"

---

## Option 3: Render (Full Control)

### Why Render?
- ✅ Full control over environment
- ✅ Persistent disk storage
- ✅ Custom domains
- ⚠️ Paid ($7/month for persistent)

### Deployment Steps

**1. Create `Dockerfile`**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Build ChromaDB
RUN python ingest_database_improved.py

# Expose port
EXPOSE 7860

# Run app
CMD ["python", "app_gradio_improved.py"]
```

**2. Create `render.yaml`**:
```yaml
services:
  - type: web
    name: sfda-inspector-assistant
    env: docker
    plan: starter
    envVars:
      - key: OPENROUTER_API_KEY
        sync: false
      - key: PYTHON_VERSION
        value: 3.9.0
    disk:
      name: chroma-data
      mountPath: /app/chroma_db
      sizeGB: 1
```

**3. Deploy to Render**:
1. Go to https://render.com
2. Sign up/Login
3. Click "New +"
4. Select "Web Service"
5. Connect your GitHub repo
6. Configure:
   - Name: `sfda-inspector-assistant`
   - Environment: Docker
   - Plan: Starter ($7/month)
7. Add environment variables
8. Click "Create Web Service"

---

## Pre-Deployment Checklist

### Code Preparation
- [ ] All secrets in `.env.example` (not `.env`)
- [ ] `.gitignore` includes `.env`
- [ ] `requirements.txt` is up to date
- [ ] Code works locally
- [ ] ChromaDB is built and tested

### Data Preparation
- [ ] Knowledge files are in `knowledge/` directory
- [ ] ChromaDB is built (or script to build on startup)
- [ ] Test queries work

### Configuration
- [ ] `config.py` reads from environment variables
- [ ] No hardcoded API keys
- [ ] No hardcoded file paths
- [ ] Logging configured (not too verbose for production)

### Testing
- [ ] Test locally with `DEBUG=False`
- [ ] Test with production API keys
- [ ] Verify memory usage (<4GB)
- [ ] Test on different queries

---

## Post-Deployment

### 1. Monitor Performance

**Metrics to Track**:
- Response time
- Error rate
- User queries
- Memory/CPU usage

**Tools**:
- Hugging Face Spaces: Built-in analytics
- Custom logging: Add to `app_gradio_improved.py`

```python
import logging
logging.basicConfig(
    filename='queries.log',
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)

# In stream_response():
logging.info(f"Query: {message} | Source: {source_choice}")
```

### 2. Collect User Feedback

Add feedback buttons to Gradio:

```python
with gr.Blocks() as demo:
    # ... existing code ...

    feedback = gr.Radio(
        choices=["👍 مفيد", "👎 غير مفيد"],
        label="هل كانت الإجابة مفيدة؟"
    )

    feedback.change(
        fn=log_feedback,
        inputs=[feedback],
        outputs=None
    )
```

### 3. Update Knowledge Base

**When regulations change:**

```bash
# Locally:
1. Add new files to knowledge/
2. Run: python ingest_database_improved.py
3. Commit chroma_db/ changes
4. Push to deployment

# Or via script:
python update_knowledge.py
```

---

## Cost Estimation

### Free Tier Options

| Platform | Cost | Limits |
|----------|------|--------|
| **Hugging Face Spaces** | Free | CPU Basic, 16GB storage, public apps |
| **Streamlit Cloud** | Free | 1GB RAM, 3 apps max |

### Paid Options

| Platform | Cost/Month | Specs |
|----------|------------|-------|
| **HF Spaces (CPU Upgraded)** | $0 | 8 CPU cores, 32GB RAM |
| **HF Spaces (GPU)** | $60 | T4 GPU, 16GB VRAM |
| **Render Starter** | $7 | 512MB RAM, 0.5 CPU |
| **Render Standard** | $25 | 2GB RAM, 1 CPU |

**Recommendation**: Start with **HF Spaces Free**, upgrade if needed.

---

## Performance Optimization

### 1. Model Optimization

**Use Smaller Embedding Model**:
```env
# .env
EMBEDDING_MODEL=intfloat/multilingual-e5-base  # 560MB vs 1.12GB
```

**Trade-off**: Slightly lower accuracy (~2%) for 50% size reduction.

### 2. Caching

**Cache LLM Responses** (for common queries):
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_response(query: str, source: str):
    # Your response generation logic
    pass
```

### 3. Lazy Loading

**Load models only when needed**:
```python
class SFDAChatbot:
    def __init__(self):
        self._llm = None
        self._embeddings = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(...)
        return self._llm
```

---

## Security Best Practices

### 1. API Key Security
- ✅ Use environment variables
- ✅ Never commit `.env` to git
- ✅ Rotate keys regularly
- ✅ Use read-only keys when possible

### 2. Input Validation
```python
def validate_query(query: str) -> bool:
    if len(query) > 500:  # Prevent abuse
        return False
    if any(char in query for char in ['<', '>', '{', '}']):  # Basic XSS
        return False
    return True
```

### 3. Rate Limiting
```python
from functools import wraps
from time import time

query_times = {}

def rate_limit(max_calls=10, time_window=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_id = get_user_id()  # From session/IP
            now = time()

            if user_id in query_times:
                recent = [t for t in query_times[user_id] if now - t < time_window]
                if len(recent) >= max_calls:
                    raise Exception("تم تجاوز الحد الأقصى للطلبات")
                query_times[user_id] = recent + [now]
            else:
                query_times[user_id] = [now]

            return func(*args, **kwargs)
        return wrapper
    return decorator
```

---

## Troubleshooting Deployment

### Common Issues

#### Issue 1: "ModuleNotFoundError"
**Solution**: Ensure `requirements.txt` includes all dependencies
```bash
pip freeze > requirements.txt
```

#### Issue 2: "ChromaDB not found"
**Solution**: Build ChromaDB during deployment or upload pre-built

#### Issue 3: "API Key Error"
**Solution**: Check environment variables are set correctly in platform

#### Issue 4: "Out of Memory"
**Solution**:
- Use smaller embedding model
- Reduce `BATCH_SIZE` in config
- Upgrade to paid tier

#### Issue 5: "Slow First Response"
**Solution**: Model cold start. Options:
- Accept it (5-10s first query)
- Use model warm-up on startup
- Upgrade to always-on instance

---

## Monitoring & Analytics

### Add Custom Analytics

**Create `analytics.py`**:
```python
import json
from datetime import datetime
from pathlib import Path

ANALYTICS_FILE = "analytics.json"

def log_query(query: str, source: str, response_time: float):
    """Log query for analytics."""
    data = []
    if Path(ANALYTICS_FILE).exists():
        with open(ANALYTICS_FILE, 'r') as f:
            data = json.load(f)

    data.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "source": source,
        "response_time": response_time
    })

    with open(ANALYTICS_FILE, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_stats():
    """Get usage statistics."""
    if not Path(ANALYTICS_FILE).exists():
        return {}

    with open(ANALYTICS_FILE, 'r') as f:
        data = json.load(f)

    return {
        "total_queries": len(data),
        "avg_response_time": sum(d["response_time"] for d in data) / len(data),
        "top_sources": ...  # Calculate most used sources
    }
```

**Add to your app**:
```python
# In stream_response():
start_time = time.time()
# ... generate response ...
response_time = time.time() - start_time
log_query(message, source_choice, response_time)
```

---

## Success Metrics

### KPIs to Track

**Technical**:
- Uptime: >99%
- Response time: <5s (95th percentile)
- Error rate: <1%

**User Engagement**:
- Daily active users
- Queries per user
- User feedback (thumbs up/down)
- Most common queries

**Business Impact**:
- Time saved per inspector
- Accuracy of responses
- Adoption rate

---

## Deployment Summary

### Recommended Path

**For Capstone Project**:
1. ✅ Deploy to **Hugging Face Spaces** (free, easy)
2. ✅ Use **CPU Basic** tier (sufficient)
3. ✅ Upload pre-built ChromaDB (faster startup)
4. ✅ Add feedback collection
5. ✅ Monitor analytics

**For Production (Real SFDA Use)**:
1. ✅ Start with HF Spaces to validate
2. ✅ Collect usage data (1-2 months)
3. ✅ Migrate to dedicated server if needed
4. ✅ Add authentication (SFDA SSO)
5. ✅ Implement advanced monitoring

---

## Deployment Checklist

- [ ] Code pushed to GitHub/HF
- [ ] Environment variables configured
- [ ] ChromaDB built
- [ ] App running on platform
- [ ] Public URL working
- [ ] Test queries verified
- [ ] Error handling tested
- [ ] Feedback mechanism added
- [ ] Analytics configured
- [ ] Documentation updated with live URL

---

**Your Deployed App**: `https://huggingface.co/spaces/YOUR_USERNAME/sfda-inspector-assistant`

**Share this URL in your presentation!** 🚀

---

**Need Help?**
- HF Spaces Docs: https://huggingface.co/docs/hub/spaces
- Gradio Deployment: https://gradio.app/guides/sharing-your-app
- Discord: Hugging Face Discord community
