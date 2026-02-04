# خطة التنفيذ - 3 أيام
**المشروع:** SFDA Legal Assistant
**الهدف:** تطبيق Best Practices + MVP Requirements

---

## 📅 الجدول الزمني

```
┌─────────────┬────────────────────────────────────┐
│   اليوم     │            المهام                 │
├─────────────┼────────────────────────────────────┤
│   Day 1     │ ✅ Core Features + MVP            │
│   (8 hrs)   │ ✅ Caching + Validation            │
├─────────────┼────────────────────────────────────┤
│   Day 2     │ ✅ Integration + Architecture      │
│   (8 hrs)   │ ✅ Agent + RAG Unified             │
├─────────────┼────────────────────────────────────┤
│   Day 3     │ ✅ Testing + Documentation         │
│   (6 hrs)   │ ✅ Deployment Ready                │
└─────────────┴────────────────────────────────────┘
```

---

## 📋 Day 1: Core Features (8 hours)

### ✅ Morning Session (4 hours)

#### 1. إنشاء ملف `core/cache.py` (45 min)
```python
# d:\last_update\core\cache.py
from datetime import datetime, timedelta
import hashlib
import json

class SmartCache:
    def __init__(self, ttl_hours=24):
        self.cache = {}
        self.ttl = timedelta(hours=ttl_hours)

    def _hash(self, query, filters):
        key = f"{query}:{json.dumps(filters, sort_keys=True)}"
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, query, filters=None):
        filters = filters or {}
        key = self._hash(query, filters)

        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry["ts"] < self.ttl:
                return entry["response"]
            else:
                del self.cache[key]
        return None

    def set(self, query, response, filters=None):
        filters = filters or {}
        key = self._hash(query, filters)
        self.cache[key] = {
            "response": response,
            "ts": datetime.now()
        }

    def stats(self):
        return {
            "size": len(self.cache),
            "memory_kb": len(str(self.cache)) / 1024
        }
```

**Test:**
```bash
python -c "from core.cache import SmartCache; c=SmartCache(); c.set('test', 'result'); print(c.get('test'))"
```

---

#### 2. إنشاء ملف `core/validator.py` (45 min)
```python
# d:\last_update\core\validator.py
import re

class QueryValidator:
    FORBIDDEN = [
        r"ignore (previous|above) instructions",
        r"<script",
        r"كيف أتهرب",
    ]
    MAX_LENGTH = 500

    @staticmethod
    def is_valid(query):
        if not query or not query.strip():
            return False, "السؤال فارغ"

        if len(query) > QueryValidator.MAX_LENGTH:
            return False, f"طويل جداً (حد أقصى {QueryValidator.MAX_LENGTH})"

        for p in QueryValidator.FORBIDDEN:
            if re.search(p, query, re.I):
                return False, "محتوى غير مسموح"

        return True, None

    @staticmethod
    def sanitize(query):
        query = re.sub(r'<[^>]+>', '', query)
        query = re.sub(r'\s+', ' ', query).strip()
        return query
```

**Test:**
```python
from core.validator import QueryValidator
v = QueryValidator()
print(v.is_valid("ما هي المادة 4؟"))  # (True, None)
print(v.is_valid("<script>alert()</script>"))  # (False, ...)
```

---

#### 3. إنشاء ملف `core/logger.py` (30 min)
```python
# d:\last_update\core\logger.py
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name="sfda"):
        logging.basicConfig(
            filename=f"logs/{name}_{datetime.now().strftime('%Y%m%d')}.log",
            level=logging.INFO,
            format='%(message)s'
        )
        self.logger = logging.getLogger(name)

    def log_query(self, query, query_type, time_ms, cache_hit=False):
        self.logger.info(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "q": query[:100],
            "type": query_type,
            "time_ms": time_ms,
            "cache": cache_hit,
            "ok": True
        }, ensure_ascii=False))

    def log_error(self, query, error):
        self.logger.error(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "q": query[:100],
            "error": str(error),
            "ok": False
        }, ensure_ascii=False))
```

**Setup:**
```bash
mkdir logs
touch logs/.gitkeep
```

---

#### 4. تحديث `.env.example` (15 min)
```bash
# API Keys
OPENROUTER_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here  # للـ Agent

# Performance
CACHE_TTL_HOURS=24
MAX_REQUESTS_PER_MINUTE=20

# Features
ENABLE_CACHE=true
ENABLE_LOGGING=true
ENABLE_AGENT=true
```

---

### ✅ Afternoon Session (4 hours)

#### 5. إنشاء ملف `core/config.py` (30 min)
```python
# d:\last_update\core\config.py
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    CHROMA_PATH: Path = BASE_DIR / "chroma_db"

    # Database
    COLLECTION_NAME: str = "sfda_collection"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"

    # LLM
    LLM_MODEL: str = "deepseek/deepseek-chat"
    LLM_TEMP: float = 0.0
    LLM_TOKENS: int = 700

    # Cache
    CACHE_TTL: int = int(os.getenv("CACHE_TTL_HOURS", 24))
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"

    # API Keys
    OPENROUTER_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    ANTHROPIC_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

config = Config()
```

---

#### 6. إنشاء ملف `core/query_router.py` (1 hour)
```python
# d:\last_update\core\query_router.py
import re

class QueryRouter:
    """Routes queries إلى Agent أو RAG"""

    @staticmethod
    def classify(query: str, selected_sources: list) -> str:
        """
        Returns: "agent" or "rag"
        """
        # ✅ إذا اختار المستخدم "تسجيل الأدوية" بس -> agent
        if "تسجيل الأدوية" in selected_sources and len(selected_sources) == 1:
            return "agent"

        # ✅ إذا فيه رقم تسجيل واضح -> agent
        if QueryRouter.is_registration_number(query):
            return "agent"

        # ✅ الباقي -> RAG
        return "rag"

    @staticmethod
    def is_registration_number(query: str) -> bool:
        """تحقق إذا السؤال فيه رقم تسجيل"""
        patterns = [
            r"\d{4,6}",  # رقم طويل
            r"رقم التسجيل",
            r"registration\s+number",
        ]
        for p in patterns:
            if re.search(p, query, re.I):
                return True
        return False

# Test:
router = QueryRouter()
print(router.classify("رقم التسجيل 123456", []))  # "agent"
print(router.classify("ما هي المادة 4؟", []))      # "rag"
```

---

#### 7. دمج كل شيء في `app_unified.py` - الهيكل (2 hours)
```python
# d:\last_update\app_unified.py
import gradio as gr
import time
from core.config import config
from core.cache import SmartCache
from core.validator import QueryValidator
from core.logger import StructuredLogger
from core.query_router import QueryRouter

# Initialize
cache = SmartCache(ttl_hours=config.CACHE_TTL) if config.ENABLE_CACHE else None
validator = QueryValidator()
logger = StructuredLogger() if config.ENABLE_LOGGING else None
router = QueryRouter()

# TODO: Import RAG & Agent handlers
# from rag_handler import handle_rag_query
# from agent_handler import handle_agent_query

def unified_query_handler(message, history, selected_sources):
    """
    الـ handler الموحد
    """
    start_time = time.time()

    # ✅ 1. Validation
    clean_msg = validator.sanitize(message)
    is_valid, error = validator.is_valid(clean_msg)
    if not is_valid:
        return f"⚠️ {error}"

    # ✅ 2. Check Cache
    cache_key_filters = {"sources": selected_sources}
    if cache:
        cached = cache.get(clean_msg, cache_key_filters)
        if cached:
            response_time = (time.time() - start_time) * 1000
            if logger:
                logger.log_query(clean_msg, "cache", response_time, cache_hit=True)
            return cached

    # ✅ 3. Route Query
    query_type = router.classify(clean_msg, selected_sources)

    # ✅ 4. Execute
    try:
        if query_type == "agent":
            # response = handle_agent_query(clean_msg)
            response = f"🔍 [Agent Mode] سيتم البحث عن: {clean_msg}"
        else:
            # response = handle_rag_query(clean_msg, selected_sources)
            response = f"📚 [RAG Mode] سيتم البحث في: {selected_sources}"

        # ✅ 5. Save Cache
        if cache:
            cache.set(clean_msg, response, cache_key_filters)

        # ✅ 6. Log
        response_time = (time.time() - start_time) * 1000
        if logger:
            logger.log_query(clean_msg, query_type, response_time, cache_hit=False)

        return response

    except Exception as e:
        if logger:
            logger.log_error(clean_msg, e)
        return f"⚠️ حدث خطأ: {str(e)}"

# ✅ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# SANAD - المساعد القانوني")

    with gr.Row():
        sources = gr.CheckboxGroup(
            choices=[
                "اللوائح التنظيمية",
                "المواد المحظورة",
                "تسجيل الأدوية (Web Search)",
            ],
            value=["اللوائح التنظيمية"],
            label="اختر المصادر"
        )

    gr.ChatInterface(
        fn=unified_query_handler,
        additional_inputs=[sources],
    )

    # ✅ Admin Tab
    with gr.Tab("📊 Stats"):
        stats_html = gr.HTML()
        refresh = gr.Button("Refresh")

        def get_stats():
            if cache:
                s = cache.stats()
                return f"""
                <h3>Cache Stats</h3>
                <ul>
                    <li>Entries: {s['size']}</li>
                    <li>Memory: {s['memory_kb']:.1f} KB</li>
                </ul>
                """
            return "Cache disabled"

        refresh.click(get_stats, outputs=stats_html)

demo.launch(share=True)
```

**Test:**
```bash
python app_unified.py
```

---

#### 8. إنشاء هيكل المجلدات (15 min)
```bash
mkdir -p core
mkdir -p logs
touch core/__init__.py
touch core/cache.py
touch core/validator.py
touch core/logger.py
touch core/config.py
touch core/query_router.py
```

---

## ✅ End of Day 1 Checklist

- [ ] `core/cache.py` يعمل
- [ ] `core/validator.py` يعمل
- [ ] `core/logger.py` يحفظ logs
- [ ] `core/config.py` يقرأ `.env`
- [ ] `core/query_router.py` يميز بين agent/rag
- [ ] `app_unified.py` يشتغل (بدون RAG/Agent الحقيقي)
- [ ] الهيكل منظم ✅

---

## 📋 Day 2: Integration (8 hours)

### Morning: RAG Handler (4 hours)

#### 1. إنشاء `handlers/rag_handler.py`
- نسخ الكود من `app_gradio.py`
- تنظيفه وتبسيطه
- إضافة article citations محسّنة

#### 2. إنشاء `handlers/agent_handler.py`
- نسخ الكود من `latest_agent.py`
- دمجه مع الـ unified system

### Afternoon: Testing & Polish (4 hours)

#### 3. Integration Testing
- اختبار كامل RAG + Agent
- اختبار Cache
- اختبار Validation

#### 4. UI Polish
- تحسين رسائل الخطأ
- إضافة source tags واضحة
- تحسين formatting

---

## 📋 Day 3: Final Polish (6 hours)

### Morning: Documentation (3 hours)

#### 1. تحديث `README.md`
#### 2. كتابة `DEPLOYMENT.md`
#### 3. إنشاء `TESTING.md`

### Afternoon: Deployment Prep (3 hours)

#### 4. إنشاء `requirements.txt` نهائي
#### 5. اختبار نهائي شامل
#### 6. Git commit + push
#### 7. نسخة demo جاهزة

---

## 🎯 Deliverables

- [ ] `app_unified.py` - التطبيق الموحد
- [ ] `core/` - المكتبات الأساسية
- [ ] `handlers/` - RAG + Agent handlers
- [ ] `logs/` - Logging system
- [ ] `README.md` - توثيق كامل
- [ ] `DEPLOYMENT.md` - دليل النشر
- [ ] `requirements.txt` - كل المكتبات
- [ ] Tests جاهزة ✅

---

## 💡 Quick Commands

```bash
# Day 1 Setup
mkdir -p core logs handlers
pip install -r requirements.txt
python app_unified.py

# Day 2 Testing
python -m pytest tests/
python test_integration.py

# Day 3 Deploy
# راجع DEPLOYMENT.md
```

---

**آخر تحديث:** 2026-01-31
**الحالة:** Ready to Execute! 🚀
