# تحسينات Architecture ضمن Best Practices
**التاريخ:** 2026-01-31
**المشروع:** SFDA Legal Assistant

---

## 📋 تحليل مقترحاتك الحالية

### ✅ نقاط قوة في مقترحاتك:

1. **Query Classification** - ممتاز!
2. **Cache Layer** - ضروري جداً!
3. **Source Tagging** - مطلوب في MVP!
4. **Checkboxes للمصادر** - UX ممتاز!

---

## 🚀 تحسينات Best Practices الإضافية

### 1. **Structured Logging & Monitoring**

```python
# ✅ Best Practice: Structured logging
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name="sfda_assistant"):
        self.logger = logging.getLogger(name)

    def log_query(self, query, query_type, response_time, cache_hit=False):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query[:100],  # أول 100 حرف فقط
            "query_type": query_type,
            "response_time_ms": response_time,
            "cache_hit": cache_hit,
            "success": True
        }
        self.logger.info(json.dumps(log_data, ensure_ascii=False))

    def log_error(self, query, error, stack_trace):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query[:100],
            "error": str(error),
            "stack_trace": stack_trace,
            "success": False
        }
        self.logger.error(json.dumps(log_data, ensure_ascii=False))

# استخدام:
logger = StructuredLogger()
logger.log_query("ما هي المادة الرابعة؟", "rag", 234, cache_hit=True)
```

**الفائدة:**
- تتبع الأداء
- تحليل الأسئلة الشائعة
- Debug سريع
- Analytics لاحقاً

---

### 2. **Cache Expiration & Invalidation Strategy**

```python
# ✅ Best Practice: Smart caching مع expiration
from datetime import datetime, timedelta
import hashlib

class SmartCache:
    def __init__(self, ttl_hours=24):
        self.cache = {}
        self.ttl = timedelta(hours=ttl_hours)

    def _hash_query(self, query, filters):
        """Create unique hash للـ query + filters"""
        key = f"{query}:{json.dumps(filters, sort_keys=True)}"
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, query, filters=None):
        filters = filters or {}
        key = self._hash_query(query, filters)

        if key in self.cache:
            entry = self.cache[key]
            # ✅ تحقق من expiration
            if datetime.now() - entry["timestamp"] < self.ttl:
                return entry["response"]
            else:
                # ✅ امسح الـ expired
                del self.cache[key]

        return None

    def set(self, query, response, filters=None):
        filters = filters or {}
        key = self._hash_query(query, filters)
        self.cache[key] = {
            "response": response,
            "timestamp": datetime.now(),
            "hit_count": 0
        }

    def invalidate_pattern(self, pattern):
        """امسح كل الـ cache المتعلق بموضوع معين"""
        keys_to_delete = [k for k in self.cache.keys() if pattern in k]
        for k in keys_to_delete:
            del self.cache[k]

    def get_stats(self):
        """إحصائيات الـ cache"""
        return {
            "total_entries": len(self.cache),
            "memory_usage_mb": len(str(self.cache)) / (1024 * 1024),
            "oldest_entry": min((v["timestamp"] for v in self.cache.values()), default=None)
        }

# استخدام:
cache = SmartCache(ttl_hours=24)

# حفظ
cache.set("ما هي المادة 4؟", "نص المادة...", {"source": "regulations"})

# استرجاع
result = cache.get("ما هي المادة 4؟", {"source": "regulations"})

# invalidate عند تحديث البيانات
cache.invalidate_pattern("المادة 4")
```

---

### 3. **Rate Limiting & Cost Control**

```python
# ✅ Best Practice: منع الاستخدام المفرط للـ API
from collections import defaultdict
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, max_requests_per_minute=10, max_cost_per_hour=1.0):
        self.requests = defaultdict(list)  # user_id -> [timestamps]
        self.costs = defaultdict(float)    # user_id -> total_cost
        self.max_rpm = max_requests_per_minute
        self.max_cost = max_cost_per_hour

    def check_rate_limit(self, user_id="default"):
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # امسح الـ old requests
        self.requests[user_id] = [
            ts for ts in self.requests[user_id]
            if ts > minute_ago
        ]

        # تحقق من الحد
        if len(self.requests[user_id]) >= self.max_rpm:
            return False, f"تجاوزت الحد المسموح ({self.max_rpm} طلب/دقيقة)"

        return True, None

    def record_request(self, user_id="default", cost=0.0):
        self.requests[user_id].append(datetime.now())
        self.costs[user_id] += cost

    def check_cost_limit(self, user_id="default"):
        if self.costs[user_id] >= self.max_cost:
            return False, f"تجاوزت ميزانية التكلفة ({self.max_cost}$/ساعة)"
        return True, None

# استخدام في Gradio:
rate_limiter = RateLimiter(max_requests_per_minute=20)

def handle_query(query, user_session):
    # ✅ تحقق من rate limit
    allowed, error = rate_limiter.check_rate_limit(user_session)
    if not allowed:
        return f"⚠️ {error}. يرجى الانتظار قليلاً."

    # معالجة...
    response = process_query(query)

    # ✅ سجل التكلفة
    rate_limiter.record_request(user_session, cost=0.001)

    return response
```

---

### 4. **Query Validation & Sanitization**

```python
# ✅ Best Practice: تنظيف وفحص المدخلات
import re

class QueryValidator:
    # ✅ قائمة بالأسئلة غير المسموحة
    FORBIDDEN_PATTERNS = [
        r"كيف أتهرب من",
        r"كيف أخالف",
        r"طريقة غش",
        r"ignore (previous|above) instructions",  # Prompt injection
        r"<script",  # XSS attempt
    ]

    # ✅ الحد الأقصى لطول السؤال
    MAX_LENGTH = 500

    @staticmethod
    def is_valid(query):
        """تحقق من صحة السؤال"""
        if not query or not query.strip():
            return False, "السؤال فارغ"

        if len(query) > QueryValidator.MAX_LENGTH:
            return False, f"السؤال طويل جداً (الحد الأقصى {QueryValidator.MAX_LENGTH} حرف)"

        # ✅ تحقق من محاولات prompt injection
        for pattern in QueryValidator.FORBIDDEN_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "السؤال يحتوي على محتوى غير مسموح"

        return True, None

    @staticmethod
    def sanitize(query):
        """تنظيف السؤال"""
        # إزالة HTML tags
        query = re.sub(r'<[^>]+>', '', query)
        # إزالة المسافات الزائدة
        query = re.sub(r'\s+', ' ', query).strip()
        # إزالة الرموز الخطرة
        query = query.replace('\x00', '')
        return query

# استخدام:
validator = QueryValidator()

def process_user_query(raw_query):
    # ✅ تنظيف
    clean_query = validator.sanitize(raw_query)

    # ✅ فحص
    is_valid, error = validator.is_valid(clean_query)
    if not is_valid:
        return f"⚠️ خطأ: {error}"

    # معالجة...
    return handle_query(clean_query)
```

---

### 5. **Graceful Degradation & Fallbacks**

```python
# ✅ Best Practice: خطة بديلة عند فشل أي component
class ResilientQueryHandler:
    def __init__(self, primary_llm, fallback_llm=None):
        self.primary = primary_llm
        self.fallback = fallback_llm
        self.cache = SmartCache()

    def handle(self, query):
        # ✅ المستوى 1: Cache
        cached = self.cache.get(query)
        if cached:
            return cached, "cache"

        # ✅ المستوى 2: Primary LLM
        try:
            response = self.primary.invoke(query)
            self.cache.set(query, response)
            return response, "primary_llm"
        except Exception as e:
            logger.log_error(query, e, traceback.format_exc())

            # ✅ المستوى 3: Fallback LLM
            if self.fallback:
                try:
                    response = self.fallback.invoke(query)
                    self.cache.set(query, response)
                    return response, "fallback_llm"
                except Exception as e2:
                    logger.log_error(query, e2, traceback.format_exc())

            # ✅ المستوى 4: Static response
            return self._get_error_message(), "error"

    def _get_error_message(self):
        return """
        ⚠️ عذراً، حدث خطأ مؤقت في النظام.

        يمكنك:
        - إعادة المحاولة بعد قليل
        - التواصل مع الدعم الفني
        - زيارة الموقع الرسمي: https://www.sfda.gov.sa
        """
```

---

### 6. **Performance Monitoring Dashboard**

```python
# ✅ Best Practice: Dashboard بسيط للمراقبة
import gradio as gr
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "avg_response_time": 0,
            "errors": 0,
            "last_update": datetime.now()
        }

    def record(self, response_time, cache_hit, error=False):
        self.metrics["total_queries"] += 1
        if cache_hit:
            self.metrics["cache_hits"] += 1
        if error:
            self.metrics["errors"] += 1

        # Update avg response time
        n = self.metrics["total_queries"]
        current_avg = self.metrics["avg_response_time"]
        self.metrics["avg_response_time"] = (
            (current_avg * (n - 1) + response_time) / n
        )
        self.metrics["last_update"] = datetime.now()

    def get_dashboard_html(self):
        cache_rate = (
            self.metrics["cache_hits"] / self.metrics["total_queries"] * 100
            if self.metrics["total_queries"] > 0 else 0
        )
        error_rate = (
            self.metrics["errors"] / self.metrics["total_queries"] * 100
            if self.metrics["total_queries"] > 0 else 0
        )

        return f"""
        <div style="padding: 20px; background: #f5f5f5; border-radius: 8px;">
            <h3>📊 Performance Metrics</h3>
            <ul>
                <li>Total Queries: {self.metrics["total_queries"]}</li>
                <li>Cache Hit Rate: {cache_rate:.1f}%</li>
                <li>Avg Response Time: {self.metrics["avg_response_time"]:.0f}ms</li>
                <li>Error Rate: {error_rate:.1f}%</li>
                <li>Last Update: {self.metrics["last_update"].strftime('%Y-%m-%d %H:%M:%S')}</li>
            </ul>
        </div>
        """

# إضافة في Gradio:
monitor = PerformanceMonitor()

with gr.Blocks() as demo:
    # ... UI الرئيسي ...

    # ✅ إضافة admin tab
    with gr.Tab("Admin Dashboard"):
        metrics_display = gr.HTML(monitor.get_dashboard_html())
        refresh_btn = gr.Button("Refresh Metrics")
        refresh_btn.click(
            lambda: monitor.get_dashboard_html(),
            outputs=metrics_display
        )
```

---

### 7. **Database Connection Pooling**

```python
# ✅ Best Practice: إعادة استخدام الـ connections
from contextlib import contextmanager

class ChromaDBPool:
    """Connection pool للـ ChromaDB"""
    def __init__(self, chroma_path, collection_name, embeddings):
        self._path = chroma_path
        self._collection = collection_name
        self._embeddings = embeddings
        self._connection = None

    @property
    def connection(self):
        """Lazy loading - اتصال واحد فقط"""
        if self._connection is None:
            self._connection = Chroma(
                collection_name=self._collection,
                embedding_function=self._embeddings,
                persist_directory=self._path,
            )
        return self._connection

    def health_check(self):
        """تحقق من صحة الاتصال"""
        try:
            count = self.connection._collection.count()
            return True, f"Connected. {count} documents."
        except Exception as e:
            return False, f"Connection error: {e}"

# استخدام:
# ✅ اتصال واحد يُعاد استخدامه
db_pool = ChromaDBPool(CHROMA_PATH, COLLECTION_NAME, embeddings_model)

def query_db(query_text):
    return db_pool.connection.similarity_search(query_text, k=5)
```

---

### 8. **Configuration Management**

```python
# ✅ Best Practice: إعدادات مركزية
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class AppConfig:
    """إعدادات التطبيق المركزية"""
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    CHROMA_PATH: Path = BASE_DIR / "chroma_db"
    KNOWLEDGE_PATH: Path = BASE_DIR / "knowledge"

    # Database
    COLLECTION_NAME: str = "sfda_collection"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"

    # LLM
    LLM_MODEL: str = "deepseek/deepseek-chat"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 700

    # Cache
    CACHE_TTL_HOURS: int = 24
    CACHE_MAX_SIZE: int = 1000

    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = 20
    MAX_COST_PER_HOUR: float = 1.0

    # Performance
    RETRIEVAL_TOP_K: int = 8
    RESPONSE_TIMEOUT_SEC: int = 30

    # Features
    ENABLE_CACHE: bool = True
    ENABLE_LOGGING: bool = True
    ENABLE_RATE_LIMITING: bool = True

    @classmethod
    def from_env(cls):
        """تحميل من environment variables"""
        return cls(
            LLM_TEMPERATURE=float(os.getenv("LLM_TEMPERATURE", 0.0)),
            CACHE_TTL_HOURS=int(os.getenv("CACHE_TTL_HOURS", 24)),
            # ... etc
        )

# استخدام:
config = AppConfig.from_env()

llm = ChatOpenAI(
    model=config.LLM_MODEL,
    temperature=config.LLM_TEMPERATURE,
    max_tokens=config.LLM_MAX_TOKENS,
)
```

---

## 🎯 خطة التنفيذ الموصى بها (3 أيام)

### اليوم 1: Core Improvements
- [ ] Smart Cache مع expiration
- [ ] Query Validation & Sanitization
- [ ] Structured Logging
- [ ] Error Handling محسّن

### اليوم 2: Performance & Security
- [ ] Rate Limiting
- [ ] Connection Pooling
- [ ] Graceful Degradation
- [ ] Configuration Management

### اليوم 3: Monitoring & Testing
- [ ] Performance Monitor
- [ ] Integration Testing
- [ ] Load Testing
- [ ] Documentation Update

---

## 📊 Metrics للنجاح

| المقياس | الهدف | الحالي |
|---------|-------|--------|
| Cache Hit Rate | > 60% | 0% |
| Avg Response Time | < 500ms | ؟ |
| Error Rate | < 1% | ؟ |
| Cost per Query | < $0.001 | ؟ |

---

## 🔒 Security Checklist

- [ ] Input validation
- [ ] Rate limiting
- [ ] API key rotation
- [ ] Logging (لا تحفظ بيانات حساسة)
- [ ] HTTPS only (في production)
- [ ] CORS configuration
- [ ] SQL injection prevention (N/A - نستخدم ChromaDB)
- [ ] XSS prevention

---

## 📚 Resources

- [LangChain Best Practices](https://python.langchain.com/docs/guides/productionization/)
- [Gradio Security Guide](https://www.gradio.app/guides/security-and-file-access)
- [ChromaDB Production Guide](https://docs.trychroma.com/deployment)

---

**آخر تحديث:** 2026-01-31
**الحالة:** Ready for Implementation ✅
