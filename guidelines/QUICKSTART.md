# 🚀 دليل البدء السريع - SFDA Agent

**ابدأ في 3 خطوات فقط!**

---

## الخطوة 1️⃣: التثبيت

```bash
# تثبيت المكتبات المطلوبة
pip install langgraph langchain-anthropic beautifulsoup4 lxml python-dotenv
```

أو باستخدام ملف requirements:
```bash
pip install -r agent_requirements.txt
```

---

## الخطوة 2️⃣: إعداد API Key

### طريقة 1: ملف .env (مُوصى بها)

```bash
# انسخ ملف .env.example
cp .env.example .env

# عدّل الملف وأضف مفتاحك
# في Windows:
notepad .env

# في Linux/Mac:
nano .env
```

أضف مفتاحك:
```
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxx
```

### طريقة 2: متغير البيئة

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-api03-xxxxxxxxxxxxx"
```

**Windows (CMD):**
```cmd
set ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxx
```

**Linux/Mac:**
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-xxxxxxxxxxxxx"
```

---

## الخطوة 3️⃣: التشغيل!

```bash
python latest_agent.py
```

---

## 🎯 أمثلة سريعة

### مثال 1: استخدام بسيط

```python
from latest_agent import create_sfda_agent, chat_with_agent

# إنشاء Agent
agent = create_sfda_agent()

# محادثة
response = chat_with_agent(agent, "ابحث عن دواء برقم التسجيل 12345")
print(response)
```

### مثال 2: محادثة متعددة

```python
agent = create_sfda_agent()

# رسالة 1
chat_with_agent(agent, "مرحباً", thread_id="session_1")

# رسالة 2 (سيتذكر السياق!)
chat_with_agent(agent, "ابحث عن دواء X", thread_id="session_1")
```

### مثال 3: عرض سجل المحادثة

```python
from latest_agent import print_conversation_history

print_conversation_history(agent, thread_id="session_1")
```

---

## ⚠️ استكشاف الأخطاء

### خطأ: ModuleNotFoundError

```bash
# تأكد من تثبيت جميع المكتبات
pip install -r agent_requirements.txt
```

### خطأ: API Key not found

```bash
# تأكد من إعداد المفتاح
echo $ANTHROPIC_API_KEY  # Linux/Mac
echo %ANTHROPIC_API_KEY%  # Windows CMD
```

### خطأ: Connection timeout

- تحقق من اتصال الإنترنت
- تحقق من إمكانية الوصول لموقع SFDA

---

## 📚 المزيد من المعلومات

راجع [AGENT_README.md](AGENT_README.md) للتفاصيل الكاملة.

---

**جاهز للبدء؟ شغّل الأمر:**

```bash
python latest_agent.py
```
