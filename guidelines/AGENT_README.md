# 🤖 SFDA Drug Search Agent

**وكيل ذكي للبحث عن الأدوية المسجلة في الهيئة العامة للغذاء والدواء السعودية**

> بُني باستخدام **LangChain 2026** مع **LangGraph StateGraph** و **MemorySaver**

---

## 📋 المحتويات

- [نظرة عامة](#نظرة-عامة)
- [المميزات](#المميزات)
- [التثبيت](#التثبيت)
- [الاستخدام](#الاستخدام)
- [البنية التقنية](#البنية-التقنية)
- [التطوير المستقبلي](#التطوير-المستقبلي)

---

## 🎯 نظرة عامة

هذا الـ Agent تم بناؤه باستخدام **أحدث تقنيات LangChain (2026)** لتوفير واجهة محادثة ذكية للبحث عن الأدوية المسجلة في موقع [الهيئة العامة للغذاء والدواء السعودية (SFDA)](https://www.sfda.gov.sa/ar/drugs-list).

### لماذا LangGraph؟

تم اختيار **LangGraph** مع **StateGraph** بدلاً من الحلول الأخرى للأسباب التالية:

1. **إدارة Workflow متقدمة**: StateGraph يوفر تحكم كامل في سير عمل الـ Agent
2. **MemorySaver المدمج**: يحفظ المحادثات تلقائياً بدون الحاجة لقواعد بيانات خارجية
3. **توافق مع LangChain 2026**: استخدام أحدث APIs والممارسات
4. **Conditional Edges**: القدرة على اتخاذ قرارات ذكية (متى نستخدم الأدوات؟)
5. **Type Safety**: استخدام TypedDict لتعريف الحالة بشكل آمن

---

## ✨ المميزات

### 1. **StateGraph من LangGraph**
```python
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
```
- إدارة الحالة بشكل منظم
- Nodes و Edges واضحة
- Conditional routing

### 2. **MemorySaver للذاكرة**
```python
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```
- يتذكر المحادثات السابقة
- كل محادثة لها `thread_id` فريد
- لا حاجة لقواعد بيانات خارجية

### 3. **أدوات مخصصة (Custom Tools)**
```python
@tool
def search_sfda_drug(registration_number: str) -> str:
    """البحث عن دواء في موقع SFDA"""
```
- استخدام `@tool` decorator من LangChain
- توثيق واضح للأدوات
- معالجة أخطاء شاملة

### 4. **معالجة الأخطاء الشاملة**
- ✅ تحقق من توفر الموقع
- ⏰ معالجة Timeout
- 🔌 معالجة فشل الاتصال
- ❌ رسائل خطأ واضحة بالعربية

---

## 🚀 التثبيت

### 1. تثبيت المكتبات المطلوبة

```bash
# تثبيت المكتبات الأساسية
pip install -r agent_requirements.txt

# أو التثبيت اليدوي:
pip install langgraph langchain-anthropic beautifulsoup4 lxml
```

### 2. إعداد مفتاح API

أنشئ ملف `.env` في المجلد الرئيسي:

```bash
# .env
ANTHROPIC_API_KEY=your-api-key-here
```

أو قم بتصدير المتغير:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

---

## 💻 الاستخدام

### الاستخدام الأساسي

```python
from latest_agent import create_sfda_agent, chat_with_agent

# إنشاء الـ Agent
agent = create_sfda_agent()

# محادثة مع الـ Agent
response = chat_with_agent(
    agent,
    "ابحث عن دواء برقم التسجيل 12345",
    thread_id="user_123"
)

print(response)
```

### تشغيل الأمثلة

```bash
python latest_agent.py
```

سيقوم البرنامج بتشغيل أمثلة توضيحية:
- ✅ محادثة عامة
- ✅ الحصول على معلومات عن موقع SFDA
- ✅ البحث عن دواء

### مثال على المحادثة

```
👤 المستخدم: مرحباً، أريد البحث عن دواء

🤖 الوكيل: أهلاً وسهلاً! يمكنني مساعدتك في البحث عن الأدوية
المسجلة في الهيئة العامة للغذاء والدواء السعودية.

للبحث الدقيق، يُفضل استخدام رقم التسجيل. هل لديك رقم تسجيل الدواء؟

---

👤 المستخدم: ابحث عن دواء برقم التسجيل 12345

🤖 الوكيل: [يستخدم أداة search_sfda_drug]
سأقوم بالبحث عن الدواء في موقع الهيئة...
```

### المحادثات المتعددة (Multi-thread)

```python
# محادثة المستخدم الأول
chat_with_agent(agent, "ابحث عن دواء X", thread_id="user_1")

# محادثة المستخدم الثاني (محادثة منفصلة)
chat_with_agent(agent, "ابحث عن دواء Y", thread_id="user_2")

# الرجوع لمحادثة المستخدم الأول (سيتذكر السياق!)
chat_with_agent(agent, "هل وجدت معلومات أخرى؟", thread_id="user_1")
```

---

## 🏗️ البنية التقنية

### معمارية الـ Agent

```
┌─────────────────────────────────────────┐
│          User Input (HumanMessage)      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         Agent Node (LLM + Tools)        │
│   - يقرأ الرسالة                         │
│   - يقرر: هل أحتاج أداة؟                 │
└──────────────────┬──────────────────────┘
                   │
           ┌───────┴────────┐
           │                │
           ▼                ▼
    ┌──────────┐      ┌─────────┐
    │   END    │      │  Tools  │
    └──────────┘      │  Node   │
                      └────┬─────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Back to     │
                    │ Agent Node  │
                    └─────────────┘
```

### Components الرئيسية

#### 1. **AgentState (TypedDict)**
```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "سجل المحادثات"]
```
- يحتفظ بجميع الرسائل
- Type-safe مع Annotated types

#### 2. **Tools (الأدوات)**

**أداة 1: `search_sfda_drug`**
- **الهدف**: البحث عن دواء برقم التسجيل
- **المدخلات**: `registration_number` (str)
- **المخرجات**: معلومات الدواء أو رسالة خطأ

**أداة 2: `get_sfda_website_info`**
- **الهدف**: معلومات عامة عن موقع SFDA
- **المدخلات**: لا شيء
- **المخرجات**: معلومات عن الموقع

#### 3. **Workflow (StateGraph)**

```python
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

# Entry point
workflow.set_entry_point("agent")

# Conditional edges
workflow.add_conditional_edges("agent", should_continue)

# Regular edge
workflow.add_edge("tools", "agent")
```

#### 4. **Memory (MemorySaver)**
```python
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```
- يحفظ الحالة تلقائياً
- كل `thread_id` له ذاكرة منفصلة

---

## 🔮 التطوير المستقبلي

### المرحلة 1: البحث الديناميكي الكامل ⚙️

**المشكلة الحالية**: موقع SFDA يستخدم JavaScript للبحث الديناميكي، لذا نحتاج لأداة تتفاعل مع الصفحة.

**الحل**: استخدام **Selenium** أو **Playwright**

#### خيار 1: Selenium
```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

@tool
def search_sfda_drug_selenium(registration_number: str) -> str:
    driver = webdriver.Chrome()
    driver.get("https://www.sfda.gov.sa/ar/drugs-list")

    # البحث في خانة رقم التسجيل
    search_box = driver.find_element(By.ID, "registration-number-input")
    search_box.send_keys(registration_number)

    # الضغط على زر البحث
    search_button = driver.find_element(By.CSS_SELECTOR, "button.search")
    search_button.click()

    # انتظار النتائج
    WebDriverWait(driver, 10).until(...)

    # استخراج النتائج من الجدول
    results = driver.find_elements(By.CSS_SELECTOR, "table tr")

    driver.quit()
    return results
```

**التثبيت**:
```bash
pip install selenium
```

#### خيار 2: Playwright (أسرع وأحدث)
```python
from playwright.sync_api import sync_playwright

@tool
def search_sfda_drug_playwright(registration_number: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("https://www.sfda.gov.sa/ar/drugs-list")

        # إدخال رقم التسجيل
        page.fill("input#registration-number", registration_number)
        page.click("button.search")

        # استخراج النتائج
        results = page.query_selector_all("table tr")

        browser.close()
        return results
```

**التثبيت**:
```bash
pip install playwright
playwright install
```

### المرحلة 2: دعم أنواع البحث المتعددة 📊

إضافة أدوات للبحث بـ:
- الاسم التجاري
- الاسم العلمي
- الشركة الصانعة
- الوكيل

```python
@tool
def search_by_trade_name(trade_name: str) -> str:
    """البحث بالاسم التجاري"""

@tool
def search_by_scientific_name(scientific_name: str) -> str:
    """البحث بالاسم العلمي"""
```

### المرحلة 3: دعم فئات الأدوية 💊

```python
from enum import Enum

class DrugCategory(Enum):
    HUMAN = "الأدوية البشرية"
    VETERINARY = "الأدوية البيطرية"
    HERBAL = "المستحضرات العشبية والفيتامينات"

@tool
def search_by_category(
    registration_number: str,
    category: DrugCategory
) -> str:
    """البحث مع تحديد فئة الدواء"""
```

### المرحلة 4: تصدير النتائج 📄

```python
@tool
def export_results_to_pdf(drug_info: dict) -> str:
    """تصدير معلومات الدواء إلى PDF"""

@tool
def export_results_to_excel(drugs: list) -> str:
    """تصدير قائمة الأدوية إلى Excel"""
```

### المرحلة 5: RAG للمعلومات الدوائية 🧠

إضافة قاعدة معرفية للأدوية:

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# بناء قاعدة معرفية من بيانات SFDA
vectorstore = Chroma.from_documents(
    documents=sfda_documents,
    embedding=HuggingFaceEmbeddings()
)

@tool
def search_drug_knowledge_base(query: str) -> str:
    """البحث في قاعدة المعرفة الدوائية"""
    results = vectorstore.similarity_search(query)
    return results
```

---

## 📚 الموارد والمراجع

### التوثيق الرسمي
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Anthropic API Documentation](https://docs.anthropic.com/)

### أمثلة مفيدة
- [LangGraph StateGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [Building Agents with LangGraph](https://blog.langchain.dev/langgraph-multi-agent-workflows/)

---

## 🤝 المساهمة

لتطوير الـ Agent:

1. Fork المشروع
2. أنشئ branch جديد (`git checkout -b feature/amazing-feature`)
3. Commit التغييرات (`git commit -m 'Add amazing feature'`)
4. Push للـ branch (`git push origin feature/amazing-feature`)
5. افتح Pull Request

---

## 📝 الترخيص

هذا المشروع مفتوح المصدر.

---

## 📞 التواصل

للأسئلة أو الاقتراحات، يرجى فتح Issue في المشروع.

---

**تم البناء باستخدام ❤️ و LangChain 2026**
