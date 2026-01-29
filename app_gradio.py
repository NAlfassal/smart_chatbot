import os
import re
import gradio as gr
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage  # ✅ مهم

# -----------------------------
# Config
# -----------------------------
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "sfda_collection"

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file.")

# -----------------------------
# Models / Vector DB
# -----------------------------
embeddings_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cpu"},
)

llm = ChatOpenAI(
    model="deepseek/deepseek-chat",
    temperature=0.0,
    api_key=OPENROUTER_KEY,
    base_url="https://openrouter.ai/api/v1",
    max_tokens=700,
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Debug (اختياري)
print("APP CWD:", os.getcwd())
print("APP CHROMA PATH:", CHROMA_PATH)
try:
    print("COUNT:", vector_store._collection.count())
except Exception as e:
    print("COUNT error:", e)

# -----------------------------
# Source display name mapping
# -----------------------------
def display_source_name(raw_source: str) -> str:
    s = os.path.basename(raw_source or "N/A").strip()
    s_lower = s.lower()

    # محظورات
    if "banned_list" in s_lower or "محظور" in s or "قائمة شاملة بالمواد المحظورة" in s:
        return "محظورات التجميل"

    # لوائح
    if "اللائحة" in s or "نظام منتجات التجميل" in s or "sfda_articles" in s_lower:
        return "لوائح التجميل"

    return "مصادر إضافية"

def sources_footer_once(docs, mode_choice: str) -> str:
    # ✅ المحظورات: لا نعرض أسماء ملفات نهائياً
    if mode_choice == "محظورات التجميل":
        return "\n\n**المصدر:** محظورات التجميل"

    seen = set()
    sources = []
    for d in docs:
        raw = d.metadata.get("source", d.metadata.get("source_file", "N/A"))
        name = display_source_name(raw)
        if name and name not in seen:
            seen.add(name)
            sources.append(name)

    return "\n\n**المصدر:** " + ("، ".join(sources) if sources else "N/A")

# -----------------------------
# Arabic article helpers
# -----------------------------
AR_WORD_TO_NUM = {
    "الأولى": "1", "الاولى": "1",
    "الثانية": "2",
    "الثالثة": "3",
    "الرابعة": "4",
    "الخامسة": "5",
    "السادسة": "6",
    "السابعة": "7",
    "الثامنة": "8",
    "التاسعة": "9",
    "العاشرة": "10",
    "الحادية عشر": "11", "الحادية عشرة": "11",
    "الثانية عشر": "12", "الثانية عشرة": "12",
    "الثالثة عشر": "13", "الثالثة عشرة": "13",
    "الرابعة عشر": "14", "الرابعة عشرة": "14",
    "الخامسة عشر": "15", "الخامسة عشرة": "15",
    "السادسة عشر": "16", "السادسة عشرة": "16",
    "السابعة عشر": "17", "السابعة عشرة": "17",
    "الثامنة عشر": "18", "الثامنة عشرة": "18",
    "التاسعة عشر": "19", "التاسعة عشرة": "19",
    "العشرون": "20",
    "الحادية والعشرون": "21",
    "الثانية والعشرون": "22",
    "الثالثة والعشرون": "23",
    "الرابعة والعشرون": "24",
    "الخامسة والعشرون": "25",
    "السادسة والعشرون": "26",
    "السابعة والعشرون": "27",
    "الثامنة والعشرون": "28",
    "التاسعة والعشرون": "29",
    "الثلاثون": "30",
}

def normalize_article_to_num(article_value: str):
    if article_value is None:
        return None
    s = str(article_value).strip()
    s = s.replace("ـ", "")
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"^\s*المادة\s+", "", s).strip()

    if re.fullmatch(r"\d+", s):
        return s
    if s in AR_WORD_TO_NUM:
        return AR_WORD_TO_NUM[s]

    words = s.split()
    for n in (4, 3, 2, 1):
        if len(words) >= n:
            cand = " ".join(words[:n])
            if cand in AR_WORD_TO_NUM:
                return AR_WORD_TO_NUM[cand]
    return None

def extract_article_number(text: str):
    text = text or ""
    m = re.search(r"المادة\s+(\d+)", text)
    if m:
        return m.group(1)

    m = re.search(r"المادة\s+([^\n،,.؟!]+)", text)
    if not m:
        return None

    phrase = re.sub(r"\s{2,}", " ", m.group(1).replace("ـ", "")).strip()
    phrase = re.sub(r"^\s*المادة\s+", "", phrase).strip()

    if phrase in AR_WORD_TO_NUM:
        return AR_WORD_TO_NUM[phrase]

    words = phrase.split()
    for n in (4, 3, 2, 1):
        if len(words) >= n:
            cand = " ".join(words[:n])
            if cand in AR_WORD_TO_NUM:
                return AR_WORD_TO_NUM[cand]
    return None

# -----------------------------
# Text formatting
# -----------------------------
def clean_repeated_characters(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1", text or "")

def merge_spaced_arabic_letters(text: str) -> str:
    if not text:
        return ""
    t = text
    for _ in range(3):
        t = re.sub(
            r"(?<![ء-ي])((?:[ء-ي]\s+){2,}[ء-ي])(?![ء-ي])",
            lambda m: m.group(1).replace(" ", ""),
            t,
        )
    return t

def pretty_arabic_text(text: str) -> str:
    if not text:
        return ""
    t = merge_spaced_arabic_letters(text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("ـ", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# -----------------------------
# ✅ Article direct fetch (لوائح فقط)
# -----------------------------
def get_article_doc(article_num: str):
    target = str(article_num).strip()
    try:
        docs = vector_store.similarity_search(
            query=f"المادة {target}",
            k=3,
            filter={"$and": [{"article": target}, {"category": "regulation"}]},
        )
        if docs:
            return docs[0]
    except Exception as e:
        print("article filter (regulation) error:", e)

    try:
        docs = vector_store.similarity_search(
            query=f"المادة {target}",
            k=3,
            filter={"article": target},
        )
        if docs:
            return docs[0]
    except Exception as e:
        print("article filter (any) error:", e)

    return None

def format_article_output(doc: Document) -> str:
    art_num = normalize_article_to_num(doc.metadata.get("article", "")) or ""
    title = f"نص المادة ({art_num}) من لوائح التجميل" if art_num else "نص المادة من لوائح التجميل"
    body = pretty_arabic_text(doc.page_content)

    # إزالة "المادة X" من بداية النص
    if art_num:
        body = re.sub(rf"^\s*المادة\s*{re.escape(art_num)}\s*\n+", "", body)
        body = re.sub(rf"^\s*المادة\s*{re.escape(art_num)}\s*[:：]?\s*", "", body)

    return f"**{title}**\n\n{body}".strip()

# -----------------------------
# Retrieval filters by user choice
# -----------------------------
def build_retriever(choice: str):
    if choice == "لوائح التجميل (PDF)":
        return vector_store.as_retriever(search_kwargs={"k": 8, "filter": {"category": "regulation"}})
    if choice == "محظورات التجميل":
        return vector_store.as_retriever(search_kwargs={"k": 8, "filter": {"category": "banned"}})
    return vector_store.as_retriever(search_kwargs={"k": 8})

def build_knowledge(docs) -> str:
    parts = []
    for d in docs:
        src = display_source_name(d.metadata.get("source", d.metadata.get("source_file", "N/A")))
        snippet = pretty_arabic_text(d.page_content)[:1400]
        parts.append(f"[{src}]\n{snippet}")
    return "\n\n".join(parts)

# -----------------------------
# ✅ Chat Logic (Streaming) - نفس طريقتك
# -----------------------------
def stream_response(message, history, source_choice):
    message = (message or "").strip()
    if not message:
        yield "اكتبي سؤالك."
        return

    # 1) سؤال مادة
    art_num = extract_article_number(message)

    if art_num:
        if source_choice == "محظورات التجميل":
            yield "عذرًا، **المواد (المادة 4...)** تكون ضمن **لوائح التجميل (PDF)**. غيّري الاختيار للأعلى."
            return

        doc = get_article_doc(art_num)
        if not doc:
            yield f"عذرًا، لم أجد نصًا صريحًا للمادة رقم {art_num} داخل لوائح التجميل."
            return

        answer = format_article_output(doc)
        answer += sources_footer_once([doc], source_choice)
        yield answer
        return

    # 2) RAG حسب اختيار المستخدم
    retriever = build_retriever(source_choice)
    retrieved_docs = retriever.invoke("query: " + message)

    if not retrieved_docs:
        yield "لم أجد نصًا صريحًا في المصادر المتاحة يجيب عن ذلك."
        return

    top_docs = retrieved_docs[:3]
    knowledge = build_knowledge(top_docs)

    generation_prompt = f"""
ROLE:
أنت مساعد امتثال يعتمد فقط على النصوص المرفقة.

RULES (مهم جداً):
- لا تضف أي معلومة من خارج "النصوص المساعدة".
- إذا لم تجد نصاً صريحاً يجيب عن السؤال، قل: "لم أجد نصاً صريحاً في المصادر المرفقة يجيب عن ذلك."
- اكتب إجابة قصيرة ومنظمة بنقاط.
- لا تذكر المصادر داخل النص (سأضيفها أنا في النهاية).

----
النصوص المساعدة:
{knowledge}

سؤال المستخدم: {message}

اكتب الإجابة الآن:
""".strip()

    final_answer = ""

    # ✅ إصلاح الخطأ: نرسل Messages بشكل صحيح
    for chunk in llm.stream([HumanMessage(content=generation_prompt)]):
        if getattr(chunk, "content", None):
            final_answer += chunk.content
            final_answer = clean_repeated_characters(final_answer)
            yield final_answer

    final_answer = final_answer.strip()
    final_answer += sources_footer_once(top_docs, source_choice)
    yield final_answer

# -----------------------------
# UI (زي السابق)
# -----------------------------
css_code = """
.gradio-container { font-family: Tahoma, sans-serif; }
"""

with gr.Blocks(css=css_code) as demo:
    gr.Markdown("# SANAD")
    gr.Markdown("اختاري مصدر البحث أولاً (لوائح / محظورات / الكل)، ثم اكتبي سؤالك.")

    source_choice = gr.Radio(
        choices=["لوائح التجميل (PDF)", "محظورات التجميل", "الكل"],
        value="لوائح التجميل (PDF)",
        label="مصدر البحث"
    )

    gr.ChatInterface(
        fn=stream_response,
        additional_inputs=[source_choice],
        textbox=gr.Textbox(
            placeholder="اكتبي سؤالك...",
            container=False,
            autoscroll=True,
            scale=7,
            label="",
            show_label=False,
        ),
        examples=[
            ["لوائح التجميل (PDF)", "ما هي المادة الرابعة؟"],
            ["لوائح التجميل (PDF)", "اذكر التزامات المُدرج في النظام"],
            ["محظورات التجميل", "هل Mercury محظور في التجميل؟"],
            ["محظورات التجميل", "اذكر لي 5 مواد محظورة تبدأ بحرف M"],
        ],
    )

demo.queue().launch(share=True, show_error=True, debug=True)
