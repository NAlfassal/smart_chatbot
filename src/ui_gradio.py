# src/ui_gradio.py

import os
import re
import sys
from typing import List, Optional, Iterator, Any

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.utils.logger import logger
from src import config


# ============================================================
# Helpers
# ============================================================
class ArabicArticleParser:
    AR_WORD_TO_NUM = {
        "الأولى": "1", "الاولى": "1", "الثانية": "2", "الثالثة": "3", "الرابعة": "4",
        "الخامسة": "5", "السادسة": "6", "السابعة": "7", "الثامنة": "8", "التاسعة": "9",
        "العاشرة": "10", "الحادية عشر": "11", "الحادية عشرة": "11", "الثانية عشر": "12",
        "الثانية عشرة": "12", "الثالثة عشر": "13", "الثالثة عشرة": "13", "الرابعة عشر": "14",
        "الرابعة عشرة": "14", "الخامسة عشر": "15", "الخامسة عشرة": "15", "السادسة عشر": "16",
        "السادسة عشرة": "16", "السابعة عشر": "17", "السابعة عشرة": "17", "الثامنة عشر": "18",
        "الثامنة عشرة": "18", "التاسعة عشر": "19", "التاسعة عشرة": "19", "العشرون": "20",
        "الحادية والعشرون": "21", "الثانية والعشرون": "22", "الثالثة والعشرون": "23",
        "الرابعة والعشرون": "24", "الخامسة والعشرون": "25", "السادسة والعشرون": "26",
        "السابعة والعشرون": "27", "الثامنة والعشرون": "28", "التاسعة والعشرون": "29", "الثلاثون": "30",
    }

    @classmethod
    def extract_article_number(cls, text: str) -> Optional[str]:
        text = text or ""
        m = re.search(r"المادة\s+(\d+)", text)
        if m:
            return m.group(1)

        m = re.search(r"المادة\s+([^\n،,.؟!]+)", text)
        if not m:
            return None

        phrase = re.sub(r"\s{2,}", " ", m.group(1).replace("ـ", "")).strip()
        phrase = re.sub(r"^\s*المادة\s+", "", phrase).strip()

        if phrase in cls.AR_WORD_TO_NUM:
            return cls.AR_WORD_TO_NUM[phrase]

        words = phrase.split()
        for n in (4, 3, 2, 1):
            if len(words) >= n:
                cand = " ".join(words[:n])
                if cand in cls.AR_WORD_TO_NUM:
                    return cls.AR_WORD_TO_NUM[cand]
        return None


class TextFormatter:
    @staticmethod
    def pretty_arabic_text(text: str) -> str:
        if not text:
            return ""
        t = text.replace("\r\n", "\n").replace("\r", "\n").replace("ـ", "")
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()


class SourceDisplayManager:
    @staticmethod
    def display_source_name_from_doc(doc: Document) -> str:
        cat = (doc.metadata.get("category") or "").lower().strip()
        if cat == "banned":
            return "محظورات التجميل"
        if cat == "regulation":
            return "لوائح التجميل"
        if cat in ("gdp", "guidelines"):
            return "الأسس (التوزيع والتخزين الجيدة)"
        raw = doc.metadata.get("source", doc.metadata.get("source_file", "N/A"))
        return os.path.basename(raw or "N/A").strip() or "مصادر إضافية"

    @staticmethod
    def sources_footer_once(docs: List[Document], chosen_sources_ui: List[str]) -> str:
        seen = set()
        sources = []
        for d in docs:
            name = SourceDisplayManager.display_source_name_from_doc(d)
            if name and name not in seen:
                seen.add(name)
                sources.append(name)
        return "\n\n**المصدر:** " + ("، ".join(sources) if sources else "N/A")


# ============================================================
# Chatbot Logic
# ============================================================
class SFDAChatbot:
    def __init__(self):
        logger.info("Initializing SFDA Chatbot...")

        self.embeddings_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=(getattr(config, "OPENROUTER_API_KEY", None) or getattr(config, "OPENAI_API_KEY", None)),
            base_url=getattr(config, "LLM_BASE_URL", None),
            max_tokens=getattr(config, "LLM_MAX_TOKENS", 1024),
        )

        self.vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings_model,
            persist_directory=str(config.CHROMA_PATH),
        )

        logger.info("✅ Chatbot initialized successfully")

    def get_article_doc(self, article_num: str) -> Optional[Document]:
        target = str(article_num).strip()
        try:
            docs = self.vector_store.similarity_search(
                query=f"المادة {target}",
                k=1,
                filter={"article": target},
            )
            if docs:
                return docs[0]
        except Exception:
            pass
        return None

    def stream_response_core(self, message: Any, source_choices: List[str]) -> Iterator[str]:
        # Robust: sometimes Gradio passes list
        if isinstance(message, list):
            message = " ".join(str(x) for x in message)

        message = str(message).strip()
        if not message:
            yield "اكتب سؤالك."
            return

        try:
            art_num = ArabicArticleParser.extract_article_number(message)
            if art_num:
                doc = self.get_article_doc(art_num)
                if doc:
                    ans = TextFormatter.pretty_arabic_text(doc.page_content)
                    yield ans + SourceDisplayManager.sources_footer_once([doc], source_choices)
                    return

            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            retrieved_docs = retriever.invoke(message)

            if not retrieved_docs:
                yield "لم أجد نصاً صريحاً في المصادر."
                return

            knowledge = "\n\n".join(
                [f"[{SourceDisplayManager.display_source_name_from_doc(d)}]\n{d.page_content[:1000]}" for d in retrieved_docs]
            )

            prompt = f"""
ROLE: مساعد ذكي لهيئة الغذاء والدواء.
CONTEXT:
{knowledge}

USER: {message}

INSTRUCTIONS:
- أجب اعتماداً على النصوص فقط.
- إذا لا توجد إجابة واضحة: قل "لا توجد معلومات في المصادر المتاحة".
- كن مختصراً.
""".strip()

            final_answer = ""
            for chunk in self.llm.stream([HumanMessage(content=prompt)]):
                if chunk.content:
                    final_answer += chunk.content
                    yield final_answer

            yield final_answer + SourceDisplayManager.sources_footer_once(retrieved_docs, source_choices)

        except Exception as e:
            logger.error(f"stream_response_core error: {e}", exc_info=True)
            yield "عذراً، حدث خطأ تقني."


# ============================================================
# CSS (Center card + all text black + inputs white)
# ============================================================
CSS_CODE = """
:root{
  --bg: #D4B2B8;            /* نفس خلفيتك الوردية */
  --card: #FFFFFF;
  --text: #000000;
  --muted: #111827;

  --input-bg: #FFFFFF;
  --input-border: #D1D5DB;

  --btn-bg: #D4B2B8;        /* زر نفس لون الخلفية */
  --btn-text: #111827;
}

.gradio-container{
  background: var(--bg) !important;
}

/* Center login card */
#login_screen{
  min-height: 100vh;
  display:flex !important;
  align-items:center !important;
  justify-content:center !important;
}

#login_card{
  width: min(760px, 92vw);
  background: var(--card) !important;
  border-radius: 44px !important;
  padding: 56px 56px 44px 56px !important;
  box-shadow: 0 22px 70px rgba(0,0,0,0.12) !important;
}

/* Title */
#login_title{
  color: var(--text) !important;
  text-align: center;
  font-size: 44px;
  font-weight: 900;
  margin-bottom: 34px;
}

/* Make all texts black */
#login_card, #login_card *{
  color: var(--text) !important;
}

/* Remove any dark panels around form groups */
#login_card .gr-form, 
#login_card .gr-box, 
#login_card .block, 
#login_card .wrap, 
#login_card .gr-panel,
#login_card .gr-group{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

/* Labels */
.gr-textbox label{
  color: var(--text) !important;
  font-weight: 800 !important;
}

/* Inputs */
.gradio-container input,
.gradio-container textarea{
  background: var(--input-bg) !important;
  border: 1px solid var(--input-border) !important;
  border-radius: 14px !important;
  color: var(--text) !important;
}

/* Login button */
#login_btn{
  background: var(--btn-bg) !important;
  color: var(--btn-text) !important;
  border: 1px solid rgba(0,0,0,0.10) !important;
  border-radius: 14px !important;
  height: 52px !important;
  font-weight: 900 !important;
  font-size: 16px !important;
}

#login_btn:hover{
  filter: brightness(0.98);
}

#login_error{
  margin-top: 12px;
  font-weight: 800;
  color: #B91C1C !important;
  text-align: center;
}

#login_hint{
  margin-top: 18px;
  text-align: center;
  color: #111827 !important;
  opacity: 0.9;
}

/* hide footer */
footer{ display:none !important; }
"""


# ============================================================
# UI
# ============================================================
def create_gradio_interface(chatbot: SFDAChatbot) -> gr.Blocks:
    # CSS in Blocks (works with many versions). Fallback if unsupported.
    try:
        demo = gr.Blocks(title="SANAD Chatbot", css=CSS_CODE)
    except TypeError:
        demo = gr.Blocks(title="SANAD Chatbot")

    with demo:
        is_logged_in = gr.State(False)

        # -----------------------------
        # LOGIN VIEW (✅ centered)
        # -----------------------------
        with gr.Column(visible=True, elem_id="login_screen") as login_view:
            with gr.Column(elem_id="login_card"):
                gr.Markdown("## اهلا بك في سند", elem_id="login_title")

                username = gr.Textbox(label="اسم المستخدم", placeholder="مثال: admin")
                password = gr.Textbox(label="كلمة السر", placeholder="••••••••", type="password")

                login_btn = gr.Button("تسجيل الدخول", elem_id="login_btn")
                login_error = gr.Markdown("", elem_id="login_error")
                gr.Markdown("اكتب بياناتك ثم اضغط تسجيل الدخول", elem_id="login_hint")

        # -----------------------------
        # CHAT VIEW
        # -----------------------------
        with gr.Column(visible=False) as chat_view:
            gr.Markdown("## 🇸🇦 SANAD - المساعد الذكي للوائح التجميل")

            source_choices = gr.CheckboxGroup(
                choices=["لوائح التجميل (PDF)", "محظورات التجميل", "الأسس (GDP)"],
                value=["لوائح التجميل (PDF)"],
                label="🔍 مصادر البحث",
            )

            # IMPORTANT: no type="messages" to stay compatible
            chatbot_ui = gr.Chatbot(label="المحادثة", height=550)

            gr.Markdown("### ✨ أمثلة جاهزة")
            examples = [
                "ما هي اشتراطات تخزين منتجات التجميل؟",
                "اذكر مسؤوليات المصنع حسب اللوائح.",
                "ماذا تقول المادة 20؟",
                "هل توجد مواد محظورة؟",
                "ما متطلبات GDP؟",
            ]

            with gr.Row():
                ex1 = gr.Button(examples[0])
                ex2 = gr.Button(examples[1])
                ex3 = gr.Button(examples[2])
            with gr.Row():
                ex4 = gr.Button(examples[3])
                ex5 = gr.Button(examples[4])

            with gr.Row():
                msg = gr.Textbox(placeholder="اكتب سؤالك هنا...", scale=4)
                send = gr.Button("إرسال", variant="primary", scale=1)

            clear = gr.Button("مسح المحادثة")

        # -----------------------------
        # Login Logic (✅ fixed password issue)
        # -----------------------------
        def do_login(u, p):
            expected_u = str(getattr(config, "UI_USERNAME", "admin")).strip()
            expected_p = str(getattr(config, "UI_PASSWORD", "admin")).strip()

            u = (u or "").strip()
            p = (p or "").strip()

            if not u or not p:
                return False, gr.update(visible=True), gr.update(visible=False), "❌ فضلاً أدخل اسم المستخدم وكلمة السر."

            if u == expected_u and p == expected_p:
                return True, gr.update(visible=False), gr.update(visible=True), ""

            return False, gr.update(visible=True), gr.update(visible=False), "❌ اسم المستخدم أو كلمة السر غير صحيحة."

        login_btn.click(do_login, [username, password], [is_logged_in, login_view, chat_view, login_error])
        password.submit(do_login, [username, password], [is_logged_in, login_view, chat_view, login_error])

        # -----------------------------
        # Examples -> Fill textbox
        # -----------------------------
        ex1.click(lambda: examples[0], None, msg)
        ex2.click(lambda: examples[1], None, msg)
        ex3.click(lambda: examples[2], None, msg)
        ex4.click(lambda: examples[3], None, msg)
        ex5.click(lambda: examples[4], None, msg)

        # -----------------------------
        # Chat callbacks (messages dict format)
        # -----------------------------
        def user_msg(user_message, history):
            if not user_message:
                return history, ""
            history = history or []
            history.append({"role": "user", "content": user_message})
            return history, ""

        def bot_msg(history, sources):
            if not history:
                return history
            last_user = history[-1]["content"]
            history.append({"role": "assistant", "content": ""})

            for chunk in chatbot.stream_response_core(last_user, sources):
                history[-1]["content"] = chunk
                yield history

        msg.submit(user_msg, [msg, chatbot_ui], [chatbot_ui, msg]).then(
            bot_msg, [chatbot_ui, source_choices], chatbot_ui
        )
        send.click(user_msg, [msg, chatbot_ui], [chatbot_ui, msg]).then(
            bot_msg, [chatbot_ui, source_choices], chatbot_ui
        )
        clear.click(lambda: [], None, chatbot_ui)

    return demo


# ============================================================
# app.py imports: from src.ui_gradio import main
# ============================================================
def main():
    bot = SFDAChatbot()
    ui = create_gradio_interface(bot)

    # Host/Port configurable (optional)
    host = str(getattr(config, "UI_HOST", "127.0.0.1")).strip()
    port = int(getattr(config, "UI_PORT", 7860))

    # Some versions don't support show_error
    try:
        ui.queue().launch(server_name=host, server_port=port, show_error=True)
    except TypeError:
        ui.queue().launch(server_name=host, server_port=port)


if __name__ == "__main__":
    main()
