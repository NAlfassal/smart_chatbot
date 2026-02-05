# src/ui_gradio.py

import os
import re
from typing import List, Optional, Iterator, Any, Dict

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
    def sources_footer_once(docs: List[Document]) -> str:
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
    UI_TO_CATEGORY = {
        "لوائح التجميل (PDF)": "regulation",
        "محظورات التجميل": "banned",
        "الأسس (GDP)": "gdp",
    }

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

    def _selected_categories(self, source_choices: Optional[List[str]]) -> List[str]:
        if not source_choices:
            return []
        cats = []
        for s in source_choices:
            c = self.UI_TO_CATEGORY.get(s)
            if c:
                cats.append(c)
        return cats

    def _build_category_filter(self, selected_cats: List[str]) -> Optional[Dict[str, Any]]:
        if not selected_cats:
            return None
        return {"$or": [{"category": c} for c in selected_cats]}

    def _and_filter(self, *parts: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        parts_clean = [p for p in parts if p]
        if not parts_clean:
            return None
        if len(parts_clean) == 1:
            return parts_clean[0]
        return {"$and": parts_clean}

    def get_article_doc(self, article_num: str, selected_cats: Optional[List[str]] = None) -> Optional[Document]:
        target = str(article_num).strip()
        try:
            cat_filter = self._build_category_filter(selected_cats or [])
            where = self._and_filter({"article": target}, cat_filter)

            docs = self.vector_store.similarity_search(
                query=f"المادة {target}",
                k=1,
                filter=where if where else {"article": target},
            )
            if docs:
                return docs[0]
        except Exception:
            pass
        return None

    def stream_response_core(self, message: Any, source_choices: List[str]) -> Iterator[str]:
        message = str(message or "").strip()
        if not message:
            yield "اكتب سؤالك."
            return

        try:
            selected_cats = self._selected_categories(source_choices)
            cat_filter = self._build_category_filter(selected_cats)

            art_num = ArabicArticleParser.extract_article_number(message)
            if art_num:
                doc = self.get_article_doc(art_num, selected_cats=selected_cats)
                if doc:
                    ans = TextFormatter.pretty_arabic_text(doc.page_content)
                    yield ans + SourceDisplayManager.sources_footer_once([doc])
                    return

            search_kwargs: Dict[str, Any] = {"k": 3}
            if cat_filter:
                search_kwargs["filter"] = cat_filter

            retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
            retrieved_docs = retriever.invoke(message)

            if not retrieved_docs:
                chosen = "، ".join(source_choices or [])
                yield f"لا توجد معلومات في المصادر المتاحة.\n\n**المصادر المختارة:** {chosen if chosen else 'الكل'}"
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

            yield final_answer + SourceDisplayManager.sources_footer_once(retrieved_docs)

        except Exception as e:
            logger.error(f"stream_response_core error: {e}", exc_info=True)
            yield "عذراً، حدث خطأ تقني."


# ============================================================
# CSS (SAFE: لا يغير هيكلة Gradio)
# ============================================================
CSS_CODE = """
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800;900&display=swap');

:root{
  --primary:#006C3A;
  --primary2:#005530;
  --bg1:#F8F9FA;
  --bg2:#E9ECEF;
  --card:#FFFFFF;
  --text:#111827;
  --muted:#6B7280;
  --border:#E5E7EB;
  --shadow:rgba(0,0,0,.10);
}

*{ font-family:'Tajawal',system-ui,-apple-system,'Segoe UI',sans-serif !important; }

.gradio-container{
  background: linear-gradient(135deg,var(--bg1) 0%,var(--bg2) 100%) !important;
}

/* LOGIN */
#login_screen{
  min-height: 100vh;
  display:flex !important;
  align-items:center !important;
  justify-content:center !important;
  padding: 18px;
}
#login_card{
  width: min(520px, 94vw);
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 22px !important;
  padding: 38px 34px !important;
  box-shadow: 0 18px 55px rgba(0,0,0,.12) !important;
}
#login_title{ text-align:center; font-size:34px; font-weight:900; color:var(--text) !important; margin:0 0 8px 0; }
#login_subtitle{ text-align:center; color:var(--muted) !important; margin:0 0 20px 0; }
#login_btn{
  background: linear-gradient(135deg,var(--primary) 0%,var(--primary2) 100%) !important;
  color:#fff !important;
  border:none !important;
  border-radius: 14px !important;
  height: 50px !important;
  font-weight: 800 !important;
}
#login_error{ color:#EF4444 !important; font-weight:700; }

/* CHAT PAGE */
#page_wrap{
  max-width: 1400px;
  margin: 0 auto;
  padding: 12px 18px 18px 18px;
}

/* Header صغير مناسب للابتوب */
#chat_header{
  background: linear-gradient(135deg,var(--primary) 0%,var(--primary2) 100%) !important;
  border-radius: 18px !important;
  padding: 16px 18px !important;
  color:#fff !important;
  box-shadow: 0 10px 26px rgba(0,0,0,.14) !important;
  margin-bottom: 10px !important;
}
#chat_header h1{ margin:0; font-size:28px; font-weight:900; }
#chat_header p{ margin:6px 0 0 0; opacity:.95; font-size:14px; }

/* Panels */
.panel{
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px !important;
  box-shadow: 0 6px 18px var(--shadow) !important;
  padding: 14px !important;
}

/* Chatbot: ارتفاع مناسب للابتوب */
#chatbot_box{
  height: 62vh;              /* ✅ يخلي الشاشة ما تحتاج سكرول كثير */
  min-height: 420px;
  border-radius: 18px !important;
  overflow: auto;
}

/* Inputs */
#send_btn{
  background: linear-gradient(135deg,var(--primary) 0%,var(--primary2) 100%) !important;
  color:#fff !important;
  border:none !important;
  border-radius: 14px !important;
  height: 48px !important;
  font-weight:800 !important;
}
#clear_btn{
  border-radius: 14px !important;
  height: 44px !important;
}

.example-btn{
  border-radius: 14px !important;
  padding: 12px 14px !important;
  min-height: 46px !important;
  font-weight:700 !important;
}

/* bot text black */
.gradio-chatbot .bot, .gradio-chatbot .bot * { color:#000 !important; }

footer{ display:none !important; }
"""


# ============================================================
# UI
# ============================================================
def create_gradio_interface(chatbot: SFDAChatbot) -> gr.Blocks:
    demo = gr.Blocks(title="SANAD Chatbot", css=CSS_CODE)

    with demo:
        is_logged_in = gr.State(False)

        # -----------------------------
        # LOGIN VIEW
        # -----------------------------
        with gr.Column(visible=True, elem_id="login_screen") as login_view:
            with gr.Column(elem_id="login_card"):
                gr.Markdown("# سَنَد", elem_id="login_title")
                gr.Markdown("المساعد الذكي للوائح التجميل", elem_id="login_subtitle")

                username = gr.Textbox(label="اسم المستخدم", placeholder="أدخل اسم المستخدم")
                password = gr.Textbox(label="كلمة المرور", placeholder="••••••••", type="password")

                login_btn = gr.Button("تسجيل الدخول", elem_id="login_btn")
                login_error = gr.Markdown("", elem_id="login_error")

        # -----------------------------
        # CHAT VIEW
        # -----------------------------
        with gr.Column(visible=False) as chat_view:
            with gr.Column(elem_id="page_wrap"):
                with gr.Column(elem_id="chat_header"):
                    gr.HTML("<h1>🇸🇦 سَنَد - SANAD</h1><p>المساعد الذكي ت</p>")

                with gr.Row():
                    # Sidebar
                    with gr.Column(scale=1):
                        with gr.Column(elem_classes=["panel"]):
                            gr.Markdown("### 🔍 مصادر البحث")
                            source_choices = gr.CheckboxGroup(
                                choices=["لوائح التجميل (PDF)", "محظورات التجميل", "الأسس (GDP)"],
                                value=["لوائح التجميل (PDF)"],
                                label="",
                                interactive=True,
                                show_label=False,
                            )

                        with gr.Column(elem_classes=["panel"]):
                            gr.Markdown("### ✨ أمثلة جاهزة")
                            examples = [
                                "ما هي اشتراطات تخزين منتجات التجميل؟",
                                "اذكر مسؤوليات المصنع حسب اللوائح.",
                                "ماذا تقول المادة 20؟",
                                "هل توجد مواد محظورة في مستحضرات التجميل؟",
                                "ما متطلبات GDP للتوزيع والتخزين؟",
                            ]
                            ex1 = gr.Button(examples[0], elem_classes=["example-btn"])
                            ex2 = gr.Button(examples[1], elem_classes=["example-btn"])
                            ex3 = gr.Button(examples[2], elem_classes=["example-btn"])
                            ex4 = gr.Button(examples[3], elem_classes=["example-btn"])
                            ex5 = gr.Button(examples[4], elem_classes=["example-btn"])

                    # Main
                    with gr.Column(scale=3):
                        with gr.Column(elem_classes=["panel"]):
                            chatbot_ui = gr.Chatbot(
                                show_label=False,
                                elem_id="chatbot_box",
                                rtl=True,
                            )

                        with gr.Column(elem_classes=["panel"]):
                            with gr.Row():
                                msg = gr.Textbox(
                                    placeholder="اكتب سؤالك هنا...",
                                    scale=4,
                                    show_label=False,
                                    container=False,
                                    rtl=True,
                                )
                                send = gr.Button("إرسال", variant="primary", scale=1, elem_id="send_btn")

                            clear = gr.Button("🗑️ مسح المحادثة", elem_id="clear_btn")

        # -----------------------------
        # Login Logic
        # -----------------------------
        def do_login(u, p):
            expected_u = str(getattr(config, "UI_USERNAME", "admin")).strip()
            expected_p = str(getattr(config, "UI_PASSWORD", "admin")).strip()

            u = (u or "").strip()
            p = (p or "").strip()

            if not u or not p:
                return False, gr.update(visible=True), gr.update(visible=False), "❌ فضلاً أدخل اسم المستخدم وكلمة المرور."

            if u == expected_u and p == expected_p:
                return True, gr.update(visible=False), gr.update(visible=True), ""

            return False, gr.update(visible=True), gr.update(visible=False), "❌ اسم المستخدم أو كلمة المرور غير صحيحة."

        login_btn.click(do_login, [username, password], [is_logged_in, login_view, chat_view, login_error])
        password.submit(do_login, [username, password], [is_logged_in, login_view, chat_view, login_error])

        # Examples -> Fill textbox
        ex1.click(lambda: examples[0], None, msg, queue=False)
        ex2.click(lambda: examples[1], None, msg, queue=False)
        ex3.click(lambda: examples[2], None, msg, queue=False)
        ex4.click(lambda: examples[3], None, msg, queue=False)
        ex5.click(lambda: examples[4], None, msg, queue=False)

        # Chat callbacks
        def user_msg(user_message, history):
            if not user_message:
                return history, ""
            history = history or []
            history.append({"role": "user", "content": user_message})
            return history, ""

        def bot_msg(history, selected_sources):
            if not history:
                return history
            last_user = history[-1]["content"]
            history.append({"role": "assistant", "content": ""})
            for chunk in chatbot.stream_response_core(last_user, selected_sources or []):
                history[-1]["content"] = chunk
                yield history

        msg.submit(user_msg, [msg, chatbot_ui], [chatbot_ui, msg], queue=False).then(
            bot_msg, [chatbot_ui, source_choices], chatbot_ui
        )
        send.click(user_msg, [msg, chatbot_ui], [chatbot_ui, msg], queue=False).then(
            bot_msg, [chatbot_ui, source_choices], chatbot_ui
        )
        clear.click(lambda: [], None, chatbot_ui, queue=False)

    return demo


def main():
    bot = SFDAChatbot()
    ui = create_gradio_interface(bot)

    host = str(getattr(config, "UI_HOST", "127.0.0.1")).strip()
    port = int(getattr(config, "UI_PORT", 7860))

    ui.queue().launch(
        server_name=host,
        server_port=port,
        show_error=True,
        share=False,
    )


if __name__ == "__main__":
    main()
