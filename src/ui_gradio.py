"""
Improved Gradio Application for SFDA Cosmetics Chatbot.

واجهة استعلام عن:
- لوائح التجميل (PDF)       category=regulation
- محظورات التجميل          category=banned
- الأسس (GDP)              category=gdp

باستخدام RAG (Retrieval Augmented Generation)
"""

import os
import re
import sys
import traceback
from typing import List, Optional, Iterator

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.utils.logger import get_logger
from src import config

logger = get_logger("sfda_app")


# ✅ (اختياري) لو تحتاج تشغيل الملف من أي مكان بدون مشاكل استيراد
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
class ArabicArticleParser:
    """Handles parsing and conversion of Arabic article numbers."""

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

    @classmethod
    def normalize_article_to_num(cls, article_value: str) -> Optional[str]:
        if article_value is None:
            return None

        s = str(article_value).strip()
        s = s.replace("ـ", "")
        s = re.sub(r"\s{2,}", " ", s)
        s = re.sub(r"^\s*المادة\s+", "", s).strip()

        if re.fullmatch(r"\d+", s):
            return s

        if s in cls.AR_WORD_TO_NUM:
            return cls.AR_WORD_TO_NUM[s]

        words = s.split()
        for n in (4, 3, 2, 1):
            if len(words) >= n:
                cand = " ".join(words[:n])
                if cand in cls.AR_WORD_TO_NUM:
                    return cls.AR_WORD_TO_NUM[cand]
        return None

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
    """Handles text formatting and cleaning operations."""

    @staticmethod
    def clean_repeated_characters(text: str) -> str:
        return re.sub(r"(.)\1{2,}", r"\1", text or "")

    @staticmethod
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

    @staticmethod
    def pretty_arabic_text(text: str) -> str:
        if not text:
            return ""
        t = TextFormatter.merge_spaced_arabic_letters(text)
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = t.replace("ـ", "")
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()


class SourceDisplayManager:
    """Display sources based on metadata category (NOT filename)."""

    @staticmethod
    def display_source_name_from_doc(doc: Document) -> str:
        cat = (doc.metadata.get("category") or "").lower().strip()

        if cat == "banned":
            return "محظورات التجميل"
        if cat == "regulation":
            return "لوائح التجميل"
        if cat in ("gdp", "guidelines", "gdp_guidelines"):
            return "الأسس (التوزيع والتخزين الجيدة)"

        raw = doc.metadata.get("source", doc.metadata.get("source_file", "N/A"))
        return os.path.basename(raw or "N/A").strip() or "مصادر إضافية"

    @staticmethod
    def sources_footer_once(docs: List[Document], chosen_sources_ui: List[str]) -> str:
        if chosen_sources_ui and set(chosen_sources_ui) == {"محظورات التجميل"}:
            return "\n\n**المصدر:** محظورات التجميل"

        seen = set()
        sources = []
        for d in docs:
            name = SourceDisplayManager.display_source_name_from_doc(d)
            if name and name not in seen:
                seen.add(name)
                sources.append(name)

        return "\n\n**المصدر:** " + ("، ".join(sources) if sources else "N/A")


# ---------------------------------------------------------------------
# Main Chatbot
# ---------------------------------------------------------------------
class SFDAChatbot:
    """Main chatbot class handling RAG operations."""

    def __init__(self):
        logger.info("Initializing SFDA Chatbot...")

        if not getattr(config, "OPENROUTER_API_KEY", None) and not getattr(config, "OPENAI_API_KEY", None):
            raise ValueError("OPENROUTER_API_KEY or OPENAI_API_KEY not found in .env file")

        logger.info("Loading embedding model...")
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
        )

        logger.info("Initializing LLM...")
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=(config.OPENROUTER_API_KEY or config.OPENAI_API_KEY),
            base_url=config.LLM_BASE_URL,
            max_tokens=config.LLM_MAX_TOKENS,
        )

        logger.info("Loading vector store...")
        self.vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings_model,
            persist_directory=str(config.CHROMA_PATH),
        )

        try:
            count = self.vector_store._collection.count()
            logger.info(f"Vector store loaded. Document count: {count}")
        except Exception as e:
            logger.warning(f"Could not get vector store count: {e}")

        logger.info("✅ Chatbot initialized successfully")

    def get_article_doc(self, article_num: str) -> Optional[Document]:
        target = str(article_num).strip()

        try:
            docs = self.vector_store.similarity_search(
                query=f"المادة {target}",
                k=3,
                filter={"$and": [{"article": target}, {"category": "regulation"}]},
            )
            if docs:
                return docs[0]
        except Exception as e:
            logger.debug(f"Regulation filter search failed: {e}")

        try:
            docs = self.vector_store.similarity_search(
                query=f"المادة {target}",
                k=3,
                filter={"article": target},
            )
            if docs:
                return docs[0]
        except Exception as e:
            logger.debug(f"Article filter search failed: {e}")

        return None

    def format_article_output(self, doc: Document) -> str:
        art_num = ArabicArticleParser.normalize_article_to_num(doc.metadata.get("article", "")) or ""
        title = f"نص المادة ({art_num}) من لوائح التجميل" if art_num else "نص المادة من لوائح التجميل"
        body = TextFormatter.pretty_arabic_text(doc.page_content)

        if art_num:
            body = re.sub(rf"^\s*المادة\s*{re.escape(art_num)}\s*\n+", "", body)
            body = re.sub(rf"^\s*المادة\s*{re.escape(art_num)}\s*[:：]?\s*", "", body)

        return f"**{title}**\n\n{body}".strip()

    def build_retriever(self, ui_choices: List[str]):
        if not ui_choices:
            return self.vector_store.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})

        selected_categories = []
        for ch in ui_choices:
            if ch == "لوائح التجميل (PDF)":
                selected_categories.append("regulation")
            elif ch == "محظورات التجميل":
                selected_categories.append("banned")
            elif ch == "الأسس (GDP)":
                selected_categories.append("gdp")

        if not selected_categories or len(selected_categories) >= 3:
            return self.vector_store.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})

        if len(selected_categories) == 1:
            return self.vector_store.as_retriever(
                search_kwargs={"k": config.RETRIEVAL_K, "filter": {"category": selected_categories[0]}}
            )

        or_filter = {"$or": [{"category": c} for c in selected_categories]}
        return self.vector_store.as_retriever(search_kwargs={"k": config.RETRIEVAL_K, "filter": or_filter})

    @staticmethod
    def build_knowledge(docs: List[Document]) -> str:
        parts = []
        for d in docs:
            src = SourceDisplayManager.display_source_name_from_doc(d)
            snippet = TextFormatter.pretty_arabic_text(d.page_content)[:1400]
            parts.append(f"[{src}]\n{snippet}")
        return "\n\n".join(parts)

    def stream_response_core(self, message: str, source_choices: List[str]) -> Iterator[str]:
        message = (message or "").strip()
        if not message:
            yield "اكتبي سؤالك."
            return

        try:
            art_num = ArabicArticleParser.extract_article_number(message)
            if art_num:
                if source_choices and set(source_choices) == {"محظورات التجميل"}:
                    yield "عذرًا، سؤال **المادة** يكون ضمن **لوائح التجميل (PDF)**. فعّلي خيار اللوائح."
                    return

                doc = self.get_article_doc(art_num)
                if not doc:
                    yield f"عذرًا، لم أجد نصًا صريحًا للمادة رقم {art_num} داخل لوائح التجميل."
                    return

                answer = self.format_article_output(doc)
                answer += SourceDisplayManager.sources_footer_once([doc], source_choices)
                yield answer
                return

            retriever = self.build_retriever(source_choices)
            retrieved_docs = retriever.get_relevant_documents(message)

            if not retrieved_docs:
                yield "لم أجد نصًا صريحًا في المصادر المتاحة يجيب عن ذلك."
                return

            top_docs = retrieved_docs[:3]
            knowledge = self.build_knowledge(top_docs)

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
            for chunk in self.llm.stream([HumanMessage(content=generation_prompt)]):
                if getattr(chunk, "content", None):
                    final_answer += chunk.content
                    final_answer = TextFormatter.clean_repeated_characters(final_answer)
                    yield final_answer

            final_answer = final_answer.strip()
            final_answer += SourceDisplayManager.sources_footer_once(top_docs, source_choices)
            yield final_answer

        except Exception as e:
            traceback.print_exc()
            logger.exception("Error generating response")

            msg = str(e)
            if "No endpoints found" in msg or "404" in msg:
                yield "⚠️ المودل غير متوفر. غيّري LLM_MODEL في .env إلى مودل شغال."
                return

            if getattr(config, "DEBUG", False):
                yield f"⚠️ خطأ: {type(e).__name__}: {e}"
            else:
                yield "عذرًا، حدث خطأ أثناء معالجة سؤالك. الرجاء المحاولة مرة أخرى."


# ---------------------------------------------------------------------
# UI (Gradio) - FIXED (clickable + correct messages format)
# ---------------------------------------------------------------------
def create_gradio_interface(chatbot: SFDAChatbot) -> gr.Blocks:
    css_code = """
.gradio-container { font-family: Tahoma, sans-serif; }

/* ✅ Center login WITHOUT position:fixed (so it won't block after hide) */
#login_row {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}
#login_card {
  width: min(520px, 92vw);
  padding: 22px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(20, 24, 33, 0.88);
  box-shadow: 0 20px 60px rgba(0,0,0,0.35);
}
"""

    with gr.Blocks(css=css_code) as demo:
        # views
        with gr.Column(visible=True) as login_view:
            with gr.Row(elem_id="login_row"):
                with gr.Column(elem_id="login_card"):
                    gr.Markdown("## Login")
                    u = gr.Textbox(label="Username", placeholder="ادخلي اسم المستخدم")
                    p = gr.Textbox(label="Password", placeholder="ادخلي كلمة المرور", type="password")
                    login_btn = gr.Button("Login", variant="primary")
                    login_msg = gr.Markdown("")

        with gr.Column(visible=False) as app_view:
            gr.Markdown("# SANAD")
            gr.Markdown("اختاري **مصدر/مصادر البحث** ثم اكتبي سؤالك واضغطي **إرسال**.")

            source_choices = gr.CheckboxGroup(
                choices=["لوائح التجميل (PDF)", "محظورات التجميل", "الأسس (GDP)"],
                value=["لوائح التجميل (PDF)"],
                label="مصادر البحث (اختيار واحد أو أكثر)",
            )

            # ✅ Gradio 6.5.1 expects messages format: list[dict(role,content)]
            chat = gr.Chatbot(label="المحادثة", height=520)
            state = gr.State([])  # list[dict]

            with gr.Row():
                msg = gr.Textbox(placeholder="اكتبي سؤالك...", show_label=False, scale=8)
                send = gr.Button("إرسال", variant="primary", scale=2)

            clear = gr.Button("مسح المحادثة")

            def init_chat():
                history = [{
                    "role": "assistant",
                    "content": (
                        "👋 **أمثلة جاهزة (انسخي/الصقي سؤالاً أو اكتبيه):**\n\n"
                        "• ما هي المادة الرابعة؟\n"
                        "• اذكر التزامات المُدرج في النظام\n"
                        "• هل Mercury محظور في التجميل؟\n"
                        "• اذكر لي 5 مواد محظورة تبدأ بحرف M\n"
                        "• ما هي متطلبات درجة الحرارة والرطوبة في المستودعات؟\n"
                    )
                }]
                return history, history

            demo.load(fn=init_chat, inputs=None, outputs=[chat, state])

            def add_user(user_message, history):
                user_message = (user_message or "").strip()
                history = history or []
                if not user_message:
                    return gr.update(value=""), history, history

                history.append({"role": "user", "content": user_message})
                history.append({"role": "assistant", "content": ""})
                return gr.update(value=""), history, history

            def stream_bot(history, choices):
                history = history or []
                if len(history) < 2:
                    yield history, history
                    return

                user_msg = history[-2]["content"]
                for chunk in chatbot.stream_response_core(user_msg, choices):
                    history[-1]["content"] = chunk
                    yield history, history

            send.click(fn=add_user, inputs=[msg, state], outputs=[msg, state, chat]).then(
                fn=stream_bot, inputs=[state, source_choices], outputs=[chat, state]
            )
            msg.submit(fn=add_user, inputs=[msg, state], outputs=[msg, state, chat]).then(
                fn=stream_bot, inputs=[state, source_choices], outputs=[chat, state]
            )

            def clear_all():
                return [], []

            clear.click(fn=clear_all, inputs=None, outputs=[chat, state])

        # ---------------- LOGIN LOGIC ----------------
        def do_login(username, password):
            username = (username or "").strip()
            password = (password or "").strip()

            if username == str(config.GRADIO_USERNAME) and password == str(config.GRADIO_PASSWORD):
                # hide login, show app, clear msg + clear fields
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(value=""),
                )

            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value="❌ بيانات غير صحيحة"),
                gr.update(value=""),
                gr.update(value=""),
            )

        login_btn.click(
            fn=do_login,
            inputs=[u, p],
            outputs=[login_view, app_view, login_msg, u, p],
        )

    return demo


def main():
    bot = SFDAChatbot()
    demo = create_gradio_interface(bot)
    demo.queue().launch(
        share=True,
        show_error=True,
        debug=True,
    )


if __name__ == "__main__":
    main()
