"""
Improved Gradio Application for SFDA Cosmetics Chatbot.
Compatible with new Project Structure (Best Practices).
No Authentication Required.
Fixed: Gradio 'messages' type error.
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

from src.utils.logger import logger
from src import config

# ---------------------------------------------------------------------
# Helpers (نفس الدوال المساعدة السابقة)
# ---------------------------------------------------------------------
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
    def normalize_article_to_num(cls, article_value: str) -> Optional[str]:
        if article_value is None: return None
        s = str(article_value).strip().replace("ـ", "")
        s = re.sub(r"\s{2,}", " ", s)
        s = re.sub(r"^\s*المادة\s+", "", s).strip()
        if re.fullmatch(r"\d+", s): return s
        if s in cls.AR_WORD_TO_NUM: return cls.AR_WORD_TO_NUM[s]
        words = s.split()
        for n in (4, 3, 2, 1):
            if len(words) >= n:
                cand = " ".join(words[:n])
                if cand in cls.AR_WORD_TO_NUM: return cls.AR_WORD_TO_NUM[cand]
        return None
    @classmethod
    def extract_article_number(cls, text: str) -> Optional[str]:
        text = text or ""
        m = re.search(r"المادة\s+(\d+)", text)
        if m: return m.group(1)
        m = re.search(r"المادة\s+([^\n،,.؟!]+)", text)
        if not m: return None
        phrase = re.sub(r"\s{2,}", " ", m.group(1).replace("ـ", "")).strip()
        phrase = re.sub(r"^\s*المادة\s+", "", phrase).strip()
        if phrase in cls.AR_WORD_TO_NUM: return cls.AR_WORD_TO_NUM[phrase]
        words = phrase.split()
        for n in (4, 3, 2, 1):
            if len(words) >= n:
                cand = " ".join(words[:n])
                if cand in cls.AR_WORD_TO_NUM: return cls.AR_WORD_TO_NUM[cand]
        return None

class TextFormatter:
    @staticmethod
    def clean_repeated_characters(text: str) -> str:
        return re.sub(r"(.)\1{2,}", r"\1", text or "")
    @staticmethod
    def pretty_arabic_text(text: str) -> str:
        if not text: return ""
        t = text.replace("\r\n", "\n").replace("\r", "\n").replace("ـ", "")
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()

class SourceDisplayManager:
    @staticmethod
    def display_source_name_from_doc(doc: Document) -> str:
        cat = (doc.metadata.get("category") or "").lower().strip()
        if cat == "banned": return "محظورات التجميل"
        if cat == "regulation": return "لوائح التجميل"
        if cat in ("gdp", "guidelines"): return "الأسس (التوزيع والتخزين الجيدة)"
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
# Main Chatbot Logic
# ---------------------------------------------------------------------
class SFDAChatbot:
    def __init__(self):
        logger.info("Initializing SFDA Chatbot...")
        if not getattr(config, "OPENROUTER_API_KEY", None) and not getattr(config, "OPENAI_API_KEY", None):
            raise ValueError("API Keys missing in .env")

        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
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

        db_path = str(config.CHROMA_PATH)
        logger.info(f"Connecting to Vector Store at: {db_path}")
        
        self.vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings_model,
            persist_directory=db_path,
        )
        try:
            if hasattr(self.vector_store, '_collection'):
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
            if docs: return docs[0]
        except Exception as e:
            logger.debug(f"Regulation search failed: {e}")
        try:
            docs = self.vector_store.similarity_search(
                query=f"المادة {target}",
                k=3,
                filter={"article": target},
            )
            if docs: return docs[0]
        except Exception as e:
            logger.debug(f"General article search failed: {e}")
        return None

    def build_retriever(self, ui_choices: List[str]):
        if not ui_choices:
            return self.vector_store.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
        selected_categories = []
        for ch in ui_choices:
            if "لوائح" in ch: selected_categories.append("regulation")
            elif "محظورات" in ch: selected_categories.append("banned")
            elif "الأسس" in ch: selected_categories.append("gdp")
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
            yield "اكتب سؤالك."
            return

        try:
            art_num = ArabicArticleParser.extract_article_number(message)
            if art_num:
                if source_choices and all("محظورات" in s for s in source_choices):
                    yield "عذرًا، سؤال **المادة** يكون ضمن **لوائح التجميل**. الرجاء تغيير المصدر."
                    return
                doc = self.get_article_doc(art_num)
                if doc:
                    answer = TextFormatter.pretty_arabic_text(doc.page_content)
                    answer += SourceDisplayManager.sources_footer_once([doc], source_choices)
                    yield answer
                    return
                else:
                    logger.info(f"Article {art_num} not found directly, switching to semantic search.")

            retriever = self.build_retriever(source_choices)
            retrieved_docs = retriever.invoke(message) 

            if not retrieved_docs:
                yield "لم أجد نصًا صريحًا في المصادر المتاحة يجيب عن ذلك."
                return

            top_docs = retrieved_docs[:3]
            knowledge = self.build_knowledge(top_docs)

            generation_prompt = f"""
ROLE: أنت مساعد ذكي لهيئة الغذاء والدواء.
CONTEXT:
{knowledge}

USER QUESTION: {message}

INSTRUCTIONS:
- أجب بدقة بناءً على النصوص المرفقة فقط.
- إذا لم تجد الإجابة قل "لا توجد معلومات في المصادر المتاحة".
- كن مختصراً ومفيداً.
""".strip()

            final_answer = ""
            for chunk in self.llm.stream([HumanMessage(content=generation_prompt)]):
                if chunk.content:
                    final_answer += chunk.content
                    yield final_answer

            yield final_answer + SourceDisplayManager.sources_footer_once(top_docs, source_choices)

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            yield "عذرًا، حدث خطأ تقني أثناء المعالجة."

# ---------------------------------------------------------------------
# UI (Gradio)
# ---------------------------------------------------------------------
def create_gradio_interface(chatbot: SFDAChatbot) -> gr.Blocks:
    css_code = """
    .gradio-container { font-family: 'Segoe UI', Tahoma, sans-serif; }
    """

    with gr.Blocks(css=css_code, title="SANAD Chatbot") as demo:
        gr.Markdown("## 🇸🇦 SANAD - المساعد الذكي للوائح التجميل")
        
        with gr.Row():
            source_choices = gr.CheckboxGroup(
                choices=["لوائح التجميل (PDF)", "محظورات التجميل", "الأسس (GDP)"],
                value=["لوائح التجميل (PDF)"],
                label="🔍 مصادر البحث",
            )
        
        # --- التعديل الجوهري هنا: type="messages" ---
        chatbot_ui = gr.Chatbot(label="المحادثة", height=550, type="messages")
        
        with gr.Row():
            msg = gr.Textbox(placeholder="اكتب سؤالك هنا...", scale=4)
            send = gr.Button("إرسال", variant="primary", scale=1)
        
        clear = gr.Button("مسح المحادثة")

        # --- Functions ---
        def user_msg_fn(user_message, history):
            if not user_message: return history, ""
            history.append({"role": "user", "content": user_message})
            return history, ""

        def bot_msg_fn(history, choices):
            if not history: return history
            last_user_msg = history[-1]["content"]
            
            history.append({"role": "assistant", "content": ""})
            
            for chunk in chatbot.stream_response_core(last_user_msg, choices):
                history[-1]["content"] = chunk
                yield history

        # --- Event Listeners ---
        msg.submit(user_msg_fn, [msg, chatbot_ui], [chatbot_ui, msg]).then(
            bot_msg_fn, [chatbot_ui, source_choices], chatbot_ui
        )
        send.click(user_msg_fn, [msg, chatbot_ui], [chatbot_ui, msg]).then(
            bot_msg_fn, [chatbot_ui, source_choices], chatbot_ui
        )
        clear.click(lambda: [], None, chatbot_ui)

    return demo

def main():
    try:
        bot = SFDAChatbot()
        demo = create_gradio_interface(bot)
        demo.queue().launch(share=False)
    except Exception as e:
        logger.critical(f"Failed to launch UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()