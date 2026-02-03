"""
Improved Gradio Application for SFDA Cosmetics Chatbot.

This module provides a web interface for querying SFDA cosmetics regulations
and banned substances using a RAG (Retrieval Augmented Generation) approach.
"""

import os
import re
import logging
from typing import List, Optional, Iterator

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO if config.DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArabicArticleParser:
    """Handles parsing and conversion of Arabic article numbers."""

    # Arabic word to number mappings
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
        """
        Convert Arabic article name to number.

        Args:
            article_value: Arabic article text

        Returns:
            Article number as string, or None if not found
        """
        if article_value is None:
            return None

        s = str(article_value).strip()
        s = s.replace("ـ", "")
        s = re.sub(r"\s{2,}", " ", s)
        s = re.sub(r"^\s*المادة\s+", "", s).strip()

        # Check if already numeric
        if re.fullmatch(r"\d+", s):
            return s

        # Direct lookup
        if s in cls.AR_WORD_TO_NUM:
            return cls.AR_WORD_TO_NUM[s]

        # Try partial matches
        words = s.split()
        for n in (4, 3, 2, 1):
            if len(words) >= n:
                cand = " ".join(words[:n])
                if cand in cls.AR_WORD_TO_NUM:
                    return cls.AR_WORD_TO_NUM[cand]
        return None

    @classmethod
    def extract_article_number(cls, text: str) -> Optional[str]:
        """
        Extract article number from user query.

        Args:
            text: User query text

        Returns:
            Article number if found, None otherwise
        """
        text = text or ""

        # Try numeric pattern first
        m = re.search(r"المادة\s+(\d+)", text)
        if m:
            return m.group(1)

        # Try text pattern
        m = re.search(r"المادة\s+([^\n،,.؟!]+)", text)
        if not m:
            return None

        phrase = re.sub(r"\s{2,}", " ", m.group(1).replace("ـ", "")).strip()
        phrase = re.sub(r"^\s*المادة\s+", "", phrase).strip()

        # Direct lookup
        if phrase in cls.AR_WORD_TO_NUM:
            return cls.AR_WORD_TO_NUM[phrase]

        # Try partial matches
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
        """Remove excessive character repetition."""
        return re.sub(r"(.)\1{2,}", r"\1", text or "")

    @staticmethod
    def merge_spaced_arabic_letters(text: str) -> str:
        """Merge Arabic letters that are incorrectly spaced."""
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
        """Format Arabic text for better display."""
        if not text:
            return ""
        t = TextFormatter.merge_spaced_arabic_letters(text)
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = t.replace("ـ", "")
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()


class SourceDisplayManager:
    """Manages display names for document sources."""

    @staticmethod
    def display_source_name(raw_source: str) -> str:
        """
        Convert raw source file name to display name.

        Args:
            raw_source: Raw source file name

        Returns:
            Formatted display name
        """
        s = os.path.basename(raw_source or "N/A").strip()
        s_lower = s.lower()

        if "banned_list" in s_lower or "محظور" in s or "قائمة شاملة بالمواد المحظورة" in s:
            return "محظورات التجميل"

        if "اللائحة" in s or "نظام منتجات التجميل" in s or "sfda_articles" in s_lower:
            return "لوائح التجميل"

        return "مصادر إضافية"

    @staticmethod
    def sources_footer_once(docs: List[Document], mode_choice: str) -> str:
        """
        Generate sources footer for display.

        Args:
            docs: Retrieved documents
            mode_choice: User's source choice

        Returns:
            Formatted sources footer
        """
        # For banned substances, don't show file names
        if mode_choice == "محظورات التجميل":
            return "\n\n**المصدر:** محظورات التجميل"

        seen = set()
        sources = []
        for d in docs:
            raw = d.metadata.get("source", d.metadata.get("source_file", "N/A"))
            name = SourceDisplayManager.display_source_name(raw)
            if name and name not in seen:
                seen.add(name)
                sources.append(name)

        return "\n\n**المصدر:** " + ("، ".join(sources) if sources else "N/A")


class SFDAChatbot:
    """Main chatbot class handling RAG operations."""

    def __init__(self):
        """Initialize chatbot with models and vector store."""
        logger.info("Initializing SFDA Chatbot...")

        # Validate API key
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")

        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
        )

        # Initialize LLM
        logger.info("Initializing LLM...")
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.LLM_BASE_URL,
            max_tokens=config.LLM_MAX_TOKENS,
        )

        # Initialize vector store
        logger.info("Loading vector store...")
        self.vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings_model,
            persist_directory=config.CHROMA_PATH,
        )

        # Log vector store stats
        try:
            count = self.vector_store._collection.count()
            logger.info(f"Vector store loaded. Document count: {count}")
        except Exception as e:
            logger.warning(f"Could not get vector store count: {e}")

        logger.info("✅ Chatbot initialized successfully")

    def get_article_doc(self, article_num: str) -> Optional[Document]:
        """
        Retrieve a specific article by number.

        Args:
            article_num: Article number to retrieve

        Returns:
            Document containing the article, or None if not found
        """
        target = str(article_num).strip()

        # Try with regulation filter first
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

        # Fallback to article-only filter
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
        """
        Format article document for display.

        Args:
            doc: Article document

        Returns:
            Formatted article text
        """
        art_num = ArabicArticleParser.normalize_article_to_num(
            doc.metadata.get("article", "")
        ) or ""
        title = f"نص المادة ({art_num}) من لوائح التجميل" if art_num else "نص المادة من لوائح التجميل"
        body = TextFormatter.pretty_arabic_text(doc.page_content)

        # Remove "المادة X" prefix from body
        if art_num:
            body = re.sub(rf"^\s*المادة\s*{re.escape(art_num)}\s*\n+", "", body)
            body = re.sub(rf"^\s*المادة\s*{re.escape(art_num)}\s*[:：]?\s*", "", body)

        return f"**{title}**\n\n{body}".strip()

    def build_retriever(self, choice: str):
        """
        Build retriever based on user's source choice.

        Args:
            choice: User's source selection

        Returns:
            Configured retriever
        """
        if choice == "لوائح التجميل (PDF)":
            return self.vector_store.as_retriever(
                search_kwargs={"k": config.RETRIEVAL_K, "filter": {"category": "regulation"}}
            )
        if choice == "محظورات التجميل":
            return self.vector_store.as_retriever(
                search_kwargs={"k": config.RETRIEVAL_K, "filter": {"category": "banned"}}
            )
        return self.vector_store.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})

    @staticmethod
    def build_knowledge(docs: List[Document]) -> str:
        """
        Build knowledge context from retrieved documents.

        Args:
            docs: Retrieved documents

        Returns:
            Formatted knowledge context
        """
        parts = []
        for d in docs:
            src = SourceDisplayManager.display_source_name(
                d.metadata.get("source", d.metadata.get("source_file", "N/A"))
            )
            snippet = TextFormatter.pretty_arabic_text(d.page_content)[:1400]
            parts.append(f"[{src}]\n{snippet}")
        return "\n\n".join(parts)

    def stream_response(
        self, message: str, history: list, source_choice: str
    ) -> Iterator[str]:
        """
        Stream chatbot response to user query.

        Args:
            message: User's question
            history: Chat history (unused for now)
            source_choice: User's source selection

        Yields:
            Partial responses as they are generated
        """
        message = (message or "").strip()
        if not message:
            yield "اكتبي سؤالك."
            return

        try:
            # Check if asking for a specific article
            art_num = ArabicArticleParser.extract_article_number(message)

            if art_num:
                if source_choice == "محظورات التجميل":
                    yield "عذرًا، **المواد (المادة 4...)** تكون ضمن **لوائح التجميل (PDF)**. غيّري الاختيار للأعلى."
                    return

                doc = self.get_article_doc(art_num)
                if not doc:
                    yield f"عذرًا، لم أجد نصًا صريحًا للمادة رقم {art_num} داخل لوائح التجميل."
                    return

                answer = self.format_article_output(doc)
                answer += SourceDisplayManager.sources_footer_once([doc], source_choice)
                yield answer
                return

            # RAG-based retrieval
            retriever = self.build_retriever(source_choice)
            retrieved_docs = retriever.invoke("query: " + message)

            if not retrieved_docs:
                yield "لم أجد نصًا صريحًا في المصادر المتاحة يجيب عن ذلك."
                return

            top_docs = retrieved_docs[:3]
            knowledge = self.build_knowledge(top_docs)

            # Generate response
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
            final_answer += SourceDisplayManager.sources_footer_once(top_docs, source_choice)
            yield final_answer

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield f"عذرًا، حدث خطأ أثناء معالجة سؤالك. الرجاء المحاولة مرة أخرى."


def create_gradio_interface(chatbot: SFDAChatbot) -> gr.Blocks:
    """
    Create and configure Gradio interface.

    Args:
        chatbot: Initialized chatbot instance

    Returns:
        Configured Gradio Blocks interface
    """
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
            fn=chatbot.stream_response,
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

    return demo


def main():
    """Main application entry point."""
    try:
        # Initialize chatbot
        chatbot = SFDAChatbot()

        # Create and launch interface
        demo = create_gradio_interface(chatbot)
        demo.queue().launch(
            share=True,
            show_error=True,
            debug=config.DEBUG
        )

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise


if __name__ == "__main__":
    main()
