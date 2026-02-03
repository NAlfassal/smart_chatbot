"""
Improved Gradio Application for SFDA Cosmetics Chatbot.

ظˆط§ط¬ظ‡ط© ط§ط³طھط¹ظ„ط§ظ… ط¹ظ†:
- ظ„ظˆط§ط¦ط­ ط§ظ„طھط¬ظ…ظٹظ„ (PDF)       category=regulation
- ظ…ط­ط¸ظˆط±ط§طھ ط§ظ„طھط¬ظ…ظٹظ„          category=banned
- ط§ظ„ط£ط³ط³ (GDP)              category=gdp

ط¨ط§ط³طھط®ط¯ط§ظ… RAG (Retrieval Augmented Generation)
"""

import os
import re
import sys
import logging
import traceback
from typing import List, Optional, Iterator

# âœ… FIX: Make project root importable so "import config" works when running from app/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

import config

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO if getattr(config, "DEBUG", False) else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sfda_app")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
class ArabicArticleParser:
    """Handles parsing and conversion of Arabic article numbers."""

    AR_WORD_TO_NUM = {
        "ط§ظ„ط£ظˆظ„ظ‰": "1", "ط§ظ„ط§ظˆظ„ظ‰": "1",
        "ط§ظ„ط«ط§ظ†ظٹط©": "2",
        "ط§ظ„ط«ط§ظ„ط«ط©": "3",
        "ط§ظ„ط±ط§ط¨ط¹ط©": "4",
        "ط§ظ„ط®ط§ظ…ط³ط©": "5",
        "ط§ظ„ط³ط§ط¯ط³ط©": "6",
        "ط§ظ„ط³ط§ط¨ط¹ط©": "7",
        "ط§ظ„ط«ط§ظ…ظ†ط©": "8",
        "ط§ظ„طھط§ط³ط¹ط©": "9",
        "ط§ظ„ط¹ط§ط´ط±ط©": "10",
        "ط§ظ„ط­ط§ط¯ظٹط© ط¹ط´ط±": "11", "ط§ظ„ط­ط§ط¯ظٹط© ط¹ط´ط±ط©": "11",
        "ط§ظ„ط«ط§ظ†ظٹط© ط¹ط´ط±": "12", "ط§ظ„ط«ط§ظ†ظٹط© ط¹ط´ط±ط©": "12",
        "ط§ظ„ط«ط§ظ„ط«ط© ط¹ط´ط±": "13", "ط§ظ„ط«ط§ظ„ط«ط© ط¹ط´ط±ط©": "13",
        "ط§ظ„ط±ط§ط¨ط¹ط© ط¹ط´ط±": "14", "ط§ظ„ط±ط§ط¨ط¹ط© ط¹ط´ط±ط©": "14",
        "ط§ظ„ط®ط§ظ…ط³ط© ط¹ط´ط±": "15", "ط§ظ„ط®ط§ظ…ط³ط© ط¹ط´ط±ط©": "15",
        "ط§ظ„ط³ط§ط¯ط³ط© ط¹ط´ط±": "16", "ط§ظ„ط³ط§ط¯ط³ط© ط¹ط´ط±ط©": "16",
        "ط§ظ„ط³ط§ط¨ط¹ط© ط¹ط´ط±": "17", "ط§ظ„ط³ط§ط¨ط¹ط© ط¹ط´ط±ط©": "17",
        "ط§ظ„ط«ط§ظ…ظ†ط© ط¹ط´ط±": "18", "ط§ظ„ط«ط§ظ…ظ†ط© ط¹ط´ط±ط©": "18",
        "ط§ظ„طھط§ط³ط¹ط© ط¹ط´ط±": "19", "ط§ظ„طھط§ط³ط¹ط© ط¹ط´ط±ط©": "19",
        "ط§ظ„ط¹ط´ط±ظˆظ†": "20",
        "ط§ظ„ط­ط§ط¯ظٹط© ظˆط§ظ„ط¹ط´ط±ظˆظ†": "21",
        "ط§ظ„ط«ط§ظ†ظٹط© ظˆط§ظ„ط¹ط´ط±ظˆظ†": "22",
        "ط§ظ„ط«ط§ظ„ط«ط© ظˆط§ظ„ط¹ط´ط±ظˆظ†": "23",
        "ط§ظ„ط±ط§ط¨ط¹ط© ظˆط§ظ„ط¹ط´ط±ظˆظ†": "24",
        "ط§ظ„ط®ط§ظ…ط³ط© ظˆط§ظ„ط¹ط´ط±ظˆظ†": "25",
        "ط§ظ„ط³ط§ط¯ط³ط© ظˆط§ظ„ط¹ط´ط±ظˆظ†": "26",
        "ط§ظ„ط³ط§ط¨ط¹ط© ظˆط§ظ„ط¹ط´ط±ظˆظ†": "27",
        "ط§ظ„ط«ط§ظ…ظ†ط© ظˆط§ظ„ط¹ط´ط±ظˆظ†": "28",
        "ط§ظ„طھط§ط³ط¹ط© ظˆط§ظ„ط¹ط´ط±ظˆظ†": "29",
        "ط§ظ„ط«ظ„ط§ط«ظˆظ†": "30",
    }

    @classmethod
    def normalize_article_to_num(cls, article_value: str) -> Optional[str]:
        if article_value is None:
            return None

        s = str(article_value).strip()
        s = s.replace("ظ€", "")
        s = re.sub(r"\s{2,}", " ", s)
        s = re.sub(r"^\s*ط§ظ„ظ…ط§ط¯ط©\s+", "", s).strip()

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

        m = re.search(r"ط§ظ„ظ…ط§ط¯ط©\s+(\d+)", text)
        if m:
            return m.group(1)

        m = re.search(r"ط§ظ„ظ…ط§ط¯ط©\s+([^\nطŒ,.طں!]+)", text)
        if not m:
            return None

        phrase = re.sub(r"\s{2,}", " ", m.group(1).replace("ظ€", "")).strip()
        phrase = re.sub(r"^\s*ط§ظ„ظ…ط§ط¯ط©\s+", "", phrase).strip()

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
                r"(?<![ط،-ظٹ])((?:[ط،-ظٹ]\s+){2,}[ط،-ظٹ])(?![ط،-ظٹ])",
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
        t = t.replace("ظ€", "")
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()


class SourceDisplayManager:
    """Display sources based on metadata category (NOT filename)."""

    @staticmethod
    def display_source_name_from_doc(doc: Document) -> str:
        cat = (doc.metadata.get("category") or "").lower().strip()

        if cat == "banned":
            return "ظ…ط­ط¸ظˆط±ط§طھ ط§ظ„طھط¬ظ…ظٹظ„"
        if cat == "regulation":
            return "ظ„ظˆط§ط¦ط­ ط§ظ„طھط¬ظ…ظٹظ„"
        if cat in ("gdp", "guidelines", "gdp_guidelines"):
            return "ط§ظ„ط£ط³ط³ (ط§ظ„طھظˆط²ظٹط¹ ظˆط§ظ„طھط®ط²ظٹظ† ط§ظ„ط¬ظٹط¯ط©)"

        raw = doc.metadata.get("source", doc.metadata.get("source_file", "N/A"))
        return os.path.basename(raw or "N/A").strip() or "ظ…طµط§ط¯ط± ط¥ط¶ط§ظپظٹط©"

    @staticmethod
    def sources_footer_once(docs: List[Document], chosen_sources_ui: List[str]) -> str:
        if chosen_sources_ui and set(chosen_sources_ui) == {"ظ…ط­ط¸ظˆط±ط§طھ ط§ظ„طھط¬ظ…ظٹظ„"}:
            return "\n\n**ط§ظ„ظ…طµط¯ط±:** ظ…ط­ط¸ظˆط±ط§طھ ط§ظ„طھط¬ظ…ظٹظ„"

        seen = set()
        sources = []
        for d in docs:
            name = SourceDisplayManager.display_source_name_from_doc(d)
            if name and name not in seen:
                seen.add(name)
                sources.append(name)

        return "\n\n**ط§ظ„ظ…طµط¯ط±:** " + ("طŒ ".join(sources) if sources else "N/A")


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
            persist_directory=str(config.CHROMA_PATH),  # âœ… important on Windows
        )

        try:
            count = self.vector_store._collection.count()
            logger.info(f"Vector store loaded. Document count: {count}")
        except Exception as e:
            logger.warning(f"Could not get vector store count: {e}")

        logger.info("âœ… Chatbot initialized successfully")

    def get_article_doc(self, article_num: str) -> Optional[Document]:
        target = str(article_num).strip()

        try:
            docs = self.vector_store.similarity_search(
                query=f"ط§ظ„ظ…ط§ط¯ط© {target}",
                k=3,
                filter={"$and": [{"article": target}, {"category": "regulation"}]},
            )
            if docs:
                return docs[0]
        except Exception as e:
            logger.debug(f"Regulation filter search failed: {e}")

        try:
            docs = self.vector_store.similarity_search(
                query=f"ط§ظ„ظ…ط§ط¯ط© {target}",
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
        title = f"ظ†طµ ط§ظ„ظ…ط§ط¯ط© ({art_num}) ظ…ظ† ظ„ظˆط§ط¦ط­ ط§ظ„طھط¬ظ…ظٹظ„" if art_num else "ظ†طµ ط§ظ„ظ…ط§ط¯ط© ظ…ظ† ظ„ظˆط§ط¦ط­ ط§ظ„طھط¬ظ…ظٹظ„"
        body = TextFormatter.pretty_arabic_text(doc.page_content)

        if art_num:
            body = re.sub(rf"^\s*ط§ظ„ظ…ط§ط¯ط©\s*{re.escape(art_num)}\s*\n+", "", body)
            body = re.sub(rf"^\s*ط§ظ„ظ…ط§ط¯ط©\s*{re.escape(art_num)}\s*[:ï¼ڑ]?\s*", "", body)

        return f"**{title}**\n\n{body}".strip()

    def build_retriever(self, ui_choices: List[str]):
        """
        âœ… IMPORTANT:
        Chroma ظ„ط§ ظٹظ‚ط¨ظ„ $or ط£ظˆ $and ط¥ط°ط§ ط§ظ„ظ‚ط§ط¦ظ…ط© ظپظٹظ‡ط§ ط´ط±ط· ظˆط§ط­ط¯ ظپظ‚ط·.
        """
        if not ui_choices:
            return self.vector_store.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})

        selected_categories = []
        for ch in ui_choices:
            if ch == "ظ„ظˆط§ط¦ط­ ط§ظ„طھط¬ظ…ظٹظ„ (PDF)":
                selected_categories.append("regulation")
            elif ch == "ظ…ط­ط¸ظˆط±ط§طھ ط§ظ„طھط¬ظ…ظٹظ„":
                selected_categories.append("banned")
            elif ch == "ط§ظ„ط£ط³ط³ (GDP)":
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
            yield "ط§ظƒطھط¨ظٹ ط³ط¤ط§ظ„ظƒ."
            return

        try:
            # 1) Article direct lookup
            art_num = ArabicArticleParser.extract_article_number(message)
            if art_num:
                if source_choices and set(source_choices) == {"ظ…ط­ط¸ظˆط±ط§طھ ط§ظ„طھط¬ظ…ظٹظ„"}:
                    yield "ط¹ط°ط±ظ‹ط§طŒ ط³ط¤ط§ظ„ **ط§ظ„ظ…ط§ط¯ط© (ظ…ط«ظ„ ط§ظ„ظ…ط§ط¯ط© ط§ظ„ط±ط§ط¨ط¹ط©)** ظٹظƒظˆظ† ط¶ظ…ظ† **ظ„ظˆط§ط¦ط­ ط§ظ„طھط¬ظ…ظٹظ„ (PDF)**. ظپط¹ظ‘ظ„ظٹ ط®ظٹط§ط± ط§ظ„ظ„ظˆط§ط¦ط­."
                    return

                doc = self.get_article_doc(art_num)
                if not doc:
                    yield f"ط¹ط°ط±ظ‹ط§طŒ ظ„ظ… ط£ط¬ط¯ ظ†طµظ‹ط§ طµط±ظٹط­ظ‹ط§ ظ„ظ„ظ…ط§ط¯ط© ط±ظ‚ظ… {art_num} ط¯ط§ط®ظ„ ظ„ظˆط§ط¦ط­ ط§ظ„طھط¬ظ…ظٹظ„."
                    return

                answer = self.format_article_output(doc)
                answer += SourceDisplayManager.sources_footer_once([doc], source_choices)
                yield answer
                return

            # 2) Retrieval
            retriever = self.build_retriever(source_choices)
            retrieved_docs = retriever.get_relevant_documents(message)

            if not retrieved_docs:
                yield "ظ„ظ… ط£ط¬ط¯ ظ†طµظ‹ط§ طµط±ظٹط­ظ‹ط§ ظپظٹ ط§ظ„ظ…طµط§ط¯ط± ط§ظ„ظ…طھط§ط­ط© ظٹط¬ظٹط¨ ط¹ظ† ط°ظ„ظƒ."
                return

            top_docs = retrieved_docs[:3]
            knowledge = self.build_knowledge(top_docs)

            generation_prompt = f"""
ROLE:
ط£ظ†طھ ظ…ط³ط§ط¹ط¯ ط§ظ…طھط«ط§ظ„ ظٹط¹طھظ…ط¯ ظپظ‚ط· ط¹ظ„ظ‰ ط§ظ„ظ†طµظˆطµ ط§ظ„ظ…ط±ظپظ‚ط©.

RULES (ظ…ظ‡ظ… ط¬ط¯ط§ظ‹):
- ظ„ط§ طھط¶ظپ ط£ظٹ ظ…ط¹ظ„ظˆظ…ط© ظ…ظ† ط®ط§ط±ط¬ "ط§ظ„ظ†طµظˆطµ ط§ظ„ظ…ط³ط§ط¹ط¯ط©".
- ط¥ط°ط§ ظ„ظ… طھط¬ط¯ ظ†طµط§ظ‹ طµط±ظٹط­ط§ظ‹ ظٹط¬ظٹط¨ ط¹ظ† ط§ظ„ط³ط¤ط§ظ„طŒ ظ‚ظ„: "ظ„ظ… ط£ط¬ط¯ ظ†طµط§ظ‹ طµط±ظٹط­ط§ظ‹ ظپظٹ ط§ظ„ظ…طµط§ط¯ط± ط§ظ„ظ…ط±ظپظ‚ط© ظٹط¬ظٹط¨ ط¹ظ† ط°ظ„ظƒ."
- ط§ظƒطھط¨ ط¥ط¬ط§ط¨ط© ظ‚طµظٹط±ط© ظˆظ…ظ†ط¸ظ…ط© ط¨ظ†ظ‚ط§ط·.
- ظ„ط§ طھط°ظƒط± ط§ظ„ظ…طµط§ط¯ط± ط¯ط§ط®ظ„ ط§ظ„ظ†طµ (ط³ط£ط¶ظٹظپظ‡ط§ ط£ظ†ط§ ظپظٹ ط§ظ„ظ†ظ‡ط§ظٹط©).

----
ط§ظ„ظ†طµظˆطµ ط§ظ„ظ…ط³ط§ط¹ط¯ط©:
{knowledge}

ط³ط¤ط§ظ„ ط§ظ„ظ…ط³طھط®ط¯ظ…: {message}

ط§ظƒطھط¨ ط§ظ„ط¥ط¬ط§ط¨ط© ط§ظ„ط¢ظ†:
""".strip()

            # 3) Stream answer
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
                yield "âڑ ï¸ڈ ط§ظ„ظ…ظˆط¯ظ„ ط؛ظٹط± ظ…طھظˆظپط± ظپظٹ OpenRouter. ط؛ظٹظ‘ط±ظٹ LLM_MODEL ظپظٹ .env ط¥ظ„ظ‰ ظ…ظˆط¯ظ„ ط´ط؛ط§ظ„ ظ…ط«ظ„: deepseek/deepseek-chat"
                return

            if getattr(config, "DEBUG", False):
                yield f"âڑ ï¸ڈ ط®ط·ط£: {type(e).__name__}: {e}"
            else:
                yield "ط¹ط°ط±ظ‹ط§طŒ ط­ط¯ط« ط®ط·ط£ ط£ط«ظ†ط§ط، ظ…ط¹ط§ظ„ط¬ط© ط³ط¤ط§ظ„ظƒ. ط§ظ„ط±ط¬ط§ط، ط§ظ„ظ…ط­ط§ظˆظ„ط© ظ…ط±ط© ط£ط®ط±ظ‰."


# ---------------------------------------------------------------------
# UI (Gradio) - centered login (working button)
# ---------------------------------------------------------------------
def create_gradio_interface(chatbot: SFDAChatbot) -> gr.Blocks:
    css_code = """
.gradio-container { font-family: Tahoma, sans-serif; }

/* âœ… Center the login card in the viewport */
#login_row {
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
}
#login_card {
  width: min(520px, 92vw);
  padding: 22px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(20, 24, 33, 0.85);
  box-shadow: 0 20px 60px rgba(0,0,0,0.35);
}
"""

    with gr.Blocks(css=css_code) as demo:
        login_view = gr.Column(visible=True)
        app_view = gr.Column(visible=False)

        # ---------------- LOGIN ----------------
        with login_view:
            with gr.Row(elem_id="login_row"):
                with gr.Column(elem_id="login_card"):
                    gr.Markdown("## Login")
                    u = gr.Textbox(label="Username", placeholder="ط§ط¯ط®ظ„ظٹ ط§ط³ظ… ط§ظ„ظ…ط³طھط®ط¯ظ…")
                    p = gr.Textbox(label="Password", placeholder="ط§ط¯ط®ظ„ظٹ ظƒظ„ظ…ط© ط§ظ„ظ…ط±ظˆط±", type="password")
                    login_btn = gr.Button("Login", variant="primary")
                    login_msg = gr.Markdown("")

        # ---------------- APP ----------------
        with app_view:
            gr.Markdown("# SANAD")
            gr.Markdown("ط§ط®طھط§ط±ظٹ **ظ…طµط¯ط±/ظ…طµط§ط¯ط± ط§ظ„ط¨ط­ط«** ط«ظ… ط§ظƒطھط¨ظٹ ط³ط¤ط§ظ„ظƒ ظˆط§ط¶ط؛ط·ظٹ **ط¥ط±ط³ط§ظ„**.")

            source_choices = gr.CheckboxGroup(
                choices=["ظ„ظˆط§ط¦ط­ ط§ظ„طھط¬ظ…ظٹظ„ (PDF)", "ظ…ط­ط¸ظˆط±ط§طھ ط§ظ„طھط¬ظ…ظٹظ„", "ط§ظ„ط£ط³ط³ (GDP)"],
                value=["ظ„ظˆط§ط¦ط­ ط§ظ„طھط¬ظ…ظٹظ„ (PDF)"],
                label="ظ…طµط§ط¯ط± ط§ظ„ط¨ط­ط« (ط§ط®طھظٹط§ط± ظˆط§ط­ط¯ ط£ظˆ ط£ظƒط«ط±)",
            )

            chat = gr.Chatbot(label="ط§ظ„ظ…ط­ط§ط¯ط«ط©", height=520)
            state = gr.State([])

            with gr.Row():
                msg = gr.Textbox(placeholder="ط§ظƒطھط¨ظٹ ط³ط¤ط§ظ„ظƒ...", show_label=False, scale=8)
                send = gr.Button("ط¥ط±ط³ط§ظ„", variant="primary", scale=2)

            clear = gr.Button("ظ…ط³ط­ ط§ظ„ظ…ط­ط§ط¯ط«ط©")

            def add_user(user_message, history_state):
                user_message = (user_message or "").strip()
                history_state = history_state or []

                if not user_message:
                    return gr.update(value=""), history_state, history_state

                history_state = history_state + [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": ""},
                ]
                return gr.update(value=""), history_state, history_state

            def stream_bot(history_state, choices):
                history_state = history_state or []

                user_message = ""
                for i in range(len(history_state) - 1, -1, -1):
                    if history_state[i].get("role") == "user":
                        user_message = history_state[i].get("content", "")
                        break

                last_assistant_idx = None
                for i in range(len(history_state) - 1, -1, -1):
                    if history_state[i].get("role") == "assistant":
                        last_assistant_idx = i
                        break

                for chunk in chatbot.stream_response_core(user_message, choices):
                    if last_assistant_idx is not None:
                        history_state[last_assistant_idx]["content"] = chunk
                    yield history_state, history_state

            send.click(fn=add_user, inputs=[msg, state], outputs=[msg, state, chat]).then(
                fn=stream_bot, inputs=[state, source_choices], outputs=[chat, state]
            )

            msg.submit(fn=add_user, inputs=[msg, state], outputs=[msg, state, chat]).then(
                fn=stream_bot, inputs=[state, source_choices], outputs=[chat, state]
            )

            def clear_all():
                return [], []

            clear.click(fn=clear_all, inputs=None, outputs=[chat, state])

            initial_examples = (
                "ًں‘‹ **ط£ظ…ط«ظ„ط© ط¬ط§ظ‡ط²ط© (ط§ظ†ط³ط®ظٹ/ط§ظ„طµظ‚ظٹ ط³ط¤ط§ظ„ط§ظ‹ ط£ظˆ ط§ظƒطھط¨ظٹظ‡):**\n\n"
                "â€¢ ظ…ط§ ظ‡ظٹ ط§ظ„ظ…ط§ط¯ط© ط§ظ„ط±ط§ط¨ط¹ط©طں\n"
                "â€¢ ط§ط°ظƒط± ط§ظ„طھط²ط§ظ…ط§طھ ط§ظ„ظ…ظڈط¯ط±ط¬ ظپظٹ ط§ظ„ظ†ط¸ط§ظ…\n"
                "â€¢ ظ‡ظ„ Mercury ظ…ط­ط¸ظˆط± ظپظٹ ط§ظ„طھط¬ظ…ظٹظ„طں\n"
                "â€¢ ط§ط°ظƒط± ظ„ظٹ 5 ظ…ظˆط§ط¯ ظ…ط­ط¸ظˆط±ط© طھط¨ط¯ط£ ط¨ط­ط±ظپ M\n"
                "â€¢ ظ…ط§ ظ‡ظٹ ظ…طھط·ظ„ط¨ط§طھ ط¯ط±ط¬ط© ط§ظ„ط­ط±ط§ط±ط© ظˆط§ظ„ط±ط·ظˆط¨ط© ظپظٹ ط§ظ„ظ…ط³طھظˆط¯ط¹ط§طھطں\n"
            )

            def init_chat():
                history = [{"role": "assistant", "content": initial_examples}]
                return history, history

            demo.load(fn=init_chat, inputs=None, outputs=[chat, state])

        # ---------------- LOGIN LOGIC ----------------
        def do_login(username, password):
            if username == config.GRADIO_USERNAME and password == config.GRADIO_PASSWORD:
                return gr.update(visible=False), gr.update(visible=True), ""
            return gr.update(visible=True), gr.update(visible=False), "â‌Œ ط¨ظٹط§ظ†ط§طھ ط؛ظٹط± طµط­ظٹط­ط©"

        login_btn.click(
            fn=do_login,
            inputs=[u, p],
            outputs=[login_view, app_view, login_msg],
        )

    return demo


def main():
    bot = SFDAChatbot()
    demo = create_gradio_interface(bot)

    # âœ… ظ…ظ‡ظ…: ظ„ط§ طھط³طھط®ط¯ظ… auth ظ‡ظ†ط§ ظ„ط£ظ† ط§ظ„ظ„ظˆظ‚ظٹظ† طµط§ط± ط¯ط§ط®ظ„ ط§ظ„ظˆط§ط¬ظ‡ط©
    demo.queue().launch(
        share=True,
        show_error=True,
        debug=getattr(config, "DEBUG", False),
    )


if __name__ == "__main__":
    main()

