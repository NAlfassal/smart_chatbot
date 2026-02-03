"""
SANAD - مساعدك الذكي للتفتيش
Beautiful, simple, and powerful AI assistant for SFDA inspectors.
"""

import os
import re
import logging
import time
from typing import List, Optional, Iterator
from datetime import datetime

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

# Import components
from app_gradio_improved import (
    ArabicArticleParser,
    TextFormatter,
    SourceDisplayManager
)


class SANADChatbot:
    """SANAD - Your intelligent inspection assistant."""

    def __init__(self):
        """Initialize SANAD chatbot."""
        logger.info("تهيئة سَنَد...")

        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")

        # Initialize embedding model
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.LLM_BASE_URL,
            max_tokens=config.LLM_MAX_TOKENS,
        )

        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings_model,
            persist_directory=config.CHROMA_PATH,
        )

        # Analytics
        self.query_count = 0
        self.total_response_time = 0.0

        logger.info("✅ سَنَد جاهز للعمل")

    def get_article_doc(self, article_num: str) -> Optional[Document]:
        """Retrieve specific article by number."""
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
            logger.debug(f"Search failed: {e}")

        try:
            docs = self.vector_store.similarity_search(
                query=f"المادة {target}",
                k=3,
                filter={"article": target},
            )
            if docs:
                return docs[0]
        except Exception as e:
            logger.debug(f"Fallback search failed: {e}")

        return None

    def format_article_output(self, doc: Document) -> str:
        """Format article for display."""
        art_num = ArabicArticleParser.normalize_article_to_num(
            doc.metadata.get("article", "")
        ) or ""
        title = f"📜 المادة ({art_num})" if art_num else "📜 المادة"
        body = TextFormatter.pretty_arabic_text(doc.page_content)

        if art_num:
            body = re.sub(rf"^\s*المادة\s*{re.escape(art_num)}\s*\n+", "", body)
            body = re.sub(rf"^\s*المادة\s*{re.escape(art_num)}\s*[:：]?\s*", "", body)

        return f"**{title}**\n\n{body}".strip()

    def build_retriever(self, sources: List[str]):
        """Build retriever based on selected sources."""
        filters = []

        if "لوائح التجميل" in sources:
            filters.append({"category": "regulation"})
        if "محظورات التجميل" in sources:
            filters.append({"category": "banned"})

        if not filters:
            return self.vector_store.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
        elif len(filters) == 1:
            return self.vector_store.as_retriever(
                search_kwargs={"k": config.RETRIEVAL_K, "filter": filters[0]}
            )
        else:
            # Multiple sources - search all
            return self.vector_store.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})

    @staticmethod
    def build_knowledge(docs: List[Document]) -> str:
        """Build knowledge context from documents."""
        parts = []
        for d in docs:
            src = SourceDisplayManager.display_source_name(
                d.metadata.get("source", d.metadata.get("source_file", "N/A"))
            )
            snippet = TextFormatter.pretty_arabic_text(d.page_content)[:1400]
            parts.append(f"[{src}]\n{snippet}")
        return "\n\n".join(parts)

    def chat(self, message: str, history: list, sources: List[str]) -> Iterator[str]:
        """
        Main chat function with SANAD personality.
        """
        start_time = time.time()
        message = (message or "").strip()

        if not message:
            yield "👋 مرحباً! أنا **سَنَد**، مساعدك الذكي.\n\nاكتب سؤالك وسأساعدك في إيجاد الإجابة من المصادر الرسمية."
            return

        if not sources:
            yield "⚠️ من فضلك، اختر على الأقل مصدراً واحداً للبحث فيه."
            return

        try:
            self.query_count += 1

            # Check for article query
            art_num = ArabicArticleParser.extract_article_number(message)

            if art_num:
                if "لوائح التجميل" not in sources:
                    yield "💡 **نصيحة:** المواد موجودة في **لوائح التجميل**. اختر هذا المصدر من الأعلى."
                    return

                yield "🔍 جاري البحث عن المادة..."
                doc = self.get_article_doc(art_num)

                if not doc:
                    yield f"😕 عذراً، لم أجد المادة رقم **{art_num}** في قاعدة البيانات.\n\nجرب صياغة السؤال بطريقة أخرى."
                    return

                answer = self.format_article_output(doc)
                answer += f"\n\n---\n📚 **المصدر:** لوائح التجميل"
                yield answer

                response_time = time.time() - start_time
                self.total_response_time += response_time
                logger.info(f"Article {art_num} | {response_time:.2f}s")
                return

            # RAG query
            yield "🔍 جاري البحث في المصادر المختارة..."

            retriever = self.build_retriever(sources)
            retrieved_docs = retriever.invoke("query: " + message)

            if not retrieved_docs:
                yield "😕 لم أجد إجابة واضحة في المصادر المتاحة.\n\n💡 **اقتراح:** جرب إعادة صياغة السؤال أو اختيار مصادر إضافية."
                return

            top_docs = retrieved_docs[:3]
            knowledge = self.build_knowledge(top_docs)

            yield "💭 جاري تحليل المعلومات وصياغة الإجابة..."

            # Generate with SANAD personality
            prompt = f"""
أنت "سَنَد"، مساعد ذكي متخصص في لوائح التجميل والمواد المحظورة.

شخصيتك:
- محترف وودود في نفس الوقت
- دقيق في المعلومات، لا تخمن أبداً
- تعتمد فقط على النصوص المرفقة
- تقدم إجابات واضحة ومنظمة

القواعد:
1. إذا لم تجد الإجابة في النصوص، قل ذلك بوضوح
2. نظم إجابتك بنقاط واضحة
3. لا تذكر المصادر في النص (سيتم إضافتها تلقائياً)
4. استخدم لغة عربية سليمة ومهنية

النصوص المرجعية:
{knowledge}

سؤال المستخدم: {message}

الإجابة:
""".strip()

            final_answer = ""
            for chunk in self.llm.stream([HumanMessage(content=prompt)]):
                if getattr(chunk, "content", None):
                    final_answer += chunk.content
                    final_answer = TextFormatter.clean_repeated_characters(final_answer)
                    yield final_answer

            # Add sources footer
            sources_list = []
            for d in top_docs:
                src = SourceDisplayManager.display_source_name(
                    d.metadata.get("source", d.metadata.get("source_file", "N/A"))
                )
                if src not in sources_list:
                    sources_list.append(src)

            final_answer = final_answer.strip()
            final_answer += f"\n\n---\n📚 **المصادر:** " + "، ".join(sources_list)
            yield final_answer

            response_time = time.time() - start_time
            self.total_response_time += response_time
            logger.info(f"RAG query | {response_time:.2f}s | {len(retrieved_docs)} docs")

        except Exception as e:
            logger.error(f"Error: {e}")
            yield f"😓 عذراً، حدث خطأ غير متوقع.\n\n**التفاصيل:** {str(e)}\n\nالرجاء المحاولة مرة أخرى."


def create_beautiful_ui(chatbot: SANADChatbot) -> gr.Blocks:
    """Create beautiful, simple, and attractive UI for SANAD."""

    # Modern, beautiful CSS
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');

    .gradio-container {
        font-family: 'Tajawal', 'Segoe UI', sans-serif !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
    }

    /* Header styling */
    .sanad-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        direction: rtl;
    }

    .sanad-logo {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .sanad-title {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .sanad-subtitle {
        font-size: 1.3rem;
        color: rgba(255,255,255,0.95);
        margin-top: 0.5rem;
        font-weight: 400;
    }

    .sanad-tagline {
        font-size: 1rem;
        color: rgba(255,255,255,0.85);
        margin-top: 1rem;
        font-style: italic;
    }

    /* Source selector styling */
    .source-selector {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        direction: rtl;
    }

    .source-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Checkbox group styling */
    .source-checkbox {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }

    /* Chat interface styling */
    .message-wrap {
        direction: rtl !important;
    }

    .message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 18px 18px 5px 18px !important;
    }

    .message.bot {
        background: #f7fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 18px 18px 18px 5px !important;
    }

    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border-right: 5px solid #fdcb6e;
        direction: rtl;
        box-shadow: 0 4px 6px rgba(253, 203, 110, 0.2);
    }

    .info-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .info-list {
        color: #2d3748;
        line-height: 1.8;
        margin: 0;
        padding-right: 1.5rem;
    }

    /* Footer styling */
    .sanad-footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid #e2e8f0;
        color: #718096;
        direction: rtl;
    }

    .footer-heart {
        color: #e53e3e;
        animation: heartbeat 1.5s ease-in-out infinite;
    }

    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }

    /* Button styling */
    .gr-button {
        border-radius: 10px !important;
        font-weight: 500 !important;
    }

    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
    }

    /* Examples styling */
    .examples {
        direction: rtl !important;
    }
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft(), title="سَنَد - مساعدك الذكي") as demo:

        # Header with SANAD branding
        gr.HTML("""
        <div class="sanad-header">
            <div class="sanad-logo">🎯</div>
            <h1 class="sanad-title">سَنَد</h1>
            <p class="sanad-subtitle">مساعدك الذكي للتفتيش على منتجات التجميل</p>
            <p class="sanad-tagline">دقيق • سريع • موثوق</p>
        </div>
        """)

        # Source selection in a beautiful box
        with gr.Group():
            gr.HTML("""
            <div class="source-selector">
                <div class="source-title">
                    📚 اختر المصادر التي تريد البحث فيها
                </div>
            </div>
            """)

            sources = gr.CheckboxGroup(
                choices=[
                    "لوائح التجميل",
                    "محظورات التجميل"
                ],
                value=["لوائح التجميل"],
                label="",
                show_label=False,
                elem_classes="source-checkbox"
            )

        # Info box with examples
        gr.HTML("""
        <div class="info-box">
            <div class="info-title">💡 كيف تستخدم سَنَد؟</div>
            <ul class="info-list">
                <li><strong>اختر المصادر:</strong> حدد المصادر التي تريد البحث فيها من الأعلى</li>
                <li><strong>اطرح سؤالك:</strong> اكتب سؤالك بوضوح باللغة العربية</li>
                <li><strong>احصل على الإجابة:</strong> سَنَد سيبحث ويجيبك مع ذكر المصادر</li>
            </ul>
        </div>
        """)

        # Chat interface
        chatbot_interface = gr.ChatInterface(
            fn=chatbot.chat,
            additional_inputs=[sources],
            textbox=gr.Textbox(
                placeholder="💬 اكتب سؤالك هنا... (مثال: ما هي المادة الرابعة؟)",
                container=False,
                scale=7,
                label="",
                show_label=False,
                rtl=True,
            ),
            examples=[
                ["ما هي المادة الرابعة؟"],
                ["ما هي التزامات المُدرج في النظام؟"],
                ["هل Mercury محظور في منتجات التجميل؟"],
                ["ما هي متطلبات التسجيل؟"],
                ["اذكر 5 مواد محظورة"],
                ["ما هي العقوبات على المخالفات؟"],
            ],
            submit_btn="📤 إرسال",
            clear_btn="🗑️ مسح المحادثة",
        )

        # Footer
        gr.HTML(f"""
        <div class="sanad-footer">
            <p><strong>سَنَد</strong> - مساعدك الذكي للتفتيش 🎯</p>
            <p>صُنع بـ <span class="footer-heart">❤️</span> لمفتشي هيئة الغذاء والدواء</p>
            <p style="font-size: 0.9rem; margin-top: 1rem; color: #a0aec0;">
                معسكر سدايا لمحترفي الذكاء الاصطناعي • {datetime.now().year}
            </p>
            <p style="font-size: 0.85rem; color: #cbd5e0; margin-top: 0.5rem;">
                DeepSeek LLM • multilingual-e5-large • ChromaDB • LangChain • Gradio
            </p>
        </div>
        """)

    return demo


def main():
    """Main entry point."""
    try:
        print("\n" + "=" * 60)
        print("🎯 سَنَد - مساعدك الذكي للتفتيش")
        print("=" * 60)
        print(f"📅 التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🧠 نموذج الـ Embeddings: {config.EMBEDDING_MODEL}")
        print(f"🤖 نموذج الـ LLM: {config.LLM_MODEL}")
        print(f"💾 قاعدة البيانات: {config.CHROMA_PATH}")
        print("=" * 60)

        # Initialize SANAD
        chatbot = SANADChatbot()

        # Create UI
        demo = create_beautiful_ui(chatbot)

        print("\n✅ سَنَد جاهز للعمل!")
        print("🌐 جاري فتح المتصفح...")
        print("=" * 60 + "\n")

        # Launch
        demo.queue().launch(
            share=True,
            show_error=True,
            server_name="0.0.0.0",
            server_port=7860,
            favicon_path=None,
        )

    except Exception as e:
        logger.error(f"فشل في تشغيل التطبيق: {e}")
        print(f"\n❌ خطأ: {e}")
        print("\n💡 استكشاف الأخطاء:")
        print("1. تأكد من وجود ملف .env مع OPENROUTER_API_KEY")
        print("2. تحقق من وجود ChromaDB (شغل: python ingest_database_improved.py)")
        print("3. تأكد من Python 3.9+")
        print("4. تأكد من تثبيت جميع المكتبات (pip install -r requirements.txt)")
        raise


if __name__ == "__main__":
    main()
