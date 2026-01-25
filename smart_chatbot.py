from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import gradio as gr
import os
import re
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas import evaluate
from datasets import Dataset
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
CHROMA_PATH = r"chroma_db"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")

if not OPENROUTER_KEY:
    print("ERROR: OPENROUTER_API_KEY not found in .env file.")
    raise SystemExit(1)

# --- Initialize Models and DB ---
device = "cpu"
print(f"Using device for embeddings: {device}")

embeddings_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": device},
)
print(f"Embeddings model loaded on {device}.")

llm = ChatOpenAI(
    model="deepseek/deepseek-r1:free",
    temperature=0.0,
    openai_api_key=OPENROUTER_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "Smart Chatbot SFDA",
    },
)
print("LLM configured for OpenRouter.")

vector_store = Chroma(
    collection_name="sfda_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)
print("ChromaDB connected.")

retriever = vector_store.as_retriever(search_kwargs={"k": 10})
print("Retriever set up with k=10.")


def clean_repeated_characters(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1", text)


# --- Chatbot Response Logic ---
def stream_response(message, history):
    print(f"\nNew query: {message}")
    query_with_prefix = f"query: {message}"
    retrieved_docs = retriever.invoke(query_with_prefix)

    if not retrieved_docs:
        yield "عفواً، لم يتم العثور على معلومات متعلقة بسؤالك في المستندات المتاحة."
        return

    knowledge = ""
    last_source_file = "N/A"
    for doc in retrieved_docs:
        last_source_file = os.path.basename(doc.metadata.get("source", "N/A"))
        knowledge += f"المصدر: [{last_source_file}]\n"
        knowledge += f"المحتوى: \"{doc.page_content}\"\n"
        knowledge += "---\n"

    relevance_check_prompt = f"""
Based ONLY on the provided "Knowledge Context", determine if it contains a plausible answer to the "User's Question".
The answer must be direct and substantive. Do not consider vague or tangential mentions as a valid answer.
Respond with only one word: "Yes" or "No".

Knowledge Context:
---
{knowledge}
---

User's Question: {message}
"""
    relevance_response = llm.invoke(relevance_check_prompt)
    is_relevant = "yes" in relevance_response.content.lower()
    print(f"Relevance check result: {'Relevant' if is_relevant else 'Not Relevant'}")

    if not is_relevant:
        yield "عفواً، لا يمكنني الإجابة على هذا السؤال لأن المعلومات المطلوبة غير متوفرة في المصادر المرفقة."
        return

    generation_prompt = f"""
ROLE AND DIRECTIVE:
You are a meticulous archival retrieval bot. Your SOLE function is to find and present information exactly as it appears in the provided documents.
You MUST NOT use any internal or external knowledge, infer, or create new text.

STRICT EXECUTION PROTOCOL:

1. LANGUAGE: All output MUST be in Arabic.

2. TARGETED EXTRACTION PRINCIPLE:
- Your primary goal is to scan all provided documents in knowledge and identify the ONE passage that is the most direct and specific answer to the user's question without any changes.

3. OUTPUT FORMAT (MANDATORY):
- Your response MUST consist of direct, verbatim quotes only
- Each quote MUST be enclosed in quotation marks (“…”).
- Immediately following EACH quote, you MUST provide an inline citation in this exact format:
  "the answer"
  [المصدر: FILE_NAME]

4. CONTEXT QUALITY CONTROL:
- You MUST NOT unnaturally repeat characters in words. Ensure the output is fluent and correctly spelled.

5. FAILURE CONDITION:
- ONLY if no relevant information is found in the sources, reply with this exact Arabic sentence and nothing else:
عفواً، لا يمكنني الإجابة على هذا السؤال لأن المعلومات المطلوبة غير متوفرة في المصادر المرفقة

----
Knowledge Context:
{knowledge}
----
User's Question: {message}

IMPORTANT: Use the source filename from the context lines like:
المصدر: [filename.pdf]
"""

    messages = [
        {"role": "system", "content": generation_prompt},
        {"role": "user", "content": message},
    ]

    final_answer = ""
    for response in llm.stream(messages):
        if response.content:
            final_answer += response.content
            yield clean_repeated_characters(final_answer)


# --- CSS Styling ---
css_code = """
.gradio-container {
    background: linear-gradient(to bottom right, #e6f4ea, #d0f0f8) !important;
    padding-top: 100px;
    position: relative;
    font-family: 'Tahoma', sans-serif;
}

.gradio-container::before {
    content: "";
    position: absolute;
    top: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: 80px;
    height: 80px;
    background-image: url(data:image/png;base64,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);
    background-repeat: no-repeat;
    background-position: center;
    z-index: 1000;
}

.message.user { background-color: #d6f5e3 !important; color: #004d26 !important; border-radius: 12px; }
.message.bot { background-color: #d0f0f7 !important; color: #003a63 !important; border-radius: 12px; }

textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #00a651 !important;
    border-radius: 6px !important;
}

.gradio-examples .gr-button {
    background-color: #00a651 !important;
    color: white !important;
    border-radius: 6px !important;
    font-weight: bold;
    border: 1px solid #00a651;
}

h1, .prose {
    color: #003a63 !important;
    text-align: center;
}
"""

# --- Launch Gradio Chatbot ---

with gr.Blocks(css=css_code) as demo:
    gr.ChatInterface(
        fn=stream_response,
        title="المساعد الذكي",
        description="أهلاً بك , هذا هو مساعدك الذكي للوائح الهيئة العامة للغذاء والدواء, يمكنك طرح سؤالك باللغة العربية للحصول على إجابة دقيقة ومباشرة من المصادر الرسمية",
        textbox=gr.Textbox(placeholder="كيف يمكنني مساعدتك ؟", container=False, scale=7),
        examples=[
            "ماهي المادة الرابعة في اللائحة التنفيذية لنظام منتجات التجميل؟",
            "ما هي الشروط التي يجب ان يشتمل عليها نظام الجودة؟"
        ],
    )

print("Launching Gradio app...")
demo.queue().launch(share=True)
