from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.config import settings


def get_llm():
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        api_key=(settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY),
        base_url=settings.LLM_BASE_URL,
        max_tokens=settings.LLM_MAX_TOKENS,
    )


def stream_answer(llm, question: str, knowledge: str):
    prompt = f"""
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

سؤال المستخدم: {question}

اكتب الإجابة الآن:
""".strip()

    for chunk in llm.stream([HumanMessage(content=prompt)]):
        if getattr(chunk, "content", None):
            yield chunk.content
