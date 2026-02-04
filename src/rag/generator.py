from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src import config
from src.utils.logger import logger

def get_llm():
    api_key = config.OPENROUTER_API_KEY or config.OPENAI_API_KEY
    
    if not api_key:
        logger.critical("No API Key found for LLM!")
        raise ValueError("OPENROUTER_API_KEY or OPENAI_API_KEY is missing.")

    return ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=api_key,
        base_url=config.LLM_BASE_URL,
        max_tokens=config.LLM_MAX_TOKENS,
    )

def stream_answer(llm, question: str, knowledge: str):
    # تحسين بسيط في التنسيق لإزالة المسافات الزائدة
    prompt = f"""
ROLE:
أنت مساعد امتثال يعتمد فقط على النصوص المرفقة.

RULES:
- لا تضف أي معلومة من خارج "النصوص المساعدة".
- إذا لم تجد نصاً صريحاً، قل: "لم أجد نصاً صريحاً في المصادر المرفقة يجيب عن ذلك."
- اكتب إجابة قصيرة ومنظمة.

----
النصوص المساعدة:
{knowledge}

سؤال المستخدم: {question}
""".strip()

    try:
        for chunk in llm.stream([HumanMessage(content=prompt)]):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        logger.error(f"LLM Generation Error: {e}")
        yield "عذراً، حدث خطأ أثناء توليد النص."