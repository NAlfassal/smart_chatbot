from typing import List
from src.config import settings


UI_TO_CATEGORY = {
    "لوائح التجميل (PDF)": "regulation",
    "محظورات التجميل": "banned",
    "الأسس (GDP)": "gdp",
}


def build_retriever(vector_store, ui_choices: List[str]):
    """
    ✅ IMPORTANT:
    Chroma لا يقبل $or/$and إذا القائمة فيها شرط واحد.
    - ولا شيء/الكل => بدون فلتر
    - اختيار واحد => filter بسيط
    - اختيارين+ => $or
    """
    k = settings.RETRIEVAL_K

    if not ui_choices:
        return vector_store.as_retriever(search_kwargs={"k": k})

    selected = [UI_TO_CATEGORY.get(x) for x in ui_choices]
    selected = [x for x in selected if x]

    if not selected or len(selected) >= 3:
        return vector_store.as_retriever(search_kwargs={"k": k})

    if len(selected) == 1:
        return vector_store.as_retriever(search_kwargs={"k": k, "filter": {"category": selected[0]}})

    or_filter = {"$or": [{"category": c} for c in selected]}
    return vector_store.as_retriever(search_kwargs={"k": k, "filter": or_filter})
