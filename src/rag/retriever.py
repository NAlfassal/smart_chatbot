from typing import List
from src.config import RETRIEVAL_K
from src.utils.logger import logger

# يجب أن تطابق هذه القيم ما يتم إرساله من Gradio CheckboxGroup
UI_TO_CATEGORY = {
    "لوائح التجميل (PDF)": "regulation",
    "محظورات التجميل": "banned",
    "الأسس (GDP)": "gdp",
}

def build_retriever(vector_store, ui_choices: List[str]):
    """
    Build a retriever with optional category filtering based on UI choices.
    """
    k = RETRIEVAL_K 

    logger.info(f"Building retriever with UI choices: {ui_choices}")

    # 1. إذا لم يختر المستخدم شيئاً، ابحث في الكل
    if not ui_choices:
        logger.info(f"No filters applied. Returning all results (k={k})")
        return vector_store.as_retriever(search_kwargs={"k": k})

    # تحويل الاختيارات العربية إلى Codes
    selected = [UI_TO_CATEGORY.get(x) for x in ui_choices]
    selected = [x for x in selected if x] # إزالة القيم الفارغة None

    # 2. إذا اختار الكل (أو أكثر من 2 خيارات)، نعتبرها بحثاً عاماً
    # (يمكنك تعديل هذا الشرط إذا أردت تفعيل الفلتر حتى لو اختار 3)
    if not selected or len(selected) >= len(UI_TO_CATEGORY):
        logger.info("All categories selected or implied. No filter applied.")
        return vector_store.as_retriever(search_kwargs={"k": k})
   
    # 3. فلتر لتصنيف واحد
    if len(selected) == 1:
        category_filter = {"category": selected[0]}
        logger.info(f"Applying single category filter: {category_filter}")
        return vector_store.as_retriever(search_kwargs={"k": k, "filter": category_filter})

    # 4. فلتر لتصنيفين ($or)
    or_filter = {"$or": [{"category": c} for c in selected]}
    logger.info(f"Applying multi-category OR filter: {or_filter}")
    return vector_store.as_retriever(search_kwargs={"k": k, "filter": or_filter})