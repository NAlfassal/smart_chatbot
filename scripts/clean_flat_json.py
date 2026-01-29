import json
import re
from pathlib import Path
from typing import Dict, Tuple


ARABIC_LETTERS = r"\u0621-\u064A"  # ء-ي
ARABIC_RE = re.compile(f"[{ARABIC_LETTERS}]")


def is_arabic_token(s: str) -> bool:
    return bool(ARABIC_RE.search(s))


def normalize_digits(s: str) -> str:
    # أرقام هندية -> عربية
    trans = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return s.translate(trans)


def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def fix_common_pdf_artifacts(s: str) -> str:
    # إصلاحات شائعة جدًا في استخراج PDF العربي
    s = s.replace("الالئحة", "اللائحة")
    s = s.replace("األ", "الأ").replace("اإل", "الإ")
    s = s.replace("االستخدام", "الاستخدام")
    s = s.replace("اإللكتروني", "الإلكتروني")

    # تحويل 4ـ1 إلى 4-1
    s = normalize_digits(s)
    s = re.sub(r"(\d)\s*ـ\s*(\d)", r"\1-\2", s)

    # إزالة علامات PDF الغريبة مثل :ـل أو ـل لو جات ملتصقة
    s = s.replace(":ـل", "")
    s = re.sub(r"\bـل\b", "", s)

    # توحيد الشرطات/البوليتس
    s = s.replace("–", "-").replace("—", "-")
    return s


def join_split_letters(s: str) -> str:
    """
    يحاول يصلح حالات مثل:
      "ع لى" -> "على"
      "ا لى" -> "الى"
      "ع لى .موقعها" -> "على .موقعها" ثم نصلح المسافة قبل النقطة لاحقًا

    ملاحظة: نتجنب دمج "و/ف/ب/ك/ل" لأنها قد تكون أدوات مستقلة.
    """
    # إصلاحات دقيقة معروفة
    s = re.sub(r"\bع\s+لى\b", "على", s)
    s = re.sub(r"\bا\s+لى\b", "الى", s)
    s = re.sub(r"\bإ\s+لى\b", "إلى", s)

    # دمج حرف عربي منفصل (ليس و ف ب ك ل) مع بداية كلمة عربية قصيرة
    # مثال: "ع لى" / "ت حدد" ... إلخ
    # نكرر كم مرة لأن بعض النصوص تكون متكسّرة أكثر من مرة
    for _ in range(3):
        s2 = re.sub(
            rf"\b(?![وفبكل])([{ARABIC_LETTERS}])\s+([{ARABIC_LETTERS}]{{1,3}})\b",
            r"\1\2",
            s,
        )
        if s2 == s:
            break
        s = s2

    return s


def unstick_words(s: str) -> str:
    """
    لو النص صار ملتصق بدون مسافات مثل:
      "تحددالهيئةالموادالمحظورة..."
    نضيف مسافة بين كلمتين عند الانتقال من:
      (حرف عربي) + (الـ)  أو (حرف عربي) + (حرف عربي كبير؟)
    ما فيه حل مثالي 100% بدون NLP، بس هذا يضبط كثير من حالات PDF.
    """
    # مسافة قبل "ال" إذا كانت ملتصقة بما قبلها: "تحددالهيئة" -> "تحدد الهيئة"
    s = re.sub(rf"([{ARABIC_LETTERS}])(?=ال[{ARABIC_LETTERS}])", r"\1 ", s)

    # مسافة بعد النقطة/الفاصلة لو ملتصقة
    s = re.sub(r"([\.،؛:])(?=\S)", r"\1 ", s)

    return s


def tidy_punctuation(s: str) -> str:
    # مسافة زائدة قبل النقطة
    s = re.sub(r"\s+\.", ".", s)
    s = re.sub(r"\s+،", "،", s)
    s = re.sub(r"\s+؛", "؛", s)
    s = re.sub(r"\s+:", ":", s)

    # توحيد المسافات
    s = re.sub(r"[ \t]+", " ", s)

    return s.strip()


def clean_arabic_pdf_text(text: str) -> str:
    if not text:
        return ""

    s = text
    s = normalize_whitespace(s)
    s = fix_common_pdf_artifacts(s)

    # نحافظ على فواصل الأسطر المهمة (خصوصًا المواد والقوائم)
    # لكن نزيل الأسطر اللي فيها شرطة فقط
    s = re.sub(r"\n\s*-\s*\n", "\n- ", s)

    s = join_split_letters(s)
    s = unstick_words(s)
    s = tidy_punctuation(s)

    # ترتيب القوائم: "- " بداية سطر
    s = re.sub(r"\s*-\s*", "\n- ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    return s


def to_by_source(flat: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    by_source: Dict[str, Dict[str, str]] = {}
    for k, v in flat.items():
        if "||" in k:
            src, art = k.split("||", 1)
        else:
            src, art = "_unknown", k
        by_source.setdefault(src, {})
        by_source[src][art] = v
    return by_source


def clean_flat_json(in_path: Path, out_flat: Path, out_by_source: Path) -> Tuple[int, int]:
    flat = json.loads(in_path.read_text(encoding="utf-8"))

    cleaned = {}
    for k, v in flat.items():
        cleaned[k] = clean_arabic_pdf_text(v)

    out_flat.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    by_source = to_by_source(cleaned)
    out_by_source.write_text(json.dumps(by_source, ensure_ascii=False, indent=2), encoding="utf-8")

    return len(cleaned), sum(len(arts) for arts in by_source.values())


if __name__ == "__main__":
    IN = Path("data/flat.json")
    OUT_FLAT = Path("data/flat_clean.json")
    OUT_BY = Path("data/by_source_clean.json")

    if not IN.exists():
        raise FileNotFoundError(f"ما لقيت {IN} — تأكدي أن flat.json موجود داخل data/")

    n_flat, n_articles = clean_flat_json(IN, OUT_FLAT, OUT_BY)
    print("✅ Cleaning done")
    print(f"- records in flat_clean.json: {n_flat}")
    print(f"- articles in by_source_clean.json: {n_articles}")
    print(f"- wrote: {OUT_FLAT}")
    print(f"- wrote: {OUT_BY}")
