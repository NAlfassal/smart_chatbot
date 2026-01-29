import re
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT = BASE_DIR / "knowledge" / "chunks_final.jsonl"
OUTPUT = BASE_DIR / "knowledge" / "clean_chunks_v2.jsonl"


def normalize_arabic(text):
    # توحيد الحروف
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)

    # إزالة التكرار المزعج
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # فصل كلمة "المادة" إذا ملتصقة
    text = re.sub(r'(الماده)(\S)', r'\1 \2', text)

    # إزالة الرموز الغريبة
    text = re.sub(r'[^\w\s\u0600-\u06FF\.\،\:\-\(\)]', ' ', text)

    # توحيد المسافات
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


with open(INPUT, "r", encoding="utf-8") as f_in, \
     open(OUTPUT, "w", encoding="utf-8") as f_out:

    for line in f_in:
        row = json.loads(line)
        text = row.get("text", "")

        cleaned = normalize_arabic(text)

        if len(cleaned) < 50:
            continue

        row["text"] = cleaned
        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Done: clean_chunks_v2.jsonl created")

