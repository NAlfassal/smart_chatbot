
# عدّل المسار لو ملفاتك في مكان ثاني
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "knowledge" / "banned_list.jsonl"
OUTPUT_PATH = BASE_DIR / "knowledge"/ "banned_list_unified_clean.jsonl"

OUTPUT_PATH.parent.mkdir(exist_ok=True)



def pick_first(*vals):
    for v in vals:
        if v and str(v).strip():
            return str(v).strip()
    return ""


def normalize_record(rec: dict) -> dict:
    keys = {k.lower(): v for k, v in rec.items()}

    # مصادر مختلفة للأسماء حسب الشيت
    generic_name = pick_first(
        keys.get("genric name"),
        keys.get("generic name"),
        keys.get("اسم المادة الكيميائية المحظورة"),
        keys.get("اسم المادة"),
        keys.get("رقم المادة"),
    )

    other_names = []
    for k, v in rec.items():
        if "other name" in k.lower():
            if v and str(v).strip():
                other_names.append(str(v).strip())

    category = pick_first(
        keys.get("sheet_name"),
        keys.get("category"),
    ).lower()

    # توحيد اسم القسم
    if "cosmatic" in category:
        category = "cosmetics"

    source = keys.get("sheet_name", "unknown")

    return {
        "generic_name": generic_name,
        "other_names": other_names,
        "category": category,
        "source": source,
    }


def main():
    cleaned = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            new_rec = normalize_record(rec)

            # نتأكد ما نضيف سجل بدون اسم
            if new_rec["generic_name"]:
                cleaned.append(new_rec)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Done. Records written: {len(cleaned)}")
    print(f"📁 Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
