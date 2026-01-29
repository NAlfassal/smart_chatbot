import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

INPUT = KNOWLEDGE_DIR / "clean_chunks_v2.jsonl"
OUTPUT = KNOWLEDGE_DIR / "chunks_final.jsonl"



def extract_article_number(text: str) -> str:
    """
    Catch many Arabic variants:
    المادة / الماده / مادة / ماده
    + digits 123 or Arabic-Indic ١٢٣
    + optional parentheses
    """
    # normalize common variants to help matching
    t = text
    t = t.replace("الماده", "المادة")
    t = t.replace("ماده", "مادة")
    t = re.sub(r"\s+", " ", t).strip()

    # 1) المادة (12) / المادة 12 / المادة: 12 / المادة-12
    m = re.search(r"(?:المادة|مادة)\s*[:\-]?\s*\(?\s*([0-9٠-٩]+)\s*\)?", t)
    if m:
        return m.group(1)

    # 2) المادة الرابعة / المادة العاشرة (كلمة بعد المادة)
    m = re.search(r"(?:المادة|مادة)\s+([^\s\-\،\.\(\)]+)", t)
    if m:
        return m.group(1)

    # 3) fallback: "رقم المادة 12" / "المادة رقم 12"
    m = re.search(r"(?:رقم\s*المادة|المادة\s*رقم)\s*[:\-]?\s*([0-9٠-٩]+)", t)
    if m:
        return m.group(1)

    return "NA"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    total = 0
    fixed = 0

    with open(INPUT, "r", encoding="utf-8") as f_in, \
         open(OUTPUT, "w", encoding="utf-8") as f_out:

        for line in f_in:
            row = json.loads(line)

            text = clean_text(row["text"])
            article_no = extract_article_number(text)

            if article_no != "NA":
                fixed += 1

            new_row = {
                "doc_name": row["doc_name"],
                "page": int(row["page"]),
                "article_no": article_no,
                "text": text,
            }

            f_out.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            total += 1

    print(f"Total chunks: {total}")
    print(f"Article numbers detected: {fixed}")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
