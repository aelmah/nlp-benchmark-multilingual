"""
=============================================================
  Script 2 : Nettoyage & Standardisation EN + ES
  Input  : data/raw/english_news.json
           data/raw/spanish_news.json
  Output : data/processed/english_news_clean.json
           data/processed/spanish_news_clean.json
           data/processed/merged_en_es_clean.json
  Format cible : [{"content": "texte propre"}, ...]
=============================================================
"""

import json
import re
import os

INPUT_DIR  = "data/raw"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)                    # HTML
    text = re.sub(r'http\S+|www\.\S+', '', text)            # URLs
    text = re.sub(r'[\\\/\|\*\#\@\$\^\&\~\`]', ' ', text)  # Caractères spéciaux
    text = text.replace('\\"', '"').replace("\\'", "'")
    text = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text.strip()


def is_valid(text: str, min_length: int = 80) -> bool:
    return len(text) >= min_length


def process_file(input_path: str, output_path: str, lang_label: str) -> list[dict]:
    print(f"\n[{lang_label}] Traitement de {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned, skipped = [], 0
    for item in data:
        text = clean_text(item.get("content", ""))
        if not is_valid(text):
            skipped += 1
            continue
        cleaned.append({"content": text})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=4)

    print(f"   {len(cleaned)} articles sauvegardés → {output_path}")
    if skipped:
        print(f"    {skipped} articles ignorés (trop courts)")
    return cleaned


def main():
    print("=" * 55)
    print("  ÉTAPE 2 : NETTOYAGE & STANDARDISATION")
    print("=" * 55)

    en_data = process_file(f"{INPUT_DIR}/english_news.json",  f"{OUTPUT_DIR}/english_news_clean.json",  "EN")
    es_data = process_file(f"{INPUT_DIR}/spanish_news.json",  f"{OUTPUT_DIR}/spanish_news_clean.json",  "ES")

    merged = en_data + es_data
    with open(f"{OUTPUT_DIR}/merged_en_es_clean.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)

    print(f"\n{'='*55}")
    print(f"   TERMINÉ — EN: {len(en_data)} | ES: {len(es_data)} | Total: {len(merged)}")
    print(f"   {OUTPUT_DIR}/merged_en_es_clean.json")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()








