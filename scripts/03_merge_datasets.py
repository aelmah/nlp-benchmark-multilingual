"""
=============================================================
  Script 3 : Fusion des datasets AR/FR + EN/ES
  Input  : data/raw/example_mixed.json       (143 articles AR/FR)
           data/processed/merged_en_es_clean.json (200 articles EN/ES)
  Output : data/processed/all_news_dataset.json  (343 articles total)
=============================================================
"""

import json
import os

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


def main():
    print("=" * 50)
    print("  ÉTAPE 3 : FUSION AR/FR + EN/ES")
    print("=" * 50)

    with open(f"{RAW_DIR}/example_mixed.json", "r", encoding="utf-8") as f:
        ar_fr_data = json.load(f)

    with open(f"{PROCESSED_DIR}/merged_en_es_clean.json", "r", encoding="utf-8") as f:
        en_es_data = json.load(f)

    all_data = ar_fr_data + en_es_data

    output_path = f"{PROCESSED_DIR}/all_news_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    print(f"  AR/FR  : {len(ar_fr_data)} articles")
    print(f"  EN/ES  : {len(en_es_data)} articles")
    print(f"  TOTAL  : {len(all_data)} articles")
    print(f"\n  Dataset final → {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()








