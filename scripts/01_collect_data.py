"""
=============================================================
  Script 1 : Collecte de données EN + ES
  Sources :
    - Anglais  : fancyzhx/ag_news (Parquet)
    - Espagnol : mteb/SpanishNewsClassification (Parquet)
  Output :
    - data/raw/english_news.json
    - data/raw/spanish_news.json
    - data/raw/merged_en_es.json
=============================================================
"""

import json
import os
from datasets import load_dataset

N_ARTICLES = 100
OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AGNEWS_LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def collect_english(n: int = 100) -> list[dict]:
    print(f"\n[EN] Chargement fancyzhx/ag_news ({n} articles)...")
    ds = load_dataset("fancyzhx/ag_news", split="test", streaming=True)
    articles = []
    for item in ds:
        if len(articles) >= n:
            break
        articles.append({
            "content": item["text"],
            "language": "en",
            "category": AGNEWS_LABEL_MAP.get(item["label"], "Unknown")
        })
    print(f"   {len(articles)} articles anglais collectés")
    return articles


def collect_spanish(n: int = 100) -> list[dict]:
    # Source 1 : mteb/SpanishNewsClassification
    try:
        print(f"\n[ES] Chargement mteb/SpanishNewsClassification ({n} articles)...")
        ds = load_dataset("mteb/SpanishNewsClassification", split="train", streaming=True)
        articles = []
        for item in ds:
            if len(articles) >= n:
                break
            content = (item.get("text") or item.get("sentence1") or "").strip()
            if len(content) < 50:
                continue
            articles.append({
                "content": content,
                "language": "es",
                "category": str(item.get("label", "Unknown"))
            })
        if articles:
            print(f"   {len(articles)} articles espagnols collectés")
            return articles
    except Exception as e:
        print(f"    Source 1 échouée : {e}")

    # Source 2 (fallback) : Helsinki-NLP/opus-100
    try:
        print(f"\n[ES] Fallback : Helsinki-NLP/opus-100 (es-en)...")
        ds = load_dataset("Helsinki-NLP/opus-100", "es-en", split="train", streaming=True)
        articles = []
        for item in ds:
            if len(articles) >= n:
                break
            content = item.get("translation", {}).get("es", "").strip()
            if len(content) < 50:
                continue
            articles.append({"content": content, "language": "es", "category": "General"})
        if articles:
            print(f"   {len(articles)} articles depuis opus-100")
            return articles
    except Exception as e:
        print(f"   Fallback échoué : {e}")

    raise RuntimeError("Aucune source espagnole disponible.")


def save_json(data: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"   Sauvegardé : {path} ({len(data)} articles)")


def main():
    print("=" * 55)
    print("  ÉTAPE 1 : COLLECTE DONNÉES EN + ES")
    print("=" * 55)

    english_articles = collect_english(N_ARTICLES)
    spanish_articles = collect_spanish(N_ARTICLES)
    merged = english_articles + spanish_articles

    save_json(english_articles, f"{OUTPUT_DIR}/english_news.json")
    save_json(spanish_articles, f"{OUTPUT_DIR}/spanish_news.json")
    save_json(merged,           f"{OUTPUT_DIR}/merged_en_es.json")

    print(f"\n{'='*55}")
    print(f"   TERMINÉ — EN: {len(english_articles)} | ES: {len(spanish_articles)}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()






