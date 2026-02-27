"""
=============================================================
  Script 05b : Preprocessing Arabe Amélioré

  Améliorations vs 05_preprocessing.py :
    1. Lemmatisation arabe via CAMeL Tools (si installé)
    2. Détection arabe standard (MSA) vs dialectal

  Installation AVANT d'exécuter (optionnel) :
    pip install camel-tools
    camel_data -i morphology-db-msa-s31

  Input  : data/processed/all_news_preprocessed.json
  Output : data/processed/all_news_preprocessed_v2.json
=============================================================
"""

import json
import os
from collections import defaultdict

DATA_PATH   = "data/processed/all_news_preprocessed.json"
OUTPUT_PATH = "data/processed/all_news_preprocessed_v2.json"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# ─────────────────────────────────────────────────────────
# PARTIE 1 — DÉTECTION ARABE STANDARD vs DIALECTAL
# ─────────────────────────────────────────────────────────

DIALECTAL_MARKERS = {
    "maghrebi": [
        "هاد", "هادا", "هادي", "واش", "بزاف", "مزيان", "كيفاش",
        "فين", "علاش", "كيما", "باش", "ماشي", "دابا", "غير",
        "نتا", "نتي", "حنا", "نتوما", "هوما", "زوين"
    ],
    "levantine": [
        "هيك", "شو", "كيف", "وين", "ليش", "هلق", "بدي",
        "رح", "عم", "مش", "هاي", "منيح", "كتير"
    ],
    "egyptian": [
        "ازيك", "عامل", "كده", "زي", "بتاع", "دلوقتي",
        "اهو", "يعني", "طب", "ايه", "فين", "امتى", "ليه"
    ],
}


def detect_arabic_variety(text: str) -> dict:
    words = text.split()
    if not words:
        return {"variety": "msa", "dialect": None, "confidence": 0.0}

    dialect_counts = {d: 0 for d in DIALECTAL_MARKERS}
    for word in words:
        for dialect, markers in DIALECTAL_MARKERS.items():
            if word in markers:
                dialect_counts[dialect] += 1

    total_dialectal = sum(dialect_counts.values())
    dialectal_ratio = total_dialectal / len(words)

    if dialectal_ratio > 0.05:
        dominant = max(dialect_counts, key=dialect_counts.get)
        return {
            "variety": "dialectal",
            "dialect": dominant if dialect_counts[dominant] > 0 else None,
            "confidence": round(dialectal_ratio, 3)
        }
    return {
        "variety": "msa",
        "dialect": None,
        "confidence": round(1 - dialectal_ratio, 3)
    }


# ─────────────────────────────────────────────────────────
# PARTIE 2 — LEMMATISATION ARABE
# ─────────────────────────────────────────────────────────

def load_camel_lemmatizer():
    try:
        from camel_tools.disambig.mle import MLEDisambiguator
        mle = MLEDisambiguator.pretrained()
        print("   CAMeL Tools chargé (lemmatisation MSA)")
        return mle
    except ImportError:
        print("     CAMeL Tools non installé → fallback normalisation simple")
        print("       Pour installer : pip install camel-tools")
        print("       Puis            : camel_data -i morphology-db-msa-s31")
        return None
    except Exception as e:
        print(f"     CAMeL Tools erreur : {e} → fallback")
        return None


def lemmatize_arabic_camel(text: str, disambiguator) -> str:
    try:
        from camel_tools.tokenizers.word import simple_word_tokenize
        tokens = simple_word_tokenize(text)
        disambig = disambiguator.disambiguate(tokens)
        lemmas = []
        for d in disambig:
            analyses = d.analyses
            if analyses:
                lemma = analyses[0].analysis.get("lex", d.word)
                lemmas.append(lemma)
            else:
                lemmas.append(d.word)
        return " ".join(lemmas)
    except Exception:
        return text


def lemmatize_arabic_fallback(text: str) -> str:
    """Normalisation arabe sans CAMeL Tools."""
    prefixes = ["ال", "وال", "بال", "كال", "فال", "لل"]
    suffixes = ["ون", "ين", "ات", "ان", "تان", "ية", "ها", "هم", "كم", "نا"]
    words = text.split()
    result = []
    for word in words:
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                word = word[len(prefix):]
                break
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break
        result.append(word)
    return " ".join(result)


def preprocess_arabic(text: str, disambiguator=None) -> dict:
    variety_info = detect_arabic_variety(text)
    if disambiguator and variety_info["variety"] == "msa":
        lemmatized = lemmatize_arabic_camel(text, disambiguator)
    else:
        lemmatized = lemmatize_arabic_fallback(text)
    return {
        "lemmatized":         lemmatized,
        "variety":            variety_info["variety"],
        "dialect":            variety_info["dialect"],
        "dialect_confidence": variety_info["confidence"],
    }


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SCRIPT 05b : PREPROCESSING ARABE AMÉLIORÉ")
    print("=" * 60 + "\n")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)
    print(f" Dataset chargé : {len(articles)} articles\n")

    print(" Chargement CAMeL Tools...")
    disambiguator = load_camel_lemmatizer()

    print("\n Traitement des articles arabes...")
    stats = {"msa": 0, "dialectal": 0, "dialectal_breakdown": defaultdict(int)}
    results = []

    for i, article in enumerate(articles):
        new_article = dict(article)

        if article["language"] == "ar":
            arabic_info = preprocess_arabic(article["content"], disambiguator)
            new_article.update({
                "lemmatized":         arabic_info["lemmatized"],
                "arabic_variety":     arabic_info["variety"],
                "arabic_dialect":     arabic_info["dialect"],
                "dialect_confidence": arabic_info["dialect_confidence"],
            })
            stats[arabic_info["variety"]] += 1
            if arabic_info["dialect"]:
                stats["dialectal_breakdown"][arabic_info["dialect"]] += 1

        results.append(new_article)

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(articles)} articles traités...")

    # Sauvegarde
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Résumé
    ar_total = stats["msa"] + stats["dialectal"]
    print(f"\n{'='*60}")
    print(f"   RÉSULTATS FINAUX")
    print(f"{'='*60}")
    print(f"\n   Articles arabes ({ar_total} total) :")
    print(f"     Arabe standard (MSA)  : {stats['msa']} ({round(stats['msa']/ar_total*100,1) if ar_total else 0}%)")
    print(f"     Arabe dialectal       : {stats['dialectal']} ({round(stats['dialectal']/ar_total*100,1) if ar_total else 0}%)")

    if stats["dialectal_breakdown"]:
        print(f"\n     Détail dialectes :")
        for dialect, count in stats["dialectal_breakdown"].items():
            print(f"       - {dialect:<12} : {count} articles")

    print(f"\n   Fichier généré : {OUTPUT_PATH}")
    print(f"\n   Prochaine étape : python scripts/06_vectorisation.py")
    print(f"      (utilise all_news_preprocessed_v2.json comme input)")

    # Aperçu
    ar_examples = [r for r in results if r["language"] == "ar"]
    if ar_examples:
        ex = ar_examples[0]
        print(f"\n── Aperçu article arabe ──────────────────────────────")
        print(f"  Variété   : {ex.get('arabic_variety')} (confiance : {ex.get('dialect_confidence')})")
        print(f"  Dialecte  : {ex.get('arabic_dialect') or 'aucun (MSA)'}")
        print(f"  Original  : {ex['content'][:120]}...")
        print(f"  Lemmatisé : {ex.get('lemmatized', '')[:120]}...")


if __name__ == "__main__":
    main()