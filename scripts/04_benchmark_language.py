"""
=============================================================
  Script 4 : Benchmark Détection de Langue
  Dataset  : data/processed/all_news_dataset.json
  Techniques : langdetect | langid | fastText
  Output   : results/benchmark_language_results.json
             results/benchmark_language_summary.csv
=============================================================
"""

import json
import time
import os
import re
import urllib.request
import pandas as pd
from collections import defaultdict

DATA_PATH   = "data/processed/all_news_dataset.json"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

FASTTEXT_MODEL_URL  = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
FASTTEXT_MODEL_PATH = "lid.176.ftz"

LANG_NORMALIZE = {
    "ar": "ar", "ara": "ar",
    "fr": "fr", "fra": "fr", "fre": "fr",
    "en": "en", "eng": "en",
    "es": "es", "spa": "es",
}

def normalize(lang: str) -> str:
    return LANG_NORMALIZE.get(lang.lower(), "other")


def load_data(path: str) -> tuple[list[str], list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for i, item in enumerate(data):
        texts.append(item["content"])
        if i >= 143 and i < 243:
            labels.append("en")
        elif i >= 243:
            labels.append("es")
        else:
            arabic_chars = len(re.findall(r'[\u0600-\u06FF]', item["content"]))
            labels.append("ar" if arabic_chars > len(item["content"]) * 0.1 else "fr")

    dist = defaultdict(int)
    for l in labels:
        dist[l] += 1
    print(f" Dataset chargé : {len(texts)} articles | Distribution : {dict(dist)}\n")
    return texts, labels


def run_langdetect(texts):
    from langdetect import detect, LangDetectException
    results, start = [], time.time()
    for text in texts:
        try:
            results.append(normalize(detect(text[:500])))
        except LangDetectException:
            results.append("other")
    return results, round((time.time() - start) / len(texts) * 1000, 3)


def run_langid(texts):
    import langid
    results, start = [], time.time()
    for text in texts:
        lang, _ = langid.classify(text[:500])
        results.append(normalize(lang))
    return results, round((time.time() - start) / len(texts) * 1000, 3)


def download_fasttext_model():
    if not os.path.exists(FASTTEXT_MODEL_PATH):
        print("   Téléchargement du modèle fastText...")
        urllib.request.urlretrieve(FASTTEXT_MODEL_URL, FASTTEXT_MODEL_PATH)
        print("    Modèle téléchargé")


def run_fasttext(texts):
    import fasttext
    import inspect
    import numpy as np

    # Patch compatibilité NumPy 2.x
    try:
        import fasttext.FastText as ft_module
        ft_path = inspect.getfile(ft_module)
        with open(ft_path, "r") as f:
            source = f.read()
        if "np.array(probs, copy=False)" in source:
            patched = source.replace("np.array(probs, copy=False)", "np.asarray(probs)")
            with open(ft_path, "w") as f:
                f.write(patched)
            import importlib
            importlib.reload(ft_module)
            importlib.reload(fasttext)
    except Exception as e:
        print(f"     Patch NumPy : {e}")

    download_fasttext_model()
    model = fasttext.load_model(FASTTEXT_MODEL_PATH)
    results, start = [], time.time()
    for text in texts:
        pred = model.predict(text[:500].replace("\n", " "), k=1)
        results.append(normalize(pred[0][0].replace("__label__", "")))
    return results, round((time.time() - start) / len(texts) * 1000, 3)


def evaluate(predictions, ground_truth):
    languages = ["ar", "fr", "en", "es"]
    correct_total = 0
    per_lang = {l: {"correct": 0, "total": 0} for l in languages}
    for pred, true in zip(predictions, ground_truth):
        if pred == true:
            correct_total += 1
        if true in per_lang:
            per_lang[true]["total"] += 1
            if pred == true:
                per_lang[true]["correct"] += 1
    return {
        "accuracy_global": round(correct_total / len(ground_truth) * 100, 2),
        "accuracy_per_lang": {
            l: round(per_lang[l]["correct"] / per_lang[l]["total"] * 100, 1)
            if per_lang[l]["total"] > 0 else 0.0 for l in languages
        }
    }


def main():
    print("=" * 60)
    print("  ÉTAPE 4 : BENCHMARK DÉTECTION DE LANGUE")
    print("  Techniques : langdetect | langid | fastText")
    print("=" * 60 + "\n")

    texts, ground_truth = load_data(DATA_PATH)
    benchmark_results = {}

    for name, func in [("langdetect", run_langdetect), ("langid", run_langid), ("fastText", run_fasttext)]:
        print(f" {name}...")
        preds, speed = func(texts)
        metrics = evaluate(preds, ground_truth)
        benchmark_results[name] = {**metrics, "speed_ms": speed}
        print(f"   Accuracy : {metrics['accuracy_global']}% | Vitesse : {speed} ms/article\n")

    # Tableau comparatif
    rows = [{"Technique": k, "Accuracy (%)": v["accuracy_global"],
             "AR (%)": v["accuracy_per_lang"]["ar"], "FR (%)": v["accuracy_per_lang"]["fr"],
             "EN (%)": v["accuracy_per_lang"]["en"], "ES (%)": v["accuracy_per_lang"]["es"],
             "Vitesse (ms)": v["speed_ms"]} for k, v in benchmark_results.items()]
    df = pd.DataFrame(rows).sort_values("Accuracy (%)", ascending=False)

    print("\n" + "=" * 60)
    print("  RÉSULTATS")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n Meilleure technique : {df.iloc[0]['Technique']}")

    # Sauvegarde
    with open(f"{RESULTS_DIR}/benchmark_language_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=4)
    df.to_csv(f"{RESULTS_DIR}/benchmark_language_summary.csv", index=False)
    print(f"\n Résultats sauvegardés dans {RESULTS_DIR}/")


if __name__ == "__main__":
    main()








