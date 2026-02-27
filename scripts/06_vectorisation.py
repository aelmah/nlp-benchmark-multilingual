"""
=============================================================
  Script 6 : Benchmark Vectorisation
  Input  : data/processed/all_news_preprocessed.json
  Output : data/processed/vectors_tfidf.npz
           data/processed/vectors_fasttext.npy
           data/processed/vectors_xlmroberta.npy
           results/benchmark_vectorisation_summary.csv
           results/pca_visualisation.png

  Techniques :
    1. TF-IDF        (baseline statistique)
    2. fastText      (embeddings denses 300D)
    3. XLM-RoBERTa   (Transformer 768D)

  Métriques :
    - Temps de vectorisation (ms/article)
    - Dimension des vecteurs
    - Visualisation PCA 2D par langue

  Installation :
    pip install scikit-learn fasttext-wheel transformers torch matplotlib
=============================================================
"""

import json
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import urllib.request

DATA_PATH   = "data/processed/all_news_preprocessed_v2.json"  # version améliorée avec lemmatisation AR
VECTORS_DIR = "data/processed"
RESULTS_DIR = "results"
os.makedirs(VECTORS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FASTTEXT_EMBED_URL  = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
FASTTEXT_MODEL_PATH = "cc.en.300.bin"     # modèle CommonCrawl EN 300D — embeddings réels

LANG_COLORS = {"ar": "#E74C3C", "fr": "#3498DB", "en": "#2ECC71", "es": "#F39C12"}
LANG_LABELS = {"ar": "Arabe", "fr": "Français", "en": "Anglais", "es": "Espagnol"}


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts  = [" ".join(item["tokens"]) if item["tokens"] else item["content"] for item in data]
    langs  = [item["language"] for item in data]
    print(f" Dataset chargé : {len(texts)} articles\n")
    return texts, langs


# ─────────────────────────────────────────────
# TECHNIQUE 1 — TF-IDF
# ─────────────────────────────────────────────

def run_tfidf(texts: list[str], langs: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import save_npz, hstack
    import numpy as np

    print(" [1/3] TF-IDF par langue (AR/FR/EN/ES séparés)...")

    # Grouper par langue
    from collections import defaultdict
    lang_groups  = defaultdict(list)
    lang_indices = defaultdict(list)
    for i, (text, lang) in enumerate(zip(texts, langs)):
        lang_groups[lang].append(text)
        lang_indices[lang].append(i)

    all_vectors = [None] * len(texts)
    dim_total   = 0
    start       = time.time()

    lang_dir = f"{VECTORS_DIR}/tfidf_by_lang"
    os.makedirs(lang_dir, exist_ok=True)

    for lang in ["ar", "fr", "en", "es"]:
        lang_texts = lang_groups[lang]
        if not lang_texts:
            continue
        max_features = 3000 if lang == "ar" else 2000
        vect    = TfidfVectorizer(max_features=max_features, sublinear_tf=True, min_df=2, max_df=0.95)
        vectors = vect.fit_transform(lang_texts)
        save_npz(f"{lang_dir}/tfidf_{lang}.npz", vectors)
        print(f"   [{lang.upper()}] {len(lang_texts)} articles → {vectors.shape[1]}D")
        for j, idx in enumerate(lang_indices[lang]):
            all_vectors[idx] = vectors[j].toarray()[0]
        dim_total = max(dim_total, vectors.shape[1])

    # Uniformiser les dimensions par padding
    max_dim = max(v.shape[0] for v in all_vectors if v is not None)
    final_vectors = []
    for v in all_vectors:
        if v is None:
            final_vectors.append(np.zeros(max_dim))
        elif v.shape[0] < max_dim:
            final_vectors.append(np.pad(v, (0, max_dim - v.shape[0])))
        else:
            final_vectors.append(v)

    vectors_array = np.array(final_vectors)
    elapsed_ms    = (time.time() - start) / len(texts) * 1000

    np.save(f"{VECTORS_DIR}/vectors_tfidf.npy", vectors_array)

    print(f"\n   Dimension totale : {vectors_array.shape[1]}D (sparse par langue)")
    print(f"   Vitesse          : {round(elapsed_ms, 3)} ms/article")
    print(f"    Sauvegardé dans : {lang_dir}/\n")

    return vectors_array, elapsed_ms, vectors_array.shape[1]


# ─────────────────────────────────────────────
# TECHNIQUE 2 — fastText Embeddings
# ─────────────────────────────────────────────

def get_fasttext_embedding(model, text: str) -> np.ndarray:
    """Moyenne des vecteurs de mots."""
    words = text.split()
    if not words:
        return np.zeros(300)
    vecs = [model.get_word_vector(w) for w in words]
    return np.mean(vecs, axis=0)


def run_fasttext(texts: list[str]):
    print(" [2/3] fastText Embeddings (via gensim)...")

    # Utilise gensim pour télécharger fasttext-wiki-news-subwords-300
    # Modèle léger ~1GB, 300D, multilingue
    import gensim.downloader as api

    print("   Chargement du modèle fasttext-wiki-news-subwords-300 (~1GB, première fois)...")
    model = api.load("fasttext-wiki-news-subwords-300")

    def embed(text: str) -> np.ndarray:
        words = text.split()
        if not words:
            return np.zeros(300)
        vecs = []
        for w in words:
            try:
                vecs.append(model[w])
            except KeyError:
                continue
        return np.mean(vecs, axis=0) if vecs else np.zeros(300)

    start = time.time()
    vectors = np.array([embed(text) for text in texts])
    elapsed_ms = (time.time() - start) / len(texts) * 1000

    np.save(f"{VECTORS_DIR}/vectors_fasttext.npy", vectors)

    print(f"   Dimension   : {vectors.shape[1]}D (dense)")
    print(f"   Vitesse     : {round(elapsed_ms, 3)} ms/article")
    print(f"    Sauvegardé : vectors_fasttext.npy\n")

    return vectors, elapsed_ms, vectors.shape[1]


# ─────────────────────────────────────────────
# TECHNIQUE 3 — XLM-RoBERTa
# ─────────────────────────────────────────────

def run_xlmroberta(texts: list[str]):
    from transformers import AutoTokenizer, AutoModel
    import torch

    print(" [3/3] XLM-RoBERTa Embeddings...")
    print("   Chargement du modèle (première fois : ~1GB)...")

    model_name = "papluca/xlm-roberta-base-language-detection"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"   Appareil    : {device}")

    def embed(text: str) -> np.ndarray:
        inputs = tokenizer(
            text[:512], return_tensors="pt",
            truncation=True, padding=True, max_length=128
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token embedding
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    start = time.time()
    vectors = np.array([embed(text) for text in texts])
    elapsed_ms = (time.time() - start) / len(texts) * 1000

    np.save(f"{VECTORS_DIR}/vectors_xlmroberta.npy", vectors)

    print(f"   Dimension   : {vectors.shape[1]}D (dense)")
    print(f"   Vitesse     : {round(elapsed_ms, 3)} ms/article")
    print(f"    Sauvegardé : vectors_xlmroberta.npy\n")

    return vectors, elapsed_ms, vectors.shape[1]


# ─────────────────────────────────────────────
# VISUALISATION PCA
# ─────────────────────────────────────────────

def plot_pca(vectors_dict: dict, langs: list[str]):
    print(" Génération de la visualisation PCA...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Visualisation PCA 2D — Séparation par langue", fontsize=14, fontweight="bold")

    for ax, (name, vectors) in zip(axes, vectors_dict.items()):
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(vectors)
        variance = pca.explained_variance_ratio_

        for lang in ["ar", "fr", "en", "es"]:
            idx = [i for i, l in enumerate(langs) if l == lang]
            ax.scatter(
                coords[idx, 0], coords[idx, 1],
                c=LANG_COLORS[lang], label=LANG_LABELS[lang],
                alpha=0.6, s=15
            )

        ax.set_title(f"{name}\n(variance : {variance[0]:.1%} + {variance[1]:.1%})", fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{RESULTS_DIR}/pca_visualisation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Sauvegardé : {path}\n")


# ─────────────────────────────────────────────
# RÉSULTATS
# ─────────────────────────────────────────────

def print_results(results: dict):
    print("\n" + "=" * 60)
    print("  RÉSULTATS — BENCHMARK VECTORISATION")
    print("=" * 60)

    rows = [{"Technique": k, "Dimension": v["dim"],
             "Type": v["type"], "Vitesse (ms)": round(v["speed_ms"], 3)}
            for k, v in results.items()]
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    fastest = min(results, key=lambda k: results[k]["speed_ms"])
    print(f"\n Plus rapide   : {fastest}")
    print(f" Plus riche    : XLM-RoBERTa (768D, contextualisé)")
    print("=" * 60)

    df.to_csv(f"{RESULTS_DIR}/benchmark_vectorisation_summary.csv", index=False)
    print(f"\n Résultats sauvegardés dans {RESULTS_DIR}/")
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ÉTAPE 6 : BENCHMARK VECTORISATION")
    print("  TF-IDF | fastText | XLM-RoBERTa")
    print("=" * 60 + "\n")

    texts, langs = load_data(DATA_PATH)

    benchmark_results = {}
    vectors_for_pca   = {}

    # 1. TF-IDF
    v_tfidf, s1, d1 = run_tfidf(texts, langs)
    benchmark_results["TF-IDF"]       = {"dim": d1, "type": "Sparse", "speed_ms": s1}
    vectors_for_pca["TF-IDF"]         = v_tfidf

    # 2. fastText
    v_ft, s2, d2 = run_fasttext(texts)
    benchmark_results["fastText"]     = {"dim": d2, "type": "Dense", "speed_ms": s2}
    vectors_for_pca["fastText"]       = v_ft

    # 3. XLM-RoBERTa
    v_xlm, s3, d3 = run_xlmroberta(texts)
    benchmark_results["XLM-RoBERTa"]  = {"dim": d3, "type": "Dense (Transformer)", "speed_ms": s3}
    vectors_for_pca["XLM-RoBERTa"]   = v_xlm

    # Résultats
    print_results(benchmark_results)

    # Visualisation PCA
    plot_pca(vectors_for_pca, langs)

    print("\n Étape 6 terminée ! Prêt pour le benchmark sentiment.")


if __name__ == "__main__":
    main()