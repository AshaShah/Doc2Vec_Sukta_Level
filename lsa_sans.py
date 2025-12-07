"""
Generate LSA embeddings (weighted + unweighted) at multiple sizes.
Outputs ONLY TSV files.
Sizes: 50, 100, 200, 300, 500, 768
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import os

# ============================================================
# 1. Load suktas
# ============================================================

def load_suktas(path="Griff_translation.txt"):
    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    return lines


# ============================================================
# 2. LSA Unweighted (CountVectorizer)
# ============================================================

def embed_lsa_unweighted(texts, size):
    vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(texts)

    svd = TruncatedSVD(
        n_components=size,
        random_state=42,
        n_iter=12
    )
    Z = svd.fit_transform(X)
    Z = normalize(Z, norm="l2", axis=1)
    return Z


# ============================================================
# 3. LSA Weighted (TF–IDF)
# ============================================================

def embed_lsa_weighted(texts, size):
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(texts)

    svd = TruncatedSVD(
        n_components=size,
        random_state=42,
        n_iter=12
    )
    Z = svd.fit_transform(X)
    Z = normalize(Z, norm="l2", axis=1)
    return Z


# ============================================================
# 4. Generate and save all embeddings
# ============================================================

def save_tsv(Z, outfile):
    np.savetxt(outfile, Z, delimiter="\t", fmt="%.6f")
    print("Saved:", outfile)


def main():
    sizes = [50, 100, 200, 300, 500, 768]
    texts = load_suktas()

    os.makedirs("embeddings-lsa-grif", exist_ok=True)

    for dim in sizes:
        # Unweighted
        Z_un = embed_lsa_unweighted(texts, size=dim)
        save_tsv(Z_un, f"embeddings-lsa-grif/lsa_unweighted_{dim:03d}.tsv")

        # Weighted
        Z_wt = embed_lsa_weighted(texts, size=dim)
        save_tsv(Z_wt, f"embeddings-lsa-grif/lsa_weighted_{dim:03d}.tsv")


if __name__ == "__main__":
    main()
