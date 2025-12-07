import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

def load_suktas(path="jemison.txt"):
    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    return lines

def embed_lsa_unweighted(texts, size=100):
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

def save_tsv(Z, outfile):
    np.savetxt(outfile, Z, delimiter="\t", fmt="%.6f")
    print("Saved:", outfile)

texts = load_suktas()

for dim in [50, 200]:
    Z = embed_lsa_unweighted(texts, size=dim)
    save_tsv(Z, f"lsa-unwe-jemison-embeddings/lsa_unweighted_{dim}.tsv")
