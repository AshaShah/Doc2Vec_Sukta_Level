"""
Generate Doc2Vec embeddings (weighted + unweighted)
for sizes: 50, 100, 200, 300, 500, 768.

Input:
    1 file where EACH LINE = ONE SUKTA

Output:
    embeddings_d2v_unweighted/<size>.tsv
    embeddings_d2v_weighted/<size>.tsv
"""

import pathlib
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# GLOBAL CONFIG
# ============================================================

TRAIN_EPOCHS = 60      # training epochs for Doc2Vec
INFER_EPOCHS = 80      # epochs for infer_vector (both variants)
DIMS = [25, 50, 100, 200, 300, 500, 768, 1024]


# ============================================================
# Load suktas (each line = one document)
# ============================================================

def load_suktas(path: pathlib.Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    docs = [l.strip() for l in lines if l.strip()]
    print(f"Loaded {len(docs)} suktas.")
    return docs


# ============================================================
# Build standard Doc2Vec (UNWEIGHTED)
# ============================================================

def build_doc2vec_unweighted(texts, size, epochs=TRAIN_EPOCHS):
    tagged = [TaggedDocument(words=t.split(), tags=[i])
              for i, t in enumerate(texts)]

    model = Doc2Vec(
        dm=1,
        vector_size=size,
        window=5,
        min_count=2,      
        negative=5,
        hs=0,
        sample=1e-4,
        workers=4,
        seed=42
    )
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=epochs)

    # use consistent INFER_EPOCHS for all docs
    Z = np.vstack([model.infer_vector(t.words, epochs=INFER_EPOCHS) for t in tagged])
    Z = normalize(Z)

    return Z


# ============================================================
# Build TF–IDF weighted Doc2Vec
# ============================================================

def build_doc2vec_weighted(texts, size, epochs=TRAIN_EPOCHS):

    # compute TF-IDF weights on raw texts
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tfidf.fit(texts)

    vocab = tfidf.vocabulary_
    idf = tfidf.idf_

    # lookup: word → idf
    idf_map = {w: idf[idx] for w, idx in vocab.items()}

    # base Doc2Vec model
    tagged = [TaggedDocument(words=t.split(), tags=[i])
              for i, t in enumerate(texts)]

    model = Doc2Vec(
        dm=1,
        vector_size=size,
        window=5,
        min_count=2,       
        negative=5,
        hs=0,
        sample=1e-4,
        workers=4,
        seed=42
    )
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=epochs)

    # weighted inference
    def infer_weighted(words):
        weighted_tokens = []
        for w in words:
            if w in idf_map:
                # scale repetition by IDF (tuned factor 3)
                reps = int(idf_map[w] * 3)
                if reps > 0:
                    weighted_tokens.extend([w] * reps)

        if not weighted_tokens:
            weighted_tokens = list(words)

        return model.infer_vector(weighted_tokens, epochs=INFER_EPOCHS)

    Z = np.vstack([infer_weighted(t.words) for t in tagged])
    Z = normalize(Z)

    return Z


# ============================================================
# Master script
# ============================================================

def main():

    infile = pathlib.Path("sanskrit.txt")  # CHANGE THIS IF NEEDED
    texts = load_suktas(infile)

    out_unweighted = pathlib.Path("sans_d2v_unwe")
    out_weighted = pathlib.Path("sans_d2v_we")
    out_unweighted.mkdir(exist_ok=True)
    out_weighted.mkdir(exist_ok=True)

    # UNWEIGHTED
    for d in DIMS:
        print(f"\n=== Building UNWEIGHTED Doc2Vec: {d} dims ===")
        Z = build_doc2vec_unweighted(texts, size=d)
        np.savetxt(out_unweighted / f"d2v_unweighted_{d}.tsv",
                   Z, fmt="%.6f", delimiter="\t")

    # WEIGHTED
    for d in DIMS:
        print(f"\n=== Building WEIGHTED Doc2Vec: {d} dims ===")
        Z = build_doc2vec_weighted(texts, size=d)
        np.savetxt(out_weighted / f"d2v_weighted_{d}.tsv",
                   Z, fmt="%.6f", delimiter="\t")

    print("\nAll embeddings generated successfully!")


if __name__ == "__main__":
    main()
