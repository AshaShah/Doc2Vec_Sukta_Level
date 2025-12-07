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
# Load suktas (each line = one document)
# ============================================================

def load_suktas(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    docs = [l.strip() for l in lines if l.strip()]
    print(f"Loaded {len(docs)} suktas.")
    return docs


# ============================================================
# Build standard Doc2Vec (UNWEIGHTED)
# ============================================================

def build_doc2vec_unweighted(texts, size, epochs=100):
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

    Z = np.vstack([model.infer_vector(t.words, epochs=30) for t in tagged])
    Z = normalize(Z)

    return Z


# ============================================================
# Build TF–IDF weighted Doc2Vec
# ============================================================

def build_doc2vec_weighted(texts, size, epochs=100):

    # compute TF-IDF weights
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tfidf.fit(texts)

    vocab = tfidf.vocabulary_
    idf = tfidf.idf_

    # simple lookup dictionary
    idf_map = {w: idf[idx] for w, idx in vocab.items()}

    # build base Doc2Vec model
    tagged = [TaggedDocument(words=t.split(), tags=[i]) 
              for i, t in enumerate(texts)]

    model = Doc2Vec(
        dm=1,
        vector_size=size,
        window=5,
        min_count=1,
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
        weighted = []
        for w in words:
            if w in idf_map:
                weighted.extend([w] * int(idf_map[w] * 3))  # boost important words
        if not weighted:
            weighted = words
        return model.infer_vector(weighted, epochs=50)

    Z = np.vstack([infer_weighted(t.words) for t in tagged])
    Z = normalize(Z)

    return Z


# ============================================================
# Master script
# ============================================================

def main():

    infile = pathlib.Path("Sanskrit.txt")  # CHANGE THIS
    texts = load_suktas(infile)

    dims = [25, 50, 100, 200, 300, 500, 768, 1024]

    out_unweighted = pathlib.Path("embeddings_d2v_unwe_sans")
    out_weighted = pathlib.Path("embeddings_d2v_we_sans")
    out_unweighted.mkdir(exist_ok=True)
    out_weighted.mkdir(exist_ok=True)

    for d in dims:
        print(f"\n=== Building UNWEIGHTED Doc2Vec: {d} dims ===")
        Z = build_doc2vec_unweighted(texts, size=d)
        np.savetxt(out_unweighted / f"d2v_unweighted_{d}.tsv", Z,
                   fmt="%.6f", delimiter="\t")

    for d in dims:
        print(f"\n=== Building WEIGHTED Doc2Vec: {d} dims ===")
        Z = build_doc2vec_weighted(texts, size=d)
        np.savetxt(out_weighted / f"d2v_weighted_{d}.tsv", Z,
                   fmt="%.6f", delimiter="\t")

    print("\nAll embeddings generated successfully!")


if __name__ == "__main__":
    main()
