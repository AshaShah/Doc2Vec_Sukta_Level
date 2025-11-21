"""
Doc2Vec on a single file where each line is one document.
Outputs both .npy and .tsv embeddings using STANDARD Doc2Vec params.
"""

import argparse, pathlib, json
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import normalize


def load_texts_single_file(path: pathlib.Path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [l.strip() for l in lines if l.strip()]
    names = [f"line_{i+1:04d}" for i in range(len(lines))]
    return lines, names


def build_doc2vec(texts, size=300, epochs=100):
    tagged = [TaggedDocument(words=t.split(), tags=[i]) for i, t in enumerate(texts)]

    model = Doc2Vec(
        dm=0,                # PV-DBOW (standard)
        vector_size=size,    # standard dimension
        window=5,            # standard window
        min_count=2,         # standard vocabulary threshold
        negative=5,          # standard negative sampling
        hs=0,                # standard: disable hierarchical softmax
        sample=1e-4,         # standard subsampling
        workers=4,
        seed=42
    )

    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=epochs)

    # standard inference using infer_vector
    Z = np.vstack([model.infer_vector(t.words, epochs=20) for t in tagged])
    Z = normalize(Z)

    return Z, model


def main_d2v():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default="mandala.txt")
    parser.add_argument("--outnpy", default="mandala_em/300_d2v.npy")
    parser.add_argument("--outtsv", default="mandala_em/300_d2v.tsv")
    parser.add_argument("--modelout", default="mandala_em/300_doc2vec.model")
    parser.add_argument("--size", type=int, default=300)   # STANDARD 300-dim
    parser.add_argument("--epochs", type=int, default=100)  # STANDARD 20 epochs
    args = parser.parse_args()

    infile = pathlib.Path(args.infile)
    outnpy = pathlib.Path(args.outnpy)
    outtsv = pathlib.Path(args.outtsv)
    outnpy.parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.modelout).parent.mkdir(parents=True, exist_ok=True)

    texts, names = load_texts_single_file(infile)
    Z, model = build_doc2vec(texts, size=args.size, epochs=args.epochs)

    np.save(outnpy, Z)
    np.savetxt(outtsv, Z, fmt="%.6f", delimiter="\t")
    model.save(args.modelout)

    manifest = {
        "docs": names,
        "shape": list(Z.shape),
        "vector_size": args.size,
        "epochs": args.epochs
    }
    (outnpy.parent / "300_d2v_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print("Embedding shape:", Z.shape)
    print("Saved TSV:", outtsv)
    print("Saved model:", args.modelout)


if __name__ == "__main__":
    main_d2v()
