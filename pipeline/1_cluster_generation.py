import numpy as np
import pandas as pd
import umap
import hdbscan
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score

from sknetwork.clustering import Leiden, Louvain, PropagationClustering, get_modularity


# ============================================================
# 0) LOAD EMBEDDINGS (NO PRE-NORMALIZATION)
# ============================================================

X = np.loadtxt("grif_d2v_unwe/d2v_unweighted_300.tsv", delimiter="\t")
print("Embedding shape:", X.shape)


# ============================================================
# 1) UMAP REDUCTION
#    10D (for clustering) + 2D (for plotting)
# ============================================================

umap_10 = umap.UMAP(
    n_neighbors=8,        
    n_components=10,
    min_dist=0.0,
    metric="euclidean",
    random_state=42
).fit_transform(X)

# # Normalize AFTER UMAP-10
X10 = normalize(umap_10, norm="l2", axis=1)

pos2d = umap.UMAP(
    n_neighbors=10,
    n_components=2,
    min_dist=0.0,
    metric="euclidean",
    random_state=42
).fit_transform(X)


# ============================================================
# 2) BUILD kNN GRAPHS (EUCLIDEAN + COSINE)
# ============================================================

nn_euc = NearestNeighbors(n_neighbors=8, metric="euclidean").fit(X10)
adj_euc = nn_euc.kneighbors_graph(mode="connectivity")
adj_euc = adj_euc.maximum(adj_euc.T)

nn_cos = NearestNeighbors(n_neighbors=8, metric="cosine").fit(X10)
adj_cos = nn_cos.kneighbors_graph(mode="connectivity")
adj_cos = adj_cos.maximum(adj_cos.T)


# ============================================================
# 3) GROUND TRUTH FOR ARI / AMI (MANDALA GROUPS)
#    Uses originalname.txt lines like "RV 1.23"
# ============================================================

def _coarse_group(book: int) -> int:
    """
    Map RV mandala number (1..10) to coarse group 0..9
    (here just identity – each mandala = one group).
    """
    if 1 <= book <= 10:
        return book - 1
    raise ValueError(book)

doc_names = [line.strip() for line in open("originalname.txt", encoding="utf-8") if line.strip()]

if len(doc_names) != X.shape[0]:
    raise ValueError(f"Mismatch: {len(doc_names)} names vs {X.shape[0]} embeddings")

import re
pat_rv = re.compile(r"^\s*RV\s+(\d{1,2})\.(\d+)\s*$")

books = []
for s in doc_names:
    m = pat_rv.match(s)
    if not m:
        raise ValueError(f"Could not parse RV label: {s!r}")
    books.append(int(m.group(1)))

y_true = np.array([_coarse_group(b) for b in books])


# ============================================================
# 4) GINI HELPER FOR 'ginicluster' (cluster size balance)
# ============================================================

def gini_from_sizes(counts: np.ndarray) -> float:
    """
    Gini-like index on cluster size distribution using
    1 - sum(p^2) (Simpson index). Larger = more balanced.
    """
    counts = counts[counts > 0]
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    return 1.0 - np.sum(p ** 2)


# ============================================================
# 5) UNIFIED EVALUATION FOR ANY LABEL VECTOR
# ============================================================

def evaluate_clustering(method_name, labels, X10, adj_euc):
    labels = np.asarray(labels)
    n_clusters = len(np.unique(labels[labels >= 0]))  # ignore noise label -1
    print(f"\n=== {method_name} ===")
    print("Clusters (excluding noise if any):", n_clusters)

    # Silhouette (only if ≥ 2 clusters and no all-noise)
    sil = None
    valid_idx = labels >= 0
    if np.unique(labels[valid_idx]).shape[0] > 1:
        try:
            sil = silhouette_score(X10[valid_idx], labels[valid_idx], metric="euclidean")
            print(f"Silhouette (euclidean, on X10, excluding noise): {sil:.3f}")
        except Exception as e:
            print("Silhouette not computed:", e)
    else:
        print("Silhouette: not defined (≤ 1 cluster).")

    # Modularity on Euclidean graph (works for any labels)
    try:
        Q = get_modularity(adj_euc, labels)
        print(f"Modularity on adj_euc: {Q:.3f}")
    except Exception as e:
        print("Modularity not computed:", e)


    try:
        ari = adjusted_rand_score(y_true, labels)
        ami = adjusted_mutual_info_score(y_true, labels, average_method="arithmetic")
        print(f"ARI: {ari:.3f}")
        print(f"AMI: {ami:.3f}")
    except Exception as e:
        print("ARI/AMI not computed:", e)

    return {
        "method": method_name,
        "n_clusters": n_clusters,
        "silhouette": sil,
        "modularity": Q if "Q" in locals() else None,
        "ari": ari if "ari" in locals() else None,
        "ami": ami if "ami" in locals() else None,
    }


# ============================================================
# 6) CLUSTERING DISPATCH
#    All methods use (or are close to) library defaults
# ============================================================

def run_clustering(method, X10, adj_euc, adj_cos):
    """
    method: one of [
        'leiden_euc', 'leiden_cos',
        'louvain_euc', 'louvain_cos',
        'hdbscan',
        'hierarchical',
        'spectral',
        'kmeans',
        'gmm',
        'propagation',
        'ginicluster'
    ]
    """

    if method == "leiden_euc":
        # Leiden defaults: modularity='Newman', resolution=1, n_iter=10
        m = Leiden(modularity='Dugue', resolution=1.0, random_state=42)                    # use library defaults
        labels = m.fit_predict(adj_euc)
        return labels, evaluate_clustering(method, labels, X10, adj_euc)

    elif method == "leiden_cos":
        m = Leiden(modularity='Dugue', resolution=1.0, random_state=42)
        labels = m.fit_predict(adj_cos)
        return labels, evaluate_clustering(method, labels, X10, adj_euc)

    elif method == "louvain_euc":
        m = Louvain(modularity='Dugue', resolution=1.2)
        labels = m.fit_predict(adj_euc)
        return labels, evaluate_clustering(method, labels, X10, adj_euc)

    elif method == "louvain_cos":
        m = Louvain(modularity='Dugue', resolution=1.2)
        labels = m.fit_predict(adj_cos)
        return labels, evaluate_clustering(method, labels, X10, adj_euc)

    elif method == "hdbscan":
        # HDBSCAN defaults: min_cluster_size=5, min_samples=None, metric='euclidean'
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=None,
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=0.0,
            alpha=1.0
        )
        labels = hdb.fit_predict(X10)
        return labels, evaluate_clustering(method, labels, X10, adj_euc)

    elif method == "hierarchical":
        # AgglomerativeClustering defaults: n_clusters=2, linkage='ward'
        cl= AgglomerativeClustering(
            n_clusters=40,
            linkage='ward'
        )
        labels = cl.fit_predict(X10)
        return labels, evaluate_clustering(method, labels, X10, adj_euc)

    elif method == "spectral":
        # SpectralClustering defaults: n_clusters=8, affinity='rbf'
        sc = SpectralClustering(
            n_clusters=40,
            affinity='nearest_neighbors',
            random_state=42
        )
        labels = sc.fit_predict(X10)
        return labels, evaluate_clustering(method, labels, X10, adj_euc)

    elif method == "kmeans":
        km = KMeans(
            n_clusters=40,
            init="k-means++",
            n_init="auto",
            max_iter=300,
            random_state=42
        )
        labels = km.fit_predict(X10)
        return labels, evaluate_clustering(method, labels, X10, adj_euc)

    elif method == "gmm":
        # GaussianMixture defaults: n_components=1, covariance_type='full'
        gmm = GaussianMixture(
            n_components=40,
            covariance_type='full',
            random_state=42
        )

        labels = gmm.fit_predict(X10)
        return labels, evaluate_clustering(method, labels, X10, adj_euc)

    elif method == "propagation":
        # PropagationClustering defaults: n_iter=20, node_order='random'
        pc = PropagationClustering()
        labels = pc.fit_predict(adj_euc)
        return labels, evaluate_clustering(method, labels, X10, adj_euc)

    elif method == "ginicluster":
        # Gini-based hierarchical: search over K and pick K with highest
        # Gini balance of cluster sizes (1 - sum p^2).
        best_g = -1.0
        best_labels = None
        best_k = None

        for k in range(10, 81, 5):  # K = 10,15,...,80
            cl = AgglomerativeClustering(n_clusters=k)  # linkage='ward' default
            lab = cl.fit_predict(X10)
            _, counts = np.unique(lab, return_counts=True)
            g = gini_from_sizes(counts)
            if g > best_g:
                best_g = g
                best_labels = lab
                best_k = k

        print(f"\n[ginicluster] chosen n_clusters={best_k} with Gini={best_g:.3f}")
        return best_labels, evaluate_clustering(method, best_labels, X10, adj_euc)

    else:
        raise ValueError("Unknown method: " + method)


# ============================================================
# 7) RUN METHODS (CHOOSE WHICH ONE TO VISUALIZE BELOW)
# ============================================================

all_methods = [
    "leiden_euc",
    "leiden_cos",
    "louvain_euc",
    "louvain_cos",
    "hdbscan",
    "hierarchical",
    "spectral",
      "kmeans",
    "gmm",
    "propagation",
    "ginicluster",
]

all_results = []
all_labels = {}

for m in all_methods:
    print("\n" + "=" * 60)
    print(f"Running method: {m}")
    labels_m, stats_m = run_clustering(m, X10, adj_euc, adj_cos)
    all_results.append(stats_m)
    all_labels[m] = labels_m

# Convert stats to DataFrame for easy comparison
results_df = pd.DataFrame(all_results)
print("\n=== METHOD COMPARISON TABLE ===")
print(results_df)

# Optionally save comparison table
results_df.to_csv("clustering_method_comparison.csv", index=False)
print("Saved → clustering_method_comparison.csv")

# ---- CHOOSE ONE METHOD TO USE FOR PLOTTING AND CSV BELOW ----
method_to_plot = "hdbscan"  # change to e.g. "ginicluster", "hdbscan", etc.

labels = all_labels[method_to_plot]
# get the corresponding stats dict
stats = [s for s in all_results if s["method"] == method_to_plot][0]



# ============================================================
# 8) VISUALIZE RESULT
# ============================================================

plt.figure(figsize=(10, 8))
clusters = np.unique(labels)
cmap = plt.cm.get_cmap("hsv", len(clusters))

for i, c in enumerate(clusters):
    idx = np.where(labels == c)
    plt.scatter(pos2d[idx, 0], pos2d[idx, 1], s=35, color=cmap(i), label=str(c))

plt.title(f"Cluster Visualization: {stats['method']}")
plt.legend(bbox_to_anchor=(1.03, 1), loc="upper left")
plt.tight_layout()
plt.show()


# ============================================================
# 9) SAVE CLUSTERS TO CSV
# ============================================================

df = pd.DataFrame({
    "document": doc_names,
    "cluster": labels
})

df.to_csv("cluster_output.csv", index=False)
print("Saved → cluster_output.csv")











#default():

# import numpy as np
# import pandas as pd
# import umap
# import hdbscan
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import normalize
# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
# from sklearn.mixture import GaussianMixture
# from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score

# from sknetwork.clustering import Leiden, Louvain, PropagationClustering, get_modularity


# # ============================================================
# # 0) LOAD EMBEDDINGS (NO PRE-NORMALIZATION)
# # ============================================================

# X = np.loadtxt("grif_d2v_unwe/d2v_unweighted_200.tsv", delimiter="\t")
# print("Embedding shape:", X.shape)


# # ============================================================
# # 1) UMAP REDUCTION
# #    10D (for clustering) + 2D (for plotting)
# # ============================================================

# umap_10 = umap.UMAP(
#     n_neighbors=8,        
#     n_components=10,
#     min_dist=0.0,
#     metric="euclidean",
#     random_state=42
# ).fit_transform(X)

# # Normalize AFTER UMAP-10, axis=0  (matches colleague recipe)
# X10 = normalize(umap_10, norm="l2", axis=0)

# pos2d = umap.UMAP(
#     n_neighbors=10,
#     n_components=2,
#     min_dist=0.0,
#     metric="euclidean",
#     random_state=42
# ).fit_transform(X)


# # ============================================================
# # 2) BUILD kNN GRAPHS (EUCLIDEAN + COSINE)
# # ============================================================

# nn_euc = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(X10)
# adj_euc = nn_euc.kneighbors_graph(mode="connectivity")
# adj_euc = adj_euc.maximum(adj_euc.T)

# nn_cos = NearestNeighbors(n_neighbors=5, metric="cosine").fit(X10)
# adj_cos = nn_cos.kneighbors_graph(mode="connectivity")
# adj_cos = adj_cos.maximum(adj_cos.T)


# # ============================================================
# # 3) GROUND TRUTH FOR ARI / AMI (MANDALA GROUPS)
# #    Uses originalname.txt lines like "RV 1.23"
# # ============================================================

# def _coarse_group(book: int) -> int:
#     """
#     Map RV mandala number (1..10) to coarse group 0..9
#     (here just identity – each mandala = one group).
#     """
#     if 1 <= book <= 10:
#         return book - 1
#     raise ValueError(book)

# doc_names = [line.strip() for line in open("originalname.txt", encoding="utf-8") if line.strip()]

# if len(doc_names) != X.shape[0]:
#     raise ValueError(f"Mismatch: {len(doc_names)} names vs {X.shape[0]} embeddings")

# import re
# pat_rv = re.compile(r"^\s*RV\s+(\d{1,2})\.(\d+)\s*$")

# books = []
# for s in doc_names:
#     m = pat_rv.match(s)
#     if not m:
#         raise ValueError(f"Could not parse RV label: {s!r}")
#     books.append(int(m.group(1)))

# y_true = np.array([_coarse_group(b) for b in books])


# # ============================================================
# # 4) GINI HELPER FOR 'ginicluster' (cluster size balance)
# # ============================================================

# def gini_from_sizes(counts: np.ndarray) -> float:
#     """
#     Gini-like index on cluster size distribution using
#     1 - sum(p^2) (Simpson index). Larger = more balanced.
#     """
#     counts = counts[counts > 0]
#     n = counts.sum()
#     if n == 0:
#         return 0.0
#     p = counts / n
#     return 1.0 - np.sum(p ** 2)
