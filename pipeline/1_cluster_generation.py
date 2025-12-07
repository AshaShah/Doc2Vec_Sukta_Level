import numpy as np
import pandas as pd
import umap
import hdbscan
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from sknetwork.clustering import Leiden, Louvain, PropagationClustering, get_modularity


X = np.loadtxt("embeddings/lsa_100_embeddings.tsv", delimiter="\t")
X = normalize(X, norm="l2", axis=1)


umap_10 = umap.UMAP(
    n_neighbors=8,
    n_components=10,
    min_dist=0.0,
    metric="euclidean",
    random_state=42
).fit_transform(X)

X10 = normalize(umap_10, norm="l2", axis=0)

pos2d = umap.UMAP(
    n_neighbors=10,
    n_components=2,
    min_dist=0.0,
    metric="euclidean",
    random_state=42
).fit_transform(X)


nn_euc = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(X10)
adj_euc = nn_euc.kneighbors_graph(mode='connectivity')
adj_euc = adj_euc.maximum(adj_euc.T)

nn_cos = NearestNeighbors(n_neighbors=5, metric='cosine').fit(X10)
adj_cos = nn_cos.kneighbors_graph(mode='connectivity')
adj_cos = adj_cos.maximum(adj_cos.T)


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
        'propagation'
    ]
    """

    if method == 'leiden_euc':
        m = Leiden(modularity='Dugue', random_state=42)
        labels = m.fit_predict(adj_euc)
        mod = get_modularity(adj_euc, labels)
        print("Modularity =", mod)
        return labels

    elif method == 'leiden_cos':
        m = Leiden(modularity='Dugue', random_state=42)
        labels = m.fit_predict(adj_cos)
        mod = get_modularity(adj_cos, labels)
        print("Modularity =", mod)
        return labels

    elif method == 'louvain_euc':
        m = Louvain(modularity='Dugue', random_state=42)
        labels = m.fit_predict(adj_euc)
        print("Modularity =", get_modularity(adj_euc, labels))
        return labels

    elif method == 'louvain_cos':
        m = Louvain(modularity='Dugue', random_state=42)
        labels = m.fit_predict(adj_cos)
        print("Modularity =", get_modularity(adj_cos, labels))
        return labels

    elif method == 'hdbscan':
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=6,
            min_samples=3,
            metric='euclidean'
        )
        return hdb.fit_predict(X10)

    elif method == 'hierarchical':
        cl = AgglomerativeClustering(
            n_clusters=50,
            linkage='ward'
        )
        return cl.fit_predict(X10)

    elif method == 'spectral':
        sc = SpectralClustering(
            n_clusters=50,
            affinity='nearest_neighbors',
            random_state=42
        )
        return sc.fit_predict(X10)

    elif method == 'kmeans':
        km = KMeans(n_clusters=50, random_state=42)
        return km.fit_predict(X10)

    elif method == 'gmm':
        gmm = GaussianMixture(n_components=50, covariance_type='full', random_state=42)
        return gmm.fit_predict(X10)

    elif method == 'propagation':
        pc = PropagationClustering()
        return pc.fit_predict(adj_euc)

    else:
        raise ValueError("Unknown method: " + method)

labels = run_clustering("leiden_euc", X10, adj_euc, adj_cos)
# labels = run_clustering("leiden_cos", X10, adj_euc, adj_cos)
# labels = run_clustering("hdbscan", X10, adj_euc, adj_cos)
# labels = run_clustering("hierarchical", X10, adj_euc, adj_cos)
# labels = run_clustering("spectral", X10, adj_euc, adj_cos)
# labels = run_clustering("louvain_cos", X10, adj_euc, adj_cos)
# labels = run_clustering("louvain_euc", X10, adj_euc, adj_cos)
# labels = run_clustering("propagation", X10, adj_euc, adj_cos)
# labels = run_clustering("kmeans", X10, adj_euc, adj_cos)
# labels = run_clustering("gmm", X10, adj_euc, adj_cos)

plt.figure(figsize=(10,8))
clusters = np.unique(labels)
cmap = plt.cm.get_cmap('hsv', len(clusters))

for i, c in enumerate(clusters):
    idx = np.where(labels == c)
    plt.scatter(pos2d[idx,0], pos2d[idx,1], s=35, color=cmap(i), label=str(c))

plt.title("Cluster Visualization")
plt.legend()
plt.show()


doc_names = open("originalname.txt").read().splitlines()

df = pd.DataFrame({
    "document": doc_names,
    "cluster": labels
})

df.to_csv("cluster_output.csv", index=False)
