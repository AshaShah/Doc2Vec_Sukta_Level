 #============================================================
# 10) TOPIC → BEST CLUSTER HIGHLIGHT PLOT (UMAP STRUCTURE)
#     - grey everything
#     - color only the 7 selected clusters (one per topic)
#     - bigger nodes for the topic suktas found in that cluster
#     - labels for those big nodes
# ============================================================

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

topics = {
    "Creation": ["RV 10.72", "RV 10.81", "RV 10.82", "RV 10.90", "RV 10.121", "RV 10.129", "RV 10.130", "RV 10.190", "RV 10.125"],
    "Marut": ["RV 1.37", "RV 1.38", "RV 1.39", "RV 1.64", "RV 1.85", "RV 1.86", "RV 1.87", "RV 1.88", "RV 1.165", "RV 1.166", "RV 2.34", "RV 8.7", "RV 5.52", "RV 5.87"],
    "Water": ["RV 10.9", "RV 7.49", "RV 10.30", "RV 10.13", "RV 6.61", "RV 7.47"],
    "Surya": ["RV 1.50", "RV 1.115", "RV 1.164", "RV 4.40", "RV 5.40", "RV 7.60", "RV 7.62", "RV 7.63", "RV 7.66", "RV 8.101", "RV 10.36", "RV 10.37", "RV 10.158", "RV 10.170", "RV 10.189", "RV 10.190"],
    "Brihaspati": ["RV 1.18", "RV 1.40", "RV 2.23", "RV 2.24", "RV 2.25", "RV 2.26", "RV 7.97", "RV 10.155", "RV 4.50"],
    "Heaven & Earth": ["RV 1.159", "RV 1.160", "RV 1.185", "RV 4.56", "RV 6.70", "RV 7.53"],
    "Funeral": ["RV 10.14", "RV 10.16", "RV 10.18", "RV 10.135", "RV 10.154", "RV 10.10", "RV 10.11", "RV 10.12", "RV 10.13", "RV 10.15", "RV 10.17", "RV 10.19"],
}

def canon_rv(s):
    if not isinstance(s, str):
        return ""
    s = s.strip().replace("\t", " ")
    s = re.sub(r"\s+", " ", s)
    m = re.search(r"RV\s*(\d+)\.(\d+)", s, flags=re.IGNORECASE)
    if not m:
        return s
    return f"RV {int(m.group(1))}.{int(m.group(2))}"

# --- build a dataframe aligned to pos2d + labels ---
plot_df = pd.DataFrame({
    "document": doc_names,    # from your code above
    "cluster": labels,        # from your method_to_plot above
    "x": pos2d[:, 0],         # from your code above
    "y": pos2d[:, 1],
})
plot_df["doc_can"] = plot_df["document"].astype(str).map(canon_rv)

doc2cluster = dict(zip(plot_df["doc_can"], plot_df["cluster"]))
cluster_sizes = plot_df["cluster"].value_counts().to_dict()

# --- pick best cluster per topic (ignore -1 unless it is the only option) ---
topic_best = {}
found_by_topic = {}

for topic, suktas in topics.items():
    cans = [canon_rv(x) for x in suktas]
    clist = [doc2cluster.get(c) for c in cans if doc2cluster.get(c) is not None]

    if len(clist) == 0:
        topic_best[topic] = None
        found_by_topic[topic] = []
        continue

    counts = pd.Series(clist).value_counts()
    counts_non = counts.drop(index=[-1], errors="ignore")

    if len(counts_non) > 0:
        best_cluster = int(counts_non.index[0])
        overlap = int(counts_non.iloc[0])
    else:
        best_cluster = int(counts.index[0])
        overlap = int(counts.iloc[0])

    topic_best[topic] = best_cluster
    found_by_topic[topic] = [c for c in cans if doc2cluster.get(c) == best_cluster]

# --- build precision/recall/F1 table exactly like your earlier logic ---
rows = []
for topic, suktas in topics.items():
    cans = [canon_rv(x) for x in suktas]
    best_cluster = topic_best.get(topic)
    total_topic = len(cans)

    if best_cluster is None:
        rows.append({
            "Topic": topic,
            "Best_Cluster": None,
            "Matched_in_Cluster": f"0 of {total_topic}",
            "Precision": 0.0, "Recall": 0.0, "Fscore": 0.0,
            "Topic_Total": total_topic, "Cluster_Size": 0, "Overlap": 0
        })
        continue

    overlap = len(found_by_topic[topic])
    cluster_size = int(cluster_sizes.get(best_cluster, 0))
    recall = overlap / total_topic if total_topic else 0.0
    precision = overlap / cluster_size if cluster_size else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    rows.append({
        "Topic": topic,
        "Best_Cluster": best_cluster,
        "Matched_in_Cluster": f"{overlap} of {total_topic}",
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "Fscore": round(f1, 3),
        "Topic_Total": total_topic,
        "Cluster_Size": cluster_size,
        "Overlap": overlap
    })

topic_table = pd.DataFrame(rows).sort_values(["Overlap", "Recall"], ascending=False).reset_index(drop=True)
topic_table.to_csv("topic_precision_recall_table.csv", index=False)
print("\nSaved → topic_precision_recall_table.csv")
print(topic_table)

# --- print the found suktas list (and save) ---
found_list = pd.DataFrame([{
    "Topic": t,
    "Selected_Cluster": topic_best[t],
    "Found_Suktas_in_Selected_Cluster": ", ".join(found_by_topic[t]) if found_by_topic[t] else "(none)"
} for t in topics.keys()])

found_list.to_csv("topic_found_suktas_in_selected_cluster.csv", index=False)
print("\nSaved → topic_found_suktas_in_selected_cluster.csv")
print(found_list)

# --- plotting rules: grey everywhere, color only selected clusters ---
topic_list = list(topics.keys())
cmap = plt.get_cmap("tab10")
topic_color = {t: cmap(i % 10) for i, t in enumerate(topic_list)}

# cluster -> color (one topic per cluster; if two topics pick same cluster, last one wins)
cluster_color = {}
for t, c in topic_best.items():
    if c is not None:
        cluster_color[c] = topic_color[t]

# bigger nodes for "found" suktas (across all topics)
matches = set()
for t in topic_list:
    matches.update(found_by_topic[t])

is_selected_cluster = plot_df["cluster"].map(lambda c: c in cluster_color)
is_match = plot_df["doc_can"].isin(matches)

base_size = 12
cluster_size = 28
match_size = 90

sizes = np.where(is_selected_cluster, cluster_size, base_size)
sizes = np.where(is_match, match_size, sizes)

default_grey = (0.7, 0.7, 0.7, 0.25)
colors = [
    cluster_color.get(c, default_grey) if sel else default_grey
    for c, sel in zip(plot_df["cluster"].values, is_selected_cluster.values)
]

plt.figure(figsize=(10, 8), dpi=200)
plt.scatter(plot_df["x"], plot_df["y"], s=sizes, c=colors, linewidths=0)

# # label only the big nodes
# for _, r in plot_df[is_match].iterrows():
#     plt.text(r["x"] + 0.02, r["y"] + 0.02, r["doc_can"], fontsize=7)

# legend = topics with selected cluster and overlap
handles, labels_ = [], []
for t in topic_list:
    c = topic_best[t]
    if c is None:
        continue
    ov = len(found_by_topic[t])
    tot = len(topics[t])
    handles.append(plt.Line2D([0],[0], marker='o', color='w',
                              markerfacecolor=topic_color[t], markersize=10))
    labels_.append(f"{t}: cluster {c} (found {ov}/{tot})")

plt.legend(handles, labels_, bbox_to_anchor=(1.03, 1), loc="upper left",
           title="Topic → Selected Cluster", fontsize=9)

plt.title(f"Topic-highlighted clusters on UMAP ({method_to_plot})\nGrey = all other nodes")
plt.xticks([]); plt.yticks([])
plt.tight_layout()

out_png = "rigveda_topic_cluster_plot_umap.png"
plt.savefig(out_png, bbox_inches="tight")
plt.show()
print(f"\nSaved → {out_png}")
