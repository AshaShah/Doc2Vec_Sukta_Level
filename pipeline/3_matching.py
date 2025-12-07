import pandas as pd

# ===========================
# 1. Load the datasets
# ===========================
leiden_df = pd.read_csv("lsa100_leiden_clusters.csv")       # columns: document, cluster
sukta_df = pd.read_csv("suktas_clusters_with_graph.csv")      # columns: Topic, Sukta, Cluster

# ===========================
# 2. Merge topics with Leiden clusters
# ===========================
merged = pd.merge(
    sukta_df, leiden_df,
    how="left",
    left_on="Sukta", right_on="document"
)
merged.rename(columns={"cluster": "Leiden_Cluster"}, inplace=True)
merged.drop(columns=["document"], inplace=True)

# ===========================
# 3. Compute Precision & Recall per Topic
# ===========================
results = []

for topic, sub in merged.groupby("Topic"):
    counts = sub["Leiden_Cluster"].dropna().value_counts()

    # --- Skip topics that have no matched clusters ---
    if counts.empty:
        results.append({
            "Topic": topic,
            "Best_Cluster": None,
            "Matched_in_Cluster": f"0 of {len(sub)}",
            "Precision": 0.0,
            "Recall": 0.0,
            "Fscore": 0.0,
            "Topic_Total": len(sub),
            "Cluster_Size": 0,
            "Overlap": 0
        })
        continue

    best_cluster = counts.idxmax()
    best_overlap = counts.max()
    total_topic = len(sub)

    total_cluster = len(leiden_df[leiden_df["cluster"] == best_cluster])
    precision = best_overlap / total_cluster
    recall = best_overlap / total_topic
    fscore = 2 * ((precision * recall) / (precision + recall))

    results.append({
        "Topic": topic,
        "Best_Cluster": best_cluster,
        "Matched_in_Cluster": f"{best_overlap} of {total_topic}",
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "Fscore": round(fscore, 3),
        "Topic_Total": total_topic,
        "Cluster_Size": total_cluster,
        "Overlap": best_overlap
    })

results_df = pd.DataFrame(results)

# ===========================
# 4. Display
# ===========================
results_df = results_df.sort_values("Recall", ascending=False).reset_index(drop=True)
display(results_df)

# ===========================
# 5. Summary
# ===========================
print("\n=== Summary by Topic ===")
for _, row in results_df.iterrows():
    print(f"{row['Topic']}: {row['Matched_in_Cluster']} hymns grouped in Leiden cluster {row['Best_Cluster']} "
          f"(Precision={row['Precision']:.3f}, Recall={row['Recall']:.3f}, Fscore={row['Fscore']:.3f})")

results_df.to_csv("topic_precision_recall_table.csv", index=False)
print("\nSaved → topic_precision_recall_table.csv")
