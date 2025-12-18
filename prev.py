import json
import pandas as pd
import re

# ============================
# Load previous clustering JSON
# ============================
with open("k8database.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Expecting: data["nodes"] = [{ "id": "RV 1.1", "group": 3, ... }, ...]
nodes = data["nodes"]

prev_df = pd.DataFrame(nodes)

# ============================
# Keep only what we need
# ============================
prev_df = prev_df[["id", "group"]].copy()
prev_df.rename(columns={"id": "document", "group": "cluster"}, inplace=True)

# ============================
# Canonicalize RV formatting
# (important for later merges)
# ============================
def canon_rv(s):
    if not isinstance(s, str):
        return ""
    s = s.strip()
    m = re.search(r"RV\s*(\d+)\.(\d+)", s, flags=re.IGNORECASE)
    if not m:
        return s
    return f"RV {int(m.group(1))}.{int(m.group(2))}"

prev_df["document"] = prev_df["document"].map(canon_rv)

# ============================
# Save
# ============================
prev_df.to_csv("Previous_cluster.csv", index=False)
print("Saved → Previous_cluster.csv")

# Quick sanity check
print(prev_df.head())
print("Total suktas:", len(prev_df))
print("Unique previous clusters:", prev_df["cluster"].nunique())
