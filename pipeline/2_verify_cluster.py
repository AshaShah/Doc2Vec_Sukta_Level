import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Load leiden clusters file ---
csv_path = Path("cluster_output.csv")  # assumes file in same folder as notebook
df = pd.read_csv(csv_path)

# Identify columns
col_doc = [c for c in df.columns if c.lower() in {"document", "doc", "name", "label", "sukta"}][0]
col_cluster = [c for c in df.columns if "cluster" in c.lower() or "label" in c.lower()][0]

# Normalize RV names
def norm_rv(s: str) -> str:
    if not isinstance(s, str): return s
    s = s.strip()
    s = re.sub(r"[,\s]+", " ", s)
    m = re.search(r"(?i)rv\s*([0-9]+)\s*\.\s*([0-9]+)", s)
    if m: return f"RV {int(m.group(1))}.{int(m.group(2))}"
    m2 = re.search(r"(?i)rv\s*([0-9]+)\s+([0-9]+)", s)
    if m2: return f"RV {int(m2.group(1))}.{int(m2.group(2))}"
    return s.upper()

df["_doc_norm"] = df[col_doc].apply(norm_rv)

# --- Topics and suktas ---
topics = {
    "Creation": ["RV 10.72", "RV 10.81", "RV 10.82", "RV 10.90", "RV 10.121", "RV 10.129", "RV 10.130", "RV 10.190", "RV 10.125"],
    "Marut": ["RV 1.37", "RV 1.38", "RV 1.39", "RV 1.64", "RV 1.85", "RV 1.86", "RV 1.87", "RV 1.88", "RV 1.165", "RV 1.166", "RV 2.34", "RV 8.7", "RV 5.52", "RV 5.87"],
    "Water": ["RV 10.9", "RV 7.49", "RV 10.30", "RV 10.13", "RV 6.61", "RV 7.47"],
    "Surya": ["RV 1.50", "RV 1.115", "RV 1.164", "RV 4.40", "RV 5.40", "RV 7.60", "RV 7.62", "RV 7.63", "RV 7.66", "RV 8.101", "RV 10.36", "RV 10.37", "RV 10.158", "RV 10.170", "RV 10.189", "RV 10.190"],
    "Brihaspati": ["RV 1.18", "RV 1.40", "RV 2.23", "RV 2.24", "RV 2.25", "RV 2.26", "RV 7.97", "RV 10.155", "RV 4.50"],
    "Heaven & Earth": ["RV 1.159", "RV 1.160", "RV 1.185", "RV 4.56", "RV 6.70", "RV 7.53"],
    "Funeral": ["RV 10.14", "RV 10.16", "RV 10.18", "RV 10.135", "RV 10.154", "RV 10.10", "RV 10.11", "RV 10.12", "RV 10.13", "RV 10.15", "RV 10.17", "RV 10.19"],
}

# Normalize all suktas
topics = {k: [norm_rv(x) for x in v] for k, v in topics.items()}

# --- Map suktas to clusters ---
records = []
for topic, suktas in topics.items():
    for s in suktas:
        found = df.loc[df["_doc_norm"] == s, col_cluster]
        if len(found) > 0:
            records.append({"Topic": topic, "Sukta": s, "Cluster": int(found.values[0])})
        else:
            records.append({"Topic": topic, "Sukta": s, "Cluster": None})

result = pd.DataFrame(records)

# --- Save minimal output in current folder ---
out_path = Path("suktas_clusters_with_graph.csv")
result.to_csv(out_path, index=False)
print(f"Saved in current directory → {out_path.resolve()}")
