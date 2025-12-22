# clustering.py
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
INPUT_CANDIDATES = ["student_dataset.xlsx", "student_dataset_50.xlsx", "student_habits.xlsx"]
OUTPUT_EXCEL = "clustering_output.xlsx"
OUTPUT_BAR_PNG = "cluster_bar_chart.png"
N_CLUSTERS = 3
RANDOM_STATE = 42
BAR_WIDTH = 20  # width of textual bar in terminal

# ---------- helpers ----------
def find_input_file(cands):
    for f in cands:
        if os.path.exists(f):
            return f
    raise FileNotFoundError("No input file found. Put one of: " + ", ".join(cands))

def find_col(df, variants):
    for v in variants:
        if v in df.columns:
            return v
    return None

# ---------- 1) Load ----------
input_file = find_input_file(INPUT_CANDIDATES)
df = pd.read_excel(input_file)
print("Using input file:", input_file)

# ---------- 2) Detect columns (exact requested names with variants) ----------
study_col = find_col(df, ["StudyHours", "Study Hours", "Study_Hours"])
work_col  = find_col(df, ["WorkHours", "Work Hours", "Work_Hours", "Other Hours", "OtherHours"])
play_col  = find_col(df, ["PlayHours", "Play Hours", "Play_Hours"])
sleep_col = find_col(df, ["SleepHour", "Sleep Hours", "SleepHours"])

marks_col = find_col(df, ["Marks", "Test Score", "TestScore", "Score", "Test_Score"])

missing = [name for name,col in [
    ("Study", study_col), ("Work/Other", work_col), ("Play", play_col), ("Sleep", sleep_col)
] if col is None]
if missing:
    raise ValueError("Missing required hour columns: " + ", ".join(missing))

# For clustering include marks if available; otherwise use PredictedMarks (ensure regression run first)
marks_for_clustering = marks_col if marks_col is not None else ("PredictedMarks" if "PredictedMarks" in df.columns else None)
if marks_for_clustering is None:
    raise ValueError("No Marks column found and no PredictedMarks present. Run regression first or include Marks.")

cluster_cols = [study_col, work_col, play_col, sleep_col, marks_for_clustering]

# ---------- 3) Prepare + scale ----------
X = df[cluster_cols].astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- 4) KMeans ----------
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ---------- 5) Rank clusters by mean marks (best -> rank 0) ----------
cluster_means = df.groupby("Cluster")[marks_for_clustering].mean().sort_values(ascending=False)
sorted_clusters = list(cluster_means.index)
cluster_to_rank = {sorted_clusters[i]: i for i in range(len(sorted_clusters))}
df["ClusterRank"] = df["Cluster"].map(cluster_to_rank)

remarks_map = {
    0: "Excellent performance",
    1: "Average performance",
    2: "Needs improvement"
}
df["Remark"] = df["ClusterRank"].map(remarks_map)

# ---------- 6) Save Excel ----------
df.to_excel(OUTPUT_EXCEL, index=False)
print("\nSaved clustering output to:", OUTPUT_EXCEL)

# ---------- 7) Create bar chart image ----------
counts = df["Cluster"].value_counts().sort_index()
clusters = counts.index.tolist()
values = counts.values.tolist()
colors = ["blue","orange","green"][:len(clusters)]

plt.figure(figsize=(6,4))
plt.bar([str(c) for c in clusters], values, color=colors)
plt.xlabel("Cluster")
plt.ylabel("Number of students")
plt.title("Cluster counts (0,1,2)")
plt.tight_layout()
plt.savefig(OUTPUT_BAR_PNG, dpi=150)
plt.close()

# auto-open image & excel (Windows)
try:
    os.startfile(OUTPUT_BAR_PNG)
except:
    pass
try:
    os.startfile(OUTPUT_EXCEL)
except:
    pass

# ---------- 8) Print textual bars in terminal ----------
max_count = max(values) if values else 1
scale = BAR_WIDTH / max_count

print("\n===== Cluster Distribution =====\n")
for c, v in zip(clusters, values):
    bar_len = int(round(v * scale))
    bar = "█" * bar_len
    # pad to BAR_WIDTH for aligned look
    pad = " " * (BAR_WIDTH - bar_len)
    print(f"Cluster {c}: {bar}{pad}  ({v} students)")
print("\n=================================\n")

# ---------- 9) Also print cluster means and counts ----------
print("Cluster mean marks (descending):")
print(cluster_means.round(2))
print("\nCounts per cluster:")
print(counts.sort_index().to_string())
