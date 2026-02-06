import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz
import os 
from matplotlib.colors import Normalize

DK = pytz.timezone("Europe/Copenhagen")

FIG_PATH = "../Results/Figures/"
os.makedirs(FIG_PATH, exist_ok=True)

CONFIGS = [
    "SB_dwell_5",
    "SB_dwell_15",
    "HIS_dwell_5",
    "HIS_dwell_15",
    "GMM"
]

OD_all = {}

# ---------------------------------------------------------
# Time window (DENMARK TIME)
# ---------------------------------------------------------
start = DK.localize(pd.Timestamp('2025-06-16'))
end   = DK.localize(pd.Timestamp('2025-06-21'))

# ---------------------------------------------------------
# LOAD OD TABLES FROM PARQUET
# ---------------------------------------------------------
for cfg in CONFIGS:
    if cfg == "GMM":
        continue
    else:
        path = f"../Results/OD_results/{cfg}/OD_{cfg}.parquet"
        df = pd.read_parquet(path)
    
        # Convert timestamps from parquet → tz-aware Denmark time
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert(DK)
        df["end_time"]   = pd.to_datetime(df["end_time"],   utc=True).dt.tz_convert(DK)
    
        # 1. Remove impossible trips
        df = df[df["end_time"] >= df["start_time"]]
    
        # 2. Restrict to desired time window
        df = df[(df["start_time"] <= end) & (df["end_time"] >= start)]
    
        # 3. Remove Wednesday (weekday=2)
        df = df[df["start_time"].dt.weekday != 2]
        df = df[df["end_time"].dt.weekday   != 2]
    
        df = df.sort_values("start_time").reset_index(drop=True)
    
        OD_all[cfg] = df
"""
    # -----------------------------------------------------
    # EAST↔WEST counts
    # -----------------------------------------------------
    east_west = len(df[(df['Origin'] < 410000) & (df['Destination'] > 410000)])
    west_east = len(df[(df['Origin'] > 410000) & (df['Destination'] < 410000)])

    print(cfg)
    print("øst→vest:", east_west)
    print("vest→øst:", west_east)


print("\nMastra comparison:")
mod_odense = np.sum([2885, 3202, 3136*0, 3132, 2620])
mod_kbh    = np.sum([2989, 3028, 3001*0, 2737, 2241])
print("øst→vest (model):", mod_odense)
print("vest→øst (model):", mod_kbh)
"""

# ---------------------------------------------------------
# BUILD OD MATRICES
# ---------------------------------------------------------
OD_matrices = {}

for cfg, df in OD_all.items():
    df2 = df[df["Origin"].notna() & df["Destination"].notna()].copy()
    df2["Origin"] = df2["Origin"].astype(int)
    df2["Destination"] = df2["Destination"].astype(int)

    mat = df2.groupby(["Origin", "Destination"]).size().unstack(fill_value=0)
    OD_matrices[cfg] = mat/4

# Align them to same zone set
all_zones = sorted(set().union(*[mat.index for mat in OD_matrices.values()]))

aligned = {
    cfg: mat.reindex(index=all_zones, columns=all_zones, fill_value=0)
    for cfg, mat in OD_matrices.items()
}

# ---------------------------------------------------------
# AGGREGATE BY MUNICIPALITY LEVEL
# ---------------------------------------------------------
def zone_map(x, level=0):
    assert 0 <= level < 4
    return (x // 10**(3 - level)) * 10**(3 - level)

OD_matrices_lvl = {}
for cfg, df in OD_all.items():

    df = df[df["Origin"].notna() & df["Destination"].notna()].copy()
    df["Origin"] = df["Origin"].astype(int).apply(zone_map)
    df["Destination"] = df["Destination"].astype(int).apply(zone_map)

    mat = df.groupby(["Origin", "Destination"]).size().unstack(fill_value=0)
    OD_matrices_lvl[cfg] = mat

all_zones_lvl = sorted(set().union(*[m.index for m in OD_matrices_lvl.values()]))

aligned_lvl = {
    cfg: m.reindex(index=all_zones_lvl, columns=all_zones_lvl, fill_value=0)
    for cfg, m in OD_matrices_lvl.items()
}

# -------------------------------------------------------------
# Load GMM matrix
# -------------------------------------------------------------
GMM_PATH = "C:/Users/b306630/OneDrive - Vejdirektoratet/Skrivebord/Master Thesis/Data/FreightTripMatrixWD.csv"
GMM = pd.read_csv(GMM_PATH)

# Standardize column names
GMM.rename(columns={
    "FromZoneID": "Origin",
    "ToZoneID": "Destination",
    "Val": "Trips"
}, inplace=True)

# Remove non-DK zones (ID >= 900000)
GMM.loc[(GMM["Origin"] > 900000), "Origin"] = 999999
GMM.loc[(GMM["Destination"] > 900000), "Destination"] = 999999

# Ensure integers
GMM["Origin"] = GMM["Origin"].astype(int)
GMM["Destination"] = GMM["Destination"].astype(int)

# Apply level=0 (municipality level)
GMM["Origin"] = GMM["Origin"].apply(zone_map)
GMM["Destination"] = GMM["Destination"].apply(zone_map)

# -------------------------------------------------------------
# Construct OD matrix
# -------------------------------------------------------------
mat = GMM.groupby(["Origin", "Destination"])["Trips"].sum().unstack(fill_value=0)

# Sort indices
origins = sorted(mat.index)
dests   = sorted(mat.columns)
mat = mat.reindex(index=all_zones_lvl, columns=dests, fill_value=0)
aligned_lvl["GMM"] = mat
# ---------------------------------------------------------
# MUNICIPALITY HEATMAPS
# ---------------------------------------------------------
# Build one list of all log matrices
all_mats_log = [np.log(aligned_lvl[cfg] + 1) for cfg in CONFIGS]

# global min and max
global_min = min(mat.min().min() for mat in all_mats_log)
global_max = max(mat.max().max() for mat in all_mats_log)

print("Global color range:", global_min, "→", global_max)

# A shared normalization object
norm = Normalize(vmin=global_min, vmax=global_max)

# plt settings
titlefontsize = 18
fig_size = (8, 6)

# Plot each CONFIG using SAME colormap scale
for cfg in CONFIGS:
    mat = np.log(aligned_lvl[cfg] + 1)

    plt.figure(figsize=fig_size)
    plt.imshow(mat.values, cmap="viridis", norm=norm, aspect="equal")

    # --- Add zone codes to axes ---
    zones = list(mat.index)          # integer zone IDs
    n = len(zones)

    # show every Nth tick if there are many zones
    step = max(1, n // 30)  # try ~25 ticks max
    tick_positions = range(0, n, step)
    tick_labels = [zones[i]//1000 for i in tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=90)
    plt.yticks(tick_positions, tick_labels)

    plt.title(f"OD Heatmap (Municipality Level) – {cfg.replace('_dwell_','-')}", fontsize = titlefontsize)
    plt.colorbar(label="log(#Trips + 1)")
    plt.xlabel("Destination Zone")
    plt.ylabel("Origin Zone")
    
    out_path = os.path.join(FIG_PATH, f"OD_heatmap_municipality_{cfg}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
# ---------------------------------------------------------
# SIMILARITY METRICS
# ---------------------------------------------------------

def cosine_similarity(matA, matB):
    A = matA.values.flatten()
    B = matB.values.flatten()
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def frobenius_similarity(matA, matB):
    A = matA.values
    B = matB.values
    dist = np.linalg.norm(A - B)
    max_val = np.linalg.norm(A) + np.linalg.norm(B) + 1e-12
    return 1 - dist / max_val

def L1_similarity(matA, matB):
    A = matA.values
    B = matB.values
    num = np.abs(A - B).sum()
    den = np.abs(A).sum() + np.abs(B).sum() + 1e-12
    return 1 - num / den

# ---------------------------------------------------------
# COSINE SIMILARITY HEATMAP
# ---------------------------------------------------------
cos_mat = pd.DataFrame(
    np.zeros((len(CONFIGS), len(CONFIGS))),
    index=CONFIGS, columns=CONFIGS
)

for a in CONFIGS:
    for b in CONFIGS:
        cos_mat.loc[a, b] = cosine_similarity(aligned_lvl[a], aligned_lvl[b])

plt.figure(figsize=(8, 6))
im = plt.imshow(cos_mat.values, cmap="Blues", vmin=0, vmax=1)
labels = [c.replace("_dwell_", "-") for c in CONFIGS]
plt.xticks(range(len(CONFIGS)), labels)
plt.yticks(range(len(CONFIGS)), labels)
plt.title("Cosine Similarity Between OD Matrices")
plt.colorbar(im)

for i in range(len(CONFIGS)):
    for j in range(len(CONFIGS)):
        val = cos_mat.values[i,j]
        color = "white" if val > 0.75 else "black"
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, "OD_matrix_heatmap_cos_similarity.png"), dpi=300)
plt.show()

# ---------------------------------------------------------
# L1 SIMILARITY
# ---------------------------------------------------------
L1_mat = pd.DataFrame(
    np.zeros((len(CONFIGS), len(CONFIGS))),
    index=CONFIGS, columns=CONFIGS
)

for a in CONFIGS:
    for b in CONFIGS:
        L1_mat.loc[a, b] = L1_similarity(aligned_lvl[a], aligned_lvl[b])

plt.figure(figsize=(8, 6))
im = plt.imshow(L1_mat.values, cmap="Blues", vmin=0, vmax=1)
labels = [c.replace("_dwell_", "-") for c in CONFIGS]
plt.xticks(range(len(CONFIGS)), labels)
plt.yticks(range(len(CONFIGS)), labels)
plt.title("L1 Similarity Between OD Matrices")
plt.colorbar(im)

for i in range(len(CONFIGS)):
    for j in range(len(CONFIGS)):
        val = L1_mat.values[i, j]
        color = "white" if val > 0.75 else "black"
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)


plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, "OD_matrix_heatmap_L1_similarity.png"), dpi=300)
plt.show()