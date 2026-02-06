import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz
import os 

DK = pytz.timezone("Europe/Copenhagen")

FIG_PATH = "../Results/Figures/"
os.makedirs(FIG_PATH, exist_ok=True)


CONFIGS = [
    "SB_dwell_5",
    "SB_dwell_15",
    "HIS_dwell_5",
    "HIS_dwell_15",
]

OD_all = {}

# Define time window IN DENMARK TIME
start = DK.localize(pd.Timestamp('2025-06-16'))
end   = DK.localize(pd.Timestamp('2025-06-20'))

for cfg in CONFIGS:
    path = f"C:/Users/b306630\Desktop/Master Thesis/Data/{cfg}_OD.csv"
    df = pd.read_csv(path)
    DK = pytz.timezone("Europe/Copenhagen")

    # Read datetimes
    df["DepartureTime"] = pd.to_datetime(df["DepartureTime"], utc=True).dt.tz_convert(DK)
    df["ArrivalTime"]   = pd.to_datetime(df["ArrivalTime"], utc=True).dt.tz_convert(DK)
    

    # -----------------------------------------------------
    # 1. Remove impossible trips: Arrival before Departure
    # -----------------------------------------------------
    df = df[df["ArrivalTime"] >= df["DepartureTime"]]

    # -----------------------------------------------------
    # 2. Keep only trips that actually occur within [start, end]
    # -----------------------------------------------------
    df = df[(df["DepartureTime"] <= end) & (df["ArrivalTime"] >= start)]

    # -----------------------------------------------------
    # 3. Sort trips by departure time (clean ordering)
    # -----------------------------------------------------
    df = df[df["DepartureTime"].dt.weekday != 2]   # remove Wednesday
    df = df[df["ArrivalTime"].dt.weekday != 2]     # also exclude trips ending on Wed

    df = df.sort_values("DepartureTime").reset_index(drop=True)

    OD_all[cfg] = df

    
    øst_vest = len(df[(df['Origin']<410000) & 
                  (df['Destination']>410000)])
    vest_øst = len(df[(df['Origin']>410000) & 
                  (df['Destination']<410000)])
    print(cfg)
    print('øst_vest',øst_vest)
    print('vest_øst',vest_øst)
print('mastra:')
mod_odense = np.sum([2885, 3202, 3136*0, 3132, 2620])
mod_københavn = np.sum([2989, 3028,3001*0, 2737, 2241])
print('øst_vest',mod_odense)
print('vest_øst',mod_københavn)
OD_matrices = {}

for cfg, df in OD_all.items():
    mat = df.groupby(["Origin", "Destination"]).size().unstack(fill_value=0)
    OD_matrices[cfg] = mat

all_zones = sorted(
    set().union(*[mat.index for mat in OD_matrices.values()])
)

aligned = {}

for cfg, mat in OD_matrices.items():
    mat_aligned = mat.reindex(index=all_zones, columns=all_zones, fill_value=0)
    aligned[cfg] = mat_aligned

diff = aligned["SB_dwell_5"] - aligned["HIS_dwell_5"]



for cfg in OD_matrices.keys():
    mat = np.log(aligned[cfg])

def cosine_similarity(matA, matB):
    A = matA.values.flatten()
    B = matB.values.flatten()
    num = np.dot(A, B)
    den = np.linalg.norm(A) * np.linalg.norm(B)
    return num / den

def frobenius_similarity(matA, matB):
    A = matA.values
    B = matB.values
    dist = np.linalg.norm(A - B)
    # scale by maximum possible distance between any two matrices
    max_val = np.linalg.norm(A) + np.linalg.norm(B) + 1e-12
    return 1 - (dist / max_val)

def L1_similarity(matA, matB):
    A = matA.values
    B = matB.values
    num = np.abs(A - B).sum()
    den = np.abs(A).sum() + np.abs(B).sum() + 1e-12
    return 1 - num/den


CONFIGS = [
    "SB_dwell_5",
    "SB_dwell_15",
    "HIS_dwell_5",
    "HIS_dwell_15",
]

# Build empty 4x4 matrix
cos_mat = pd.DataFrame(
    np.zeros((len(CONFIGS), len(CONFIGS))),
    index=CONFIGS,
    columns=CONFIGS
)

# Fill diagonal = 1, others = computed similarities
for a in CONFIGS:
    for b in CONFIGS:
        cos_mat.loc[a, b] = cosine_similarity(aligned[a], aligned[b])


plt.figure(figsize=(8, 6))
im = plt.imshow(cos_mat.values, cmap="viridis", vmin=0, vmax=1)

# Tick labels
plt.xticks(ticks=range(len(CONFIGS)), labels=CONFIGS, rotation=45, ha="right")
plt.yticks(ticks=range(len(CONFIGS)), labels=CONFIGS)

plt.title("Cosine Similarity Between OD Matrices", fontsize=14)
plt.colorbar(im, label="Cosine Similarity")

# --- Annotate each cell ---
for i in range(len(CONFIGS)):
    for j in range(len(CONFIGS)):
        value = cos_mat.values[i, j]
        plt.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=10)
plt.tight_layout()
out_path = os.path.join(FIG_PATH, "OD_matrix_heatmap_cos_Similarity .png")
plt.savefig(out_path, dpi=300)
plt.show()

# Fill diagonal = 1, others = computed similarities
for a in CONFIGS:
    for b in CONFIGS:
        cos_mat.loc[a, b] = L1_similarity(aligned[a], aligned[b])


plt.figure(figsize=(8, 6))
im = plt.imshow(cos_mat.values, cmap="viridis", vmin=0, vmax=1)

# Tick labels
plt.xticks(ticks=range(len(CONFIGS)), labels=CONFIGS, rotation=45, ha="right")
plt.yticks(ticks=range(len(CONFIGS)), labels=CONFIGS)

plt.title(r"$\mathcal{L}_1$ Similarity Between OD Matrices", fontsize=14)
plt.colorbar(im, label="Cosine Similarity")

# --- Annotate each cell ---
for i in range(len(CONFIGS)):
    for j in range(len(CONFIGS)):
        value = cos_mat.values[i, j]
        plt.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=10)

plt.tight_layout()
out_path = os.path.join(FIG_PATH, "OD_matrix_heatmap_L1_Similarity .png")
plt.savefig(out_path, dpi=300)
plt.show()


def zone_map(x, level=0):
    # Validate level
    assert isinstance(level, int), "level must be an integer"
    assert 0 <= level < 4, "level must be between 0 and 3"
    tmp = 3 - level
    result = (x // 10**tmp) * 10**tmp
    return result

OD_matrices = {}

for cfg, df in OD_all.items():
    
    df["Origin"] = df["Origin"].apply(zone_map)
    df["Destination"] = df["Destination"].apply(zone_map)

    
    mat = df.groupby(["Origin", "Destination"]).size().unstack(fill_value=0)
    OD_matrices[cfg] = mat

all_zones = sorted(
    set().union(*[mat.index for mat in OD_matrices.values()])
)
aligned = {}

for cfg, mat in OD_matrices.items():
    mat_aligned = mat.reindex(index=all_zones, columns=all_zones, fill_value=0)
    aligned[cfg] = mat_aligned

for cfg in OD_matrices.keys():
    mat = np.log(aligned[cfg])
    
    plt.figure(figsize=(12, 10))
    plt.imshow(mat.values, cmap="viridis", aspect="auto")
    plt.title(f"OD Matrix Heatmap – {cfg}")
    plt.colorbar(label="Number of Trips")
    plt.xlabel("Destination Zone")
    plt.ylabel("Origin Zone")
    plt.tight_layout()
    out_path = os.path.join(FIG_PATH, f"OD_matrix_heatmap_municipality_{cfg}.png")
    plt.savefig(out_path, dpi=300)
    plt.show()