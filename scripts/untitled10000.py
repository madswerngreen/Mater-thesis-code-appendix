import pandas as pd
import numpy as np
import glob
import os
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# ---------------------------------------------------------
# CONFIGS & PATHS
# ---------------------------------------------------------
CONFIGS = [
    "SB_dwell_5",
    "SB_dwell_15",
    "HIS_dwell_5",
    "HIS_dwell_15",
]

RAW_PATH = "C:/Users/b306630/Desktop/Master Thesis/Data/parquet_trucks/truck_*.parquet"
TRIPS_ROOT = "../Data/truck_trips/"

# ---------------------------------------------------------
# 1. Load all Truck IDs
# ---------------------------------------------------------
raw_files = glob.glob(RAW_PATH)
truck_ids = sorted(int(os.path.basename(f).split("_")[1].split(".")[0]) for f in raw_files)

print(f"Found {len(truck_ids)} trucks in raw data.")


# ---------------------------------------------------------
# Helper: Count trips for a single truck in one config
# ---------------------------------------------------------
def count_trips(cfg, truck_id):
    path = f"{TRIPS_ROOT}/{cfg}/truck_{truck_id}_trips.parquet"
    if not os.path.exists(path):
        return 0
    df = pd.read_parquet(path)
    return df["TripID"].nunique()

def count_GPS(cfg, truck_id):
    path = f"{TRIPS_ROOT}/{cfg}/truck_{truck_id}_trips.parquet"
    if not os.path.exists(path):
        return 0
    df = pd.read_parquet(path)
    return len(df["TripID"])


# ---------------------------------------------------------
# VALIDATION LOOP
# ---------------------------------------------------------
summary = []

for cfg in CONFIGS:
    print(f"\n=== VALIDATING {cfg} ===")

    # Parallel trip counting
    counts = Parallel(n_jobs=8)(
        delayed(count_trips)(cfg, tid) for tid in tqdm(truck_ids, desc=f"{cfg}")
    )

    counts = np.array(counts)
    zero_trips = (counts == 0).sum()
    total_trips = counts.sum()

    # Quantiles
    q_min  = counts.min()
    q_25   = np.percentile(counts, 25)
    q_50   = np.percentile(counts, 50)
    q_75   = np.percentile(counts, 75)
    q_max  = counts.max()
    
    # Parallel trip counting
    counts = Parallel(n_jobs=8)(
        delayed(count_GPS)(cfg, tid) for tid in tqdm(truck_ids, desc=f"{cfg}")
    )
    
    num_gps = np.array(counts).sum()
    
    print(f" Trucks with zero trips:  {zero_trips}")
    print(f" Total trips:             {total_trips:,}")
    print(f' Total os GPS points:     {num_gps}')
    print(f" Trip count quantiles:")
    print(f"   min: {q_min}")
    print(f"   25%: {q_25:.1f}")
    print(f"   50%: {q_50:.1f}")
    print(f"   75%: {q_75:.1f}")
    print(f"   max: {q_max}")

    summary.append({
        "Config": cfg,
        "Total Trips": total_trips,
        "Zero-Trip Trucks": zero_trips,
        "Min": q_min,
        "Q25": q_25,
        "Median": q_50,
        "Q75": q_75,
        "Max": q_max,
    })

# ---------------------------------------------------------
# SUMMARY TABLE
# ---------------------------------------------------------
summary_df = pd.DataFrame(summary)
summary_df["Zero-Trip Trucks %"]  =summary_df["Zero-Trip Trucks"] / len(truck_ids) 
print("\n==================== SUMMARY ====================\n")
print(summary_df.to_string(index=False))
