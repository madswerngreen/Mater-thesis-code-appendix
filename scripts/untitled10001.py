import pandas as pd
import numpy as np
import glob
from tqdm.auto import tqdm
import os
import pytz

from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib


# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
DK = pytz.timezone("Europe/Copenhagen")

CONFIGS = [
    {"name": "SB_dwell_5"},
    {"name": "SB_dwell_15"},
    {"name": "HIS_dwell_5"},
    {"name": "HIS_dwell_15"},
]

# ---------------------------------------------------------------
# WEEK WINDOW (DENMARK TIME)
# ---------------------------------------------------------------
week_start = DK.localize(pd.Timestamp("2025-06-16 00:00:00"))
week_end   = DK.localize(pd.Timestamp("2025-06-20 00:00:00"))


# ---------------------------------------------------------------
# TRIP-LEVEL STAT EXTRACTION
# ---------------------------------------------------------------
def extract_trip_level_stats(df):
    """
    Extract per-trip distance AFTER filtering:
      • Trip must overlap desired week
      • Trip must NOT begin or end on Wednesday
    """
    out = []

    for trip_id, grp in df.groupby("TripID"):

        if len(grp) < 2:
            continue

        # Convert index → Denmark time
        t_start = grp.index.min().tz_convert(DK)
        t_end   = grp.index.max().tz_convert(DK)

        # Remove Wednesday trips
        if t_start.weekday() == 2 or t_end.weekday() == 2:
            continue

        # Remove trips outside desired week
        if (t_end < week_start) or (t_start > week_end):
            continue

        # Compute total distance in km
        trip_km = grp["dist"].sum() / 1000.0
        out.append((trip_id, trip_km))

    return pd.DataFrame(out, columns=["TripID", "trip_distance_km"])


# ---------------------------------------------------------------
# WORKER FUNCTION (processing one parquet file)
# ---------------------------------------------------------------
def process_single_file(file_path):
    """
    Runs on worker process:
      • Reads file
      • Extracts valid trips according to rules
      • Returns DataFrame or None
    """
    try:
        df = pd.read_parquet(file_path)

        trip_stats = extract_trip_level_stats(df)
        if len(trip_stats) == 0:
            return None

        truck_id = int(file_path.split("_")[-2])
        trip_stats["truck"] = truck_id

        return trip_stats

    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return None


# ---------------------------------------------------------------
# COMPUTE TOTAL TRIP DISTANCE FOR ONE CONFIG (PARALLEL)
# ---------------------------------------------------------------
def compute_trip_stats(config_name, n_jobs=8):

    print("\n" + "=" * 80)
    print(f" Processing CONFIG: {config_name}")
    print("=" * 80)

    trip_files = glob.glob(
        f"C:/Users/b306630/Desktop/Mater thesis back uup/Data/truck_trips/{config_name}/truck_*_trips.parquet"
    )

    # Run all files in parallel
    with tqdm_joblib(tqdm(total=len(trip_files), desc=f"Trips ({config_name})")):
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_single_file)(f) for f in trip_files
        )

    # Remove empty results
    results = [r for r in results if r is not None]

    if not results:
        print("No valid trips!")
        return 0.0

    # Combine trip-level stats
    trip_level = pd.concat(results, ignore_index=True)

    # Trim extremely long trips (99.5 percentile)
    d_hi = np.percentile(trip_level["trip_distance_km"], 99.5)
    trip_level.loc[trip_level["trip_distance_km"] > d_hi, "trip_distance_km"] = d_hi

    # Sum total km for this config
    return trip_level["trip_distance_km"].sum() / 3


# ---------------------------------------------------------------
# RUN FOR ALL CONFIGS
# ---------------------------------------------------------------
if __name__ == "__main__":
    loads = {}

    for cfg in CONFIGS:
        name = cfg["name"]
        loads[name] = compute_trip_stats(name)

    print("\n=== Total valid trip distance per config (km) ===")
    gmm = "GMM"
    gmm_km = 12_210_815
    print(f"{gmm:15s}: {gmm_km:,.0f} km")
    for k, v in loads.items():
        print(f"{k:15s}: {v:,.0f} km")
