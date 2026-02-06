import pandas as pd
import geopandas as gpd
import numpy as np
import glob
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm.auto import tqdm
from shapely.ops import linemerge


# --------------------------------------------------
# Compute cumulative trip distance (m)
# --------------------------------------------------
def trip_distance(geoms):
    """
    Compute cumulative distance along the trip geometry.
    CRS must be metric (EPSG:25832 etc.)
    """
    if len(geoms) < 2:
        return 0.0
    
    dist = 0.0
    for i in range(1, len(geoms)):
        dist += geoms[i].distance(geoms[i-1])
    return float(dist)


# --------------------------------------------------
# Extract endpoints + times + distance from ONE file
# --------------------------------------------------
def extract_from_file(path, cfg_name):
    df = gpd.read_parquet(path)
    if df.empty:
        return pd.DataFrame()

    rows = []
    trip_groups = df.groupby("TripID")

    for tid, trip in trip_groups:
        geoms = trip.geometry.values
        times = trip.index.values

        if len(geoms) == 0:
            continue

        start_geom = geoms[0]
        end_geom   = geoms[-1]

        dist_m = trip_distance(geoms)

        rows.append({
            "Config": cfg_name,
            "TripID": tid,
            "start_time": times[0],
            "end_time": times[-1],
            "start_x": start_geom.x,
            "start_y": start_geom.y,
            "end_x": end_geom.x,
            "end_y": end_geom.y,
            "distance_m": dist_m,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Process one configuration
# --------------------------------------------------
def process_config(cfg_name, trip_root):
    print(f"\n=== Processing: {cfg_name} ===")

    trip_files = glob.glob(os.path.join(trip_root, cfg_name, "truck_*_trips.parquet"))

    with Pool(cpu_count()) as pool:
        worker_func = partial(extract_from_file, cfg_name=cfg_name)

        results = list(tqdm(
            pool.imap(worker_func, trip_files),
            total=len(trip_files),
            desc=cfg_name
        ))

    df_out = pd.concat(results, ignore_index=True)

    out_path = f"../Results/Endpoints_{cfg_name}.parquet"
    df_out.to_parquet(out_path)

    print(f"✓ Saved → {out_path} ({len(df_out)} rows)")
    return df_out


# --------------------------------------------------
# Run all configurations
# --------------------------------------------------
if __name__ == "__main__":
    TRIP_ROOT = "../Data/truck_trips/"

    configs = ["SB_dwell_5", "SB_dwell_15", "HIS_dwell_5", "HIS_dwell_15"]
    all_outputs = []

    for cfg in configs:
        df_cfg = process_config(cfg, TRIP_ROOT)
        all_outputs.append(df_cfg)

    OUTPUT_ROOT = "../Results/Endpoints/"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for cfg, df_cfg in zip(configs, all_outputs):
        cfg_folder = os.path.join(OUTPUT_ROOT, cfg)
        os.makedirs(cfg_folder, exist_ok=True)

        out_file = os.path.join(cfg_folder, f"Endpoints_{cfg}.parquet")
        df_cfg.to_parquet(out_file)
        print(f"✓ Saved {cfg} → {out_file} ({len(df_cfg)} rows)")
