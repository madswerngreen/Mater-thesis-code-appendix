import pandas as pd
import geopandas as gpd
import numpy as np
import glob
import os
from shapely.strtree import STRtree
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm.auto import tqdm

# ------------------------------
# Global variables for workers
# ------------------------------
zones = None
zone_geoms = None
zone_ids = None
tree = None

def init_worker(zone_file):
    """
    Each worker runs this exactly ONCE.
    Heavy objects are built only once per worker.
    """
    global zones, zone_geoms, zone_ids, tree

    zones = gpd.read_file(zone_file).to_crs(25832)
    zones["zoneid"] = zones["zoneid"].astype(int)

    zone_geoms = zones.geometry.values
    zone_ids = zones.zoneid.values

    # Build STRtree once per worker = huge speedup
    tree = STRtree(zone_geoms)


# -----------------------------------------------------
# Zone lookup (uses worker-level globals)
# -----------------------------------------------------
def find_zone(point):
    global zone_geoms, zone_ids, tree

    hit_idxs = tree.query(point)
    if len(hit_idxs) == 0:
        return None

    for idx in hit_idxs:
        geom = zone_geoms[idx]
        if geom.contains(point):
            return int(zone_ids[idx])

    return None


# -----------------------------------------------------
# OD extraction for one file
# -----------------------------------------------------
def extract_OD_from_file(path, cfg_name):
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

        # Origin
        origin = next((find_zone(g) for g in geoms if find_zone(g) is not None), None)

        # Destination
        dest = next((find_zone(g) for g in geoms[::-1] if find_zone(g) is not None), None)

        rows.append({
            "Config": cfg_name,
            "TripID": tid,
            "start_time": times[0],
            "end_time": times[-1],
            "Origin": origin,
            "Destination": dest,
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------
# MAIN LOOP USING MULTIPROCESSING POOL
# -----------------------------------------------------
def process_config(cfg_name, trip_root, zones_path):
    print(f"\n=== Processing: {cfg_name} ===")

    trip_files = glob.glob(os.path.join(trip_root, cfg_name, "truck_*_trips.parquet"))

    with Pool(cpu_count(), initializer=init_worker, initargs=(zones_path,)) as pool:
        worker_func = partial(extract_OD_from_file, cfg_name=cfg_name)

        results = list(tqdm(pool.imap(worker_func, trip_files),
                            total=len(trip_files), desc=cfg_name))

    df_out = pd.concat(results, ignore_index=True)
    out_path = f"../Results/OD_{cfg_name}.parquet"
    df_out.to_parquet(out_path)

    print(f"✓ Saved → {out_path} ({len(df_out)} rows)")
    return df_out


# -----------------------------------------------------
# RUN ALL CONFIGS
# -----------------------------------------------------
if __name__ == "__main__":
    ZONES_PATH = "C:/Users/b306630/Desktop/Mater thesis back uup/Data/Geospatial_info/zones.gpkg"
    TRIP_ROOT = "../Data/truck_trips/"

    configs = ["SB_dwell_5", "SB_dwell_15", "HIS_dwell_5", "HIS_dwell_15"]

    all_results = []
    for cfg in configs:
        df_cfg = process_config(cfg, TRIP_ROOT, ZONES_PATH)
        all_results.append(df_cfg)

    OUTPUT_ROOT = "../Results/OD_results/"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    configs = ["SB_dwell_5", "SB_dwell_15", "HIS_dwell_5", "HIS_dwell_15"]
    
    # all_results = [df_SB5, df_SB15, df_HIS5, df_HIS15]
    # Already produced by your code.
    
    for cfg, df_cfg in zip(configs, all_results):
    
        cfg_folder = os.path.join(OUTPUT_ROOT, cfg)
        os.makedirs(cfg_folder, exist_ok=True)
    
        out_file = os.path.join(cfg_folder, f"OD_{cfg}.parquet")
    
        df_cfg.to_parquet(out_file)
        print(f"✓ Saved {cfg} → {out_file} ({len(df_cfg)} rows)")


