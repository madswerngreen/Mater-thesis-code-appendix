import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
from joblib import Parallel, delayed
import glob
import os
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from shapely.ops import unary_union

# ---------------------------------------------------------
# CONFIGURATIONS
# ---------------------------------------------------------
CONFIGS = {
    "SB_dwell_5":  {"method": "SB",  "dwell": 5 * 60},
    "SB_dwell_15": {"method": "SB",  "dwell": 15 * 60},
    "HIS_dwell_5": {"method": "HIS", "dwell": 5 * 60},
    "HIS_dwell_15": {"method": "HIS", "dwell": 15 * 60},
}

OUTPUT_ROOT = "../Data/truck_trips/"

ferry_connections = {0:1,1:0,       # Helingør - Helsingborg
                     7:8,8:7,       # Aarhus - Odden
                     17:65,65:17,   # Kalundborg - Ballen
                     46:47,47:46,   # Sælvig - Hou
                     74:75,75:74,   # Læsø - Frederikshavn
                     67:68,68:67,   # Køge - Rønne
                     28:85,85:28,   # Spodsbjerg - Tårs
                     37:38,38:37,   # Fynshav - Bøjden
                     60:61,61:60,   # Branden - Fur
                     }

# ---------------------------------------------------------
# Load polygons where trip ends should NOT create a boundary
# ---------------------------------------------------------
NO_END_PATH = "C:/Users/b306630/Desktop/Master Thesis/Data/Geospatial_info/Datalag.gpkg"
fuel  = gpd.read_file(NO_END_PATH, layer="OSMFuelProcessed").to_crs(25832)
fuel["geometry"] = fuel.geometry.buffer(25)
rest  = gpd.read_file(NO_END_PATH, layer="Rastepladser").to_crs(25832)
no_end_zones = pd.concat([fuel, rest], ignore_index=True)

ferry = gpd.read_file(NO_END_PATH, layer="Havne").to_crs(25832)
ferry["geometry"] = ferry.geometry.buffer(50)

highways_file = 'C:/Users/b306630/Desktop/Master Thesis/Data/Geospatial_info/highways_shape.gpkg'
highways = gpd.read_file(highways_file).to_crs(25832)
"""
# Precompute ferry water zones
ferry_water = {}

for a, b in ferry_connections.items():
    if a < b:  # avoid duplicates (0→1 == 1→0)
        A = ferry.loc[a, "geometry"]
        B = ferry.loc[b, "geometry"]
        hull = unary_union([A, B]).convex_hull.buffer(300)  # 300m buffer example
        ferry_water[(a, b)] = hull
        ferry_water[(b, a)] = hull  # symmetric
"""

CHAIN_MAX_GAP_SEC = 1 * 3600
CHAIN_MAX_DIST_M  = 250


QUANTILE_PATH = "rolling_quantiles.pkl"

# ---------------------------------------------------------
# Load quantiles or create them if missing
# ---------------------------------------------------------
if os.path.exists(QUANTILE_PATH):
    with open(QUANTILE_PATH, "rb") as f:
        qdata = pickle.load(f)
    speed_cdf = qdata["speed_cdf"]
    heading_cdf = qdata["heading_cdf"]
    print("✓ Loaded quantiles.pkl")
else:
    raise ValueError('you must run Quantiles_speed_and_heading.py')


# ---------------------------------------------------------
# Read raw truck data
# ---------------------------------------------------------
def read_truck(truck_id, speed_from_file=True):
    df = pd.read_parquet(f"C:/Users/b306630\Desktop/Master Thesis/Data/parquet_trucks/truck_{truck_id}.parquet")

    df["latitude"]  /= 1e6
    df["longitude"] /= 1e6

    df = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    ).to_crs(25832)

    df["dist"] = df.geometry.distance(df.geometry.shift())
    df["time_delta"] = (df["time"] - df["time"].shift()).dt.total_seconds()

    if speed_from_file:
        df["speed_estimate"] = df["speed"] * 0.036
        df = df[df["speed_estimate"] < 130].copy()

    else:
        df["speed_estimate"] = df["dist"] / df["time_delta"] * 3.6

    df["dx"] = df.geometry.x.diff()
    df["dy"] = df.geometry.y.diff()
    df["heading_estimate"] = np.degrees(np.arctan2(df["dx"], df["dy"]))
    df["heading_estimate"] = (df["heading_estimate"] + 360) % 360
    df["delta_heading"] = ((df["heading_estimate"].diff() + 180) % 360) - 180

    # ====== FILTER BY DATE ======
    # Keep only June 2025
    df = df.set_index("time")
    df = df.loc["2025-06-01":"2025-06-30"]

    return df.dropna()
# ---------------------------------------------------------
# Trip segmentation (SB or HIS)
# ---------------------------------------------------------

def segment_trips(df, method, dwell_sec):
    # Rolling features
    
    df["delta_heading_var"] = (
        df["delta_heading"].rolling("1min", center=True, min_periods=2).var(ddof=0)
    )
    df["speed_estimate"] = (
        df["speed_estimate"].rolling("1min", center=True, min_periods=1).mean()
    )

    
    df = df.dropna().copy()

    # Thresholds
    SPEED_THRESHOLD = speed_cdf[30] 
    HEADING_THRESHOLD = heading_cdf[90]
    HARD_SPEED = speed_cdf[25]

    # ---------------------------------------------------------
    # 1. STOP CLASSIFICATION (method-dependent)
    # ---------------------------------------------------------

    if method == "SB":
        # SB: only speed threshold
        df["stop_candidate"] = df["speed_estimate"] < HARD_SPEED

    elif method == "HIS":
        # HIS: speed + heading variance
        df["stop_candidate"] = (
            (df["speed_estimate"] < SPEED_THRESHOLD) &
            (df["delta_heading_var"] > HEADING_THRESHOLD)
        )
        # very low speed is always stop
        df.loc[df["speed_estimate"] < HARD_SPEED, "stop_candidate"] = True

    else:
        raise ValueError("Unknown method")

    df["stop_candidate"] = df["stop_candidate"].astype(bool)

    # Make a working copy
    df_moving = df[~df["stop_candidate"]].copy()
    df_moving["time_gap"] = df_moving.index.to_series().diff().dt.total_seconds()
    df_moving["new_trip"] = (df_moving["time_gap"] > dwell_sec).fillna(True)
    df_moving["TripID"] = df_moving["new_trip"].cumsum() +1 


    return df_moving


# ---------------------------------------------------------
# Trip filtering + chaining
# ---------------------------------------------------------
def postprocess_and_chain(df_moving, truck_id):
    # ------------------------------
    # 0. Filter short / tiny trips
    # ------------------------------
    tmp = df_moving.reset_index()
    trip_stats = tmp.groupby("TripID").agg(
        start_time=("time", "min"),
        end_time=("time", "max"),
        distance=("dist", "sum")
    )
    trip_stats["duration"] = (trip_stats["end_time"] - trip_stats["start_time"]).dt.total_seconds()

    invalid = trip_stats[(trip_stats["duration"] < 60) | (trip_stats["distance"] < 500)].index
    df_moving = df_moving[~df_moving["TripID"].isin(invalid)].copy()

    # ------------------------------
    # 1. Make TripIDs unique
    # ------------------------------
    df_moving["TripID"] += truck_id * 10_000

    # =====================================================
    # === CHAINING STEP 1 — SB/HIS NORMAL MERGING =========
    # =====================================================
    tmp = df_moving.reset_index()

    trip_stats_local = (
        tmp.groupby("TripID")
           .agg(start_time=("time", "min"), end_time=("time", "max"))
           .sort_values("start_time")
    )
    ordered = trip_stats_local.index.to_numpy()

    # Build GeoDataFrames for join
    trip_starts = tmp.groupby("TripID").head(1)[["TripID", "time", "geometry"]]
    trip_ends   = tmp.groupby("TripID").tail(1)[["TripID", "time", "geometry"]]
    
    trip_starts = gpd.GeoDataFrame(trip_starts, geometry="geometry", crs=df_moving.crs)
    trip_ends   = gpd.GeoDataFrame(trip_ends,   geometry="geometry", crs=df_moving.crs)

    # Spatial joins
    ends_in   = trip_ends.sjoin(no_end_zones,   how="left", predicate="within")
    end_in_map   = (~ends_in["index_right"].isna()).groupby(ends_in["TripID"]).first().to_dict()

    # Geometry maps
    start_geom_map = trip_starts.set_index("TripID")["geometry"].to_dict()
    end_geom_map   = trip_ends.set_index("TripID")["geometry"].to_dict()

    merged_id = {tid: tid for tid in ordered}

    # Normal chaining
    for prev_tid, curr_tid in zip(ordered[:-1], ordered[1:]):
        prev_in = end_in_map.get(prev_tid, False)
        if prev_in:
            gap = (trip_stats_local.loc[curr_tid, "start_time"] -
                   trip_stats_local.loc[prev_tid, "end_time"]).total_seconds()

            try:
                dist = end_geom_map[prev_tid].distance(start_geom_map[curr_tid])
            except:
                dist = np.inf

            if gap <= CHAIN_MAX_GAP_SEC and dist <= CHAIN_MAX_DIST_M:
                merged_id[curr_tid] = merged_id[prev_tid]

    df_moving["TripID"] = df_moving["TripID"].map(merged_id)

    # =====================================================
    # === CHAINING STEP 2 — SIMPLE FERRY MERGING ==========
    # =====================================================
    
    tmp = df_moving.reset_index()
    
    trip_stats_local = (
        tmp.groupby("TripID")
           .agg(start_time=("time", "min"),
                end_time=("time", "max"))
           .sort_values("start_time")
    )
    ordered = list(trip_stats_local.index)
    
    # Build GeoDataFrames
    trip_starts = gpd.GeoDataFrame(
        tmp.groupby("TripID").head(1)[["TripID", "time", "geometry"]],
        geometry="geometry", crs=df_moving.crs
    )
    trip_ends = gpd.GeoDataFrame(
        tmp.groupby("TripID").tail(1)[["TripID", "time", "geometry"]],
        geometry="geometry", crs=df_moving.crs
    )
    
    # Spatial join with ferry ports
    starts_ferry = trip_starts.sjoin(ferry, how="left", predicate="within")
    ends_ferry   = trip_ends.sjoin(ferry,   how="left", predicate="within")
    
    start_ferry_map = starts_ferry["index_right"].groupby(starts_ferry["TripID"]).first().to_dict()
    end_ferry_map   = ends_ferry["index_right"].groupby(ends_ferry["TripID"]).first().to_dict()
    
    merged_id = {tid: tid for tid in ordered}
    
    for i in range(1, len(ordered)):
        curr_tid = ordered[i]
        prev_tid = ordered[i-1]
    
        start_port = start_ferry_map.get(curr_tid, None)
        if start_port is None:
            continue
    
        # Find the expected arrival port
        expected_arrival_port = ferry_connections.get(start_port, None)
        if expected_arrival_port is None:
            continue
    
        # ----------------------------------------------
        # Rule 1: Previous trip ends at arrival port?
        # ----------------------------------------------
        if end_ferry_map.get(prev_tid, None) == expected_arrival_port:
            merged_id[curr_tid] = merged_id[prev_tid]
            continue
    
        # ----------------------------------------------
        # Rule 2: Two-trip ferry:
        # T_{i-2} ends at arrival port AND T_{i-1} is water
        # ----------------------------------------------
        if i >= 2:
            prevprev_tid = ordered[i-2]
    
            if end_ferry_map.get(prevprev_tid, None) == expected_arrival_port:
                merged_id[curr_tid] = merged_id[prevprev_tid]
                merged_id[prev_tid] = merged_id[prevprev_tid]
    
    df_moving["TripID"] = df_moving["TripID"].map(merged_id)
    
    # =====================================================
    # === CHAINING STEP 3 — HIGHWAY MERGING ==============
    # =====================================================
    
    tmp = df_moving.reset_index()
    
    trip_stats_local = (
        tmp.groupby("TripID")
           .agg(start_time=("time", "min"),
                end_time=("time", "max"))
           .sort_values("start_time")
    )
    ordered = trip_stats_local.index.to_numpy()
    
    # Build GeoDataFrames for trip start/end points
    trip_starts = gpd.GeoDataFrame(
        tmp.groupby("TripID").head(1)[["TripID", "time", "geometry"]],
        geometry="geometry", crs=df_moving.crs
    )
    trip_ends = gpd.GeoDataFrame(
        tmp.groupby("TripID").tail(1)[["TripID", "time", "geometry"]],
        geometry="geometry", crs=df_moving.crs
    )
    
    # Identify if start/end points lie on the highway
    starts_hw = trip_starts.sjoin(highways, how="left", predicate="within")
    ends_hw   = trip_ends.sjoin(highways,   how="left", predicate="within")
    
    start_on_hw = (~starts_hw["index_right"].isna()).groupby(starts_hw["TripID"]).first().to_dict()
    end_on_hw   = (~ends_hw["index_right"].isna()).groupby(ends_hw["TripID"]).first().to_dict()
    
    # Merge dictionary for this stage
    merged_id_hw = {tid: tid for tid in ordered}
    
    for prev_tid, curr_tid in zip(ordered[:-1], ordered[1:]):
    
        # Must begin/end on highways
        prev_on = end_on_hw.get(prev_tid, False)
        curr_on = start_on_hw.get(curr_tid, False)
    
        if not (prev_on and curr_on):
            continue
    
        # Time gap constraint: must be reasonably continuous (≤ 2 hours)
        gap = (trip_stats_local.loc[curr_tid, "start_time"] -
               trip_stats_local.loc[prev_tid, "end_time"]).total_seconds()
    
        if gap > 2 * 3600:   # more than 2 hours → DO NOT merge
            continue
    
        # Distance constraint: must be close (≤ 1 km)
        end_geom = trip_ends.loc[trip_ends.TripID == prev_tid, "geometry"].values[0]
        start_geom = trip_starts.loc[trip_starts.TripID == curr_tid, "geometry"].values[0]
    
        dist = end_geom.distance(start_geom)
    
        if dist <= 1000:     # ≤ 1 km
            merged_id_hw[curr_tid] = merged_id_hw[prev_tid]
    
    # Apply highway merge
    df_moving["TripID"] = df_moving["TripID"].map(merged_id_hw)


    # ---------------------------------------------
    # Final selected columns
    # ---------------------------------------------
    return df_moving[[
        'TruckID', 'TripID', 'speed_estimate', 'heading_estimate',
        'delta_heading', 'delta_heading_var', 'weightLimits',
        'CO2EmissionClass', 'dist', 'time_delta', 'geometry'
    ]]

# ---------------------------------------------------------
# Full pipeline for one truck + one config
# ---------------------------------------------------------
def get_trips_for_config(truck_id, cfg_name, cfg_params):
    try:
        df = read_truck(truck_id)
        df_seg = segment_trips(df, cfg_params["method"], cfg_params["dwell"])
        df_final = postprocess_and_chain(df_seg, truck_id)

        if len(df_final) == 0:
            print(f"{cfg_name}: No valid trips for truck {truck_id}")
            return None

        return df_final

    except Exception as e:
        print(f"❌ Error in {cfg_name} for truck {truck_id}: {e}")
        return None


# ---------------------------------------------------------
# Main: TEST MODE — only truck 3, all configs in one GPKG
# ---------------------------------------------------------
if __name__ == "__main__":

    TRUCK_ID = 3
    OUTPUT_GPKG = "truck3_all_configs.gpkg"

    # Remove old file (GeoPackage can't append to existing layer names)
    if os.path.exists(OUTPUT_GPKG):
        os.remove(OUTPUT_GPKG)

    for cfg_name, cfg_params in CONFIGS.items():
        print(f"\n=== Running CONFIG: {cfg_name} ===\n")

        # run segmentation + chaining
        df_result = get_trips_for_config(TRUCK_ID, cfg_name, cfg_params)

        if df_result is None or len(df_result) == 0:
            print(f"{cfg_name}: no output written")
            continue

        # Write GeoPackage layer
        df_result.to_file(
            OUTPUT_GPKG,
            layer=cfg_name,
            driver="GPKG"
        )

        print(f"✓ Saved {cfg_name} to {OUTPUT_GPKG}")
