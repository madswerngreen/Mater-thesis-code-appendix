import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import os


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
# ---------------------------------------------------------
# Load polygons where trip ends should NOT create a boundary
# ---------------------------------------------------------

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
    df = pd.read_parquet(f"C:/Users/b306630/OneDrive - Vejdirektoratet/Skrivebord/Master Thesis/Data/parquet_trucks/truck_{truck_id}.parquet")

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
        df_final = segment_trips(df, cfg_params["method"], cfg_params["dwell"])

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
