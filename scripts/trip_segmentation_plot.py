import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import matplotlib.pyplot as plt
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
NO_END_PATH = "K:/TPD/PA/TAN/Studentermedhjælper/Mads/OSM/Datalag.gpkg"

fuel = gpd.read_file(NO_END_PATH, layer="OSMFuelProcessed").to_crs(25832)
rest = gpd.read_file(NO_END_PATH, layer="Rastepladser").to_crs(25832)

fuel["geometry"] = fuel.geometry.buffer(25)
no_end_zones = pd.concat([fuel, rest], ignore_index=True)

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

    df = df.set_index("time").dropna()
    return df

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
    tmp = df_moving.reset_index()
    trip_stats = tmp.groupby("TripID").agg(
        start_time=("time", "min"),
        end_time=("time", "max"),
        distance=("dist", "sum")
    )
    trip_stats["duration"] = (trip_stats["end_time"] - trip_stats["start_time"]).dt.total_seconds()

    invalid = trip_stats[
        (trip_stats["duration"] < 60) | (trip_stats["distance"] < 500)
    ].index

    df_moving = df_moving[~df_moving["TripID"].isin(invalid)].copy()
    #if df_moving.empty:
        #return df_moving
    
    # Make TripIDs unique per truck
    df_moving["TripID"] += truck_id * 10_000
    """
    # Trip chaining
    tmp = df_moving.reset_index()
    trip_stats_local = (
        tmp.groupby("TripID")
           .agg(start_time=("time", "min"), end_time=("time", "max"))
           .sort_values("start_time")
    )
    ordered = trip_stats_local.index.to_numpy()

    trip_starts = tmp.groupby("TripID").head(1)[["TripID", "time", "geometry"]]
    trip_ends   = tmp.groupby("TripID").tail(1)[["TripID", "time", "geometry"]]

    trip_starts = gpd.GeoDataFrame(trip_starts, geometry="geometry", crs=df_moving.crs)
    trip_ends   = gpd.GeoDataFrame(trip_ends,   geometry="geometry", crs=df_moving.crs)

    starts_in = trip_starts.sjoin(no_end_zones, how="left", predicate="within")
    ends_in   = trip_ends.sjoin(no_end_zones,   how="left", predicate="within")

    start_in_map = (~starts_in["index_right"].isna()).groupby(starts_in["TripID"]).first().to_dict()
    end_in_map   = (~ends_in["index_right"].isna()).groupby(ends_in["TripID"]).first().to_dict()

    start_geom_map = trip_starts.set_index("TripID")["geometry"].to_dict()
    end_geom_map   = trip_ends.set_index("TripID")["geometry"].to_dict()

    merged_id = {tid: tid for tid in ordered}

    for prev_tid, curr_tid in zip(ordered[:-1], ordered[1:]):
        prev_in = end_in_map.get(prev_tid, False)
        curr_in = start_in_map.get(curr_tid, False)

        gap = (trip_stats_local.loc[curr_tid, "start_time"] -
               trip_stats_local.loc[prev_tid, "end_time"]).total_seconds()

        try:
            dist = end_geom_map[prev_tid].distance(start_geom_map[curr_tid])
        except:
            dist = np.inf

        if prev_in and curr_in and gap <= CHAIN_MAX_GAP_SEC and dist <= CHAIN_MAX_DIST_M:
            merged_id[curr_tid] = merged_id[prev_tid]
    df_moving["TripID"] = df_moving["TripID"].map(merged_id).astype(int)
    """
    df_moving.to_file("test.gpkg", layer="HIS_dwell_5", driver="GPKG")
    return df_moving



# ---------------------------------------------------------
# Full pipeline for one truck + one config
# ---------------------------------------------------------
def get_trips_for_config(truck_id, cfg_params):
    try:
        df = read_truck(truck_id)
        df_seg = segment_trips(df, cfg_params["method"], cfg_params["dwell"])
        df_final = postprocess_and_chain(df_seg, truck_id)


    except Exception as e:
        print(f"❌ Error  for truck {truck_id}: {e}")


    return df_final, df 


def plot_truck_timeseries(df, all_points, truck_id, 
                          speed_threshold, hard_speed_threshold, heading_threshold, method, title, cfg_name,
                          textsize = 12):
    """
    df          = segmented trips (moving points only)
    all_points  = full time series including stops
    """

    fig, ax1 = plt.subplots(figsize=(10,4))
    start_time = pd.Timestamp("2025-06-21 16:20:00")
    end_time   = pd.Timestamp("2025-06-21 19:20:00")
    # ---------------------------------------------------------
    # 4. Plot trip intervals as green bands
    # ---------------------------------------------------------
    for trip_id, grp in df.groupby("TripID"):
        start = grp.index.min()
        end   = grp.index.max()

        ax1.axvspan(start, end, color="lightgreen", alpha=0.25)
    # ---------------------------------------------------------
    # 1. Speed on left axis
    # ---------------------------------------------------------
    #ax1.set_title(title, fontsize=14)

    ax1.scatter(all_points.index, all_points["speed_estimate"],
                s=12, color="steelblue", label="Speed [km/h]")

    ax1.set_ylabel("Speed [km/h]", color="steelblue", fontsize = textsize)
    ax1.set_xlabel("Time")

    # ---------------------------------------------------------
    # 2. Speed threshold lines
    # ---------------------------------------------------------
    if method == 'HIS':
        ax1.axhline(speed_threshold, color="navy", linestyle="--", linewidth=1.5,
                    label=f"Speed threshold = {speed_threshold:.1f} km/h")
    ax1.axhline(hard_speed_threshold, color="navy", linestyle=":", linewidth=1.5,
                label=f"Hard speed threshold = {hard_speed_threshold:.1f} km/h")
    
    if method == 'HIS':
        # ---------------------------------------------------------
        # 3. Heading variance on right axis
        # ---------------------------------------------------------
        ax2 = ax1.twinx()
    
        ax2.scatter(all_points.index, all_points["delta_heading_var"],
                    s=10, color="orange", alpha=0.8, label="Heading variance")
    
        ax2.set_ylabel("Heading variance", color="orange", fontsize = textsize)
    
        # Threshold line
        ax2.axhline(heading_threshold, color="orange", linestyle="--", linewidth=1.5,
                    label=f"Heading var. threshold = {heading_threshold:.1f}")

    # ---------------------------------------------------------
    # Legend handling
    # ---------------------------------------------------------
    handles1, labels1 = ax1.get_legend_handles_labels()
    if method == 'HIS':
        ax2.grid(False)
        handles2, labels2 = ax2.get_legend_handles_labels()
        leg = ax1.legend(handles1 + handles2, labels1 + labels2,
                         loc="upper left",
                         frameon=True,
                         facecolor="white",
                         framealpha=1.0,
                         fontsize = textsize)   # fully opaque background
        ax2.set_xlim(start_time, end_time)  # right axis must match
    else:
        leg = ax1.legend(handles1, labels1,
                         loc="upper left",
                         frameon=True,
                         facecolor="white",
                         framealpha=1.0,
                         fontsize = textsize)   # fully opaque background
    ax1.grid(False)
    ax1.set_xlim(start_time, end_time)
    plt.tight_layout()
    plt.savefig(f"../Results/Figures/Trip_Intervals_plot_{cfg_name}")
    plt.show()
    
for cfg_name, cfg_params in CONFIGS.items():
    print(f"\n=== Running CONFIG: {cfg_name} ===\n")  
    df, all_points = get_trips_for_config(truck_id = 3, cfg_params = cfg_params)
    
    # Rolling features
    all_points["delta_heading_var"] = (
        all_points["delta_heading"].rolling("1min", center=True, min_periods=2).var(ddof=0)
    )
    all_points["speed_estimate"] = (
        all_points["speed_estimate"].rolling("1min", center=True, min_periods=1).mean()
    )
    df = df[['TruckID', 'TripID', 'speed_estimate', 'heading_estimate',
           'delta_heading', 'delta_heading_var', 'weightLimits',
           'CO2EmissionClass', 'dist', 'time_delta', 'geometry']]
    
    plot_truck_timeseries(
        df=df,
        all_points=all_points,
        truck_id=3,
        speed_threshold=speed_cdf[30],
        hard_speed_threshold=speed_cdf[25],
        heading_threshold=heading_cdf[90],
        method=cfg_params['method'],
        title = f"Truck 3 — Speed & Heading Variance with Trip Intervals\n{cfg_name}",
        cfg_name = cfg_name
    )
