#==============================================================================
# 0. INITIALIZATION
#==============================================================================
import pandas as pd
import geopandas as gpd
import numpy as np
import os

CFG = 'SB_dwell_5'

OUT_GPKG_PATH  = f"../Results/StopClusters_{CFG}.gpkg"
OUT_GPKG_PATH  = f"K:/PLA/PA/TAN/Studentermedhjælper/Mads/Master Thesis/Præsentation/Results/StopClusters_{CFG}.gpkg"
min_stop_time = 0.5  * 3600    # 15 min
max_stop_time = 2 * 3600    # 2 hours
CRS_EPSG = 25832  # metric CRS (ETRS89 / UTM 32N)

#==============================================================================
# 1. PREPARING DATAFRAME
#==============================================================================
df = pd.read_parquet(f"../Results/Endpoints_{CFG}.parquet")
df["TruckID"] = df["TripID"] //10_000
# Sort by truck, then by time
df = df.sort_values(["TruckID", "start_time"]).reset_index(drop=True)

# Shift columns to align “next trip” information
df["next_TruckID"] = df["TruckID"].shift(-1)
df["next_start_time"] = df["start_time"].shift(-1)

# Stop duration only when next trip is same truck
same_truck = df["TruckID"] == df["next_TruckID"]
df["duration"] = df["next_start_time"] - df["end_time"]
df.loc[~same_truck, "duration"] = pd.NaT

# Remove invalid stops (no duration)
df_stops = df.dropna(subset=["duration"]).copy()

# Convert to seconds
df_stops["duration_s"] = df_stops["duration"].dt.total_seconds()

# Filter 30 min to 24 hours
df_stops = df_stops[
    (df_stops["duration_s"] >= min_stop_time) &
    (df_stops["duration_s"] <= max_stop_time)
].copy()

# Filter stops where next start is spatially near the stop end
threshold = 1000  # meters

df_stops["next_start_x"] = df_stops["start_x"].shift(-1)
df_stops["next_start_y"] = df_stops["start_y"].shift(-1)

# Distance formula for 2D CRS
dx = df_stops["next_start_x"] - df_stops["end_x"]
dy = df_stops["next_start_y"] - df_stops["end_y"]
df_stops["restart_dist_m"] = np.sqrt(dx*dx + dy*dy)

# Keep only spatially valid stops
df_stops = df_stops[df_stops["restart_dist_m"] <= threshold]


print("Number of valid stops:", len(df_stops))
#==============================================================================
# 2. CLUSTERING OF STOPS
#==============================================================================
"""
import hdbscan

X = df_stops[["end_x", "end_y"]].values

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    metric='euclidean'
)

labels = clusterer.fit_predict(X)
df_stops["cluster"] = labels
"""
from sklearn.cluster import DBSCAN

X = df_stops[["end_x", "end_y"]].values

clusterer = DBSCAN(
    eps=500,                # maximum cluster radius in meters
    min_samples=10,         # minimum stops to define a valid hub
    algorithm = "kd_tree"
)

labels = clusterer.fit_predict(X)
df_stops["cluster"] = labels


#==============================================================================
# 3. OUTPUT
#==============================================================================
gdf_stops = gpd.GeoDataFrame(
    df_stops,
    geometry=gpd.points_from_xy(df_stops["end_x"], df_stops["end_y"]),
    crs=f"EPSG:{CRS_EPSG}"
)

# Keep only useful columns for QGIS (but you can keep more)
cols_keep = [
    "TruckID",
    "TripID",
    "start_time",
    "end_time",
    "next_start_time",
    "duration_s",
    "cluster",
    "geometry",
]
gdf_stops = gdf_stops[cols_keep]

mask_clusters = gdf_stops["cluster"] != -1
df_clu = gdf_stops[mask_clusters].copy()

if not df_clu.empty:
    agg = (
        df_clu
        .groupby("cluster")
        .agg(
            x_mean=("geometry", lambda s: np.mean([p.x for p in s])),
            y_mean=("geometry", lambda s: np.mean([p.y for p in s])),
            n_stops=("TruckID", "size"),
            n_trucks=("TruckID", "nunique"),
            total_duration_h=("duration_s", lambda s: s.sum() / 3600.0),
            mean_duration_h=("duration_s", lambda s: s.mean() / 3600.0),
        )
        .reset_index()
    )

    gdf_clusters = gpd.GeoDataFrame(
        agg,
        geometry=gpd.points_from_xy(agg["x_mean"], agg["y_mean"]),
        crs=f"EPSG:{CRS_EPSG}"
    )
else:
    gdf_clusters = gpd.GeoDataFrame(
        columns=[
            "cluster", "x_mean", "y_mean",
            "n_stops", "n_trucks",
            "total_duration_h", "mean_duration_h",
            "geometry",
        ],
        geometry="geometry",
        crs=f"EPSG:{CRS_EPSG}"
    )

# --------------------------------------------------
# 3.1 Filter: Top 250 clusters by number of stops
# --------------------------------------------------
if not gdf_clusters.empty:
    gdf_clusters_top250 = (
        gdf_clusters
        .sort_values(by="n_stops", ascending=False)
        .head(250)
        .copy()
    )
else:
    gdf_clusters_top250 = gdf_clusters.copy()
# --------------------------------------------------
# 3.2 Filter stops belonging only to the top 250 clusters
# --------------------------------------------------
cluster_ids_top250 = gdf_clusters_top250["cluster"].unique()

gdf_stops_top250 = gdf_stops[
    gdf_stops["cluster"].isin(cluster_ids_top250)
].copy()

# --------------------------------------------------
# 3.3 Write to GeoPackage
# --------------------------------------------------
# Remove existing file to avoid layer overwrite issues
if os.path.exists(OUT_GPKG_PATH):
    os.remove(OUT_GPKG_PATH)

gdf_stops.to_file(OUT_GPKG_PATH, layer="stops", driver="GPKG")
gdf_stops_top250.to_file(OUT_GPKG_PATH, layer="stops_top250", driver="GPKG")
gdf_clusters.to_file(OUT_GPKG_PATH, layer="clusters", driver="GPKG")
gdf_clusters_top250.to_file(OUT_GPKG_PATH, layer="clusters_top250", driver="GPKG")

print(f"✓ GeoPackage written: {OUT_GPKG_PATH}")
print("Layers written: 'stops', 'stops_top250', 'clusters', 'clusters_top250'")
