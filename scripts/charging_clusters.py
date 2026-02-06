import pickle
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Load background zones
file = '../Data/Geospatial_info/Zoneslevel3_GMM4_with_CRS.gpkg'
zones = gpd.read_file(file)
zones = zones.loc[zones['zoneid'].astype(int) < 900000].copy()


charger_odense   = (55.39489472457706, 10.188725367076998)
charger_limfjord = (56.8844274871181 , 8.59086656628868)
# Load charger locations
def read_chargers(path="../Data/Geospatial_info/Truck_Chargers.xlsx", new_chargers = None):
    df = pd.read_excel(path)

    df[["lat", "lon"]] = (
        df["xy"]
        .str.split(",", expand=True)
        .astype(float)
    )

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )
    chargers = gdf.to_crs("EPSG:25832")[["geometry"]]
    
    if new_chargers:
        # Convert to GeoDataFrame in WGS84, then reproject
        gdf_new = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lat, lon in new_chargers],
            crs="EPSG:4326"
        ).to_crs("EPSG:25832")
        chargers = pd.concat([chargers, gdf_new], ignore_index=True)
    return chargers.drop_duplicates(subset="geometry").reset_index(drop=True)

def read_public():
    # Load charger locations
    with open("../Results/ChargingSimulation/charging_results_SB_dwell_5.pkl", "rb") as f:
        data = pickle.load(f)

    Results = data["Results"]
    config  = data["config"]
    # Extract baseline public charging events (ε = 1)
    public = Results["epsilon_1"].copy()
    
    public = gpd.GeoDataFrame(
        public,
        geometry=gpd.points_from_xy(public.x, public.y),
        crs="EPSG:25832"
    )
    public = public.reset_index(drop=True)

    return public

def get_DBSCAN_clusters(df):
    coords = np.vstack([
        df.geometry.x.values,
        df.geometry.y.values
    ]).T

    db = DBSCAN(
        eps=1000,        # meters
        min_samples=30,
        metric="euclidean"
    )
    df["cluster"] = db.fit_predict(coords)
    
    return df

def extract_cluster_info(df, col):
    
    clusters = (
        df[df["cluster"] >= 0]
        .groupby("cluster")
        .agg(
            n_events=("cluster", "size"),
            mean_dist_km=(col, "mean"),
            median_dist_km=(col, "median"),
            x_mean=("geometry", lambda g: g.x.mean()),
            y_mean=("geometry", lambda g: g.y.mean())
        )
        .reset_index()
    )

    clusters_gdf = gpd.GeoDataFrame(
        clusters,
        geometry=gpd.points_from_xy(clusters.x_mean, clusters.y_mean),
        crs="EPSG:25832"
    )
    return clusters_gdf

def plot(clusters, chargers, col, new_chargers = None):
    fig, ax = plt.subplots(figsize=(8, 10))

    # --- Background GMM level-3 zones ---
    zones.plot(
        ax=ax,
        color='lightgray',
        edgecolor=None,
        alpha=0.6
    )

    # Plot cluster centroids
    clusters.plot(
        ax=ax,
        column=col,
        cmap="viridis",
        markersize=clusters["n_events"] * 0.5,
        legend=True,
        legend_kwds={
            "label": "Mean distance to nearest charger [km]",
            "shrink": 0.7
        },
        alpha=0.9
    )

    # Plot chargers as small black markers (triangles)
    chargers.plot(
        ax=ax,
        color="red",
        marker="^",        # ▲ triangle marker
        markersize=20,
        alpha=0.8,
        label="Existing chargers"
    )
    if new_chargers is not None:
        new_chargers.plot(
            ax=ax,
            color='orange',
            marker="^",
            markersize=20,    # maybe larger?
            alpha=0.9,
            label="New chargers"
        )
        
        # --- Manual legend for chargers ---
        legend_items = [
            Line2D([0],[0], marker='^', color='red', linestyle='None', markersize=8, label='Existing chargers'),
            Line2D([0],[0], marker='^', color='orange', linestyle='None', markersize=10, label='New chargers')
        ]

        ax.legend(handles=legend_items, loc='upper right', frameon=True)
    else:
        # --- Manual legend for chargers ---
        charger_legend = Line2D(
            [0], [0],
            marker="^",
            color="red",
            linestyle="None",
            markersize=8,
            label="Existing chargers"
        )

        ax.legend(
            handles=[charger_legend],
            loc="upper right",
            frameon=True
        )
    # Axis limits (Denmark)
    ax.set_xlim(400_000, 800_000)
    ax.set_ylim(6_025_000, 6_425_000)

    ax.set_title(
        "Clusters of public charging demand",
        fontsize=12
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

    plt.tight_layout()
    if new_chargers is not None:
        plt.savefig(
            "../Results/Figures/cluster_mean_distance_map.png",
            dpi=300,
            bbox_inches="tight"
        )
    else:
        plt.savefig(
            "../Results/Figures/cluster_mean_distance_map_new.png",
            dpi=300,
            bbox_inches="tight"
        )
    plt.show()

#==============================================================================
# MAIN 
#==============================================================================


chargers = read_chargers()
public = read_public()

# Distance to nearest charger (fast & exact)
public_dist = gpd.sjoin_nearest(
    public,
    chargers,
    how="left",
    distance_col="dist_to_charger"
).drop(columns="index_right")
public_dist["dist_to_charger"] = public_dist["dist_to_charger"] / 1000

public_dist = get_DBSCAN_clusters(public_dist)
clusters = extract_cluster_info(public_dist, "dist_to_charger")
plot(clusters, chargers,"mean_dist_km")
print('now')
print(public_dist["dist_to_charger"].sum())

# Distance to nearest charger (fast & exact)
public_dist = gpd.sjoin_nearest(
    public,
    read_chargers(new_chargers = [charger_odense]),
    how="left",
    distance_col="dist_to_charger"
).drop(columns="index_right")
public_dist["dist_to_charger"] = public_dist["dist_to_charger"] / 1000
print('odense')
print(public_dist["dist_to_charger"].sum())

# Distance to nearest charger (fast & exact)
public_dist = gpd.sjoin_nearest(
    public,
    read_chargers(new_chargers = [charger_limfjord]),
    how="left",
    distance_col="dist_to_charger"
).drop(columns="index_right")
public_dist["dist_to_charger"] = public_dist["dist_to_charger"] / 1000
print('lim')
print(public_dist["dist_to_charger"].sum())

# Distance to nearest charger (fast & exact)
public_dist = gpd.sjoin_nearest(
    public,
    read_chargers(new_chargers = [charger_odense,charger_limfjord]),
    how="left",
    distance_col="dist_to_charger"
).drop(columns="index_right")
public_dist["dist_to_charger"] = public_dist["dist_to_charger"] / 1000
public_dist = get_DBSCAN_clusters(public_dist)

clusters = extract_cluster_info(public_dist, "dist_to_charger")
new_chargers = [charger_odense,charger_limfjord]
new_chargers = gpd.GeoDataFrame(
    geometry=[Point(lon, lat) for lat, lon in new_chargers],
    crs="EPSG:4326"
).to_crs("EPSG:25832")

plot(clusters, chargers,"mean_dist_km", new_chargers)
print('both')
print(public_dist["dist_to_charger"].sum())