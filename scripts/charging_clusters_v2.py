import pickle
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
raise ValueError('Breakpoint')
#==============================================
# LOAD GMM ZONES (background plotting)
#==============================================
file = '../Data/Geospatial_info/Zoneslevel3_GMM4_with_CRS.gpkg'
zones = gpd.read_file(file)
zones = zones.loc[zones['zoneid'].astype(int) < 900000].copy()

#==============================================
# LOAD EXISTING CHARGERS
#==============================================
def read_chargers(path="../Data/Geospatial_info/Truck_Chargers.xlsx",
                  new_chargers=None):

    df = pd.read_excel(path)
    df[["lat", "lon"]] = df["xy"].str.split(",", expand=True).astype(float)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )
    chargers = gdf.to_crs("EPSG:25832")[["geometry"]]

    if new_chargers:
        gdf_new = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lat, lon in new_chargers],
            crs="EPSG:4326"
        ).to_crs("EPSG:25832")
        chargers = pd.concat([chargers, gdf_new], ignore_index=True)

    return chargers.drop_duplicates(subset="geometry").reset_index(drop=True)

#==============================================
# LOAD PUBLIC CHARGING EVENTS (ε = 1 scenario)
#==============================================
def read_public():
    with open("../Results/ChargingSimulation/charging_results_SB_dwell_5.pkl", "rb") as f:
        data = pickle.load(f)

    public = data["Results"]["epsilon_1"].copy()

    public = gpd.GeoDataFrame(
        public,
        geometry=gpd.points_from_xy(public.x, public.y),
        crs="EPSG:25832"
    ).reset_index(drop=True)

    return public

#==============================================
# DBSCAN CLUSTERING
#==============================================
def get_DBSCAN_clusters(df):
    coords = np.vstack([df.geometry.x.values, df.geometry.y.values]).T
    db = DBSCAN(eps=1000, min_samples=30, metric="euclidean")
    df["cluster"] = db.fit_predict(coords)
    return df

#==============================================
# EXTRACT CLUSTER STATS
#==============================================
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

#==============================================
# PLOTTING
#==============================================
def plot(clusters, chargers, col, new_chargers=None, name = None,
         title="Clusters of public charging demand"):
    fig, ax = plt.subplots(figsize=(8, 7))

    zones.plot(ax=ax, color='lightgray', edgecolor=None, alpha=0.6)

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

    # existing chargers
    chargers.plot(
        ax=ax,
        color="red",
        marker="^",
        markersize=20,
        alpha=0.8
    )

    # new chargers
    if new_chargers is not None:
        new_chargers.plot(
            ax=ax,
            color='orange',
            marker="^",
            markersize=30,
            alpha=0.9
        )
        legend_items = [
            Line2D([0], [0], marker='^', color='red', linestyle='None', markersize=8, label='Existing chargers'),
            Line2D([0], [0], marker='^', color='orange', linestyle='None', markersize=10, label='New chargers')
        ]
        ax.legend(handles=legend_items, loc='upper right', frameon=True)
    else:
        legend_items = [
            Line2D([0], [0], marker='^', color='red', linestyle='None', markersize=8, label='Existing chargers')
        ]
        ax.legend(handles=legend_items, loc='upper right', frameon=True)

    ax.set_xlim(400_000, 800_000)
    ax.set_ylim(6_025_000, 6_425_000)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    plt.tight_layout()
    plt.savefig(f"../Results/Figures/cluster_mean_distance_{name}.png", dpi=300)
    plt.show()

def build_distance_matrix(clusters, public_dist, existing_col="dist_to_charger"):
    """
    Build D matrix for MILP:
    D[i,j] = distance from cluster i to candidate j
    Last column j=N is distance to existing chargers from original data.
    """
    clusters = clusters.copy().reset_index(drop=True)

    # === Number of clusters ===
    N = len(clusters)

    # === Cluster demand vector ===
    n_vec = clusters["n_events"].values.astype(int)

    # === Extract cluster centroids ===
    xs = clusters.geometry.x.values
    ys = clusters.geometry.y.values

    # === Compute distance between every centroid pair ===
    # D_candidate[i,j] = distance from cluster i to cluster j
    # shape: N x N
    cx = xs.reshape(-1, 1)
    cy = ys.reshape(-1, 1)
    cX = xs.reshape(1, -1)
    cY = ys.reshape(1, -1)

    dist_candidates = np.sqrt((cx - cX)**2 + (cy - cY)**2) / 1000.0  # meters → km

    # === Last column: distance to existing chargers ===
    # mean fallback distance per cluster
    # group the already computed distances
    d_existing = (
        public_dist.groupby("cluster")[existing_col]
        .mean()
        .reindex(range(N))     # ensure ordering matches clusters df
        .fillna(0)
        .values
    ).reshape(-1, 1)

    # === Concatenate into final D ===
    D = np.hstack([dist_candidates, d_existing])

    return D, n_vec

#==============================================
# CURRENT STATE
#==============================================
chargers = read_chargers()
public = read_public()

#--- Initial distances ---
public_dist = gpd.sjoin_nearest(
    public, chargers, how="left", distance_col="dist_to_charger"
).drop(columns="index_right")

public_dist["dist_to_charger"] /= 1000
public_dist = get_DBSCAN_clusters(public_dist)

# Compute cluster-level metrics
clusters = extract_cluster_info(public_dist, col="dist_to_charger")

plot(clusters, chargers, col="mean_dist_km",name = 'existing_chargers')

best_costs = []
Ks = [1,2,3,4,5,6,7,8,9,10]
for k in Ks:
    #==============================================
    # OPTIMIZATION IMPORTS
    #==============================================
    from pyomo.environ import (
        ConcreteModel, Var, Objective, Constraint, RangeSet,
        Binary, SolverFactory, summation
    )
    #==============================================
    #MODEL, INDEX SETS AND PARAMETERS
    #==============================================
    D, n =  build_distance_matrix(clusters, public_dist, existing_col="dist_to_charger")
    
    N = D.shape[0]         # number of clusters
    J = N + 1              # number of "locations" (N candidates + 1 existing)
    
    model = ConcreteModel()
    
    model.I = RangeSet(0, N-1)     # Demand clusters   (Python idx)
    model.J = RangeSet(0, J-1)     # Candidate + 1 existing
    
    model.D = D                   # shape N x (N+1)
    model.n = n                   # shape N
    model.k = k                   # max number of new chargers
    #==============================================
    # VARIABLES
    #==============================================
    model.x = Var(model.I, model.J, domain=Binary)
    #==============================================
    # OBJECTIVE FUNCTION
    #==============================================
    def total_cost(model):
        return sum(model.n[i] * model.D[i][j] * model.x[i,j]
                   for i in model.I for j in model.J)
    
    model.OBJ = Objective(rule=total_cost)
    
    #==============================================
    # CONSTRAINTS
    #==============================================
    
    # CONSTRAINT 1: Each cluster chooses exactly one location
    def assign_once(model, i):
        return sum(model.x[i,j] for j in model.J) == 1
    
    model.AssignOnce = Constraint(model.I, rule=assign_once)
    
    # CONSTRAINT 2: Can only assign to opened sites (j != N)
    def open_logic(model, i, j):
        if j < N:   # candidate locations
            return model.x[i,j] <= model.x[j,j]
        return Constraint.Skip  # skip the existing network column
    
    model.OpenLogic = Constraint(model.I, model.J, rule=open_logic)
    
    # CONSTRAINT 3: At most k candidate chargers
    def k_limit(model):
        return sum(model.x[j,j] for j in range(N)) <= model.k
    
    model.Klimit = Constraint(rule=k_limit)
    
    #==============================================
    # SOLVE
    #==============================================
    #solver = SolverFactory("highs")  # or "cbc" or "glpk"
    solver = SolverFactory("appsi_highs")
    result = solver.solve(model, tee=True)
    #print(result.solver.status)
    
    #==============================================
    # EXTRACT SOLUTION
    #==============================================
    opened = [j for j in range(N) if model.x[j,j].value == 1]
    
    #print("Opened charger locations at clusters:", opened)
    
    assignments = np.array([[model.x[i,j].value for j in range(J)] for i in range(N)])
    best_cost = model.OBJ()
    best_costs.append(best_cost)
    print("Total cost:", best_cost)
 
    # Extract selected cluster centroids
    new_chargers = clusters.loc[opened].copy()
    
    new_chargers = gpd.GeoDataFrame(
        new_chargers,
        geometry=new_chargers.geometry,
        crs="EPSG:25832"
    )
    all_chargers = pd.concat(
        [chargers, new_chargers[["geometry"]]],
        ignore_index=True
    )
    public_dist_new = gpd.sjoin_nearest(
        public,
        all_chargers,
        how="left",
        distance_col="dist_to_charger"
    ).drop(columns="index_right")
    
    public_dist_new["dist_to_charger"] /= 1000  # meters → km
    public_dist_new["cluster"] = public_dist["cluster"].values
    clusters_new = extract_cluster_info(
        public_dist_new,
        col="dist_to_charger"
    )
    
    plot(
        clusters_new,
        chargers,
        col="mean_dist_km",
        new_chargers=new_chargers,
        name=f"optimal_chargers_updated_dist_k_{k}",
        title=f"Optimal for k={k} New Charger locations"
    )

plt.figure(figsize=(5, 4))

plt.plot(
    Ks,
    best_costs,
    marker="o",
    linewidth=2
)

plt.xlabel("Number of new chargers (k)")
plt.ylabel("Objective value")
plt.title("Optimal objective value as a function of k")

plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"../Results/Figures/objective_value.png", dpi=300)
plt.show()

