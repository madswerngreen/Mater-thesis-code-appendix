#==============================================================================
# 0. INITIALIZATION
#==============================================================================
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
CFG = 'SB_dwell_5'
EV_RANGE_KM = 400
Dwell_time_depot = 6 # [h]
epsilon = 1 # public charge fraction 
df = pd.read_parquet(f"../Results/Endpoints_{CFG}.parquet")
df['TruckID'] = df['TripID'] // 10_000

#==============================================================================
# 1. Functions
#==============================================================================

def build_stops(df):
    df = df.copy()

    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time']   = pd.to_datetime(df['end_time'])

    df = df.sort_values(by="end_time").reset_index(drop=True)

    df["dwell_hours"] = (
        df['start_time'].shift(-1) - df['end_time']
    ).dt.total_seconds() / 3600 # trip time [h]
    # stop time
    df['time'] = df['end_time'] 
    # stop location
    df['x'] = df['end_x'] 
    df['y'] = df['end_y']
    # next trip length
    df['next_leg_km'] = df['distance_m'].shift(-1) / 1000

    return df[['time','x','y','next_leg_km','dwell_hours']]


def simulate_truck(stops, ev_range_km=EV_RANGE_KM):
    
    stops = stops.sort_values("time").copy()
    stops.reset_index(inplace=True, drop=True)

    stops["charge"] = False
    stops["charge_type"] = None
    stops["SoC_km"] = np.nan
    
    soc = ev_range_km
    
    # First observed stop = start with full battery
    first_idx = stops.index[0]
    stops.loc[first_idx, "charge"] = True
    stops.loc[first_idx, "charge_type"] = "start_full"
    stops.loc[first_idx, "SoC_km"] = soc
    
    for idx, row in stops.iterrows():
        
        if pd.isna(row["next_leg_km"]) or row["next_leg_km"] == 0:
            stops.loc[idx, "SoC_km"] = soc
            continue
        
        dist = row["next_leg_km"]
        
        # ---- RULE 1: long break -> depot charging ----
        if not pd.isna(row["dwell_hours"]) and row["dwell_hours"] >= Dwell_time_depot:
            stops.loc[idx, "charge"] = True
            stops.loc[idx, "charge_type"] = "depot_long_break"
            soc = ev_range_km # charge to full
        
        # ---- RULE 2: insufficient SoC -> public charge ----
        if dist > soc:
            stops.loc[idx, "charge"] = True
            stops.loc[idx, "charge_type"] = "public"
            soc = epsilon * ev_range_km   # charge to epsilon * full
        
        # ---- Drive the leg ----
        soc -= dist
    
    return stops

#==============================================================================
# 2. Main
#==============================================================================

# Run Model on ALL TRUCKS
results = []

# tqdm wrapped around the TruckID loop
# - shows how many trucks processed
# - estimates remaining time
truck_groups = df.groupby("TruckID")

for truck_id, df_truck in tqdm(truck_groups, total=len(truck_groups), desc="Simulating trucks"):
    
    # Build sequential stop timeline
    stops = build_stops(df_truck)
    
    # Run charging simulation
    sim = simulate_truck(stops)
    
    # Store truck ID for later identification
    sim["TruckID"] = truck_id
    
    results.append(sim)

# Combine all simulation outputs
sim_all = pd.concat(results, ignore_index=True)

# Extract Charging Points
charges = sim_all[sim_all["charge"]].copy()

# Split types (useful for different maps or analysis)
public = charges[charges["charge_type"] == "public"].copy()
depot  = charges[charges["charge_type"] == "depot_long_break"]
public["time"] = public["time"].dt.tz_localize("UTC").dt.tz_convert("Europe/Copenhagen")
# ================================
# Simple Temporal Intensity Plot
# ================================
public["hour"] = public["time"].dt.hour

hour_counts = (
    public["hour"]
    .value_counts()
    .reindex(range(24), fill_value=0)  # ensure full 0–23 coverage
    .sort_index()
)

plt.figure(figsize=(10,4))
plt.bar(hour_counts.index, hour_counts.values)
plt.xticks(range(24))
plt.xlabel("Hour of Day")
plt.ylabel("Number of Stops / Charging Events")
plt.title("Temporal Distribution of Public Charging Events")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    "../Results/Figures/charging_temporal_intensities.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

#------------------------------------------------------------------------------
# Convert to GeoDataFrame (choose which dataset you want to visualize)
# Here: ONLY public charging demand
#------------------------------------------------------------------------------
gdf = gpd.GeoDataFrame(
    public,
    geometry=gpd.points_from_xy(public.x, public.y),
    crs="EPSG:25832"
)

# --------------------------------------------------
# 1) Prepare coordinates
# --------------------------------------------------
coords = np.vstack([gdf.x.values, gdf.y.values]).T  # shape (n_samples, 2)

# Choose bandwidth in meters (same for x & y)
bandwidth = 5_000  # e.g. 10 km smoothing radius

# Fit isotropic Gaussian KDE
kde = KernelDensity(
    bandwidth=bandwidth,   # in meters
    kernel='gaussian'
).fit(coords)

# --------------------------------------------------
# 2) Build evaluation grid
# --------------------------------------------------
x_min, x_max = gdf.x.min(), gdf.x.max()
y_min, y_max = gdf.y.min(), gdf.y.max()

# choose resolution
nx, ny = 800, 800

x_grid = np.linspace(x_min, x_max, nx)
y_grid = np.linspace(y_min, y_max, ny)
xx, yy = np.meshgrid(x_grid, y_grid)

grid_points = np.vstack([xx.ravel(), yy.ravel()]).T  # (nx*ny, 2)

# --------------------------------------------------
# 3) Evaluate KDE (returns log-density)
# --------------------------------------------------
log_dens = kde.score_samples(grid_points)
dens = np.exp(log_dens).reshape(ny, nx)  # reshape to grid
#dens = log_dens.reshape(ny, nx)  # reshape to grid
# --------------------------------------------------
# 4) Plot with equal aspect
# --------------------------------------------------
plt.figure(figsize=(6, 4))
plt.imshow(
    dens,
    origin='lower',
    extent=[x_min, x_max, y_min, y_max],
    cmap='inferno',
    aspect='equal'  # <-- equal scaling in x and y
)
plt.colorbar(label="Relative density (charging events)")
plt.title(f"Isotropic KDE (bandwidth = {bandwidth/1000:.1f} km)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.xlim([400000,800000])
plt.ylim([6000000, 6400734])
plt.savefig(
    "../Results/Figures/charging_kde.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
#============================================================
# Temporal KDE plots (hourly)
#============================================================
public["hour"] = public["time"].dt.hour

fig, axes = plt.subplots(4, 6, figsize=(18, 12))
axes = axes.ravel()

# Store per-hour density grids
dens_per_hour = []

for hour in range(24):
    subset = public[public.hour == hour]
    ax = axes[hour]

    if subset.empty:
        dens_per_hour.append(None)
        ax.set_title(f"{hour:02d}:00\n(no data)")
        ax.axis("off")
        continue

    coords = np.column_stack([subset.x.values, subset.y.values])

    # KDE per hour (you can tweak bandwidth)
    kde_h = KernelDensity(
        bandwidth=bandwidth,
        kernel="gaussian"
    ).fit(coords)

    log_dens = kde_h.score_samples(grid_points)
    d = np.exp(log_dens).reshape(ny, nx)
    dens_per_hour.append(d)

# Global max for shared color scale
global_max = max(d.max() for d in dens_per_hour if d is not None)

# Plot
im = None
for hour in range(24):
    ax = axes[hour]
    d = dens_per_hour[hour]
    
    if d is None:
        # already handled above
        continue
    plt.figure()
    plt.imshow(d,
               origin="lower",
               extent=[x_min, x_max, y_min, y_max],
               cmap="inferno",
               aspect="equal",
               vmin=0,
               )
    plt.xlim(400000, 800000)
    plt.ylim(6000000, 6400734)
    plt.title((f"{hour:02d}:00"))
    plt.tight_layout()
    plt.savefig(
        f"../Results/Figures/charging_kde{hour:02d}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    
    im = ax.imshow(d,
                   origin="lower",
                   extent=[x_min, x_max, y_min, y_max],
                   cmap="inferno",
                   aspect="equal",
                   vmin=0,
    )
    
    ax.set_xlim(400000, 800000)
    ax.set_ylim(6000000, 6400734)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(f"{hour:02d}:00")

plt.tight_layout()
if im is not None:
    fig.colorbar(im, ax=axes.tolist(), label="Relative density")
plt.show()
plt.savefig(
    "../Results/Figures/charging_kde.png",
    dpi=300,
    bbox_inches="tight"
)


