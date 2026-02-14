#==============================================================================
# 0. INITIALIZATION
#==============================================================================
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from tqdm.auto import tqdm
import os
import pickle
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.neighbors import KernelDensity
sim = 'load' #'compute'

CFG = 'SB_dwell_5'

EV_RANGE_KM = 400 # [km]
EV_RANGE_KMs = [300,400,500,600]

Dwell_time_depot = 6 # [h]
Dwell_time_depots = [4,6,8,10]

epsilon = 1 # public charge fraction 
epsilons = [0.7, 0.8, 0.9, 1]

df = pd.read_parquet(f"../Results/Endpoints_{CFG}.parquet")
df['TruckID'] = df['TripID'] // 10_000

file = '../Data/Geospatial_info/Zoneslevel3_GMM4_with_CRS.gpkg'
zones = gpd.read_file(file)
zones = zones.loc[zones['zoneid'].astype(int) < 900000].copy()
# Dissolve to single geometry
dk = zones.dissolve().explode(ignore_index=True)

#==============================================================================
# 1. Functions
#==============================================================================

def read_chargers(path = '../Data/Geospatial_info/Truck_Chargers.xlsx'):
    df = pd.read_excel(path)
    # split string -> two columns
    df[["lat", "lon"]] = (
        df["xy"]
        .str.split(",", expand=True)
        .astype(float)
    )
    df["geometry"] = [
        Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])
    ]
    gdf = gpd.GeoDataFrame(
        df,
        geometry="geometry",
        crs="EPSG:4326"   # lat/lon
    )
    gdf = gdf.to_crs("EPSG:25832")  # ETRS89 / UTM zone 32N
    gdf = gdf.drop_duplicates(subset="geometry")

    return gdf.geometry


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


def simulate_truck(stops, 
                   ev_range_km=EV_RANGE_KM,
                   Dwell_time_depot = Dwell_time_depot,
                   epsilon=epsilon):
    
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
        stops.loc[idx, "SoC_km"] = soc
        if pd.isna(row["next_leg_km"]) or row["next_leg_km"] == 0:
            dist = 0
        else:
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

if sim == 'compute':
    # Run Model on ALL TRUCKS
    stops = []
    Results = {}
    # tqdm wrapped around the TruckID loop
    # - shows how many trucks processed
    # - estimates remaining time
    truck_groups = df.groupby("TruckID")
    
    for truck_id, df_truck in tqdm(truck_groups, total=len(truck_groups), desc="Prepare stops"):
        # Build sequential stop timeline
        temp = build_stops(df_truck)
        temp["TruckID"] = truck_id
        stops.append(temp)
        
    for eps in epsilons :
        results_temp = []
        for stop_df in tqdm(stops, total=len(stops), desc="Simulating trucks"):
            # Run charging simulation
            sim = simulate_truck(stop_df,epsilon=eps)
            
            results_temp.append(sim)
        
        # Combine all simulation outputs
        sim_all = pd.concat(results_temp, ignore_index=True)
        
        # Extract Charging Points
        charges = sim_all[sim_all["charge"]].copy()
        # Split types (useful for different maps or analysis)
        public = charges[charges["charge_type"] == "public"].copy()
        depot  = charges[charges["charge_type"] == "depot_long_break"]
        public["time"] = public["time"].dt.tz_localize("UTC").dt.tz_convert("Europe/Copenhagen")
        Results[f'epsilon_{eps}'] = public
        
        if eps == 1:
            Results['depot_base'] = depot
    for EV_RANGE in EV_RANGE_KMs:
        results_temp = []
        for stop_df in tqdm(stops, total=len(stops), desc="Simulating trucks"):
            # Run charging simulation
            sim = simulate_truck(stop_df,ev_range_km=EV_RANGE)
            
            results_temp.append(sim)
        
        # Combine all simulation outputs
        sim_all = pd.concat(results_temp, ignore_index=True)
        
        # Extract Charging Points
        charges = sim_all[sim_all["charge"]].copy()
        # Split types (useful for different maps or analysis)
        public = charges[charges["charge_type"] == "public"].copy()
        depot  = charges[charges["charge_type"] == "depot_long_break"]
        public["time"] = public["time"].dt.tz_localize("UTC").dt.tz_convert("Europe/Copenhagen")
        Results[f'EV_RANGE_{EV_RANGE}'] = public
    
    for Dwell in Dwell_time_depots:
        results_temp = []
        for stop_df in tqdm(stops, total=len(stops), desc="Simulating trucks"):
            # Run charging simulation
            sim = simulate_truck(stop_df,Dwell_time_depot=Dwell)
            
            results_temp.append(sim)
        
        # Combine all simulation outputs
        sim_all = pd.concat(results_temp, ignore_index=True)
        
        # Extract Charging Points
        charges = sim_all[sim_all["charge"]].copy()
        # Split types (useful for different maps or analysis)
        public = charges[charges["charge_type"] == "public"].copy()
        depot  = charges[charges["charge_type"] == "depot_long_break"]
        public["time"] = public["time"].dt.tz_localize("UTC").dt.tz_convert("Europe/Copenhagen")
        Results[f'Dwell_{Dwell}'] = public
    #==============================================================================
    # 3. SAVE RESULTS
    #==============================================================================
    
    # Output directory
    OUT_DIR = "../Results/ChargingSimulation"
    os.makedirs(OUT_DIR, exist_ok=True)

    outfile = f"{OUT_DIR}/charging_results_{CFG}.pkl"
    
    # Bundle results + metadata (important for reproducibility)
    payload = {
        "Results": Results,
        "config": {
            "CFG": CFG,
            "EV_RANGE_KMs": EV_RANGE_KMs,
            "Dwell_time_depots": Dwell_time_depots,
            "epsilons": epsilons,
            "default_EV_RANGE_KM": EV_RANGE_KM,
            "default_Dwell_time_depot": Dwell_time_depot,
            "default_epsilon": epsilon,
        }
    }
    
    # Save
    with open(outfile, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"[OK] Results saved to: {outfile}")
    

#==============================================================================
# 4. OPEN RESULTS
#==============================================================================
with open("../Results/ChargingSimulation/charging_results_SB_dwell_5.pkl", "rb") as f:
    data = pickle.load(f)

Results = data["Results"]
config  = data["config"]

#============================================================
# TEMPORAL PLOT
#============================================================
def baseline_temporal_plot():
    key = "epsilon_1"
    df = Results[key].copy()

    df["hour"] = df["time"].dt.hour

    hourly = (
        df["hour"]
        .value_counts()
        .reindex(range(24), fill_value=0)
        .sort_index()
    )

    plt.figure(figsize=(6, 4))
    plt.bar(hourly.index, hourly.values, label='Number of charging events')

    plt.xticks(range(24))
    plt.xlabel("Hour of day")
    plt.ylabel("Public charging events")
    plt.title("Temporal distribution of charging events")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("../Results/Figures/charging_temporal_intensities.png", dpi=300)
    plt.show()

baseline_temporal_plot()


def plot_temporal_comparison(Results, keys, names, title, outfile):
    plt.figure(figsize=(6,3))

    for key, name in zip(keys,names):
        df = Results[key].copy()
        df["hour"] = df["time"].dt.hour

        hourly = (
            df["hour"]
            .value_counts()
            .reindex(range(24), fill_value=0)
            .sort_index()
        )

        plt.plot(hourly.index, hourly.values, marker="o", label=name)

    plt.xticks(range(24))
    plt.xlabel("Hour of day")
    plt.ylabel("Public charging events")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.show()

# ε sensitivity
plot_temporal_comparison(
    Results,
    keys=[f"epsilon_{e}" for e in epsilons],
    names = [f'$\epsilon$={e}' for e in epsilons],
    title="Temporal distribution – ε sensitivity",
    outfile="../Results/Figures/temporal_eps_comparison.png"
)

# EV range sensitivity
plot_temporal_comparison(
    Results,
    keys=[f"EV_RANGE_{r}" for r in EV_RANGE_KMs],
    names=[rf"$R_{{\mathrm{{max}}}}={r}$" for r in EV_RANGE_KMs],
    title="Temporal distribution – $R_{max}$ sensitivity",
    outfile="../Results/Figures/temporal_range_comparison.png"
)

# Depot dwell sensitivity
plot_temporal_comparison(
    Results,
    keys=[f"Dwell_{d}" for d in Dwell_time_depots],
    names=[rf"$\tau_{{\mathrm{{depot}}}}={d}$" for d in Dwell_time_depots],
    title=r"Temporal distribution – $\tau_{{\mathrm{{depot}}}}$ threshold",
    outfile="../Results/Figures/temporal_dwell_comparison.png"
)

#=========================================================
# BINARY DIFFERENCE OF HOTSPOTS PLOT
#=========================================================
def kde_hotspot_mask(dens, q=0.95):
    thr = np.quantile(dens, q)
    return dens >= thr

def hotspot_difference(mask_scn, mask_base):
    # +1 = new hotspot, -1 = lost hotspot, 0 = unchanged
    return mask_scn.astype(int) - mask_base.astype(int)

def compute_kde_from_df(df, bandwidth=2500, nx=600, ny=600):
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.x, df.y),
        crs="EPSG:25832"
    )

    coords = np.vstack([gdf.x.values, gdf.y.values]).T
    kde = KernelDensity(bandwidth=bandwidth).fit(coords)

    # --------------------------------------------------
    # GLOBAL KDE GRID (fixed for all scenarios)
    # --------------------------------------------------
    x_min, x_max = 400_000, 800_000
    y_min, y_max = 6_025_000, 6_425_000

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, nx),
        np.linspace(y_min, y_max, ny)
    )
    grid = np.vstack([xx.ravel(), yy.ravel()]).T

    dens = np.exp(kde.score_samples(grid)).reshape(ny, nx)

    extent = (x_min, x_max, y_min, y_max)
    return dens, extent


# --------------------------------------------------
# Plot
# --------------------------------------------------
dens, extent = compute_kde_from_df(Results['epsilon_1'], bandwidth=5000, nx=600, ny=600)
plt.figure(figsize=(6, 4))

dk.plot(
    facecolor='none',      # transparent
    edgecolor='white',
    linewidth=0.6,
    alpha=0.9
)
plt.imshow(
    dens,
    origin="lower",
    extent=extent,
    cmap="inferno",
    aspect="equal"
)

plt.colorbar(label="Relative density (charging demand)")
plt.title(f"KDE Plot (bandwidth = {5000/1000:.1f} km)")
plt.xticks([400_000, 500_000, 600_000, 700_000, 800_000])
plt.xlabel("X (m)")
plt.ylabel("Y (m)")

plt.xlim([400_000, 800_000])
plt.ylim([6_025_000, 6_425_000])

plt.savefig(
    "../Results/Figures/charging_kde.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

dens, extent = compute_kde_from_df(Results['depot_base'], bandwidth=5000, nx=600, ny=600)
plt.figure(figsize=(6, 4))

dk.plot(
    facecolor='none',      # transparent
    edgecolor='white',
    linewidth=0.6,
    alpha=0.9
)

plt.imshow(
    dens,
    origin="lower",
    extent=extent,
    cmap="inferno",
    aspect="equal"
)

plt.colorbar(label="Relative density (charging demand)")
plt.title(f"KDE Plot - Depot (bandwidth = {5000/1000:.1f} km)")
plt.xticks([400_000, 500_000, 600_000, 700_000, 800_000])
plt.xlabel("X (m)")
plt.ylabel("Y (m)")

plt.xlim([400_000, 800_000])
plt.ylim([6_025_000, 6_425_000])

plt.savefig(
    "../Results/Figures/charging_kde_depot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

def plot_hotspot_comparison(
    Results,
    baseline_key,
    scenario_keys,
    scenario_labels,
    bandwidth=5000,
    q=0.95,
    outfile=None
):
    n = len(scenario_keys)
    fig, axes = plt.subplots(
        nrows=n,
        ncols=3,
        figsize=(4*3, 4*n),
        sharex=True,
        sharey=True
    )

    # --- Baseline ---
    dens_base, extent = compute_kde_from_df(
        Results[baseline_key], bandwidth=bandwidth
    )
    mask_base = kde_hotspot_mask(dens_base, q=q)

    for i, (key, label) in enumerate(zip(scenario_keys, scenario_labels)):

        # --- Scenario KDE & mask ---
        dens_scn, _ = compute_kde_from_df(
            Results[key], bandwidth=bandwidth
        )
        mask_scn = kde_hotspot_mask(dens_scn, q=q)

        # --- Difference ---
        diff = hotspot_difference(mask_scn, mask_base)

        # === Column 1: Baseline ===
        ax = axes[i, 0]
        # DK outline
        dk.plot(
            ax=ax,
            facecolor='none',
            edgecolor='gray',
            linewidth=0.6,
            alpha=0.9
        )
        ax.imshow(
            mask_base,
            origin="lower",
            extent=extent,
            cmap="gray_r",
            aspect="equal"
        )
        ax.set_ylabel(label,fontsize=20)

        # === Column 2: Scenario ===
        ax = axes[i, 1]
        # DK outline
        dk.plot(
            ax=ax,
            facecolor='none',
            edgecolor='gray',
            linewidth=0.6,
            alpha=0.9
        )
        ax.imshow(
            mask_scn,
            origin="lower",
            extent=extent,
            cmap="gray_r",
            aspect="equal"
        )

        # === Column 3: Difference ===
        ax = axes[i, 2]
        # DK outline
        dk.plot(
            ax=ax,
            facecolor='none',
            edgecolor='gray',
            linewidth=0.6,
            alpha=0.9
        )        
        ax.imshow(
            diff,
            origin="lower",
            extent=extent,
            cmap="RdBu",
            vmin=-1,
            vmax=1,
            aspect="equal"
        )

    # --- Column titles ---
    axes[0, 0].set_title("Baseline\n(top 5%)",fontsize = 20)
    axes[0, 1].set_title("Scenario\n(top 5%)",fontsize = 20)
    axes[0, 2].set_title("Difference to baseline",fontsize = 20)

    # --- Axis limits (Denmark) ---
    for ax in axes.ravel():
        ax.set_xlim(400000, 800000)
        ax.set_ylim(6_025_000, 6_425_000)

    # --- Annotation explaining difference colors ---
    axes[0, 2].text(
        0.02, 0.95,
        "Red: new hotspot\nBlue: lost hotspot",
        transform=axes[0, 2].transAxes,
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
    )

    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")

    plt.show()

baseline_key = "epsilon_1"

scenario_keys = ["epsilon_0.7", "epsilon_0.8", "epsilon_0.9"]
scenario_labels = [r"$\epsilon = 0.7$", r"$\epsilon = 0.8$", r"$\epsilon = 0.9$"]

plot_hotspot_comparison(
    Results,
    baseline_key=baseline_key,
    scenario_keys=scenario_keys,
    scenario_labels=scenario_labels,
    q=0.95,
    outfile="../Results/Figures/hotspot_eps_comparison.png"
)

baseline_key = "EV_RANGE_400"

scenario_keys = ["EV_RANGE_300", "EV_RANGE_500", "EV_RANGE_600"]
scenario_labels = [
    "300 km",
    "500 km",
    "600 km"
]

plot_hotspot_comparison(
    Results,
    baseline_key=baseline_key,
    scenario_keys=scenario_keys,
    scenario_labels=scenario_labels,
    q=0.95,
    outfile="../Results/Figures/hotspot_range_comparison.png"
)

baseline_key = "Dwell_6"

scenario_keys = ["Dwell_4", "Dwell_8", "Dwell_10"]
scenario_labels = [
    "4 h",
    "8 h",
    "10 h"
]

plot_hotspot_comparison(
    Results,
    baseline_key=baseline_key,
    scenario_keys=scenario_keys,
    scenario_labels=scenario_labels,
    q=0.95,
    outfile="../Results/Figures/hotspot_dwell_comparison.png"
)
