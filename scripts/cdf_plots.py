import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm.auto import tqdm  
import os
import pickle   

FORCE_RECALCULATION = False

CONFIGS = [
    {"name": "SB_dwell_5"},
    {"name": "SB_dwell_15"},
    {"name": "HIS_dwell_5"},
    {"name": "HIS_dwell_15"},
]
DTU_COLORS = [
    "#E57200",  # DTU Orange
    "#79A70A",  # DTU Green
    "#003D73",  # DTU Blue
    "#C8102E",  # DTU Red
]

def extract_trip_level_stats(df):
    """
    Input: df containing multiple trips for one truck.
    Index must be datetime. Column 'TripID' identifies trips.
    Column 'distance' is per-point distance to previous point.
    
    Output: DataFrame with one row per trip.
    """
    out = []

    for trip_id, grp in df.groupby("TripID"):
        if len(grp) < 2:
            continue

        # --- Trip time ---
        t0 = grp.index.min()
        t1 = grp.index.max()
        trip_minutes = (t1 - t0).total_seconds() / 60.0

        # --- Trip distance ---
        trip_km = grp["dist"].sum() / 1000.0   # convert meters → km

        out.append((trip_id, trip_minutes, trip_km))

    return pd.DataFrame(out, columns=["TripID", "trip_time_min", "trip_distance_km"])

def compute_trip_stats(config_name):
    print("\n" + "="*80)
    print(f" Processing CONFIG: {config_name}")
    print("="*80)

    trip_files = glob.glob( f"C:/Users/b306630/Desktop/Mater thesis back uup/Data/truck_trips/{config_name}/truck_*_trips.parquet")

    stats = []
    trip_level_frames = []

    for f in tqdm(trip_files, desc=f"Counting trips ({config_name})"):
        df = pd.read_parquet(f)
        truck = int(f.split("_")[-2])

        stats.append((truck, df["TripID"].nunique()))

        trip_stats = extract_trip_level_stats(df)
        trip_stats["truck"] = truck
        trip_level_frames.append(trip_stats)

    # ---------------------------
    # Trip count dataframe
    # ---------------------------
    trip_counts = pd.DataFrame(stats, columns=["truck", "n_trips"])

    have_file = set(trip_counts["truck"])
    expected  = set(range(1, max(have_file) + 1))
    missing   = sorted(list(expected - have_file))
    zero_df = pd.DataFrame({"truck": missing, "n_trips": 0})
    trip_counts = pd.concat([trip_counts, zero_df], ignore_index=True)

    # ---------------------------
    # Trip-level data (if exists)
    # ---------------------------
    if len(trip_level_frames) == 0:
        print("No trip-level data!")
        return None

    trip_level = pd.concat(trip_level_frames, ignore_index=True)

    # thresholds
    t_lo, t_hi = np.percentile(trip_level["trip_time_min"], [0.5, 99.5])
    d_lo, d_hi = np.percentile(trip_level["trip_distance_km"], [0.5, 99.5])

    # extreme trips BEFORE trimming
    t_cut = t_hi
    d_cut = d_hi

    extreme_trips = trip_level[
        (trip_level["trip_time_min"] >= t_cut) |
        (trip_level["trip_distance_km"] >= d_cut)
    ][["truck", "TripID", "trip_time_min", "trip_distance_km"]]

    extreme_trips = extreme_trips.sort_values(
        by=["trip_time_min", "trip_distance_km"], ascending=False
    ).reset_index(drop=True)

    # trimming
    trip_level = trip_level[
        (trip_level["trip_time_min"].between(t_lo, t_hi)) &
        (trip_level["trip_distance_km"].between(d_lo, d_hi))
    ]

    # ---------------------------
    # Compute quantiles only (fast store)
    # ---------------------------
    quantiles = np.arange(0, 100.25, 0.25)

    quant_dict = {
        "config": config_name,
        "trip_count_quantiles": {
            float(q): float(np.percentile(trip_counts["n_trips"], q))
            for q in quantiles
        },
        "trip_time_quantiles": {
            float(q): float(np.percentile(trip_level["trip_time_min"], q))
            for q in quantiles
        },
        "trip_distance_quantiles": {
            float(q): float(np.percentile(trip_level["trip_distance_km"], q))
            for q in quantiles
        },
        "trim_thresholds": {
            "t_lo": float(t_lo), "t_hi": float(t_hi),
            "d_lo": float(d_lo), "d_hi": float(d_hi)
        },
        "extreme_trips": extreme_trips
    }

    return quant_dict

def plot_from_quantiles(
    result,
    var="trip_time_quantiles",
    annotate_quantiles=[25, 50, 75, 95],
    fig_dir = '../Results/Figures/',
    config = None,
):
    q = np.array(list(result[var].keys()))
    vals = np.array(list(result[var].values()))

    # ----------------------------------------
    # Plot style and setup
    # ----------------------------------------
    plt.figure(figsize=(7, 6))
    plt.style.use("seaborn-v0_8")

    colors = {
        "trip_time_quantiles":   ("#1f77b4", "#aec7e8"),
        "trip_distance_quantiles": ("#d62728", "#ff9896"),
        "trip_count_quantiles":  ("#2ca02c", "#98df8a")
    }
    main_color, shade_color = colors[var]

    # ----------------------------------------
    # Plot curve
    # ----------------------------------------
    if var == "trip_count_quantiles":
        plt.step(vals, q, where="post", lw=2, color=main_color, label="Empirical CDF")
        plt.fill_between(vals, q, step="post", color=shade_color, alpha=0.20)
    else:
        plt.plot(vals, q, lw=2, color=main_color, label="Empirical CDF")
        plt.fill_between(vals, q, color=shade_color, alpha=0.20)

    # ----------------------------------------
    # Labels
    # ----------------------------------------
    xlabel_map = {
        "trip_time_quantiles": "Trip time [minutes]",
        "trip_distance_quantiles": "Trip distance [km]",
        "trip_count_quantiles": "Number of trips per truck"
    }
    plt.xlabel(xlabel_map[var], fontsize=10)
    plt.ylabel("Percentile [%]", fontsize=10)
    plt.title(f"CDF of {xlabel_map[var]} – {result['config']}", fontsize=12, pad=10)

    plt.grid(ls="--", alpha=0.35)
    plt.ylim(0, 100)
    plt.xlim(left=0)

    # ----------------------------------------
    # Annotations (like your example plot)
    # ----------------------------------------
    for p in annotate_quantiles:
        if p not in result[var]:
            continue  # skip if not in quantile dict

        x_val = result[var][p]
        y_val = p

        # guide lines
        plt.axvline(x_val, ymin=0, ymax=y_val/100, color="gray", ls="--", alpha=0.8)
        plt.axhline(y_val, xmin=0, xmax=x_val/np.max(vals), color="gray", ls="--", alpha=0.8)

        # label
        plt.text(
            x_val-1,
            y_val + 0.8,
            f"{p}%\n({x_val:.0f})",
            ha="center",
            va="bottom",
            fontsize=12,
            color="gray",
            fontweight="bold"
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir + var[:-10] +'_CDF_' + config)
    print(fig_dir + var[:-10] +'_CDF_' + config)
    plt.show()


def plot_all_configs(all_results, var="trip_time_quantiles", 
                     fig_dir="../Results/Figures/"):
    
    plt.figure(figsize=(7, 6))
    plt.style.use("seaborn-v0_8")

    # consistent colors across configs
    color_cycle = DTU_COLORS[:len(all_results)]


    # xlabel map
    xlabel_map = {
        "trip_time_quantiles": "Trip time [minutes]",
        "trip_distance_quantiles": "Trip distance [km]",
        "trip_count_quantiles": "Number of trips per truck"
    }
    xlab = xlabel_map[var]

    # ---------------------------------------------------
    # Loop through configs and add all CDF curves
    # ---------------------------------------------------
    for (cfg_name, res), col in zip(all_results.items(), color_cycle):
        q = np.array(list(res[var].keys()))
        vals = np.array(list(res[var].values()))

        # step only for trip counts
        if var == "trip_count_quantiles":
            plt.step(vals, q, where="post", lw=2, color=col, label=cfg_name)
        else:
            plt.plot(vals, q, lw=2, color=col, label=cfg_name)

    # ---------------------------------------------------
    # formatting
    # ---------------------------------------------------
    plt.xlabel(xlab, fontsize=13)
    plt.ylabel("Percentile [%]", fontsize=13)
    plt.title(f"CDF of {xlab} – All Configurations", fontsize=16, pad=10)

    plt.grid(ls="--", alpha=0.35)
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(title="Configurations", fontsize=11)

    plt.tight_layout()

    # save figure
    out_path = fig_dir + f"ALL_{var[:-10]}_CDF.png"
    plt.show()
    #plt.savefig(out_path, dpi=200)
    #plt.close()

    print(f"Saved: {out_path}")

def plot_all_configs_side_by_side(all_results, fig_dir="../Results/Figures/"):

    vars_to_plot = [
        ("trip_time_quantiles",    "Time"),
        ("trip_distance_quantiles","Distance"),
        ("trip_count_quantiles",   "Count")
    ]

    color_cycle = DTU_COLORS[:len(all_results)]

    # NEW: use constrained_layout
    fig, axes = plt.subplots(
        1, 3, figsize=(7, 3), sharey=True
    )
    plt.style.use("seaborn-v0_8")

    xlabel_map = {
        "trip_time_quantiles": "Trip time [minutes]",
        "trip_distance_quantiles": "Trip distance [km]",
        "trip_count_quantiles": "Number of trips per truck"
    }

    for ax, (var, title) in zip(axes, vars_to_plot):

        for (cfg_name, res), col in zip(all_results.items(), color_cycle):
            q = np.array(list(res[var].keys()))
            vals = np.array(list(res[var].values()))

            if var == "trip_count_quantiles":
                ax.step(vals, q, where="post", lw=1, color=col, label=cfg_name)
            else:
                ax.plot(vals, q, lw=1, color=col, label=cfg_name.replace('_dwell_','-'))

        ax.set_xlabel(xlabel_map[var])#, fontsize=18)
        ax.set_title(title)#, fontsize=25)
        ax.grid(ls="--", alpha=0.35)
        ax.set_ylim(0, 100)
        ax.set_xlim(left=0)

    axes[0].set_ylabel("Percentile [%]")#, fontsize=18)

    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(all_results),
        #fontsize=18,
        frameon=True
    )

    # Now set the shared title (this works perfectly)
    """
    fig.suptitle(
        "CDF Comparison of Trip Characteristics Across Configurations",
        fontsize=18,
        fontweight="bold"
    )
    """
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    out_path = fig_dir + "ALL_CONFIGS_side_by_side.png"
    fig.savefig(out_path, dpi=200)
    plt.show()

    print(f"Saved: {out_path}")






all_results = {}

os.makedirs("../Results/Quantiles", exist_ok=True)

for cfg in CONFIGS:
    name = cfg["name"]
    pkl_path = f"../Results/Quantiles/{name}_quantiles.pkl"

    # -------------------------------------------------
    # If pickle exists → load instead of recomputing
    # -------------------------------------------------
    if os.path.exists(pkl_path) and not FORCE_RECALCULATION:
        print(f"Loading cached quantiles for {name} ...")
        with open(pkl_path, "rb") as f:
            res = pickle.load(f)

    else:
        print(f"Computing quantiles for {name} ...")
        res = compute_trip_stats(name)

        with open(pkl_path, "wb") as f:
            pickle.dump(res, f)

    all_results[name] = res


    # -------------------------------------------------
    # Example: plot one configuration (fast)
    # -------------------------------------------------
    res = all_results[name]
    
    #plot_from_quantiles(res, var="trip_time_quantiles"      ,config=name)
    #plot_from_quantiles(res, var="trip_distance_quantiles"  ,config=name)
    #plot_from_quantiles(res, var="trip_count_quantiles"     ,config=name)

#plot_all_configs(all_results, var="trip_time_quantiles")
#plot_all_configs(all_results, var="trip_distance_quantiles")
#plot_all_configs(all_results, var="trip_count_quantiles")
plot_all_configs_side_by_side(all_results)
