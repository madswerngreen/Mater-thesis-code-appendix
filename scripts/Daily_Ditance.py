import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import glob
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

# ---------------------------------------------------------
# Read a single truck and return daily km list
# ---------------------------------------------------------
def read_truck(truck_id):
    df = pd.read_parquet(
        f"C:/Users/b306630/OneDrive - Vejdirektoratet/Master Thesis/Data/parquet_trucks/truck_{truck_id}.parquet"
    )

    # coords → meters
    df["latitude"]  /= 1e6
    df["longitude"] /= 1e6
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    ).to_crs(25832)

    # time UTC → DK
    gdf["time"] = (
        pd.to_datetime(gdf["time"], utc=True)
          .dt.tz_convert("Europe/Copenhagen")
    ).rename("time")
    gdf["time_delta"] = (gdf["time"] - gdf["time"].shift()).dt.total_seconds()
    gdf = gdf.set_index("time")

    # target days
    days = ["2025-06-15", "2025-06-16", "2025-06-18"]
    days = pd.to_datetime(days).tz_localize("Europe/Copenhagen")

    # distance between points
    gdf["dist_m"] = gdf.geometry.distance(gdf.geometry.shift()).fillna(0)
    gdf.loc[gdf["dist_m"] / gdf["time_delta"] > 100 / 3.6 ,"dist_m"] = 0
    
    # sum per day (km)
    return [
        gdf[gdf.index.normalize() == d.normalize()]["dist_m"].sum() / 1000
        for d in days
    ]


# ---------------------------------------------------------
# Wrapper for parallel
# ---------------------------------------------------------
def get_dist(truck_id):
    try:
        return truck_id, read_truck(truck_id)
    except Exception:
        return truck_id, None

def compute_daily_km_quantiles(vals):
    quantiles = np.arange(0, 100.25, 0.25)
    return {
        float(q): float(np.percentile(vals, q))
        for q in quantiles
    }

def plot_daily_km_cdf(
    quantile_dict,
    annotate_quantiles=[25, 50, 75, 95],
    fig_dir="../Results/Figures/"
):
    q = np.array(list(quantile_dict.keys()))
    vals = np.array(list(quantile_dict.values()))

    plt.figure(figsize=(7,6))
    plt.style.use("seaborn-v0_8")

    main_color = "#003D73"   # DTU blue
    shade_color = "#AEC7E8"  # light shade

    # Main curve
    plt.plot(vals, q, lw=2, color=main_color, label="Empirical CDF")
    plt.fill_between(vals, q, color=shade_color, alpha=0.25)

    plt.xlabel("Daily distance driven [km]", fontsize=11)
    plt.ylabel("Percentile [%]", fontsize=11)
    plt.title("CDF of Daily Distance", fontsize=14, pad=10)

    plt.grid(ls="--", alpha=0.35)
    plt.ylim(0, 100)
    plt.xlim(left=0)

    # Annotate quantiles (matching your style)
    for p in annotate_quantiles:
        if p not in quantile_dict:
            continue
        x_v = quantile_dict[p]
        y_v = p

        # Guide lines
        plt.axvline(x_v, ymin=0, ymax=y_v/100, color="gray", ls="--", alpha=0.8)
        plt.axhline(y_v, xmin=0, xmax=x_v/np.max(vals), color="gray", ls="--", alpha=0.8)

        # Numeric label
        plt.text(
            x_v, y_v + 1,
            f"{p}%\n({x_v:.0f} km)",
            ha="center",
            va="bottom",
            fontsize=12,
            color="gray",
            fontweight="bold"
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir + "DailyKmCDF.png", dpi=200)
    plt.show()

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    files = glob.glob(
        "C:/Users/b306630/OneDrive - Vejdirektoratet/Master Thesis/Data/parquet_trucks/truck_*.parquet"
    )
    truck_ids = sorted(int(f.split("_")[-1].split(".")[0]) for f in files)

    with tqdm_joblib(tqdm(total=len(truck_ids), desc="distance")):
        results = Parallel(n_jobs=8, backend="loky")(
            delayed(get_dist)(tid)
            for tid in truck_ids
        )

    # Clean and store in dict {truck_id: [km15, km16, km18]}
    km_dict = {tid: vals for tid, vals in results if vals is not None}
    
    # Flatten to one list of km values
    vals = np.array([x for lst in km_dict.values() for x in lst if x > 0])

    quantiles = compute_daily_km_quantiles(vals)
    
    plot_daily_km_cdf(quantiles)
