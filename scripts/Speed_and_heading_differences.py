import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

INPUT_DIR  = "C:/Users/b306630/OneDrive - Vejdirektoratet/Skrivebord/Master Thesis/Data/parquet_trucks/"

# -----------------------------------------------------------
# Your existing process_file function
# -----------------------------------------------------------
def process_file(path):
    try:
        truck_id = os.path.basename(path).split("_")[1].split(".")[0]

        df = pd.read_parquet(path)

        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        df = df.sort_values("time")

        df["latitude_deg"]  = df["latitude"]  / 1e6
        df["longitude_deg"] = df["longitude"] / 1e6

        import geopandas as gpd
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude_deg"], df["latitude_deg"]),
            crs="EPSG:4326"
        ).to_crs(25832)

        x = gdf.geometry.x.values
        y = gdf.geometry.y.values

        dx = np.insert(np.diff(x), 0, np.nan)
        dy = np.insert(np.diff(y), 0, np.nan)

        gdf["dx"] = dx
        gdf["dy"] = dy
        gdf["dist"] = np.sqrt(dx**2 + dy**2)

        dt = gdf["time"].diff().dt.total_seconds().values
        dt[dt <= 0] = np.nan
        gdf["speed_estimate"] = (gdf["dist"].values / dt) * 3.6
        
        gdf["speed"] = gdf["speed"] * 0.036
        
        heading = np.degrees(np.arctan2(dx, dy))
        heading = (heading + 360) % 360
        gdf["heading_estimate"] = heading
        
        gdf.loc[gdf["speed"] == -1, "speed"] = np.nan
        gdf.loc[gdf["heading"] == -1, "heading"] = np.nan
        speed_diff = gdf["speed"] - gdf["speed_estimate"]
        speed_diff = speed_diff.clip(lower=-130, upper=130).astype("float32")

        heading_diff = (gdf["heading"] / 10 - gdf["heading_estimate"]).astype("float32")
        heading_diff = ((heading_diff + 180) % 360) - 180
        
        return speed_diff.values, heading_diff.values

    except Exception as e:
        print(f"❌ Error in {path}: {e}")
        return np.array([]), np.array([])

# -----------------------------------------------------------
# Run parallel processing
# -----------------------------------------------------------
paths = glob.glob(os.path.join(INPUT_DIR, "*.parquet"))
print(f"Found {len(paths)} files")

with tqdm_joblib(tqdm(desc="Processing trucks", total=len(paths))):
    results = Parallel(n_jobs=-1)(
        delayed(process_file)(p) for p in paths
    )

# -----------------------------------------------------------
# Combine all diffs
# -----------------------------------------------------------
all_speed_diff = np.concatenate([r[0] for r in results])
all_heading_diff = np.concatenate([r[1] for r in results])

# Remove NaN
all_speed_diff = all_speed_diff[~np.isnan(all_speed_diff)]
all_heading_diff = all_heading_diff[~np.isnan(all_heading_diff)]

print("Total points:", len(all_speed_diff), "(speed diff)")
print("Total points:", len(all_heading_diff), "(heading diff)")

# -----------------------------------------------------------
# Compute quantiles (0%, 1%, 2%, ..., 100%)
# -----------------------------------------------------------
q_vals = np.linspace(0, 1, 101)

speed_quantiles = np.quantile(all_speed_diff, q_vals)
heading_quantiles = np.quantile(all_heading_diff, q_vals)


# -----------------------------------------------------------
# Plot CDF
# -----------------------------------------------------------
def plot_cdf_quantiles(q_vals, q_data, title, xlabel, path):
    plt.figure(figsize=(5, 4))
    plt.plot(q_data, q_vals, linewidth=2)   # x = quantiles, y = CDF
    plt.grid(alpha=0.3)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(path)
    plt.show()


plot_cdf_quantiles(
    q_vals, 
    speed_quantiles, 
    "CDF of Speed Difference (Quantile Smoothed)", 
    "Speed difference (km/h)",
    "C:/Users/b306630/OneDrive - Vejdirektoratet/Skrivebord/Mater thesis back uup/Results/Figures/CDF_difference_speed.png"
)

plot_cdf_quantiles(
    q_vals, 
    heading_quantiles, 
    "CDF of Heading Difference (Quantile Smoothed)", 
    "Heading difference (deg)",
    "C:/Users/b306630/OneDrive - Vejdirektoratet/Skrivebord/Mater thesis back uup/Results/Figures/CDF_difference_heading.png"
)
