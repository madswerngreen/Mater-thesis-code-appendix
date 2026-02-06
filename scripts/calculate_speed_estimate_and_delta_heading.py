import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

INPUT_DIR  = "C:/Users/b306630/Desktop/Master Thesis/Data/parquet_trucks/"
OUTPUT_DIR = "../Data/parquet_trucks_enriched/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_file(path):
    try:
        truck_id = os.path.basename(path).split("_")[1].split(".")[0]

        # -------------------------------------------------
        # Load parquet file
        # -------------------------------------------------
        df = pd.read_parquet(path)

        # Ensure time is naive timestamp
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        df = df.sort_values("time")

        # -------------------------------------------------
        # Convert microdegrees → degrees
        # -------------------------------------------------
        df["latitude_deg"]  = df["latitude"]  / 1e6
        df["longitude_deg"] = df["longitude"] / 1e6

        # Project to EPSG:25832
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude_deg"], df["latitude_deg"]),
            crs="EPSG:4326"
        ).to_crs(25832)

        # -------------------------------------------------
        # Compute dx, dy, distance
        # -------------------------------------------------
        x = gdf.geometry.x.values
        y = gdf.geometry.y.values

        dx = np.insert(np.diff(x), 0, np.nan)
        dy = np.insert(np.diff(y), 0, np.nan)

        gdf["dx"] = dx
        gdf["dy"] = dy
        gdf["dist"] = np.sqrt(dx**2 + dy**2)

        # -------------------------------------------------
        # Time delta + speed estimate
        # -------------------------------------------------
        dt = gdf["time"].diff().dt.total_seconds().values
        dt[dt <= 0] = np.nan
        gdf["speed_estimate"] = (gdf["dist"].values / dt) * 3.6
        
        # speed measuere used
        gdf['speed'] = gdf['speed'] * 0.036
        # -------------------------------------------------
        # Heading + delta heading
        # -------------------------------------------------
        heading = np.degrees(np.arctan2(dx, dy))
        heading = (heading + 360) % 360
        gdf["heading_estimate"] = heading

        delta_heading = ((heading - np.roll(heading, 1) + 180) % 360) - 180
        delta_heading[0] = np.nan
        gdf["delta_heading"] = delta_heading

        # -------------------------------------------------
        # Rolling features → requires datetime index
        # -------------------------------------------------
        # remove unlikely speeds
        gdf = gdf[gdf["speed"] <= 130].copy()
        gdf = gdf.set_index("time")
        
        # 1-minute rolling variance of delta_heading
        gdf["delta_heading_var"] = (
            gdf["delta_heading"]
            .rolling("1min", center=True, min_periods=2)
            .var(ddof=0)
        )


        # 1-minute rolling mean of speed estimate
        gdf["speed_estimate_mean"] = (
            gdf["speed"]
            .rolling("1min", center=True, min_periods=1)
            .mean()
        )

        # -------------------------------------------------
        # Save output
        # -------------------------------------------------
        out_path = os.path.join(OUTPUT_DIR, f"truck_{truck_id}.parquet")
        gdf_export = gdf[["delta_heading_var", "speed_estimate_mean"]].copy()
        gdf_export.to_parquet(out_path, index=True)   # keep time index
    except Exception as e:
        print(f"❌ Error in {path}: {e}")



# -------------------------------------------------
# MAIN (parallel)
# -------------------------------------------------
if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "truck_*.parquet")))
    print(f"Processing {len(files)} trucks in parallel...")

    # Parallel loop with progress bar
    with tqdm_joblib(tqdm(total=len(files), desc="Processing trucks")):
        Parallel(n_jobs=8, backend="loky")(
            delayed(process_file)(f) for f in files
        )

    print("✅ Finished!")
