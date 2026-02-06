import dask.dataframe as dd
import datashader as ds
import datashader.transfer_functions as tf
import colorcet as cc
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns
from dask.diagnostics import ProgressBar
import pytz
import glob
import os
from joblib import Parallel, delayed
from tqdm.auto import tqdm
# Path to your Parquet file
filename = "C:/Users/b306630/Desktop/Master Thesis/Data/parquet_trucks"
Num_rows = True
Datespan = True
density_plot = True

# ------------------------------------------------------------------------------------
# 1. Number of rows in data (total GPS-points)
# ------------------------------------------------------------------------------------
if Num_rows:
    df = dd.read_parquet(filename, columns=["TruckID"])  # load minimal column
    row_count = df.shape[0].compute()
    print(f"Total rows: {row_count:,}")  # formatted with commas
    
# ------------------------------------------------------------------------------------
# 2. histrogram of clases
# ------------------------------------------------------------------------------------

RAW_PATH = "C:/Users/b306630/Desktop/Master Thesis/Data/parquet_trucks/truck_*.parquet"
raw_files = glob.glob(RAW_PATH)

print(f"Found {len(raw_files)} trucks.")
# Helper: safely extract per-truck metadata

def extract_metadata(path):
    truck_id = int(os.path.basename(path).split("_")[1].split(".")[0])

    # Load minimal columns
    df = pd.read_parquet(path, columns=["time", "weightLimits", "CO2EmissionClass"])

    # Drop NaNs (if any)
    w = df["weightLimits"].dropna().unique()
    c = df["CO2EmissionClass"].dropna().unique()

    # Hard checks
    if len(w) != 1 or len(c) != 1:
        # Filter to the desired window
        mask = (df["time"] >= "2025-06-15") & (df["time"] <= "2025-06-22")
        df = df.loc[mask]

        if df.empty:
            return truck_id, None, None

        # Drop NaNs (if any)
        w = df["weightLimits"].dropna().unique()
        c = df["CO2EmissionClass"].dropna().unique()

    return truck_id, w[0], c[0]


# Extract values for all trucks
results = Parallel(n_jobs=8)(
    delayed(extract_metadata)(f) for f in tqdm(raw_files)
)

df_meta = pd.DataFrame(results, columns=["TruckID", "WeightLimit", "CO2Class"])

print(df_meta.head())
print(df_meta.describe(include="all"))


# Print all trucks where metadata could not be determined
invalid_trucks = df_meta[(df_meta["WeightLimit"].isna()) | (df_meta["CO2Class"].isna())]

if invalid_trucks.empty:
    print("\nNo trucks returned None for metadata. All trucks validated successfully.")
else:
    print("\n❗ Trucks with missing or inconsistent metadata (returned None):")
    print(invalid_trucks.to_string(index=False))
    print(f"\nTotal invalid trucks: {len(invalid_trucks)}")


# Map weight limits to official weight classes

def map_weight_class(w):
    if 1200 <= w <= 1799:
        return "12,000–17,999 kg"
    elif 1800 <= w <= 3200:
        return "18,000–32,000 kg"
    elif w > 3200:
        return "Over 32,000 kg"
    else:
        return None

df_meta["WeightClass"] = df_meta["WeightLimit"].apply(map_weight_class)

# Distribution
class_counts = df_meta["WeightClass"].value_counts().reindex([
    "12,000–17,999 kg",
    "18,000–32,000 kg",
    "Over 32,000 kg"
])

# Plot
plt.figure(figsize=(10, 5))
class_counts.plot(kind="bar")

plt.title("Distribution of Weight Classes Across Trucks")
plt.xlabel("Weight Class")
plt.ylabel("Number of Trucks")
plt.grid(axis="y", alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('C:/Users/b306630/Desktop/Mater thesis back uup/Results/Figures/Weight_Class.png')
plt.show()

plt.figure(figsize=(10, 5))

counts = df_meta["CO2Class"].value_counts().sort_index()
counts.plot(kind="bar")

plt.title("Distribution of CO₂ Emission Class Across Trucks")
plt.xlabel("CO₂ Emission Class")
plt.ylabel("Number of Trucks")
plt.grid(axis="y", alpha=0.3)

# Force x-ticks to be integers
plt.xticks(
    ticks=range(len(counts)),
    labels=counts.index.astype(int),
    rotation=0
)

plt.tight_layout()
plt.savefig('C:/Users/b306630/Desktop/Mater thesis back uup/Results/Figures/CO2_Emission_Class.png')
plt.show()


# ------------------------------------------------------------------------------------
# 3. Density plot
# ------------------------------------------------------------------------------------
if density_plot:

    ddf = dd.read_parquet(
        filename,
        columns=["longitude", "latitude", "time"]
    )

    # Ensure datetime + timezone
    
    DK = pytz.timezone("Europe/Copenhagen")
    start_time = DK.localize(pd.Timestamp("2025-06-16 00:00:00"))
    end_time   = DK.localize(pd.Timestamp("2025-06-21 00:00:00"))
    # Convert to Denmark time
    ddf["tme"] = ddf["time"].dt.tz_convert(DK)
    
    # Apply time filter (THIS is what you asked about)
    ddf = ddf[
        (ddf["time"] >= start_time) &
        (ddf["time"] <  end_time)
    ]


    def add_utm_partition(df):
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        
        # Convert from microdegrees → degrees
        lon = df["longitude"].to_numpy(dtype="float64") / 1e6
        lat = df["latitude"].to_numpy(dtype="float64") / 1e6
        mask = np.isfinite(lon) & np.isfinite(lat)
        x = np.full(lon.shape, np.nan, dtype="float64")
        y = np.full(lat.shape, np.nan, dtype="float64")
        if mask.any():
            x_valid, y_valid = transformer.transform(lon[mask], lat[mask])
            x[mask] = x_valid
            y[mask] = y_valid
        return pd.DataFrame({"x": x, "y": y}, index=df.index)
    
    meta = {"x": "f8", "y": "f8"}
    ddf_xy = ddf.map_partitions(add_utm_partition, meta=meta)
    
    x_range = (425000.0, 900000.0)
    y_range = (6020000.0, 6440000.0)
    
    canvas = ds.Canvas(
        plot_width=1130,
        plot_height=1000,
        x_range=x_range,
        y_range=y_range
    )
    
    agg = canvas.points(ddf_xy, "x", "y", ds.count())
    
    img = tf.shade(agg, cmap=cc.fire, how="log", alpha=255)
    pil_img = img.to_pil()
    
    # ----------------------------
    # Matplotlib plot with colorbar
    # ----------------------------

    with ProgressBar():
        agg_np = agg.compute().values

    vmin = np.nanmin(agg_np[agg_np > 0])
    vmax = np.nanmax(agg_np)
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    
    cmap = cc.cm.fire
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Black background only inside axes
    ax.set_facecolor("black")
    
    # Show Datashader image (no origin flip)
    ax.imshow(pil_img, extent=[*x_range, *y_range])
    
    # Keep correct aspect ratio
    ax.set_aspect("equal")
    # REMOVE axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")   # remove label
    ax.set_ylabel("")   # remove label
    ax.set_title("GPS Point Density",fontsize = 20)
    # Colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.88, aspect=10)
    cbar.set_label("Number of points (log scale)")
    
    plt.tight_layout()
    plt.grid(visible=False)
    plt.show()
    
    fig.savefig("C:/Users/b306630/Desktop/Mater thesis back uup/Results/Figures/gps_density_denmark_with_colorbar.png", dpi=300)
    
     
# ------------------------------------------------------------------------------------
# 4. Category values
# ------------------------------------------------------------------------------------
if False:
    # Read only needed columns
    ddf = dd.read_parquet(
        filename,
        columns=['vihicle_class', 'euro_value', 'co2_emission_class']
    )
    
    # Compute unique values for each column
    unique_vals = {
        col: ddf[col].unique().compute().tolist()
        for col in ['vihicle_class', 'euro_value', 'co2_emission_class']
    }
    
    print(unique_vals)
        
# ------------------------------------------------------------------------------------
# 5.Temporal plot
# ------------------------------------------------------------------------------------

# Use Parquet if possible (faster for large datasets)
df = dd.read_parquet(filename)
DK = pytz.timezone("Europe/Copenhagen")

# If `time` is currently UTC:
df["event_ts"] = df["time"].dt.tz_convert(DK)

start_time = DK.localize(pd.Timestamp("2025-06-16 00:00:00"))
end_time   = DK.localize(pd.Timestamp("2025-06-21 00:00:00"))
df = df[(df['event_ts'] >= start_time) & (df['event_ts'] < end_time)]

df['hour'] = df['event_ts'].dt.hour
df['weekday'] = df['event_ts'].dt.weekday  # Monday=0
# Count number of events per hour and weekday
pivot = df.groupby(['hour', 'weekday'])['TruckID'].count().compute()

# Reshape to 2D for heatmap
pivot_table = pivot.unstack(level='weekday')

weekday_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

plt.figure(figsize=(10,6))
ax = sns.heatmap(pivot_table, cmap='viridis', annot=False)
# Set weekday labels
ax.set_xticks(np.arange(len(weekday_labels)) + 0.5)
ax.set_xticklabels(weekday_labels, rotation=0, ha='center')
plt.xlabel("Weekday")
plt.ylabel("Hour of Day")
plt.title("Truck Activity by Hour and Weekday")
plt.tight_layout()
plt.savefig("C:/Users/b306630/Desktop/Mater thesis back uup/Results/Figures/temporal.png",dpi=300)
plt.show()



plt.figure(figsize=(10,4))
ax = sns.heatmap(pivot_table.T, cmap='viridis', annot=False)
# Set weekday labels
ax.set_yticks(np.arange(len(weekday_labels)) + 0.5)
ax.set_yticklabels(weekday_labels, rotation=0)
plt.xlabel("Hour of Day")
plt.ylabel("Weekday")
plt.title("Truck Activity by Hour and Weekday")
plt.tight_layout()
plt.savefig("C:/Users/b306630/Desktop/Mater thesis back uup/Results/Figures/temporal_Transpose.png",dpi=300)
plt.show()



# Extract date only (no time)
df['date'] = df['event_ts'].dt.date
# Count events per date
daily_counts = df.groupby('date')['TruckID'].count().compute()
daily_unique_trucks = df.groupby('date')['TruckID'].nunique().compute()

plt.figure(figsize=(12,6))
plt.plot(daily_counts.index, daily_counts.values, marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Number of GPS points')
plt.title('Daily Truck Activity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


 


