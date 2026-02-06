import geopandas as gpd
import os

# ==================================================
# CONFIG
# ==================================================
CFG = "SB_dwell_5"
TRUCK_ID = 16198
CRS_EPSG = 25832
TRIP_FILE = f"../Data/truck_trips/{CFG}/truck_{TRUCK_ID}_trips.parquet"
OUT_GPKG = "../Results/Truck_for_QGIS.gpkg"


# ==================================================
# Load data
# ==================================================
if not os.path.exists(TRIP_FILE):
    raise FileNotFoundError(f"File not found: {TRIP_FILE}")

df = gpd.read_parquet(TRIP_FILE)

if df.empty:
    raise ValueError(f"No data found for TruckID {TRUCK_ID}")

print(f"Loaded {len(df)} GPS points and {df['TripID'].nunique()} trips.")


# ==================================================
# Convert timestamp index from UTC → Denmark
# ==================================================
# (must be timezone-aware first, which it is)
df.index = df.index.tz_convert("Europe/Copenhagen")


# ==================================================
# Ensure CRS is set
# ==================================================
df = df.set_geometry("geometry")
df = df.set_crs(epsg=CRS_EPSG)


# ==================================================
# Save to GPKG
# ==================================================
if os.path.exists(OUT_GPKG):
    os.remove(OUT_GPKG)

df.to_file(OUT_GPKG, layer="points", driver="GPKG")

print(f"✓ Written: {OUT_GPKG} (with timestamps in Denmark local time)")
