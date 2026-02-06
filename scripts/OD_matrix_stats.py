import pandas as pd
import numpy as np
import pytz
import os

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DK = pytz.timezone("Europe/Copenhagen")

SB_CFG = "SB_dwell_5"
SB_OD_PATH = f"../Results/OD_results/{SB_CFG}/OD_{SB_CFG}.parquet"

GMM_PATH = r"C:/Users/b306630/Desktop/Master Thesis/Data/FreightTripMatrixWD.csv"

OUT_DIR = "../Results/OD_diagnostics/"
os.makedirs(OUT_DIR, exist_ok=True)

n_weekday = 3

# Week window in DK time (your setup)
start = DK.localize(pd.Timestamp("2025-06-16"))
end   = DK.localize(pd.Timestamp("2025-06-20"))  # we'll remove Wed => divide by n_weekday for daily

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def zone_map(x, level=0):
    """Aggregate zone IDs to coarser levels. level=0 -> municipality level as you used."""
    tmp = 3 - level
    return (int(x) // (10**tmp)) * (10**tmp)

def top_producers_attractors(mat: pd.DataFrame, topn=10):
    """
    Producers: row sums (outgoing)
    Attractors: col sums (incoming)
    Returns two DataFrames: top producers, top attractors
    """
    prod = mat.sum(axis=1).sort_values(ascending=False).head(topn)
    attr = mat.sum(axis=0).sort_values(ascending=False).head(topn)

    prod_df = prod.reset_index()
    prod_df.columns = ["Zone", "Trips_out"]

    attr_df = attr.reset_index()
    attr_df.columns = ["Zone", "Trips_in"]

    return prod_df, attr_df

def melt_matrix(mat: pd.DataFrame, value_name="val"):
    """Convert OD matrix to long format with (Origin, Destination, value)."""
    long = mat.stack().reset_index()
    long.columns = ["Origin", "Destination", value_name]
    return long

# ---------------------------------------------------------
# 1) Load SB-5 OD table and build daily municipality OD matrix
# ---------------------------------------------------------
sb = pd.read_parquet(SB_OD_PATH)

# tz convert (your OD parquet stores times; safe even if already tz-aware UTC)
sb["start_time"] = pd.to_datetime(sb["start_time"], utc=True).dt.tz_convert(DK)
sb["end_time"]   = pd.to_datetime(sb["end_time"],   utc=True).dt.tz_convert(DK)

# basic cleaning
sb = sb[sb["end_time"] >= sb["start_time"]]
sb = sb[(sb["start_time"] <= end) & (sb["end_time"] >= start)]
sb = sb[sb["start_time"].dt.weekday != 2]  # remove Wed
sb = sb[sb["end_time"].dt.weekday   != 2]

# keep valid OD
sb = sb[sb["Origin"].notna() & sb["Destination"].notna()].copy()
sb["Origin"] = sb["Origin"].astype(int).apply(zone_map)
sb["Destination"] = sb["Destination"].astype(int).apply(zone_map)

# REMOVE non-DK OD pairs
sb = sb[(sb["Origin"] < 900000) & (sb["Destination"] < 900000)]

# municipality OD counts and convert to daily (n weekdays)
SB_mat = (sb.groupby(["Origin", "Destination"]).size() / n_weekday).unstack(fill_value=0)

# ---------------------------------------------------------
# 2) Load GMM and build municipality OD matrix
# ---------------------------------------------------------
gmm = pd.read_csv(GMM_PATH)
gmm = gmm.rename(columns={"FromZoneID": "Origin", "ToZoneID": "Destination", "Val": "Trips"})

# map foreign/transfer zones
# REMOVE non-DK OD pairs entirely
gmm = gmm[(gmm["Origin"] < 900000) & (gmm["Destination"] < 900000)]


gmm["Origin"] = gmm["Origin"].astype(int).apply(zone_map)
gmm["Destination"] = gmm["Destination"].astype(int).apply(zone_map)

GMM_mat = gmm.groupby(["Origin", "Destination"])["Trips"].sum().unstack(fill_value=0)

# ---------------------------------------------------------
# 3) Align matrices to common zone universe
# ---------------------------------------------------------
all_zones = sorted(set(SB_mat.index).union(SB_mat.columns).union(GMM_mat.index).union(GMM_mat.columns))

SB_mat = SB_mat.reindex(index=all_zones, columns=all_zones, fill_value=0.0)
GMM_mat = GMM_mat.reindex(index=all_zones, columns=all_zones, fill_value=0.0)

# ---------------------------------------------------------
# 4) Top 10 Producers / Attractors
# ---------------------------------------------------------
sb_prod, sb_attr = top_producers_attractors(SB_mat, topn=10)
gmm_prod, gmm_attr = top_producers_attractors(GMM_mat, topn=10)

print("\n=== Top 10 Producers (SB-5, daily) ===")
print(sb_prod.to_string(index=False))

print("\n=== Top 10 Attractors (SB-5, daily) ===")
print(sb_attr.to_string(index=False))

print("\n=== Top 10 Producers (GMM) ===")
print(gmm_prod.to_string(index=False))

print("\n=== Top 10 Attractors (GMM) ===")
print(gmm_attr.to_string(index=False))

# Save
sb_prod.to_csv(os.path.join(OUT_DIR, "top10_producers_SB5_daily.csv"), index=False)
sb_attr.to_csv(os.path.join(OUT_DIR, "top10_attractors_SB5_daily.csv"), index=False)
gmm_prod.to_csv(os.path.join(OUT_DIR, "top10_producers_GMM.csv"), index=False)
gmm_attr.to_csv(os.path.join(OUT_DIR, "top10_attractors_GMM.csv"), index=False)

# ---------------------------------------------------------
# 5) Mismatch per OD cell + top corridors
# ---------------------------------------------------------
# Absolute mismatch: SB - GMM
diff = SB_mat - GMM_mat

# Long format (absolute)
diff_long = melt_matrix(diff, value_name="diff_abs")

# Add SB and GMM values for context
sb_long  = melt_matrix(SB_mat, value_name="SB_daily")
gmm_long = melt_matrix(GMM_mat, value_name="GMM")

merged = diff_long.merge(sb_long, on=["Origin", "Destination"]).merge(gmm_long, on=["Origin", "Destination"])

# Remove zero-zero cells to keep it readable
merged_nonzero = merged[(merged["SB_daily"] > 0) | (merged["GMM"] > 0)].copy()

# Top 10 overpredicted (SB much larger than GMM)
top_over = merged_nonzero.sort_values("diff_abs", ascending=False).head(10)

# Top 10 underpredicted (SB much smaller than GMM)
top_under = merged_nonzero.sort_values("diff_abs", ascending=True).head(10)

print("\n=== Top 10 Under-predicted corridors (SB-5 - GMM) ===")
print(top_over[["Origin","Destination","SB_daily","GMM","diff_abs"]].to_string(index=False))

print("\n=== Top 10 Over-predicted corridors (SB-5 - GMM) ===")
print(top_under[["Origin","Destination","SB_daily","GMM","diff_abs"]].to_string(index=False))

top_over.to_csv(os.path.join(OUT_DIR, "top10_overpredicted_corridors_abs.csv"), index=False)
top_under.to_csv(os.path.join(OUT_DIR, "top10_underpredicted_corridors_abs.csv"), index=False)

# ---------------------------------------------------------
# 6) Optional: Relative mismatch (useful if you care about proportional differences)
#     rel = (SB - GMM) / (GMM + eps)
# ---------------------------------------------------------
eps = 1e-6
merged_nonzero["diff_rel"] = (merged_nonzero["SB_daily"] - merged_nonzero["GMM"]) / (merged_nonzero["GMM"] + eps)

top_over_rel = merged_nonzero.sort_values("diff_rel", ascending=False).head(10)
top_under_rel = merged_nonzero.sort_values("diff_rel", ascending=True).head(10)

top_over_rel.to_csv(os.path.join(OUT_DIR, "top10_overpredicted_corridors_rel.csv"), index=False)
top_under_rel.to_csv(os.path.join(OUT_DIR, "top10_underpredicted_corridors_rel.csv"), index=False)

print("\n✓ Saved outputs to:", os.path.abspath(OUT_DIR))

# ---------------------------------------------------------
# TOTAL NUMBER OF TRIPS (DAILY)
# ---------------------------------------------------------
total_SB5 = SB_mat.values.sum()
total_GMM = GMM_mat.values.sum()

print("\n=== Total daily number of trips ===")
print(f"SB-5 (daily): {total_SB5:,.0f}")
print(f"GMM  (daily): {total_GMM:,.0f}")



# ---------------------------------------------------------
# REMOVE intra-zonal (diagonal) trips
# ---------------------------------------------------------
merged_inter = merged_nonzero[
    merged_nonzero["Origin"] != merged_nonzero["Destination"]
].copy()

# Top 10 overpredicted inter-zonal corridors
top_over = (
    merged_inter
    .sort_values("diff_abs", ascending=False)
    .head(10)
)

# Top 10 underpredicted inter-zonal corridors
top_under = (
    merged_inter
    .sort_values("diff_abs", ascending=True)
    .head(10)
)

print("\n=== Top 10 Over-predicted INTER-ZONAL corridors (SB-5 - GMM) ===")
print(top_over[["Origin","Destination","SB_daily","GMM","diff_abs"]]
      .to_string(index=False))

print("\n=== Top 10 Under-predicted INTER-ZONAL corridors (SB-5 - GMM) ===")
print(top_under[["Origin","Destination","SB_daily","GMM","diff_abs"]]
      .to_string(index=False))

