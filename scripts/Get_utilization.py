import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# =========================================================
# CONFIG
# =========================================================
SB_CFG = "SB_dwell_5"
SB_OD_PATH = f"../Results/OD_results/{SB_CFG}/OD_{SB_CFG}.parquet"

TRANSFER_THRESHOLD = 999_000

T_START = pd.Timestamp("2025-06-16 00:00")
T_END   = pd.Timestamp("2025-06-21 00:00")

# =========================================================
# 1) LOAD DATA
# =========================================================
sb = pd.read_parquet(SB_OD_PATH)

# TruckID from TripID
sb["TruckID"] = sb["TripID"] // 10_000

# ---------------------------------------------------------
# Trip classification
# ---------------------------------------------------------
sb["is_origin_DK"] = sb["Origin"] <= TRANSFER_THRESHOLD
sb["is_dest_DK"]   = sb["Destination"] <= TRANSFER_THRESHOLD

sb["trip_type"] = np.select(
    [
        sb["is_origin_DK"] & sb["is_dest_DK"],
        sb["is_origin_DK"] & ~sb["is_dest_DK"],
        ~sb["is_origin_DK"] & sb["is_dest_DK"],
    ],
    [
        "DK_to_DK",
        "DK_to_OUT",
        "OUT_to_DK",
    ],
    default="OUT_to_OUT"
)

print("\nTrip type shares:")
print(sb["trip_type"].value_counts(normalize=True))

# ---------------------------------------------------------
# Trip → state mapping
# ---------------------------------------------------------
trip_state_map = {
    "DK_to_DK": "active",
    "DK_to_OUT": "unknown",
    "OUT_to_DK": "unknown",
    "OUT_to_OUT": "unknown",
}

sb["trip_state"] = sb["trip_type"].map(trip_state_map)

sb = sb.sort_values(["TruckID", "start_time"])

# =========================================================
# 2) BUILD RAW STATE INTERVALS
# =========================================================
records = []

for truck_id, g in tqdm(
    sb.groupby("TruckID", sort=False),
    total=sb["TruckID"].nunique(),
    desc="Building state timelines"
):
    g = g.sort_values("start_time").reset_index(drop=True)

    for i, row in g.iterrows():

        # --- Trip interval ---
        records.append({
            "TruckID": truck_id,
            "state": row["trip_state"],
            "t_start": row["start_time"],
            "t_end": row["end_time"],
        })

        # --- Gap to next trip ---
        if i < len(g) - 1:
            next_row = g.loc[i + 1]

            gap_start = row["end_time"]
            gap_end   = next_row["start_time"]

            if gap_end > gap_start:
                in_dk_before = row["is_dest_DK"]
                in_dk_after  = next_row["is_origin_DK"]

                gap_state = "rest" if (in_dk_before and in_dk_after) else "unknown"

                records.append({
                    "TruckID": truck_id,
                    "state": gap_state,
                    "t_start": gap_start,
                    "t_end": gap_end,
                })

state_ts = pd.DataFrame(records)

state_ts["t_start"] = pd.to_datetime(state_ts["t_start"])
state_ts["t_end"]   = pd.to_datetime(state_ts["t_end"])

# =========================================================
# 3) CLIP TO ANALYSIS WINDOW
# =========================================================
mask = (
    (state_ts["t_end"] > T_START) &
    (state_ts["t_start"] < T_END)
)
state_ts = state_ts.loc[mask].copy()

state_ts["t_start"] = state_ts["t_start"].clip(lower=T_START)
state_ts["t_end"]   = state_ts["t_end"].clip(upper=T_END)

# =========================================================
# 4) COMPLETE WITH BOUNDARY REST STATES
# =========================================================
completed = []

for truck_id, g in state_ts.groupby("TruckID", sort=False):

    g = g.sort_values("t_start").reset_index(drop=True)

    # ---- start boundary ----
    if g.loc[0, "t_start"] > T_START:
        completed.append({
            "TruckID": truck_id,
            "state": "rest",
            "t_start": T_START,
            "t_end": g.loc[0, "t_start"],
        })

    # ---- observed states ----
    completed.extend(g.to_dict("records"))

    # ---- end boundary ----
    if g.loc[len(g) - 1, "t_end"] < T_END:
        completed.append({
            "TruckID": truck_id,
            "state": "rest",
            "t_start": g.loc[len(g) - 1, "t_end"],
            "t_end": T_END,
        })

state_ts = pd.DataFrame(completed)

# =========================================================
# 5) SANITY CHECKS
# =========================================================
assert (state_ts["t_start"] >= T_START).all()
assert (state_ts["t_end"]   <= T_END).all()
assert (state_ts["t_end"] > state_ts["t_start"]).all()

# No gaps per truck
ok = (
    state_ts.sort_values(["TruckID", "t_start"])
            .groupby("TruckID")
            .apply(lambda x: (
                x["t_start"].iloc[1:].values ==
                x["t_end"].iloc[:-1].values
            ).all())
)
assert ok.all()
##
# =========================================================
# 6) UTILIZATION ANALYSIS
# =========================================================
state_ts["duration_h"] = (
    state_ts["t_end"] - state_ts["t_start"]
).dt.total_seconds() / 3600

# Aggregate per truck and state
truck_state_time = (
    state_ts
    .groupby(["TruckID", "state"])["duration_h"]
    .sum()
    .unstack(fill_value=0)
)

# Time observed in Denmark
truck_state_time["time_in_DK"] = (
    truck_state_time.get("active", 0) +
    truck_state_time.get("rest", 0)
)

# Utilization conditional on being in Denmark
truck_state_time["utilization"] = (
    truck_state_time["active"] / truck_state_time["time_in_DK"]
)

truck_state_time = truck_state_time.replace([np.inf, np.nan], 0)

# =========================================================
# 7) SUMMARY OUTPUT
# =========================================================
print("\nNumber of trucks:", truck_state_time.shape[0])
print("Trucks never observed in DK:",
      (truck_state_time["time_in_DK"] == 0).sum())

print("\nUtilization summary (conditional on DK):")
print(
    truck_state_time.loc[truck_state_time["time_in_DK"] > 0, "utilization"]
    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
)

print("\nTotal time by state (hours):")
print(
    state_ts.groupby("state")["duration_h"]
            .sum()
            .sort_values(ascending=False)
)
