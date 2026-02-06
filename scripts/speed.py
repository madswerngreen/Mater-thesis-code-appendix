import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

plt.style.use("seaborn-v0_8")

CONFIGS = [
    "SB_dwell_5",
    "SB_dwell_15",
    "HIS_dwell_5",
    "HIS_dwell_15",
]

FIG_DIR = "../Results/Figures/"
os.makedirs(FIG_DIR, exist_ok=True)

def plot_speed_cdf_for_config(cfg):
    print(f"\nProcessing {cfg} ...")

    # Load lazy Dask DataFrame
    path = f"C:/Users/b306630/Desktop/Mater thesis back uup/Data/truck_trips/{cfg}/truck_*_trips.parquet"
    ddf = dd.read_parquet(path)

    # Compute metadata
    n_trips  = ddf["TripID"].nunique().compute()
    n_trucks = ddf["TruckID"].nunique().compute()
    n_points = len(ddf)

    print(f" - Points: {n_points:,}")
    print(f" - Trucks: {n_trucks:,}")
    print(f" - Trips:  {n_trips:,}")

    # Percentiles
    percentiles = np.linspace(0, 1, 101)
    quantile_values = ddf["speed_estimate"].quantile(percentiles, method = 'tdigest').compute()

    x = quantile_values.values
    y = percentiles * 100   # convert to %

    # --- Plot ---
    plt.figure(figsize=(12, 7))

    plt.plot(x, y, lw=2, color="steelblue", label=f"{cfg}")
    plt.fill_between(x, y, color="lightblue", alpha=0.3)

    # Annotated key quantiles
    key_p = [25, 50, 75, 95]
    for p in key_p:
        val = np.percentile(x, p)
        plt.hlines(y=p, xmin=0, xmax=val, colors="grey", linestyles="--", alpha=0.7)
        plt.vlines(x=val, ymin=0, ymax=p, colors="grey", linestyles="--", alpha=0.7)
        plt.text(val, p + 1, f"{p}%\n({val:.1f} km/h)",
                 ha="left", va="bottom", fontsize=10, color="grey")

    # Formatting
    plt.xlabel("Speed [km/h]", fontsize=12)
    plt.ylabel("Percentile [%]", fontsize=12)
    plt.title(f"Speed CDF — {cfg}", fontsize=15)
    plt.grid(ls="--", alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 100)
    plt.legend()

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"speed_CDF_{cfg}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")
    return x, y


def save_speed_quantiles(cfg, x, y, out_dir="../Results/Speed_Quantiles/"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cfg}_speed_quantiles.pkl")

    data = {
        "config": cfg,
        "x": x,   # quantile values
        "y": y    # percentiles (%)
    }

    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved speed quantiles → {out_path}")

# --- Run for all CONFIGS ---
for cfg in CONFIGS:
    x, y = plot_speed_cdf_for_config(cfg)
    save_speed_quantiles(cfg, x, y)