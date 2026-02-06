import dask.dataframe as dd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

ENRICHED_DIR = "../Data/parquet_trucks_enriched/"
QUANTILE_PATH = "rolling_quantiles.pkl"
FIG_DIR = "../Results/Figures/"
os.makedirs(FIG_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8")

# ---------------------------------------------------------
# Load enriched parquet files
# ---------------------------------------------------------
ddf = dd.read_parquet(
    ENRICHED_DIR,
    engine="pyarrow",
    columns=["delta_heading_var", "speed_estimate_mean"]
)

# Remove NaNs (rolling windows produce NaN at edges)
ddf = ddf.dropna()

# ---------------------------------------------------------
# Compute quantiles 0–100%
# ---------------------------------------------------------
q = np.linspace(0, 1, 101)

delta_heading_var_q = ddf["delta_heading_var"].quantile(q, method = 'tdigest').compute()
speed_estimate_mean_q = ddf["speed_estimate_mean"].quantile(q, method = 'tdigest').compute()

# ---------------------------------------------------------
# Save quantiles
# ---------------------------------------------------------
"""
qdata = {
    'heading_cdf': delta_heading_var_q.values,
    'speed_cdf'  : speed_estimate_mean_q.values,
}

with open(QUANTILE_PATH, "wb") as f:
    pickle.dump(qdata, f)

print("✅ Saved quantiles to", QUANTILE_PATH)

"""
FIG_DIR = "../Results/Figures/"
os.makedirs(FIG_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8")

qs = np.linspace(0, 100, 101)


# ---------------------------------------------------------
# Helper function (NO NameError anymore)
# ---------------------------------------------------------
def plot_single_cdf(values, variable_name, xlabel, filename,
                    main_color="#003D73", shade_color="#AEC7E8",
                    annotate_qs = [25, 30, 90]):  

    plt.figure(figsize=(7, 6))

    # line
    plt.plot(values, qs, lw=2, color=main_color, label="Empirical CDF")

    # fill
    plt.fill_between(values, qs, color=shade_color, alpha=0.25)

    # labels
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel("Percentile [%]", fontsize=11)
    plt.title(f"CDF of {variable_name}", fontsize=14)

    plt.grid(ls="--", alpha=0.35)
    plt.ylim(0, 100.1)

    # -------------------------------------
    # Add quantile annotations
    # -------------------------------------
    for p in annotate_qs:
        idx = int(p)    # qs = 0–100 so direct index works
        x_val = values[idx]
        y_val = p

        plt.axvline(x_val, ymin=0, ymax=y_val/100, ls="--", color="gray", alpha=0.7)
        plt.axhline(y_val, xmin=0, xmax=x_val/np.max(values), ls="--", color="gray", alpha=0.7)

        plt.text(
            x_val,
            y_val + 1,
            f"{p}%\n({x_val:.2f})",
            ha="center",
            fontsize=10,
            color="gray",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()
    """
    out = FIG_DIR + filename
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)
    """

# ---------------------------------------------------------
# Actual calls for your two variables
# ---------------------------------------------------------

plot_single_cdf(
    values=delta_heading_var_q.values,
    variable_name="delta_heading_var",
    xlabel="delta_heading_var",
    filename="delta_heading_var_CDF.png",
    main_color="#E57200",   # DTU orange
    shade_color="#F3B87A",
    annotate_qs = [90]
)

plot_single_cdf(
    values=speed_estimate_mean_q.values,
    variable_name="speed_estimate_mean",
    xlabel="speed_estimate_mean (km/h)",
    filename="speed_estimate_mean_CDF.png",
    main_color="#003D73",   # DTU blue
    shade_color="#A8C4DD",
    annotate_qs = [25, 30]
)


