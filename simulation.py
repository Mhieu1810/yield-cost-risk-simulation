import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# GLOBAL ASSUMPTIONS
# =========================

PRODUCTION_VOLUME = 10000
UNIT_VALUE = 15
REWORK_LABOR_COST = 6
EXTRA_MATERIAL_COST = 4
INSPECTION_COST = 0.6

SIM_RUNS = 1000
COST_THRESHOLD = 35000

# =========================
# ENGINEERING MODELS
# =========================

def cpk_to_scrap(cpk):
    """
    Map Cp/Cpk to scrap rate (engineering assumption)
    Higher Cpk -> exponentially lower scrap
    """
    return max(0.002, np.exp(-2.5 * cpk))

def calculate_total_cost(scrap_rate, rework_rate):
    scrap_cost = PRODUCTION_VOLUME * scrap_rate * UNIT_VALUE
    rework_labor = PRODUCTION_VOLUME * rework_rate * REWORK_LABOR_COST
    extra_material = PRODUCTION_VOLUME * rework_rate * EXTRA_MATERIAL_COST
    inspection = PRODUCTION_VOLUME * INSPECTION_COST
    return scrap_cost + rework_labor + extra_material + inspection

# =========================
# MONTE CARLO SIMULATION
# =========================

def run_simulation(cpk_mean, label):
    data = []

    for _ in range(SIM_RUNS):
        cpk = random.normalvariate(cpk_mean, 0.15)
        scrap_rate = cpk_to_scrap(cpk)
        rework_rate = scrap_rate + random.uniform(0.01, 0.04)

        total_cost = calculate_total_cost(scrap_rate, rework_rate)

        data.append({
            "cpk": cpk,
            "scrap_rate": scrap_rate,
            "rework_rate": rework_rate,
            "total_cost": total_cost
        })

    df = pd.DataFrame(data)
    df["scenario"] = label
    return df

# =========================
# SCENARIOS
# =========================

before_df = run_simulation(cpk_mean=1.1, label="Before Improvement")
after_df  = run_simulation(cpk_mean=1.6, label="After Improvement")

df = pd.concat([before_df, after_df])

# =========================
# NUMERICAL RESULTS
# =========================

summary = df.groupby("scenario")["total_cost"].agg(
    mean_cost="mean",
    worst_case="max",
    risk_probability=lambda x: (x > COST_THRESHOLD).mean()
)

print("\n=== COST SIMULATION SUMMARY ===\n")
print(summary.round(2))

# =========================
# HISTOGRAM – COST DISTRIBUTION
# =========================

plt.figure()
plt.hist(before_df["total_cost"], bins=40, alpha=0.6, label="Before")
plt.hist(after_df["total_cost"], bins=40, alpha=0.6, label="After")
plt.axvline(COST_THRESHOLD, linestyle="--")
plt.xlabel("Total Monthly Cost (USD)")
plt.ylabel("Frequency")
plt.title("Cost Distribution – Monte Carlo Simulation")
plt.legend()
plt.show()

# =========================
# TORNADO CHART – RISK DRIVER
# =========================

drivers = {
    "Scrap Rate": df["scrap_rate"].corr(df["total_cost"]),
    "Rework Rate": df["rework_rate"].corr(df["total_cost"]),
    "Cpk": df["cpk"].corr(df["total_cost"])
}

tornado = pd.Series(drivers).sort_values()

plt.figure()
tornado.plot(kind="barh")
plt.xlabel("Correlation with Total Cost")
plt.title("Tornado Chart – Cost Risk Drivers")
plt.show()
