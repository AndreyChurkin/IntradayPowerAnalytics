"""
Detects anomalous intraday trading sessions in a given intraday continuous market dataset.
Isolation Forest algorithm is used for anomaly detection here.

Input:  pre-computed session features CSV
Output: session anomaly scores CSV and visualisations

Negative anomaly scores indicate more anomalous sessions. 
The final classification of anomalies depends on the contamination threshold.

Andrey Churkin
https://andreychurkin.ru/

"""

import os
import time

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

start_time = time.time()


# ── Parameters ────────────────────────────────────────────────────────────────

N_ESTIMATORS  = 200    # number of isolation trees used in the ensemble
CONTAMINATION = 0.05   # expected fraction of anomalous sessions
RANDOM_STATE  = 42     # random seed for reproducibility
FZ            = 14     # base font size for all plots


# ── Load features CSC ─────────────────────────────────────────────────────────

df_features = pd.read_csv("../results/session_features_8737_v1.csv")
print(f"\nFeatures loaded: {len(df_features)} sessions, {df_features.shape[1]} features")

X = df_features.values


# ── Isolation Forest ──────────────────────────────────────────────────────────

iso_forest = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE
)
iso_forest.fit(X)

df_features["Anomaly"]       = iso_forest.predict(X)           # -1 = anomaly, 1 = normal
df_features["Anomaly_score"] = iso_forest.decision_function(X) # continuous score

anomaly_sessions = df_features[df_features["Anomaly"] == -1]
print(f"\nAnomalous sessions detected: {len(anomaly_sessions)} "
      f"({100 * len(anomaly_sessions) / len(df_features):.1f}%)")
print(anomaly_sessions.sort_values("Anomaly_score").to_string())


# ── Save results ──────────────────────────────────────────────────────────────

output_csv = "../results/session_anomaly_scores_v1.csv"
df_features.to_csv(output_csv, index=False)
print(f"\nCSV with anomaly scores saved to:: {output_csv}")


# ── Plot 1: Scatter — Bid mean vs Ask mean, coloured by anomaly score ─────────

plt.rcParams["font.family"] = "Courier New"

fig1, ax1 = plt.subplots(figsize=(8, 8))

sc = ax1.scatter(
    df_features["Bid_mean_price"],
    df_features["Ask_mean_price"],
    c=df_features["Anomaly_score"],
    cmap="coolwarm_r",
    s=5,
)
ax1.set_xlabel("Bid mean price, EUR/MWh", fontsize=FZ)
ax1.set_ylabel("Ask mean price, EUR/MWh", fontsize=FZ)
ax1.tick_params(labelsize=FZ)
ax1.set_aspect("equal", adjustable="box")

cbar = fig1.colorbar(sc, fraction=0.046, pad=0.04)
cbar.set_label("Anomaly score", fontsize=FZ)

fig1.tight_layout()
fig1_name = "anomaly_score_scatter_bid_vs_ask"
fig1.savefig(f"../results/{fig1_name}.png", dpi=300, bbox_inches="tight")
fig1.savefig(f"../results/{fig1_name}.pdf", bbox_inches="tight")
fig1.savefig(f"../results/{fig1_name}.svg", bbox_inches="tight")
print(f"\nAnomaly score plot saved to: ../results/{fig1_name}")
plt.show()


# ── Plot 2: Sequential line — anomaly score across all sessions ───────────────

fig2, ax2 = plt.subplots(figsize=(14, 5))

ax2.plot(range(len(df_features)), df_features["Anomaly_score"],
         color="k", linewidth=0.8)
ax2.axhline(0, color="red", linewidth=1.0, linestyle="--", label="Anomaly threshold")

ax2.set_xlabel("Trading session #", fontsize=FZ)
ax2.set_ylabel("Anomaly score", fontsize=FZ)
ax2.set_title("Isolation Forest anomaly scores for intraday sessions "
              "(negative values = anomalous)", fontsize=FZ - 2)
ax2.tick_params(labelsize=FZ)
ax2.set_xlim(0, len(df_features))
ax2.legend(fontsize=FZ - 2)
ax2.grid(which="major", alpha=0.5)
ax2.minorticks_on()
ax2.grid(which="minor", alpha=0.15)

fig2.tight_layout()
fig2_name = "anomaly_score_sequential_line"
fig2.savefig(f"../results/{fig2_name}.png", dpi=300, bbox_inches="tight")
fig2.savefig(f"../results/{fig2_name}.pdf", bbox_inches="tight")
fig2.savefig(f"../results/{fig2_name}.svg", bbox_inches="tight")
print(f"\nAnomaly score plot saved to: ../results/{fig2_name}")
plt.show()


# ── Timing ────────────────────────────────────────────────────────────────────

print(f"\n⏱️  Total time taken: {time.time() - start_time:.2f} seconds")
