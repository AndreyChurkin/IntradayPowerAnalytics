"""
Correlates session anomaly scores with optimal BESS trading profitability to investigate
whether anomalous market sessions offer systematically different profit opportunities.

Inputs:
  - Session anomaly scores CSV  (output of the anomaly detection pipeline)
  - Session trading profits CSV (output of the optimal backtesting pipeline)

Outputs saved to results/:
  - session_anomaly_score_and_profit_merged.csv          — merged dataset (anomaly scores + profits)
  - session_anomaly_score_and_profit_statistics.csv      — group statistics and correlation coefficients
  - scatter_anomaly_score_vs_profit                      — PNG, PDF, SVG
  - violin_profit_by_anomaly_group                       — PNG, PDF, SVG
  - ........TO UPDATE!

Analysis steps:
  1. Merge the two datasets on the shared session identifier
  2. Compute Pearson and Spearman correlations between anomaly score and profit
  3. Compare profit distributions between anomalous and normal sessions using descriptive statistics and a Mann-Whitney U test
  4. Visualise results (scatter, violin, histogram)

Andrey Churkin
https://andreychurkin.ru/

"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

start_time = time.time()


# ── PARAMETERS & SETTINGS ─────────────────────────────────────────────────────

ANOMALY_SCORES_CSV  = "../../anomaly_detection/results/session_anomaly_scores_8737_v1.csv"
TRADING_PROFITS_CSV = "../../trading/results/bess_optimal_backtest_summary_multi_1h_sessions_8737_results.csv"

SESSION_KEY       = "delivery_start"  # shared column used to merge the two datasets
ANOMALY_SCORE_COL = "Anomaly_score"   # continuous value: negative = anomalous, positive = normal
ANOMALY_LABEL_COL = "Anomaly"         # binary label: -1 = anomalous, 1 = normal
PROFIT_COL        = "profit"          # optimal trading profit per session

LABEL_NORMAL    =  1
LABEL_ANOMALOUS = -1
COLOR_NORMAL    = "#2196F3"   # blue
COLOR_ANOMALOUS = "#F44336"   # red

FZ = 14   # base font size for all plots


# ── Step 1: Load and merge datasets ───────────────────────────────────────────

df_anomaly = pd.read_csv(ANOMALY_SCORES_CSV)
df_profit  = pd.read_csv(TRADING_PROFITS_CSV)

print(f"\nAnomaly scores file : {len(df_anomaly)} sessions")
print(f"Trading profits file: {len(df_profit)} sessions")

df = pd.merge(
    df_anomaly[[SESSION_KEY, ANOMALY_SCORE_COL, ANOMALY_LABEL_COL]],
    df_profit[[SESSION_KEY, PROFIT_COL]],
    on=SESSION_KEY,
    how="inner"
)

print(f"Merged dataset      : {len(df)} sessions")
if len(df) < len(df_anomaly) or len(df) < len(df_profit):
    print(f"  ⚠️  Warning: {max(len(df_anomaly), len(df_profit)) - len(df)} sessions lost "
          f"during merge — check SESSION_KEY alignment between the two files.")

os.makedirs("../results", exist_ok=True)
df.to_csv("../results/session_anomaly_score_and_profit_merged.csv", index=False)
print("Saved: ../results/session_anomaly_score_and_profit_merged.csv")


# ── Step 2: Correlation — anomaly score vs profit ─────────────────────────────

# Pearson: measures linear correlation
pearson_r,  pearson_p  = pearsonr(df[ANOMALY_SCORE_COL],  df[PROFIT_COL])
# Spearman: rank-based, more robust when profit distribution is skewed
spearman_r, spearman_p = spearmanr(df[ANOMALY_SCORE_COL], df[PROFIT_COL])

print(f"\n── Correlation: anomaly score vs profit ──────────────────────────────────")
print(f"  Pearson  r = {pearson_r:+.4f}   p = {pearson_p:.2e}")
print(f"  Spearman r = {spearman_r:+.4f}   p = {spearman_p:.2e}")


# ── Step 3: Group comparison — anomalous vs normal sessions ───────────────────

df_normal    = df[df[ANOMALY_LABEL_COL] == LABEL_NORMAL]
df_anomalous = df[df[ANOMALY_LABEL_COL] == LABEL_ANOMALOUS]

def group_stats(group_df, label):
    s = group_df[PROFIT_COL]
    return {
        "group":  label,
        "count":  len(group_df),
        "mean":   round(s.mean(),   2),
        "median": round(s.median(), 2),
        "std":    round(s.std(),    2),
        "min":    round(s.min(),    2),
        "max":    round(s.max(),    2),
    }

stats = pd.DataFrame([
    group_stats(df_normal,    "normal"),
    group_stats(df_anomalous, "anomalous"),
])

print(f"\n── Profit statistics by group ────────────────────────────────────────────")
print(stats.to_string(index=False))

# Mann-Whitney U test: non-parametric test for whether the two profit
# distributions are significantly different (does not assume normality)
mw_stat, mw_p = mannwhitneyu(
    df_normal[PROFIT_COL],
    df_anomalous[PROFIT_COL],
    alternative="two-sided"
)
print(f"\n── Mann-Whitney U test (normal vs anomalous profit) ──────────────────────")
print(f"  U = {mw_stat:.1f}   p = {mw_p:.2e}")
if mw_p < 0.05:
    print("  ✅ Distributions are significantly different (p < 0.05)")
else:
    print("  ❌ No significant difference detected (p >= 0.05)")

# Append correlation and test results to the statistics table and save
stats["pearson_r"]     = pearson_r
stats["pearson_p"]     = pearson_p
stats["spearman_r"]    = spearman_r
stats["spearman_p"]    = spearman_p
stats["mannwhitney_U"] = mw_stat
stats["mannwhitney_p"] = mw_p

stats.to_csv("../results/session_anomaly_score_and_profit_statistics.csv", index=False)
print("\nSaved: ../results/session_anomaly_score_and_profit_statistics.csv")


# ── Step 4: Visualisation ─────────────────────────────────────────────────────

plt.rcParams["font.family"] = "Courier New"
plt.rcParams["font.size"]   = FZ

def save_figure(fig, name):
    """Save figure as PNG, PDF, and SVG to results/."""
    os.makedirs("../results", exist_ok=True)
    for fmt in ["png", "pdf", "svg"]:
        path = f"../results/{name}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")


# -- Plot 1: Scatter — anomaly score vs profit ---------------------------------

fig1, ax1 = plt.subplots(figsize=(10, 7))

ax1.scatter(df_normal[ANOMALY_SCORE_COL],    df_normal[PROFIT_COL],
            color=COLOR_NORMAL,    alpha=0.5, s=10, label="Normal")
ax1.scatter(df_anomalous[ANOMALY_SCORE_COL], df_anomalous[PROFIT_COL],
            color=COLOR_ANOMALOUS, alpha=0.7, s=15, label="Anomalous")

# Reference lines
ax1.axvline(0, color="k",    linewidth=0.9, linestyle="--", label="Anomaly threshold (score = 0)")
ax1.axhline(0, color="grey", linewidth=0.9, linestyle="--", label="Zero profit")

ax1.set_xlabel("Anomaly score  (negative = anomalous session)", fontsize=FZ)
ax1.set_ylabel("Optimal trading profit, EUR", fontsize=FZ)
ax1.tick_params(labelsize=FZ)
ax1.legend(fontsize=FZ - 3)

# Correlation annotation
ax1.text(0.02, 0.97,
         f"Pearson  r = {pearson_r:+.3f}  (p = {pearson_p:.1e})\n"
         f"Spearman r = {spearman_r:+.3f}  (p = {spearman_p:.1e})",
         transform=ax1.transAxes, fontsize=FZ - 3,
         verticalalignment="top",
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

ax1.grid(which="major", alpha=0.4)
ax1.minorticks_on()
ax1.grid(which="minor", alpha=0.15)

fig1.tight_layout()
print("\n── Saving Plot 1: Scatter ────────────────────────────────────────────────")
save_figure(fig1, "scatter_anomaly_score_vs_profit")
plt.show()


# -- Plot 2: Violin — profit distribution by anomaly group --------------------

fig2, ax2 = plt.subplots(figsize=(8, 7))

violin_data   = [df_normal[PROFIT_COL].values, df_anomalous[PROFIT_COL].values]
violin_colors = [COLOR_NORMAL, COLOR_ANOMALOUS]

parts = ax2.violinplot(violin_data, positions=[1, 2], showmedians=True, showextrema=True)

for body, color in zip(parts["bodies"], violin_colors):
    body.set_facecolor(color)
    body.set_alpha(0.6)
for part in ["cmedians", "cbars", "cmins", "cmaxes"]:
    parts[part].set_color("k")
    parts[part].set_linewidth(1.2)

median_normal    = df_normal[PROFIT_COL].median()
median_anomalous = df_anomalous[PROFIT_COL].median()
mean_normal      = df_normal[PROFIT_COL].mean()
mean_anomalous   = df_anomalous[PROFIT_COL].mean()

# Colored dots for medians (overlaid on the median line drawn by violinplot)
ax2.scatter([1], [median_normal],    color=COLOR_NORMAL,    s=60, zorder=5,
            marker="o", edgecolors="k", linewidths=0.7,
            label=f"Normal median: {median_normal:.1f} EUR")
ax2.scatter([2], [median_anomalous], color=COLOR_ANOMALOUS, s=60, zorder=5,
            marker="o", edgecolors="k", linewidths=0.7,
            label=f"Anomalous median: {median_anomalous:.1f} EUR")

# Colored X markers for means
ax2.scatter([1], [mean_normal],    color=COLOR_NORMAL,    s=60, zorder=5,
            marker="X", edgecolors="k", linewidths=0.7,
            label=f"Normal mean: {mean_normal:.1f} EUR")
ax2.scatter([2], [mean_anomalous], color=COLOR_ANOMALOUS, s=60, zorder=5,
            marker="X", edgecolors="k", linewidths=0.7,
            label=f"Anomalous mean: {mean_anomalous:.1f} EUR")

ax2.set_xticks([1, 2])
ax2.set_xticklabels([
    f"Normal\n(n={len(df_normal)})",
    f"Anomalous\n(n={len(df_anomalous)})"
], fontsize=FZ)
ax2.set_ylabel("Optimal trading profit, EUR", fontsize=FZ)
ax2.tick_params(labelsize=FZ)
ax2.axhline(0, color="grey", linewidth=0.9, linestyle="--")

from matplotlib.lines import Line2D
mw_entry = Line2D([], [], color="none", label=f"Mann-Whitney p = {mw_p:.1e}")
handles, _ = ax2.get_legend_handles_labels()
ax2.legend(handles=[*handles, mw_entry], fontsize=FZ - 3, loc="upper left")

ax2.grid(which="major", alpha=0.4)
ax2.minorticks_on()
ax2.grid(which="minor", alpha=0.15)

fig2.tight_layout()
print("\n── Saving Plot 2: Violin ────────────────────────────────────────────────")
save_figure(fig2, "violin_profit_by_anomaly_group")
plt.show()


# -- Plot 3: Profit vs session index, coloured by anomaly label ---------------
# Shows whether anomalous sessions cluster in time or are randomly distributed

fig3, ax3 = plt.subplots(figsize=(14, 6))

ax3.scatter(df_normal.index,    df_normal[PROFIT_COL],
            color=COLOR_NORMAL,    alpha=0.4, s=6,  label=f"Normal  (n={len(df_normal)})")
ax3.scatter(df_anomalous.index, df_anomalous[PROFIT_COL],
            color=COLOR_ANOMALOUS, alpha=0.8, s=12, label=f"Anomalous  (n={len(df_anomalous)})")

ax3.axhline(0, color="grey", linewidth=0.8, linestyle="--")

ax3.set_xlabel("Trading session index (chronological order)", fontsize=FZ)
ax3.set_ylabel("Optimal trading profit, EUR", fontsize=FZ)
ax3.tick_params(labelsize=FZ)
ax3.set_xlim(0, len(df))
ax3.legend(fontsize=FZ - 3)
ax3.grid(which="major", alpha=0.4)
ax3.minorticks_on()
ax3.grid(which="minor", alpha=0.15)

fig3.tight_layout()
print("\n── Saving Plot 3: Profit vs session index ───────────────────────────────")
save_figure(fig3, "profit_vs_session_index_by_anomaly_group")
plt.show()


# ── Timing ────────────────────────────────────────────────────────────────────

print(f"\n⏱️  Total time taken: {time.time() - start_time:.2f} seconds")
