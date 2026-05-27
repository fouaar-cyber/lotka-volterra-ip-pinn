#!/usr/bin/env python3
"""
statistical_analysis.py
Statistical validation and figure generation for SC-PINN revision.
Generates Q-Q plots, swarm plots, and performs all statistical tests.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os

# Setup publication-quality plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'pdf.fonttype': 42,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading experimental data...")
with open('data/all_results.json', 'r') as f:
    data = json.load(f)['all_results']

baseline = np.array(data['baseline'])
soft = np.array(data['soft_ip_pinn'])
sc = np.array(data['sc_pinn'])

print(f"Baseline: n={len(baseline)}, mean={np.mean(baseline):.4e}, std={np.std(baseline):.4e}")
print(f"Soft:     n={len(soft)}, mean={np.mean(soft):.4e}, std={np.std(soft):.4e}")
print(f"SC-PINN:  n={len(sc)}, mean={np.mean(sc):.4e}, std={np.std(sc):.4e}")

# ============================================================================
# 1. STATISTICAL TESTS
# ============================================================================
print("\n" + "="*70)
print("STATISTICAL VALIDATION REPORT")
print("="*70)

# Shapiro-Wilk tests
shapiro_results = {}
for name, arr in [("Baseline PINN", baseline), ("Soft IP-PINN", soft), ("SC-PINN", sc)]:
    w_stat, p_val = stats.shapiro(arr)
    shapiro_results[name] = {'W': float(w_stat), 'p': float(p_val)}
    print(f"{name}:")
    print(f"  Shapiro-Wilk W = {w_stat:.4f}, p = {p_val:.4f}")
    print(f"  Mean ± SD: {np.mean(arr):.4e} ± {np.std(arr):.4e}")
    print()

# Welch's t-tests
t_bs, p_bs = stats.ttest_ind(baseline, soft, equal_var=False)
t_bc, p_bc = stats.ttest_ind(baseline, sc, equal_var=False)
t_sc, p_sc = stats.ttest_ind(soft, sc, equal_var=False)

# Cohen's d
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled_std

print("Pairwise Comparisons (Welch's t-test):")
print(f"  Baseline vs Soft: t={t_bs:.3f}, p={p_bs:.4e}, Cohen's d={cohens_d(baseline, soft):.3f}")
print(f"  Baseline vs SC:   t={t_bc:.3f}, p={p_bc:.4e}, Cohen's d={cohens_d(baseline, sc):.3f}")
print(f"  Soft vs SC:       t={t_sc:.3f}, p={p_sc:.4e}, Cohen's d={cohens_d(soft, sc):.3f}")

# Bootstrap confidence intervals
np.random.seed(42)
B = 10000

def bootstrap_ci(data, B=10000, alpha=0.05):
    n = len(data)
    means = np.array([np.mean(np.random.choice(data, size=n, replace=True)) for _ in range(B)])
    return np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])

ci_base = bootstrap_ci(baseline, B)
ci_soft = bootstrap_ci(soft, B)
ci_sc = bootstrap_ci(sc, B)

print(f"\nBootstrap 95% Confidence Intervals (B={B}):")
print(f"  Baseline: [{ci_base[0]:.4f}, {ci_base[1]:.4f}]")
print(f"  Soft:     [{ci_soft[0]:.4f}, {ci_soft[1]:.4f}]")
print(f"  SC-PINN:  [{ci_sc[0]:.4f}, {ci_sc[1]:.4f}]")

# Non-parametric tests
print("\n" + "="*70)
print("NON-PARAMETRIC VALIDATION (Mann-Whitney U)")
print("="*70)

u_bs, p_mw_bs = stats.mannwhitneyu(baseline, soft, alternative='two-sided')
u_bc, p_mw_bc = stats.mannwhitneyu(baseline, sc, alternative='two-sided')
u_sc, p_mw_sc = stats.mannwhitneyu(soft, sc, alternative='two-sided')

print(f"Baseline vs Soft: U={u_bs:.1f}, p={p_mw_bs:.4e}")
print(f"Baseline vs SC:   U={u_bc:.1f}, p={p_mw_bc:.4e}")
print(f"Soft vs SC:       U={u_sc:.1f}, p={p_mw_sc:.4e}")

# Cliff's delta
def cliffs_delta(x, y):
    x, y = np.array(x), np.array(y)
    nx, ny = len(x), len(y)
    dominance = 0
    for xi in x:
        for yj in y:
            if xi > yj: dominance += 1
            elif xi < yj: dominance -= 1
    return dominance / (nx * ny)

print(f"\nCliff's Delta (non-parametric effect size):")
print(f"  Baseline vs Soft: {cliffs_delta(baseline, soft):.3f}")
print(f"  Baseline vs SC:   {cliffs_delta(baseline, sc):.3f}")
print(f"  Soft vs SC:       {cliffs_delta(soft, sc):.3f}")
print("  (|d|>0.33 medium, |d|>0.474 large effect)")

# Save statistics
stats_report = {
    'shapiro_wilk': shapiro_results,
    'welch_ttest': {
        'baseline_vs_soft': {'t': float(t_bs), 'p': float(p_bs), 'cohens_d': float(cohens_d(baseline, soft))},
        'baseline_vs_sc': {'t': float(t_bc), 'p': float(p_bc), 'cohens_d': float(cohens_d(baseline, sc))},
        'soft_vs_sc': {'t': float(t_sc), 'p': float(p_sc), 'cohens_d': float(cohens_d(soft, sc))}
    },
    'bootstrap_ci': {
        'baseline': [float(ci_base[0]), float(ci_base[1])],
        'soft': [float(ci_soft[0]), float(ci_soft[1])],
        'sc': [float(ci_sc[0]), float(ci_sc[1])]
    },
    'mann_whitney': {
        'baseline_vs_soft': {'U': float(u_bs), 'p': float(p_mw_bs)},
        'baseline_vs_sc': {'U': float(u_bc), 'p': float(p_mw_bc)},
        'soft_vs_sc': {'U': float(u_sc), 'p': float(p_mw_sc)}
    },
    'cliffs_delta': {
        'baseline_vs_soft': float(cliffs_delta(baseline, soft)),
        'baseline_vs_sc': float(cliffs_delta(baseline, sc)),
        'soft_vs_sc': float(cliffs_delta(soft, sc))
    }
}

with open('data/statistical_validation.json', 'w') as f:
    json.dump(stats_report, f, indent=2)
print("\nSaved: data/statistical_validation.json")

# ============================================================================
# 2. Q-Q PLOTS (Appendix B)
# ============================================================================
print("\nGenerating Q-Q plots...")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

groups = [('Baseline PINN', baseline, '#ff9999'), 
          ('Soft IP-PINN', soft, '#66b3ff'), 
          ('SC-PINN', sc, '#99ff99')]

for ax, (name, data_arr, color) in zip(axes, groups):
    stats.probplot(data_arr, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(color)
    ax.get_lines()[0].set_markeredgecolor('black')
    ax.get_lines()[0].set_markersize(8)
    ax.get_lines()[1].set_color('red')
    ax.get_lines()[1].set_linewidth(2)
    ax.set_title(f'{name}\nShapiro-Wilk p={stats.shapiro(data_arr)[1]:.4f}', fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/appendix_qq_plots.pdf')
plt.savefig('figures/appendix_qq_plots.png')
plt.close()
print("Saved: figures/appendix_qq_plots.pdf")

# ============================================================================
# 3. SWARM PLOT / BAR CHART (Figure 3 Revised)
# ============================================================================
print("\nGenerating revised Figure 3 (swarm plot)...")
fig, ax = plt.subplots(figsize=(8, 6))

labels = ['Baseline PINN', 'Soft IP-PINN', 'SC-PINN (Ours)']
means = [np.mean(baseline), np.mean(soft), np.mean(sc)]
stds = [np.std(baseline), np.std(soft), np.std(sc)]
colors = ['#ff9999', '#66b3ff', '#99ff99']
edgecolors = ['#cc0000', '#0055cc', '#00cc00']

# Bar chart with error bars (SD)
bars = ax.bar(labels, means, yerr=stds, capsize=10, color=colors, 
              edgecolor=edgecolors, linewidth=2, alpha=0.7, zorder=1)

# Add individual data points (swarm-like)
np.random.seed(42)
for i, (data_arr, color) in enumerate(zip([baseline, soft, sc], colors)):
    x_jitter = np.random.normal(i, 0.08, size=len(data_arr))
    ax.scatter(x_jitter, data_arr, color='black', alpha=0.6, s=50, zorder=3, 
               edgecolors='white', linewidth=0.5)
    ax.hlines(np.mean(data_arr), i-0.3, i+0.3, colors='red', linestyles='-', linewidth=2, zorder=4)

# Value labels on bars
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.025, 
            f'{mean:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('RMS Invariant Deviation', fontsize=12)
ax.set_title('Statistical Comparison with Individual Run Data (n=15)', fontsize=13, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, 0.45)

# Add annotation about non-normality
ax.text(0.98, 0.98, 'Note: Soft IP-PINN and SC-PINN\nviolate normality (Shapiro-Wilk p<0.05)', 
        transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('figures/figure3_statistics_revised.pdf')
plt.savefig('figures/figure3_statistics_revised.png')
plt.close()
print("Saved: figures/figure3_statistics_revised.pdf")

# ============================================================================
# DONE
# ============================================================================
print("\n" + "="*70)
print("ALL STATISTICAL ANALYSES COMPLETE")
print("="*70)
print("Generated files:")
print("  - figures/appendix_qq_plots.pdf")
print("  - figures/figure3_statistics_revised.pdf")
print("  - data/statistical_validation.json")