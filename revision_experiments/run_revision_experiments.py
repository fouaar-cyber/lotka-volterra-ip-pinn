import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy import stats
import json
import os
import time

# ============================================================================
# MASTER REVISION SCRIPT - Addresses Reviewer 1 & Reviewer 2
# ============================================================================
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

os.makedirs('figures_revision', exist_ok=True)
os.makedirs('data_revision', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALPHA, BETA, DELTA, GAMMA = 1.0, 0.1, 0.075, 0.75

def lotka_volterra(t, y):
    x, y = y
    return [ALPHA * x - BETA * x * y, DELTA * x * y - GAMMA * y]

def compute_invariant(x, y):
    eps = 1e-8
    x = torch.clamp(x, min=eps) if isinstance(x, torch.Tensor) else np.clip(x, eps, None)
    y = torch.clamp(y, min=eps) if isinstance(y, torch.Tensor) else np.clip(y, eps, None)
    if isinstance(x, torch.Tensor):
        return DELTA * x + BETA * y - GAMMA * torch.log(x) - ALPHA * torch.log(y)
    else:
        return DELTA * x + BETA * y - GAMMA * np.log(x) - ALPHA * np.log(y)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================
class BaselinePINN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 2)
        )
        self.scale = torch.nn.Parameter(torch.ones(2))
    def forward(self, t):
        return torch.nn.functional.softplus(self.net(t)) * torch.nn.functional.softplus(self.scale)

class SC_PINN(torch.nn.Module):
    """SC-PINN with configurable lambda and optional refinement ablation"""
    def __init__(self, lambda_corr=0.1):
        super().__init__()
        self.log_net = torch.nn.Sequential(
            torch.nn.Linear(1, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 2)
        )
        self.V0 = torch.nn.Parameter(torch.tensor(1.0))
        self.lambda_corr = lambda_corr

    def forward(self, t):
        log_xy = self.log_net(t)
        xy = torch.exp(log_xy)
        x, y = xy[:, 0:1], xy[:, 1:2]
        V_current = compute_invariant(x, y)
        delta_V = V_current - self.V0
        correction = torch.exp(-self.lambda_corr * delta_V)
        return xy * correction

def train_model(model, mode='baseline', use_refinement=True, epochs=15000, T_train=100, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1000, factor=0.5, threshold=1e-7
    )

    t_train = torch.linspace(0, T_train, 400, requires_grad=True).reshape(-1, 1).to(device)
    sol = solve_ivp(lotka_volterra, [0, T_train], [2.0, 2.0],
                    t_eval=t_train.detach().cpu().numpy().flatten(),
                    method='RK45', rtol=1e-8, atol=1e-8)
    y_ref = torch.tensor(sol.y.T, dtype=torch.float32).to(device)

    if mode == 'sc':
        with torch.no_grad():
            V_ref = compute_invariant(y_ref[:, 0:1], y_ref[:, 1:2])
            model.V0.data = torch.mean(V_ref)

    best_loss = float('inf')
    patience = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(t_train)
        x, y = y_pred[:, 0:1], y_pred[:, 1:2]

        data_loss = torch.mean((y_pred - y_ref)**2)

        dxdt = torch.autograd.grad(x, t_train, torch.ones_like(x), create_graph=True)[0]
        dydt = torch.autograd.grad(y, t_train, torch.ones_like(y), create_graph=True)[0]
        res_x = dxdt - (ALPHA * x - BETA * x * y)
        res_y = dydt - (DELTA * x * y - GAMMA * y)
        physics_loss = torch.mean(res_x**2 + res_y**2)

        loss = data_loss + 0.1 * physics_loss

        if mode == 'sc' and use_refinement:
            V_pred = compute_invariant(x, y)
            loss = loss + 0.5 * torch.mean((V_pred - model.V0)**2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            patience = 0
        else:
            patience += 1
            if patience > 3000:
                break

    return model

def compute_rms_drift(model, T=50, n_points=500):
    t_eval = torch.linspace(0, T, n_points).reshape(-1, 1).to(device)
    with torch.no_grad():
        pred = model(t_eval)
        V = compute_invariant(pred[:, 0:1], pred[:, 1:2]).cpu().numpy()
    sol = solve_ivp(lotka_volterra, [0, T], [2.0, 2.0],
                    t_eval=np.linspace(0, T, n_points), method='RK45')
    V_ref = np.mean(compute_invariant(sol.y[0], sol.y[1]))
    return np.sqrt(np.mean((V - V_ref)**2))

def compute_trajectory_deviation(model, T=50, n_points=500):
    """Returns max Euclidean distance from reference RK4 trajectory"""
    t_eval = torch.linspace(0, T, n_points).reshape(-1, 1).to(device)
    with torch.no_grad():
        pred = model(t_eval).cpu().numpy()
    sol = solve_ivp(lotka_volterra, [0, T], [2.0, 2.0],
                    t_eval=np.linspace(0, T, n_points), method='RK45')
    ref = sol.y.T
    distances = np.sqrt(np.sum((pred - ref)**2, axis=1))
    return np.max(distances), np.mean(distances), np.std(distances), distances

# ============================================================================
# EXPERIMENT 1: SENSITIVITY ANALYSIS (Reviewer 1, Point 11 / Reviewer 2, Point 7)
# ============================================================================
def run_sensitivity_analysis():
    print("\n" + "="*70)
    print("EXPERIMENT 1: SENSITIVITY ANALYSIS (lambda vs RMS drift)")
    print("="*70)

    lambda_values = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    n_runs = 5  # Per lambda value (adjust based on time)

    results = {lam: [] for lam in lambda_values}

    for lam in lambda_values:
        print(f"\nTesting lambda = {lam}...")
        for run in range(n_runs):
            model = SC_PINN(lambda_corr=lam).to(device)
            model = train_model(model, mode='sc', use_refinement=True, 
                              epochs=8000, T_train=100, seed=42+run)
            drift = compute_rms_drift(model, T=50)
            results[lam].append(drift)
            print(f"  Run {run+1}: RMS drift = {drift:.4e}")

    # Plot
    means = [np.mean(results[lam]) for lam in lambda_values]
    stds = [np.std(results[lam]) for lam in lambda_values]

    plt.figure(figsize=(8, 5))
    plt.errorbar(lambda_values, means, yerr=stds, marker='o', markersize=8, 
                 capsize=5, linewidth=2, color='#2E86AB', ecolor='#A23B72',
                 markerfacecolor='#F18F01', markeredgewidth=1.5)
    plt.axvline(x=0.1, color='red', linestyle='--', linewidth=1.5, label='Chosen λ = 0.1')
    plt.axhspan(0, 0.15, alpha=0.1, color='green', label='Stable region (RMS < 0.15)')
    plt.xlabel('Projection Strength λ')
    plt.ylabel('RMS Invariant Deviation')
    plt.title('Sensitivity of RMS Drift to Projection Strength λ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('figures_revision/sensitivity_lambda.pdf')
    plt.savefig('figures_revision/sensitivity_lambda.png')
    plt.close()

    with open('data_revision/sensitivity_results.json', 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print("\nSaved: figures_revision/sensitivity_lambda.pdf")
    print("Saved: data_revision/sensitivity_results.json")

    return results

# ============================================================================
# EXPERIMENT 2: LONG-TIME INTEGRATION T=500 (Reviewer 1, Point 9)
# ============================================================================
def run_longtime_integration():
    print("\n" + "="*70)
    print("EXPERIMENT 2: LONG-TIME INTEGRATION (T=500)")
    print("="*70)

    # Train models on [0, 100] as before
    print("Training Baseline PINN...")
    base = BaselinePINN().to(device)
    base = train_model(base, mode='baseline', epochs=10000, T_train=100, seed=42)

    print("Training SC-PINN...")
    sc = SC_PINN(lambda_corr=0.1).to(device)
    sc = train_model(sc, mode='sc', use_refinement=True, epochs=10000, T_train=100, seed=42)

    # Evaluate on extended domain
    T_long = 500
    n_points = 2000
    t_eval = torch.linspace(0, T_long, n_points).reshape(-1, 1).to(device)
    t_np = t_eval.cpu().numpy().flatten()

    sol = solve_ivp(lotka_volterra, [0, T_long], [2.0, 2.0],
                    t_eval=t_np, method='RK45', rtol=1e-8, atol=1e-8)
    V_ref = np.mean(compute_invariant(sol.y[0], sol.y[1]))

    with torch.no_grad():
        V_base = compute_invariant(base(t_eval)[:,0:1], base(t_eval)[:,1:2]).cpu().numpy()
        V_sc = compute_invariant(sc(t_eval)[:,0:1], sc(t_eval)[:,1:2]).cpu().numpy()

    # Compute rolling RMS drift over windows
    window = 500
    rms_base = []
    rms_sc = []
    centers = []
    for i in range(0, len(t_np)-window, window//2):
        rms_base.append(np.sqrt(np.mean((V_base[i:i+window] - V_ref)**2)))
        rms_sc.append(np.sqrt(np.mean((V_sc[i:i+window] - V_ref)**2)))
        centers.append(t_np[i+window//2])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: Invariant evolution
    axes[0].plot(t_np, V_base, 'r--', lw=1, alpha=0.7, label='Baseline PINN')
    axes[0].plot(t_np, V_sc, 'b-', lw=1, alpha=0.9, label='SC-PINN (Ours)')
    axes[0].axhline(y=V_ref, color='k', linestyle=':', alpha=0.8, label=f'Target H₀ ≈ {V_ref:.3f}')
    axes[0].set_ylabel('Invariant Value (H)')
    axes[0].set_title(f'Invariant Evolution over Extended Horizon (T={T_long})')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Bottom: Rolling RMS drift
    axes[1].semilogy(centers, rms_base, 'r--', marker='o', markersize=4, label='Baseline PINN')
    axes[1].semilogy(centers, rms_sc, 'b-', marker='s', markersize=4, label='SC-PINN (Ours)')
    axes[1].set_xlabel('Time (t)')
    axes[1].set_ylabel('Rolling RMS Drift')
    axes[1].set_title('Local RMS Drift over Sliding Windows')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures_revision/longtime_T500.pdf')
    plt.savefig('figures_revision/longtime_T500.png')
    plt.close()

    # Summary stats
    print(f"\nT={T_long} Summary:")
    print(f"  Baseline overall RMS: {np.sqrt(np.mean((V_base - V_ref)**2)):.4e}")
    print(f"  SC-PINN overall RMS:  {np.sqrt(np.mean((V_sc - V_ref)**2)):.4e}")
    print(f"  Baseline max |H-H0|:  {np.max(np.abs(V_base - V_ref)):.4e}")
    print(f"  SC-PINN max |H-H0|:   {np.max(np.abs(V_sc - V_ref)):.4e}")
    print("\nSaved: figures_revision/longtime_T500.pdf")

# ============================================================================
# EXPERIMENT 3: TRAJECTORY DEVIATION & CONFIDENCE BANDS (Reviewer 1, Point 8)
# ============================================================================
def run_trajectory_confidence():
    print("\n" + "="*70)
    print("EXPERIMENT 3: TRAJECTORY CONFIDENCE BANDS (n=15 runs)")
    print("="*70)

    T = 50
    n_points = 500
    n_runs = 15
    t_eval = torch.linspace(0, T, n_points).reshape(-1, 1).to(device)
    t_np = t_eval.cpu().numpy().flatten()

    sol = solve_ivp(lotka_volterra, [0, T], [2.0, 2.0],
                    t_eval=t_np, method='RK45')
    ref_traj = sol.y.T

    base_trajectories = []
    sc_trajectories = []
    base_devs = []
    sc_devs = []

    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}...")

        base = BaselinePINN().to(device)
        base = train_model(base, mode='baseline', epochs=8000, T_train=100, seed=42+run)
        with torch.no_grad():
            base_pred = base(t_eval).cpu().numpy()
        base_trajectories.append(base_pred)
        base_devs.append(np.sqrt(np.sum((base_pred - ref_traj)**2, axis=1)))

        sc = SC_PINN(lambda_corr=0.1).to(device)
        sc = train_model(sc, mode='sc', use_refinement=True, epochs=8000, T_train=100, seed=42+run)
        with torch.no_grad():
            sc_pred = sc(t_eval).cpu().numpy()
        sc_trajectories.append(sc_pred)
        sc_devs.append(np.sqrt(np.sum((sc_pred - ref_traj)**2, axis=1)))

    base_trajectories = np.array(base_trajectories)  # (15, 500, 2)
    sc_trajectories = np.array(sc_trajectories)
    base_devs = np.array(base_devs)
    sc_devs = np.array(sc_devs)

    # Phase space plot with confidence bands
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Baseline
    axes[0].plot(ref_traj[:,0], ref_traj[:,1], 'k-', lw=2, label='Reference (RK4)')
    for i in range(n_runs):
        axes[0].plot(base_trajectories[i,:,0], base_trajectories[i,:,1], 'r-', alpha=0.2, lw=0.5)
    axes[0].plot([], [], 'r-', alpha=0.4, label='Baseline PINN (n=15)')
    axes[0].set_title('Baseline PINN: Phase-Space Variability')
    axes[0].set_xlabel('Prey (x)')
    axes[0].set_ylabel('Predator (y)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # SC-PINN
    axes[1].plot(ref_traj[:,0], ref_traj[:,1], 'k-', lw=2, label='Reference (RK4)')
    for i in range(n_runs):
        axes[1].plot(sc_trajectories[i,:,0], sc_trajectories[i,:,1], 'b-', alpha=0.2, lw=0.5)
    axes[1].plot([], [], 'b-', alpha=0.4, label='SC-PINN (n=15)')
    axes[1].set_title('SC-PINN: Phase-Space Variability')
    axes[1].set_xlabel('Prey (x)')
    axes[1].set_ylabel('Predator (y)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures_revision/trajectory_confidence_bands.pdf')
    plt.close()

    # Deviation plot
    fig, ax = plt.subplots(figsize=(8, 5))
    mean_base = np.mean(base_devs, axis=0)
    std_base = np.std(base_devs, axis=0)
    mean_sc = np.mean(sc_devs, axis=0)
    std_sc = np.std(sc_devs, axis=0)

    ax.plot(t_np, mean_base, 'r--', lw=2, label='Baseline PINN (mean)')
    ax.fill_between(t_np, mean_base-2*std_base, mean_base+2*std_base, alpha=0.2, color='red')
    ax.plot(t_np, mean_sc, 'b-', lw=2, label='SC-PINN (mean)')
    ax.fill_between(t_np, mean_sc-2*std_sc, mean_sc+2*std_sc, alpha=0.2, color='blue')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Euclidean Distance from Reference')
    ax.set_title(f'Trajectory Deviation from RK4 Reference (±2σ bands, n={n_runs})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('figures_revision/trajectory_deviation.pdf')
    plt.close()

    # Max deviation statistics
    max_base = [np.max(d) for d in base_devs]
    max_sc = [np.max(d) for d in sc_devs]
    print(f"\nMax Euclidean Distance from Reference (T={T}):")
    print(f"  Baseline PINN: {np.mean(max_base):.4f} ± {np.std(max_base):.4f}")
    print(f"  SC-PINN:       {np.mean(max_sc):.4f} ± {np.std(max_sc):.4f}")
    print("\nSaved: figures_revision/trajectory_confidence_bands.pdf")
    print("Saved: figures_revision/trajectory_deviation.pdf")

# ============================================================================
# EXPERIMENT 4: ABLATION STUDY WITH STATISTICAL TEST (Reviewer 1, Point 12)
# ============================================================================
def run_ablation_study():
    print("\n" + "="*70)
    print("EXPERIMENT 4: ABLATION STUDY (Refinement Term)")
    print("="*70)

    n_runs = 15  # Match main experiments
    full_results = []
    no_ref_results = []

    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}...")

        # Full model (with refinement)
        sc_full = SC_PINN(lambda_corr=0.1).to(device)
        sc_full = train_model(sc_full, mode='sc', use_refinement=True, 
                             epochs=8000, T_train=100, seed=42+run)
        drift_full = compute_rms_drift(sc_full, T=50)
        full_results.append(drift_full)

        # Without refinement
        sc_no_ref = SC_PINN(lambda_corr=0.1).to(device)
        sc_no_ref = train_model(sc_no_ref, mode='sc', use_refinement=False,
                               epochs=8000, T_train=100, seed=42+run)
        drift_no_ref = compute_rms_drift(sc_no_ref, T=50)
        no_ref_results.append(drift_no_ref)

        print(f"  Full: {drift_full:.4e} | No refinement: {drift_no_ref:.4e}")

    full_arr = np.array(full_results)
    no_ref_arr = np.array(no_ref_results)

    # Statistical tests
    t_stat, p_val = stats.ttest_ind(full_arr, no_ref_arr, equal_var=False)
    cohen_d = (np.mean(no_ref_arr) - np.mean(full_arr)) / np.sqrt(
        (np.std(no_ref_arr, ddof=1)**2 + np.std(full_arr, ddof=1)**2) / 2
    )

    print(f"\nAblation Results (n={n_runs}):")
    print(f"  Full model:      {np.mean(full_arr):.4e} ± {np.std(full_arr):.4e}")
    print(f"  No refinement:   {np.mean(no_ref_arr):.4e} ± {np.std(no_ref_arr):.4e}")
    print(f"  Degradation:     {(np.mean(no_ref_arr)-np.mean(full_arr))/np.mean(full_arr)*100:.1f}%")
    print(f"  Welch's t-test:  t={t_stat:.3f}, p={p_val:.4e}")
    print(f"  Cohen's d:       {cohen_d:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    positions = [1, 2]
    bp = ax.boxplot([full_arr, no_ref_arr], positions=positions, widths=0.6, 
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    colors = ['#99ff99', '#ffcc99']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Individual points
    for i, data in enumerate([full_arr, no_ref_arr]):
        x_jitter = np.random.normal(positions[i], 0.04, size=len(data))
        ax.scatter(x_jitter, data, color='black', alpha=0.5, s=40, zorder=3)

    ax.set_xticklabels(['Full Model\n(Eq. 4)', 'Without\nRefinement'])
    ax.set_ylabel('RMS Invariant Deviation')
    ax.set_title(f'Ablation Study: Refinement Term Effect\n(p={p_val:.4e}, Cohen\'s d={cohen_d:.2f})')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('figures_revision/ablation_study.pdf')
    plt.close()

    with open('data_revision/ablation_results.json', 'w') as f:
        json.dump({
            'full_model': full_results,
            'no_refinement': no_ref_results,
            'statistics': {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'cohens_d': float(cohen_d),
                'mean_full': float(np.mean(full_arr)),
                'std_full': float(np.std(full_arr)),
                'mean_no_ref': float(np.mean(no_ref_arr)),
                'std_no_ref': float(np.std(no_ref_arr))
            }
        }, f, indent=2)
    print("\nSaved: figures_revision/ablation_study.pdf")
    print("Saved: data_revision/ablation_results.json")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print(" SC-PINN MAJOR REVISION - COMPREHENSIVE EXPERIMENTAL SUITE")
    print(" Addresses Reviewer 1 & Reviewer 2")
    print("="*70)

    # Run all experiments (comment out as needed)
    run_sensitivity_analysis()
    run_longtime_integration()
    run_trajectory_confidence()
    run_ablation_study()

    print("\n" + "="*70)
    print(" ALL EXPERIMENTS COMPLETE!")
    print(" Check figures_revision/ and data_revision/ directories")
    print("="*70)
