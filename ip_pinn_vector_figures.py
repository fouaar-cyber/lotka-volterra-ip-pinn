import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy import stats
import json
import os
import time
from tqdm import tqdm

# ============================================================================
# PUBLICATION-QUALITY VECTOR FIGURE CONFIGURATION
# ============================================================================
plt.rcParams.update({
    # Font settings (Times New Roman or fallback)
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,           # Base font size for readability
    'axes.titlesize': 11,       # Slightly larger titles
    'axes.labelsize': 10,       # Axes labels
    'xtick.labelsize': 9,       # Tick labels
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    
    # Vector output settings
    'pdf.fonttype': 42,         # Embed TrueType fonts (editable text)
    'ps.fonttype': 42,          # For EPS if needed
    
    # Figure quality
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',    # Vector PDF by default
    'savefig.transparent': False,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    
    # Visual clarity
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
})

# ============================================================================
# DIRECTORY SETUP & GLOBAL PARAMETERS
# ============================================================================
os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

ALPHA, BETA, DELTA, GAMMA = 1.0, 0.1, 0.075, 0.75
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# DIRECTORY SETUP & GLOBAL PARAMETERS
# ============================================================================
os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

ALPHA, BETA, DELTA, GAMMA = 1.0, 0.1, 0.075, 0.75
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# LOTKA-VOLTERRA DYNAMICS (ADD THIS BLOCK)
# ============================================================================
def lotka_volterra(t, y):
    """Lotka-Volterra predator-prey dynamics"""
    x, y = y
    dxdt = ALPHA * x - BETA * x * y
    dydt = DELTA * x * y - GAMMA * y
    return [dxdt, dydt]


# ============================================================================
# MODEL DEFINITIONS (UNCHANGED)
# ============================================================================
class BaselinePINN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 50), torch.nn.Tanh(),
            torch.nn.Linear(50, 50), torch.nn.Tanh(),
            torch.nn.Linear(50, 2)
        )
    
    def forward(self, t):
        return torch.abs(self.net(t))

class IP_Pinn(torch.nn.Module):
    def __init__(self, use_feature=True, use_intrusive=True):
        super().__init__()
        self.use_feature = use_feature
        self.use_intrusive = use_intrusive
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(1, 50), torch.nn.Tanh(),
            torch.nn.Linear(50, 50), torch.nn.Tanh()
        )
        self.baseline_head = torch.nn.Linear(50, 2)
        if use_intrusive:
            self.invariant_head = torch.nn.Linear(50, 1)
            self.invariant_weight = torch.nn.Parameter(torch.tensor(1.0))
        if use_feature:
            self.feature_head = torch.nn.Linear(50, 1)
            self.feature_weight = torch.nn.Parameter(torch.tensor(1.0))
    
    def forward(self, t):
        features = self.shared(t)
        baseline_output = torch.abs(self.baseline_head(features))
        if not self.use_intrusive and not self.use_feature:
            return baseline_output
        total_loss = 0
        if self.use_intrusive:
            invariant_pred = self.invariant_head(features)
            total_loss = total_loss + self.invariant_weight * invariant_pred
        if self.use_feature:
            feature_pred = self.feature_head(features)
            total_loss = total_loss + self.feature_weight * feature_pred
        correction = 1.0 / (1.0 + torch.abs(total_loss))
        return baseline_output * correction

# ============================================================================
# TRAINING & EVALUATION (UNCHANGED)
# ============================================================================
def train_model(model, epochs=5000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    t_train = torch.linspace(0, 50, 200).reshape(-1, 1).to(device)
    sol = solve_ivp(lotka_volterra, [0, 50], [2.0, 2.0], 
                    t_eval=t_train.cpu().numpy().flatten(), 
                    method='RK45', rtol=1e-8, atol=1e-8)
    y_ref = torch.tensor(sol.y.T, dtype=torch.float32).to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(t_train)
        data_loss = torch.mean((y_pred - y_ref)**2)
        
        if t_train.requires_grad == False:
            t_train.requires_grad = True
        
        y_ode = model(t_train)
        x, y = y_ode[:, 0], y_ode[:, 1]
        dxdt = torch.autograd.grad(x, t_train, torch.ones_like(x), create_graph=True)[0]
        dydt = torch.autograd.grad(y, t_train, torch.ones_like(y), create_graph=True)[0]
        
        residual_x = dxdt - (ALPHA * x - BETA * x * y)
        residual_y = dydt - (DELTA * x * y - GAMMA * y)
        physics_loss = torch.mean(residual_x**2 + residual_y**2)
        
        V = DELTA * x + BETA * y - GAMMA * torch.log(x) - ALPHA * torch.log(y)
        invariant_loss = torch.var(V)
        
        total_loss = data_loss + 0.1 * physics_loss + 0.01 * invariant_loss
        total_loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        final_pred = model(t_train)
        return torch.mean((final_pred - y_ref)**2).item()

def calculate_invariant_drift(model, t_span=[0, 100], n_points=1000):
    model.eval()
    with torch.no_grad():
        t_test = torch.linspace(t_span[0], t_span[1], n_points).reshape(-1, 1).to(device)
        y_test = model(t_test)
        x, y = y_test[:, 0], y_test[:, 1]
        V = DELTA * x + BETA * y - GAMMA * torch.log(x) - ALPHA * torch.log(y)
        return torch.std(V).item()
    model.train()

def run_experiment():
    results = {'baseline': [], 'ip_pinn': []}
    seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021, 
             2223, 2425, 2627, 2829, 3031]
    
    for run_idx in tqdm(range(15), desc="Running experiments"):
        print(f"\n--- Run {run_idx+1}/15 ---")
        torch.manual_seed(seeds[run_idx])
        np.random.seed(seeds[run_idx])
        
        baseline = BaselinePINN().to(device)
        mse_baseline = train_model(baseline)
        results['baseline'].append(mse_baseline)
        print(f"Baseline MSE: {mse_baseline:.4e}")
        
        ip_pinn = IP_Pinn(use_feature=True, use_intrusive=True).to(device)
        mse_ip_pinn = train_model(ip_pinn)
        results['ip_pinn'].append(mse_ip_pinn)
        print(f"IP-PINN MSE: {mse_ip_pinn:.4e}")
    
    return results

# ============================================================================
# PUBLICATION-QUALITY PLOTTING FUNCTIONS (VECTOR PDF)
# ============================================================================
def plot_phase_space(results, save_path='figures/phase_space_comparison.pdf'):
    """Generate publication-quality vector phase space plot"""
    print(f"Generating vector phase space plot...")
    
    t_ref = np.linspace(0, 50, 500)
    sol_ref = solve_ivp(lotka_volterra, [0, 50], [2.0, 2.0], 
                        t_eval=t_ref, rtol=1e-8, atol=1e-8)
    t_test = torch.linspace(0, 50, 500).reshape(-1, 1).to(device)
    
    baseline = BaselinePINN().to(device)
    train_model(baseline)
    baseline.eval()
    with torch.no_grad():
        pred_baseline = baseline(t_test).cpu().numpy()
    
    ip_pinn = IP_Pinn(use_feature=True, use_intrusive=True).to(device)
    train_model(ip_pinn)
    ip_pinn.eval()
    with torch.no_grad():
        pred_ip_pinn = ip_pinn(t_test).cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2), constrained_layout=True)
    
    axes[0].plot(sol_ref.y[0], sol_ref.y[1], 'k-', linewidth=1.5, label='Reference')
    axes[0].set_title('Reference Solution', fontsize=11)
    axes[0].set_xlabel('Prey', labelpad=2)
    axes[0].set_ylabel('Predator', labelpad=2)
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.3, linewidth=0.5)
    
    axes[1].plot(pred_baseline[:, 0], pred_baseline[:, 1], 'r--', linewidth=1.5, label='Baseline PINN')
    axes[1].set_title('Baseline PINN', fontsize=11)
    axes[1].set_xlabel('Prey', labelpad=2)
    axes[1].set_ylabel('Predator', labelpad=2)
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.3, linewidth=0.5)
    
    axes[2].plot(pred_ip_pinn[:, 0], pred_ip_pinn[:, 1], 'b-', linewidth=1.5, label='IP-PINN')
    axes[2].set_title('Invariant-Preserving PINN', fontsize=11)
    axes[2].set_xlabel('Prey', labelpad=2)
    axes[2].set_ylabel('Predator', labelpad=2)
    axes[2].legend(frameon=False)
    axes[2].grid(True, alpha=0.3, linewidth=0.5)
    
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Vector phase space plot saved: {save_path}")

def plot_invariant_drift(results, save_path='figures/invariant_drift_comparison.pdf'):
    """Generate publication-quality vector invariant drift plot"""
    print(f"Generating vector invariant drift plot...")
    
    seeds = [42, 123, 456]
    drift_results = {'baseline': [], 'ip_pinn': []}
    
    for seed in seeds:
        torch.manual_seed(seed)
        baseline = BaselinePINN().to(device)
        train_model(baseline, epochs=2000)
        drift_results['baseline'].append(calculate_invariant_drift(baseline))
        
        ip_pinn = IP_Pinn(use_feature=True, use_intrusive=True).to(device)
        train_model(ip_pinn, epochs=2000)
        drift_results['ip_pinn'].append(calculate_invariant_drift(ip_pinn))
    
    drift_results['baseline'] = [np.mean(drift_results['baseline'])]
    drift_results['ip_pinn'] = [np.mean(drift_results['ip_pinn'])]
    
    fig, ax = plt.subplots(figsize=(3.25, 2.5), constrained_layout=True)
    
    models = ['Baseline PINN', 'IP-PINN']
    drift_values = [drift_results['baseline'][0], drift_results['ip_pinn'][0]]
    colors = ['#D62728', '#1F77B4']
    
    bars = ax.bar(models, drift_values, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=0.8)
    
    ax.set_ylabel('Invariant Drift (SD)', fontsize=10)
    ax.set_title('Invariant Preservation Comparison', fontsize=11, pad=10)
    ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    
    # FIX: Increase vertical offset and add y-axis headroom
    max_height = max(drift_values)
    offset = max_height * 0.15  # 15% offset instead of fixed 0.02
    
    for bar, value in zip(bars, drift_values):
        # Place text above bar with dynamic offset
        ax.text(bar.get_x() + bar.get_width()/2, value + offset,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # FIX: Add padding to y-axis limits
    ax.set_ylim(0, max_height * 1.25)  # 25% headroom
    
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Vector invariant drift plot saved: {save_path}")

def plot_three_species_comparison(save_path='figures/three_species_comparison.pdf'):
    """Generate publication-quality vector three-species plot"""
    print(f"Generating vector three-species plot...")
    
    def lotka_volterra_3d(t, y):
        x, y, z = y
        dxdt = x * (1 - x - 0.5*y - 0.3*z)
        dydt = y * (1 - 0.4*x - y - 0.2*z)
        dzdt = z * (1 - 0.3*x - 0.3*y - z)
        return [dxdt, dydt, dzdt]
    
    t_span = [0, 50]
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    y0 = [0.5, 0.5, 0.5]
    sol_ref = solve_ivp(lotka_volterra_3d, t_span, y0, t_eval=t_eval, 
                        method='RK45', rtol=1e-8, atol=1e-8)
    
    V_ref = np.sum(sol_ref.y, axis=0)
    drift_ref = np.std(V_ref)
    
    fig = plt.figure(figsize=(6.5, 4.5), constrained_layout=True)
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(sol_ref.t, sol_ref.y[0], 'b-', linewidth=1.5, label='Species 1')
    ax1.plot(sol_ref.t, sol_ref.y[1], 'r-', linewidth=1.5, label='Species 2')
    ax1.plot(sol_ref.t, sol_ref.y[2], 'g-', linewidth=1.5, label='Species 3')
    ax1.set_xlabel('Time', fontsize=10)
    ax1.set_ylabel('Population', fontsize=10)
    ax1.set_title('Three-Species Lotka–Volterra System', fontsize=11, pad=8)
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(sol_ref.t, V_ref, 'k-', linewidth=1.5)
    ax2.set_xlabel('Time', fontsize=10)
    ax2.set_ylabel('Total Population Invariant', fontsize=10)
    ax2.set_title(f'Invariant Drift (σ = {drift_ref:.3f})', fontsize=11, pad=8)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Vector three-species plot saved: {save_path}")
    return drift_ref

def print_statistical_summary(results):
    """Generate statistical summary"""
    print(f"\n{'='*60}")
    print("STATISTICAL SUMMARY - 2D LOTKA-VOLTERRA")
    print(f"{'='*60}")
    
    baseline_mean = np.mean(results['baseline'])
    baseline_std = np.std(results['baseline'], ddof=1)
    ip_pinn_mean = np.mean(results['ip_pinn'])
    ip_pinn_std = np.std(results['ip_pinn'], ddof=1)
    
    t_stat, p_value = stats.ttest_ind(results['baseline'], results['ip_pinn'], equal_var=False)
    
    pooled_std = np.sqrt((len(results['baseline']) - 1) * baseline_std**2 + 
                         (len(results['ip_pinn']) - 1) * ip_pinn_std**2) / \
                 (len(results['baseline']) + len(results['ip_pinn']) - 2)
    cohens_d = (baseline_mean - ip_pinn_mean) / pooled_std
    
    n_bootstrap = 10000
    baseline_boot = np.random.choice(results['baseline'], 
                                     size=(n_bootstrap, len(results['baseline'])), 
                                     replace=True)
    ip_pinn_boot = np.random.choice(results['ip_pinn'], 
                                    size=(n_bootstrap, len(results['ip_pinn'])), 
                                    replace=True)
    diff_boot = np.mean(baseline_boot, axis=1) - np.mean(ip_pinn_boot, axis=1)
    ci_lower, ci_upper = np.percentile(diff_boot, [2.5, 97.5])
    
    print(f"Baseline PINN: Mean ± SD = {baseline_mean:.2e} ± {baseline_std:.2e}")
    print(f"IP-PINN: Mean ± SD = {ip_pinn_mean:.2e} ± {ip_pinn_std:.2e}")
    print(f"Welch's t-test: p = {p_value:.2e}")
    print(f"Cohen's d = {cohens_d:.2f}")
    print(f"Bootstrap CI: [{ci_lower:.2e}, {ci_upper:.2e}]")

def save_raw_data(results, save_path='data/raw_experimental_data.json'):
    """Save raw experimental data"""
    data = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(device),
            'n_runs': 15
        },
        'results': {
            'baseline': results['baseline'],
            'ip_pinn': results['ip_pinn']
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Raw data saved: {save_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(" IP-PINN EXPERIMENTAL PIPELINE - JCP SUBMISSION VERSION")
    print(f" Device: {device}")
    print(f" Runs per configuration: 15")
    print(f" Figures: Vector PDF with embedded fonts")
    print(f"{'='*70}\n")
    
    results = run_experiment()
    
    print(f"\n{'='*70}")
    print(" GENERATING PUBLICATION-QUALITY VECTOR FIGURES")
    print(f"{'='*70}")
    
    plot_phase_space(results)
    plot_invariant_drift(results)
    drift_3species = plot_three_species_comparison()
    print(f"3-Species invariant drift: {drift_3species:.2e}")
    
    print_statistical_summary(results)
    save_raw_data(results)
    
    print(f"\n{'='*70}")
    print(" FIGURE GENERATION COMPLETE")
    print(f"{'='*70}")
    print("✅ All figures are vector-based PDFs with embedded fonts")
    print("✅ Font: Times New Roman (or fallback DejaVu Serif)")
    print("✅ Output: figures/ directory")