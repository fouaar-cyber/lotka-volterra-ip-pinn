# Hard-Constrained Invariant-Preserving Neural Networks for Conservative Dynamical Systems

[![DOI](https://zenodo.org/badge/XXXXX.svg)](https://doi.org/10.5281/zenodo.18259016)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📄 **Quick reference:** See [HIGHLIGHTS](HIGHLIGHTS) for key findings


**Publication:** "Hard-Constrained Invariant-Preserving Neural Networks for Conservative Dynamical Systems: A Statistically Validated Approach"

A **reproducible** framework demonstrating that hard structural constraints (multiplicative manifold projection) dramatically outperform soft penalty methods for invariant preservation in neural network integrators.

---

## 🎯 Key Innovation

**Hard IP-PINN vs. Soft IP-PINN vs. Baseline:**

| Method | Invariant Drift | Improvement | p-value | Cohen's d |
|--------|----------------|-------------|---------|-----------|
| Baseline PINN | 0.343 ± 0.062 | — | — | — |
| Soft IP-PINN | 0.288 ± 0.086 | 16.1% | 0.56 (n.s.) | 0.73 |
| **Hard IP-PINN (Ours)** | **0.095 ± 0.032** | **72.2%** | **9.9×10⁻¹²** | **5.03** |

**Key finding:** Hard structural constraints achieve **4.5× greater improvement** than soft penalties, with overwhelming statistical significance.

---

## 🚀 Quick Start

### Prerequisites

# Python 3.11+ recommended
pip install -r requirements.txt

# Run Complete Analysis
# Execute full pipeline (≈30 minutes on CPU)
python ippinn_hard_constraint.py

# Expected Output

✅ 3 publication-quality figures in figures/
✅ Raw experimental data in data/all_results.json
✅ Statistical summary: p < 10⁻¹¹, Cohen's d = 5.03
 
# Repository Structure

├── ippinn_hard_constraint.py    # Main experimental pipeline (FINAL VERSION)
├── requirements.txt             # Python dependencies
├── main.tex                     # LaTeX manuscript (submission-ready)
├── refs.bib                     # Bibliography
├── HIGHLIGHTS                   # Key findings summary for quick reference  <-- ADDED
├── figures/                     # Generated vector PDFs
│   ├── figure1_phase_space.pdf
│   ├── figure2_invariant_evolution.pdf
│   └── figure3_statistics.pdf
├── data/                        # Experimental results
│   └── all_results_GOOD_72percent.json  # <-- KEY FILE
├── CITATION.cff                 # Citation metadata
├── LICENSE                      # MIT License
└── .gitignore                   # Repository hygiene


# Methodology

Hard Constraint Architecture
Unlike soft penalty methods that add invariant terms to the loss function, our Hard IP-PINN structurally projects network outputs onto the invariant manifold:

# Forward pass with multiplicative projection
z_θ = exp(N_θ(t))  # Base network output
Ĥ = H(z_θ)         # Current invariant estimate
ẑ = z_θ · exp(-λ(Ĥ - H₀))  # Projective correction
  
Where:

    λ = 0.1 (fixed projection strength, not tuned per run)
    H₀ = H(x₀, y₀) (physical invariant from initial condition)
    Projection is scalar rescaling, not exact orthogonal projection

#  Statistical Protocol

    n = 15 independent runs with deterministic seeding (seed = 42 + i)
    Three-way comparison: Baseline vs. Soft IP-PINN vs. Hard IP-PINN
    Welch's t-test with variance heterogeneity correction
    Bootstrap confidence intervals (B = 10,000 replicates)
    Effect size: Cohen's d = 5.03 (very large)

# Results Summary

Phase-Space Preservation
Hard IP-PINN maintains stable periodic orbits tightly confined to invariant level sets, while baseline exhibits drift-induced distortion.
Long-Time Stability
Over T = 100 time units:

    Baseline: RMS invariant deviation = 0.343 (substantial drift)
    Hard IP-PINN: RMS invariant deviation = 0.095 (tight confinement)

# Comparison to Prior Approaches

| Feature                    | Soft Penalty \[Jin et al.] | Hard Constraint (Ours)  |
| -------------------------- | -------------------------- | ----------------------- |
| Mechanism                  | Loss function penalty      | Forward-pass projection |
| Exactness                  | Approximate                | Structural (empirical)  |
| Improvement                | 16% (not significant)      | **72% (p < 10⁻¹¹)**     |
| Hyperparameter sensitivity | High (λ tuning critical)   | Low (fixed λ = 0.1)     |


# Citation

@article{ouaar2026hard,
  title={Hard-Constrained Invariant-Preserving Neural Networks for Conservative Dynamical Systems: A Statistically Validated Approach},
  author={OUAAR, Fatima},
  journal={Chaos, Solitons \& Fractals},
  year={2026},
  volume={XX},
  pages={XXXXX},
  doi={10.1016/j.chaos.2026.XXXXX}
}

 # Contact

Fatima OUAAR
Laboratory of Mathematical Analysis, Probability and Optimization
Mohamed Khider University, Biskra, Algeria
Email: f.ouaar@univ-biskra.dz
GitHub: @fouaar-cyber

#  Version History 

    v2.0 (Current): Hard constraint architecture with 72% improvement
        Three-way statistical comparison (Baseline/Soft/Hard)
        Corrected projection formulation (non-circular)
        Physically consistent H₀ from initial conditions
    v1.0 (Previous): Soft penalty approach with inconsistent results
        16% improvement (not statistically significant)
        Hyperparameter sensitivity issues





















