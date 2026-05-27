# Structurally Corrected Invariant-Preserving Neural Networks for Conservative Dynamical Systems: A Statistically Validated Approach

> **Published in:** Scientific Journal of King Faisal University: Basic and Applied Sciences (2026) 27:7  
> **DOI:** [10.1007/s44523-026-00007-7](https://doi.org/10.1007/s44523-026-00007-7)  
> **Zenodo Archive:** [10.5281/zenodo.18259016](https://doi.org/10.5281/zenodo.18259016)  
> **GitHub:** [fouaar-cyber/lotka-volterra-ip-pinn](https://github.com/fouaar-cyber/lotka-volterra-ip-pinn)

Quick reference: See [`highlights.txt`](highlights.txt) for key findings.

---

## Key Innovation

| Method | Invariant Drift | Improvement | p-value | Cohen's d |
|:---|:---|:---|:---|:---|
| Baseline PINN | 0.343 ± 0.062 | — | — | — |
| Soft IP-PINN | 0.288 ± 0.086 | 16.1% | 0.063 (n.s.) | 0.73 |
| **SC-PINN (Ours)** | **0.095 ± 0.032** | **72.2%** | **9.9×10⁻¹²** | **5.03** |

**Key finding:** Structural correction achieves **4.5× greater improvement** than soft penalties, with overwhelming statistical significance (n = 15 independent runs).

---

## Quick Start

### Prerequisites

Python 3.11+ recommended.

```bash
pip install -r requirements.txt
```

### Main Experimental Pipeline

```bash
# Original three-way comparison (Table 1, Figures 1–3)
python original_experiments/ippinn_WORKING_72percent.py

# Revision experiments (Figures 4–6, Appendices A–B)
python revision_experiments/run_revision_experiments.py
```

### Generate Figures Separately

```bash
# Original publication figures
python original_experiments/generate_figures.py

# Revision figures (sensitivity, ablation, long-time, trajectories)
python revision_experiments/run_revision_experiments.py
```

---

## Repository Structure

```
├── original_experiments/
│   ├── ippinn_WORKING_72percent.py   # Main pipeline (Table 1, baseline/soft/SC)
│   └── generate_figures.py             # Original figure generation
├── revision_experiments/
│   ├── run_revision_experiments.py     # Sensitivity, ablation, T=500, trajectories
│   └── statistical_analysis.py         # Q-Q plots, normality diagnostics
├── data/
│   ├── all_results.json                # Original 15-run comparison
│   ├── ablation_results.json           # Ablation study (n=15, full vs no-refinement)
│   ├── sensitivity_results.json        # λ sensitivity analysis
│   └── statistical_validation.json     # Shapiro-Wilk, Welch, Mann-Whitney tests
├── figures/
│   ├── figure1_phase_space.pdf         # Phase-plane trajectories (original)
│   ├── figure2_invariant_evolution.pdf # H(t) over T=50 (original)
│   ├── figure2_phase_plane_revised.pdf # Phase-plane with confidence bands (new)
│   ├── figure2b_trajectory_deviation.pdf # Trajectory deviation ±2σ (new)
│   ├── figure3_statistics_revised.pdf  # Bar chart with swarm overlay (revised)
│   ├── figure4_sensitivity_lambda.pdf  # λ sensitivity analysis (new)
│   ├── figure5_longtime_T500.pdf       # Long-time integration T=500 (new)
│   ├── appendix_ablation_study.pdf     # Ablation box plots (new)
│   └── appendix_qq_plots.pdf           # Normality Q-Q plots (new)
├── main.tex                            # LaTeX manuscript
├── main.pdf                            # Compiled paper
├── highlights.txt                      # Key findings summary
├── requirements.txt                    # Python dependencies
├── CITATION.cff                        # Citation metadata
├── LICENSE                             # MIT License
└── README.md                           # This file
```

---

## Methodology

### Forward pass with multiplicative correction

```
z_θ = exp(N_θ(t))          # Base network output
Ĥ = H(z_θ)                  # Current invariant estimate
ẑ = z_θ · exp(-λ(Ĥ - H₀)) # Structural correction
```

Where:
- **λ = 0.1** (fixed projection strength)
- **H₀ = H(x₀, y₀)** (physical invariant from initial conditions)
- Correction is scalar rescaling, not exact orthogonal projection

---

## Statistical Protocol

- **n = 15** independent runs with deterministic seeding (`seed = 2 + i`)
- Three-way comparison: Baseline vs. Soft IP-PINN vs. SC-PINN
- Welch's t-test with variance heterogeneity correction
- Bootstrap confidence intervals (**B = 10,000** replicates)
- Effect size: Cohen's d = 5.03 (very large)
- Non-parametric validation: Mann–Whitney U + Cliff's delta (Appendix B)

### Honest Statistical Reporting

| Test | Baseline | Soft IP-PINN | SC-PINN |
|:---|:---|:---|:---|
| Shapiro-Wilk W | 0.920 | 0.799 | 0.717 |
| Shapiro-Wilk p | 0.191 | 0.004 | < 0.001 |
| Normality | Normal | Non-normal | Non-normal |

> **Note:** Welch's t-test is robust to moderate non-normality at n = 15. All conclusions confirmed by non-parametric tests.

---

## Results Summary

### Main Results (Table 1)

| Feature | Soft Penalty | Structural Correction (Ours) |
|:---|:---|:---|
| Mechanism | Loss function penalty | Forward-pass correction |
| Exactness | Approximate | Structural (empirical) |
| Improvement | 16.1% (not significant) | 72.2% (p < 10⁻¹¹) |
| Hyperparameter sensitivity | High (tuning critical) | Low (fixed λ = 0.1) |

### Ablation Study (Appendix A)

Comparison of full model (with refinement loss) vs. architectural correction alone:

| Configuration | Mean ± SD | Welch p | Cohen's d |
|:---|:---|:---|:---|
| Full model | 0.922 ± 0.112 | — | — |
| Without refinement | 0.919 ± 0.115 | 0.935 | 0.03 |

**Conclusion:** The architectural correction alone provides the dominant invariant stabilization. The refinement loss serves as an auxiliary optimization stabilizer during training.

### Sensitivity Analysis (Figure 6)

Systematic evaluation of projection strength λ ∈ [0.01, 2.0] (n = 5 per value):
- **Robust region:** λ ∈ [0.05, 0.5] yields comparable performance
- **Selected value:** λ = 0.1 (conservative middle ground)
- **Instability:** λ > 1.0 introduces training stiffness

### Long-Time Integration (Figure 5)

Models trained on [0, 100], evaluated on [0, 500]:
- **SC-PINN:** Maintains bounded deviation (stable behavior)
- **Baseline:** Exhibits unbounded drift

### Trajectory Confidence (Figure 2 revised)

Maximum Euclidean distance from RK4 reference over T = 50 (n = 15):

| Method | Max Distance | Mean Distance |
|:---|:---|:---|
| Baseline PINN | 2.34 ± 0.41 | 0.89 ± 0.15 |
| **SC-PINN** | **0.87 ± 0.19** | **0.31 ± 0.08** |

---

## Figure Generation Guide

| Figure | Description | Source Script |
|:---|:---|:---|
| Figure 1 | Phase-space trajectories | `original_experiments/generate_figures.py` |
| Figure 2 | Invariant evolution T=50 | `original_experiments/generate_figures.py` |
| Figure 2 revised | Phase-plane with confidence bands | `revision_experiments/run_revision_experiments.py` |
| Figure 2b | Trajectory deviation ±2σ | `revision_experiments/run_revision_experiments.py` |
| Figure 3 revised | Statistical comparison with swarm | Manual plotting from `data/all_results.json` |
| Figure 4 | λ sensitivity analysis | `revision_experiments/run_revision_experiments.py` |
| Figure 5 | Long-time integration T=500 | `revision_experiments/run_revision_experiments.py` |
| Appendix A | Ablation study box plots | `revision_experiments/run_revision_experiments.py` |
| Appendix B | Q-Q plots for normality | `revision_experiments/statistical_analysis.py` |

---

## Citation

If you use this code or data, please cite both the paper and the software archive:

### Published article

```bibtex
@article{ouaar2026structurally,
  title={Structurally corrected invariant-preserving neural networks for conservative dynamical systems: a statistically validated approach},
  author={Ouaar, Fatima},
  journal={Scientific Journal of King Faisal University: Basic and Applied Sciences},
  year={2026},
  volume={27},
  pages={7},
  doi={10.1007/s44523-026-00007-7}
}
```

### Software archive

```bibtex
@software{ouaar2026sc_pinn,
  author={Ouaar, Fatima},
  title={Structurally corrected invariant-preserving neural networks for conservative dynamical systems},
  year={2026},
  doi={10.5281/zenodo.18259016},
  url={https://github.com/fouaar-cyber/lotka-volterra-ip-pinn}
}
```

---

## License

MIT License — see [`LICENSE`](LICENSE) file for details.

Copyright (c) 2026 Fatima Ouaar

---

## Version History

| Version | Date | Description |
|:---|:---|:---|
| **v3.0 (Current)** | **2026-05** | **Published in SJKFU: Basic Appl. Sci. 27:7** |
| | | - Final published version with de-anonymized links |
| | | - Added sensitivity analysis (Figure 6) |
| | | - Added long-time integration T=500 (Figure 5) |
| | | - Added trajectory confidence bands (Figure 2 revised) |
| | | - Added ablation study with n=15 (Appendix A) |
| | | - Added normality diagnostics and non-parametric validation (Appendix B) |
| | | - Honest statistical reporting (Shapiro-Wilk, Mann–Whitney U, Cliff's delta) |
| | | - Expanded mathematical derivations (Poisson tensor, convergence analysis) |
| | | - Added generalization discussion for multiple invariants |
| v2.0 | 2026-02 | Structural correction architecture with 72% improvement |
| | | - Three-way statistical comparison (Baseline/Soft/SC-PINN) |
| | | - Corrected projection formulation |
| | | - Physically consistent H₀ from initial conditions |
| v1.0 | 2025-XX | Superseded: Soft penalty approach with inconsistent results |
| | | - 16% improvement (not statistically significant) |
| | | - Hyperparameter sensitivity issues |

---

## Contact

**Fatima Ouaar**  
Mathematics Department, University of Biskra, Biskra, Algeria  
📧 f.ouaar@univ-biskra.dz  
🐙 GitHub: [@fouaar-cyber](https://github.com/fouaar-cyber)
