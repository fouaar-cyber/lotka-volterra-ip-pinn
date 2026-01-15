# Statistically Validated IP-PINN for Conservative Dynamical Systems

A **reproducible** framework for invariant-preserving physics-informed neural networks (IP-PINN), validated on Lotka–Volterra systems with rigorous statistical protocols.

https://doi.org/10.5281/zenodo.18259016

## 🎯 What This Repository Provides

- **Complete experimental pipeline** with 15 independent runs per configuration
- **Statistical validation**: Welch's t-test, bootstrap CI, Cohen's d effect sizes
- **Publication-ready figures**: Vector PDFs with embedded fonts
- **JCP compliance**: Exceeds Journal of Computational Physics reproducibility standards

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies (Python 3.11+ recommended)
pip install -r requirements.txt

### Run Complete Analysis

# Execute full pipeline (≈25 minutes on CPU)
python ip_pinn_vector_figures.py

### Expected Output

    ✅ 3 publication-quality figures in figures/
    ✅ Raw experimental data in data/raw_experimental_data.json
    ✅ Statistical summary with p < 0.001, Cohen's d = 2.59

###  Key Results (Invariant Drift)
| Method         | Mean ± SD       | p-value     | Cohen's d | Improvement |
| -------------- | --------------- | ----------- | --------- | ----------- |
| Baseline PINN  | 20.7 ± 0.0      | —           | —         | —           |
| IP-PINN (Ours) | **5.25 ± 8.44** | **2.71e-6** | **2.59**  | **75% ↓**   |

### Repository Structure
├── ip_pinn_vector_figures.py    # Main experimental pipeline
├── requirements.txt             # Python dependencies
├── main.tex                     # LaTeX manuscript
├── refs.bib                     # Bibliography
├── figures/                     # Generated vector PDFs
│   ├── phase_space_comparison.pdf
│   ├── invariant_drift_comparison.pdf
│   └── three_species_comparison.pdf
├── data/                        # Raw experimental data
│   └── raw_experimental_data.json
├── CITATION.cff                 # Citation metadata
├── LICENSE                      # MIT License
└── .gitignore                   # Repository hygiene

### Citation
@article{ouaar2026ippinn,
  title={Statistically Validated Invariant-Preserving Physics-Informed Neural Networks for Conservative Dynamical Systems},
  author={OUAAR, Fatima},
  journal={Journal of Computational Physics},
  year={2026},
  doi={10.1016/j.jcp.2026.XXXXX}  # Update after acceptance
}

### Reproducibility

    Hardware: CPU-only (no GPU required)
    Seeding: Deterministic (fixed seeds = 42 to 3031)
    Data: All 15-run raw data archived in JSON
    Figures: Vector PDFs with editable text


### 📧 Contact
Fatima OUAAR
Mohamed Khider University, Biskra, Algeria
Email: f.ouaar@univ-biskra.dz