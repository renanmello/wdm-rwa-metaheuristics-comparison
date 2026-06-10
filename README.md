# RWA-WDM: Statistical Analysis of Bio-Inspired Algorithms for Optical Networks

**RWA-WDM** is a comprehensive computational framework for simulation and comparative analysis of bio-inspired algorithms applied to the Routing and Wavelength Assignment (RWA) problem in WDM networks with dynamic traffic.

---

## 📄 Problem Description

The Routing and Wavelength Assignment (RWA) problem consists of establishing optical connections (lightpaths) between source-destination pairs while respecting two fundamental constraints:

1. **Wavelength continuity**: The same wavelength must be used throughout the entire route
2. **Wavelength distinctness**: Two lightpaths sharing a link cannot use the same wavelength

This project implements and compares three meta-heuristics for solving the RWA problem in WDM networks with dynamic traffic:

- **Genetic Algorithm (GA)**
- **Particle Swarm Optimization (PSO)**
- **Differential Evolution (DE)**

---

## 🏗️ Framework Architecture
````
RWA-WDM Framework
├── sim-high-resolution.py # Main simulation engine
├── optimized-bioinspired.py # Hyperparameter optimization module
└── pos_process.py # Statistical post-processing
````

### 1. `sim-high-resolution.py` - Main Simulation Engine

Responsible for executing high-resolution experiments (loads from 1 to 200/400 Erlangs).

**Features:**
- Network topology construction (JANET6, RedCLARA, IPÊ)
- K-shortest path calculation using Yen's algorithm
- Execution of three bio-inspired algorithms with optimized parameters
- Dynamic traffic simulation based on Poisson process
- Statistical data collection in CSV and JSON formats

### 2. `optimized-bioinspired.py` - Hyperparameter Optimization

Uses the Optuna framework to find optimal parameters for each algorithm.

**Optimized parameters:**

| Algorithm | Parameters | Search Space |
|:---|:---|:---|
| **GA** | population, generations, crossover, mutation, tournament | [50-200], [20-100], [0.1-0.9], [0.01-0.5], [2-10] |
| **PSO** | population, iterations, inertia, acceleration | [50-200], [20-100], [0.4-0.9], [1.0-2.5] |
| **DE** | population, iterations, CR, F | [50-200], [20-100], [0.3-0.9], [0.3-0.9] |

### 3. `pos_process.py` - Statistical Post-Processing

Generates comprehensive reports and statistical tests.

**Statistical analyses:**
- Friedman test (global comparison of three algorithms)
- Wilcoxon signed-rank test (pairwise comparisons)
- Effect size (Cohen's d)
- Bonferroni correction for multiple comparisons
- Inflection point (load where BP exceeds 1%)

---

## 📊 Evaluated Metrics

| Metric | Description |
|:---|:---|
| **Blocking Probability (BP)** | Proportion of unsuccessful connection requests |
| **Inflection Point** | Load at which BP reaches 1% |
| **Execution Time** | Computational cost per algorithm |
| **Fairness** | BP variation across different O-D pairs |

---

## 🚀 How to Run

### Prerequisites

```
pip install networkx numpy matplotlib pandas scipy pymoo optuna seaborn
```
## 1. Hyperparameter Optimization (one-time execution)
```
python optimized-bioinspired.py
```
## 2. Main Experiment Execution
```
python sim-high-resolution.py
```
## 3. Statistical Report Generation
```
python pos_process.py
```

📁 Output Structure
```
results_GA_highres/
results_PSO_highres/
results_DE_highres/
├── *_raw_*.csv              # Raw data (20 executions × loads)
├── *_stats_*.csv            # Descriptive statistics
├── *_curve_*.png            # Blocking probability curves with 95% CI
├── por_par/                 # Per O-D pair data
│   ├── 0_12/
│   ├── 2_6/
│   └── ...
├── execution_times.csv      # Execution time records
└── pairs_summary_*.csv      # Per-pair comparative summary

relatorio_final/
├── complete_statistical_report_*.txt
├── per_pair_statistical_report_*.txt
├── comparison_*.png
└── metrics_summary_*.csv
```
