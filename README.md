# RWA-WDM: Statistical Analysis of Bio-Inspired Algorithms for Optical Networks

Simulation and comparative-analysis framework for **Genetic Algorithm (GA)**, **Particle Swarm Optimization (PSO)** and **Differential Evolution (DE)** applied to the **Routing and Wavelength Assignment (RWA)** problem in WDM optical networks under dynamic traffic.

The three algorithms are evaluated under a **unified experimental protocol**: same solution representation, same fitness function, same candidate-route space, same traffic model and **Common Random Numbers (CRN)** for paired statistical comparison.

## Contents

- [Problem description](#problem-description)
- [Repository structure](#repository-structure)
- [Experimental protocol](#experimental-protocol)
- [Metrics and statistical analysis](#metrics-and-statistical-analysis)
- [Installation](#installation)
- [How to run](#how-to-run)
- [Output structure](#output-structure)
- [Reproducibility notes](#reproducibility-notes)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Problem description

RWA consists of establishing lightpaths between source-destination (S-D) pairs by jointly choosing a route and a wavelength for each connection request, subject to:

1. **Wavelength distinctness**: two lightpaths sharing a link cannot use the same wavelength.
2. **Wavelength assignment with conversion**: the wavelength is assigned link by link. The previous wavelength is kept whenever it is free; otherwise the first free wavelength of the link is used and **one conversion is counted**. A request is blocked only if at least one link of the route has no free wavelength. Conversion is allowed at any intermediate node.

Connection requests arrive dynamically (Poisson arrivals, exponential holding times) and are spread uniformly over five predefined S-D pairs per topology.

## Repository structure

```text
.
├── sim-high-resolution.py     # Simulation engine (GA, PSO, DE, traffic, outputs)
├── optimized-bioinspired.py   # Hyperparameter calibration with Optuna
├── pos_process.py             # Statistical post-processing and reports
├── diagnose_pairing.py        # (optional) consistency check of result files
└── LICENSE
```

### `sim-high-resolution.py`

- Topologies: Janet6, RedCLARA and Rede Ipê.
- Candidate routes: up to K shortest simple paths per S-D pair (Yen's algorithm, NetworkX), stored in a look-up table (LUT).
- Solution encoding: integer vector with one LUT route index per S-D pair. PSO and DE work in continuous space and are discretized with `clip(round(x), 0, K_i - 1)`.
- Dynamic traffic simulation and collection of blocking probability, conversions and timings.

### `optimized-bioinspired.py`

Calibrates the hyperparameters with [Optuna](https://optuna.org/) in two stages:

1. Global parameters shared by all algorithms (fitness weights, K, population size, generations).
2. Algorithm-specific parameters.

Each stage minimizes the blocking probability (100 trials, TPE sampler with seed 42, `MedianPruner`) for every topology and wavelength scenario. The selected values are aggregated by the **median** across topologies.

### `pos_process.py`

Reads the `results_*_highres/` folders and generates statistical reports, comparison plots and summary tables (see [Output structure](#output-structure)).

## Experimental protocol

### Fitness function

For each S-D pair the fitness follows Equation 1 of the paper:

```text
fit = 1 / (alpha * nc + beta * rl)
```

- `rl`: number of links (hops) of the selected route.
- `nc`: average number of wavelength conversions **actually observed** per established connection of that S-D pair.
- `alpha = 0.45` (conversion weight) and `beta = 0.55` (hop weight).

The total fitness of a candidate solution is the sum over its five S-D pairs. `nc` is measured by a **short dynamic simulation** of the whole candidate solution (all five routes together, so shared links and congestion matter), using the same wavelength assignment as the final simulation:

- 400 requests per replica, the first 100 discarded as warm-up, 2 independent replicas;
- evaluation load calibrated per topology/wavelength scenario as the load where shortest-path routing reaches about 2 % blocking;
- fitness values are cached per distinct solution.

### Global parameters (same for GA, PSO and DE)

| Parameter | Value |
|:---|:---|
| Link (hop) weight, beta | 0.55 |
| Conversion weight, alpha | 0.45 |
| Maximum candidate routes, K | 150 |
| Population / swarm size | 120 |
| Generations / iterations | 35 |

### Algorithm-specific parameters

| Algorithm | Parameter | 40 wavelengths | 80 wavelengths |
|:---|:---|:---|:---|
| GA | Crossover rate | 0.2710 | 0.6451 |
| GA | Mutation rate | 0.2952 | 0.1997 |
| GA | Tournament size | 4 | 3 |
| DE | Mutation factor F | 0.4713 | 0.5436 |
| DE | Recombination rate CR | 0.5355 | 0.3402 |
| PSO | Inertia weight | 0.6719 | 0.7333 |
| PSO | Cognitive coefficient c1 | 1.9636 | 1.6527 |
| PSO | Social coefficient c2 | 1.4178 | 2.0948 |

GA uses tournament selection, single-point crossover, per-gene mutation and elitism (about 10 % of the population). DE uses the `DE/rand/1/bin` variant. PSO and DE are provided by [pymoo](https://pymoo.org/).

### Scenarios

| Topology | Wavelengths | Loads (Erlangs) | Executions |
|:---|:---|:---|:---|
| Janet6, RedCLARA, Rede Ipê | 40 | 1 to 200 | 20 |
| Janet6, RedCLARA, Rede Ipê | 80 | 1 to 400 | 20 |

Each simulation processes 5,000 connection requests.

## Metrics and statistical analysis

| Metric | Description |
|:---|:---|
| **Blocking probability (BP)** | Blocked requests / total requests, aggregated over the five S-D pairs and averaged over the 20 executions (95 % CI with Student's t, 19 degrees of freedom) |
| **BT1%** | 1 % Blocking Threshold: lowest load where the mean BP reaches or exceeds 1 % |
| **Variability** | Standard deviation of BP across executions, averaged over the evaluated loads |
| **Execution time** | Algorithm time, simulation time and total time per execution |
| **Conversions** | Mean conversions per established connection and fraction of connections that needed conversion |
| **Per-pair blocking** | BP of each S-D pair, to inspect how blocking is distributed |

Statistical analysis (paired through CRN):

- Friedman test (global comparison of the three algorithms)
- Wilcoxon signed-rank test (pairwise comparisons)
- Effect size
- Bonferroni correction for multiple comparisons

## Installation

Requires Python 3.10 or newer.

```bash
pip install networkx numpy matplotlib pandas scipy pymoo optuna
```

To reproduce the exact environment, install the versions listed in `requirements.txt`.

## How to run

The workflow has two required steps, in this order: **simulation**, then **post-processing**. Run every command from the repository root.

### Step 1 (optional) - Hyperparameter calibration

The final hyperparameters used in the paper are already set in `sim-high-resolution.py` and listed in [Experimental protocol](#experimental-protocol). Run this step only if you want to recalibrate them.

```bash
python optimized-bioinspired.py
```

### Step 2 - Run the simulations

```bash
python sim-high-resolution.py
```

This runs every scenario defined in `main()`: 3 topologies x {40 wavelengths up to 200 Erlangs, 80 wavelengths up to 400 Erlangs}, 20 executions each. For every scenario the three algorithms (PSO, DE and GA) are executed with the same traffic seeds (CRN).

The scenarios and the number of executions are configured at the top of `main()`:

```python
NUM_EXECUTIONS = 20
experiments = [
    # (name, graph function, S-D pairs, wavelengths, max load)
    ('JANET6', get_janet6_graph, JANET_PAIRS, 40, 200),
    # ... remaining scenarios
]
```

Results are written to `results_PSO_highres/`, `results_DE_highres/` and `results_GA_highres/` (see [Output structure](#output-structure)). The full run takes several hours, depending on your hardware.

**Quick test (recommended before the full run):** in `main()`, set `NUM_EXECUTIONS = 2` and keep a single short scenario, for example:

```python
NUM_EXECUTIONS = 2
experiments = [
    ('JANET6', get_janet6_graph, JANET_PAIRS, 40, 20),
]
```

Run the script, check that the three `results_*_highres/` folders were created, and restore the original configuration before the real run.

### Step 3 - Statistical post-processing

Run it from the same folder that contains the `results_*_highres/` folders:

```bash
python pos_process.py
```

It reads the three result folders and writes everything to `relatorio_final/` (statistical reports, comparison plots and summary tables).

To analyze one specific run, set `TIMESTAMP_FILTER` at the top of `pos_process.py` (for example `"20260930"`).

> **Before a new full run**, move or delete the previous `results_*_highres/` folders. The post-processing uses the most recent file of each scenario and could otherwise mix results from different runs.

### Workflow summary

```text
sim-high-resolution.py  -->  results_{PSO,DE,GA}_highres/  -->  pos_process.py  -->  relatorio_final/
     (simulation)             (raw data, stats, solutions)       (statistics)        (reports, plots, tables)
```

## Output structure

```text
results_GA_highres/   results_PSO_highres/   results_DE_highres/
├── *_raw_*.csv              # BP per execution (rows) and load (columns)
├── *_stats_*.csv            # Mean, standard deviation, 95% CI and related statistics per load
├── *_best_solutions_*.csv   # Selected route index per S-D pair, hops, fitness, observed nc, timings
├── *_conversions_*.csv      # Conversions per established connection, per load
├── *_curve_*.png            # BP curves with 95% CI
├── execution_times.csv      # Execution time records
├── pairs_summary_*.csv      # Per-pair comparative summary
└── por_par/                 # Per S-D pair data
    ├── 0_12/
    ├── 2_6/
    └── ...

relatorio_final/
├── relatorio_estatistico_completo_*.txt      # Friedman, Wilcoxon, effect size, Bonferroni
├── relatorio_estatistico_por_par_*.txt       # Same analysis per S-D pair
├── comparacao_*.png                          # Algorithm comparison plots
├── resumo_metricas_*.csv                     # Global metrics summary
├── resumo_metricas_por_par_*.csv             # Per-pair metrics summary
├── resumo_metricas_por_par_pivot_*.csv       # Per-pair pivot table
├── resumo_solucoes_*.csv                     # Chosen solutions per algorithm
└── resumo_conversoes_*.csv                   # Conversion summary
```

## Reproducibility notes

- **Common Random Numbers**: for each (execution, load) the traffic seed is the same for GA, PSO and DE; only the algorithm seed differs. This is what makes the Friedman and Wilcoxon tests paired.
- **Fitness evaluation traffic** uses its own random generator and seeds that never coincide with the seeds of the final evaluation, so the optimization does not see the evaluation traffic.
- All hyperparameters are fixed before the comparison; the Optuna sampler uses seed 42.
- The best solution of each execution is saved in `*_best_solutions_*.csv`, so every reported curve can be traced back to the routes that produced it.

## Limitations

- Three topologies, three classical metaheuristics and one traffic model (Poisson arrivals, exponential holding times).
- Five predefined S-D pairs per topology and one fixed route per pair.
- Wavelength conversion is assumed possible at every intermediate node.
- Results should not be generalized automatically to other topologies, traffic models or optimization strategies.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{teixeira2026unified,
  title   = {A Unified Experimental Comparison of Bio-Inspired Algorithms for Routing and Wavelength Assignment under Dynamic Traffic in WDM Optical Networks},
  author  = {Teixeira, Diego Bento Aires and Mello, Renan Pereira and dos Santos, Albert Einstein Coutinho and Ara{\'u}jo, Josivaldo de Souza and Costa, Fernando Augusto Ribeiro and Lobato, Fabricio Rossy de Lima and Seruffo, Marcos C{\'e}sar da Rocha},
  year    = {2026},
  note    = {Update with journal and DOI when available}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file.

## Acknowledgments

- Network topologies: Janet6, RedCLARA and Rede Ipê.
- Libraries: [pymoo](https://pymoo.org/) (PSO and DE), [Optuna](https://optuna.org/) (hyperparameter optimization), [NetworkX](https://networkx.org/) (graphs and Yen's algorithm).
- Supported in part by CAPES (Finance Code 001), CNPq and the Federal University of Pará (UFPA).

## Contact

- Corresponding author: Diego Bento Aires Teixeira (diegoaires@gmail.com)
- Institution: Federal University of Pará (UFPA), Belém, Brazil
