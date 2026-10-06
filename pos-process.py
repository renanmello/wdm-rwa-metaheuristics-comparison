"""
COMPREHENSIVE STATISTICAL POST-PROCESSING WITH STATISTICAL TESTS
Reads files inside results_*_highres folders
Includes: Wilcoxon, p-value, Friedman, CI, etc.
Supports GLOBAL and per-OD-PAIR data

IMPORTANT — PAIRING (COMMON RANDOM NUMBERS):
    Friedman and Wilcoxon signed-rank tests require that the N executions
    of GA, PSO, and DE were submitted to the SAME traffic realization
    (Common Random Numbers). This is guaranteed by sim-high-resolution.py
    through the separation between algo_seed (different per algorithm) and
    traffic_seed (SAME for the 3 algorithms in each (execution, load)).

    If the simulator is changed and pairing is broken, the tests
    lose validity and should be replaced by Kruskal-Wallis and
    Mann-Whitney U (independent samples).

CORRECTIONS APPLIED (v3):
- NST / inflection point: first load with mean blocking >= 1% (as in the paper).
- New: summary of best solutions (*_best_solutions_*.csv written by the simulator).

CORRECTIONS APPLIED (v2):
- Files are now sorted by modification time (most recent is used).
- Optional timestamp filter (TIMESTAMP_FILTER) to analyze a specific run.
- File path is stored in the data dict for auditing.
- Clearer warning when no data is found.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from scipy.stats import wilcoxon, friedmanchisquare

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {'PSO': 'blue', 'DE': 'green', 'GA': 'red'}
MARKERS = {'PSO': 'o', 'DE': 's', 'GA': '^'}

# ============================================
# OPTIONAL: TIMESTAMP FILTER
# ============================================
# Set to None to use the most recent files (default).
# Set to a string like "20260915" to filter by timestamp.
TIMESTAMP_FILTER = None


# ============================================
# ALGORITHM NAME NORMALIZATION
# ============================================
ALGO_ALIASES = {
    'AG': 'GA',   # normalize AG -> GA
    'GA': 'GA',
    'PSO': 'PSO',
    'DE': 'DE'
}

ALGORITHMS = ['PSO', 'DE', 'GA']


def normalize_algo(name: str) -> str:
    """Normalize algorithm name to canonical form (GA, PSO, DE)."""
    return ALGO_ALIASES.get(name.upper(), name.upper())


def _get_most_recent_files(files, timestamp_filter=None):
    """Sort files by modification time (most recent last) and apply filter."""
    if timestamp_filter is not None:
        files = [f for f in files if timestamp_filter in f]
    return sorted(files, key=os.path.getmtime)


# ============================================
# DATA LOADING
# ============================================

def load_raw_data(base_dir=".", timestamp_filter=TIMESTAMP_FILTER):
    """Load RAW data (global) from folders."""
    raw_data = {}
    result_dirs = glob.glob("results_*_highres")

    print(f"Folders found: {result_dirs}")

    for dir_name in result_dirs:
        raw_files = _get_most_recent_files(
            glob.glob(f"{dir_name}/*_raw_*.csv"), timestamp_filter
        )

        for file in raw_files:
            basename = os.path.basename(file).replace('.csv', '')
            parts = basename.split('_')

            if len(parts) >= 6:
                algoritmo = normalize_algo(parts[0])
                rede = parts[2]
                lambdas = parts[3].replace('l', '')
                max_load = parts[4].replace('loads', '')

                df = pd.read_csv(file)
                df.columns = df.columns.astype(str)

                key = f"{rede}_{lambdas}l"
                raw_data.setdefault(key, {}).setdefault(max_load, {})

                raw_data[key][max_load][algoritmo] = {
                    'type': 'global',
                    'data': df,
                    'file': file
                }

                print(f"  Loaded (global): {algoritmo} - {rede} - {lambdas}l - {max_load} loads")
                print(f"    from: {os.path.basename(file)}")

    return raw_data


def load_raw_data_by_pair(base_dir=".", timestamp_filter=TIMESTAMP_FILTER):
    """Load RAW data per OD pair."""
    raw_data_by_pair = {}
    result_dirs = glob.glob("results_*_highres")

    for dir_name in result_dirs:
        for pair_dir in glob.glob(f"{dir_name}/por_par/*"):
            pair_name = os.path.basename(pair_dir)
            raw_files = _get_most_recent_files(
                glob.glob(f"{pair_dir}/*_raw_*.csv"), timestamp_filter
            )

            for file in raw_files:
                basename = os.path.basename(file).replace('.csv', '')
                parts = basename.split('_')

                if len(parts) >= 6:
                    algoritmo = normalize_algo(parts[0])
                    rede = parts[2]
                    lambdas = parts[3].replace('l', '')
                    max_load = parts[4].replace('loads', '')

                    df = pd.read_csv(file)
                    df.columns = df.columns.astype(str)

                    key = f"{rede}_{lambdas}l"
                    raw_data_by_pair.setdefault(key, {}).setdefault(max_load, {}).setdefault(pair_name, {})

                    raw_data_by_pair[key][max_load][pair_name][algoritmo] = df

                    print(f"  Loaded (pair {pair_name}): {algoritmo} - {rede} - {lambdas}l - {max_load} loads")

    return raw_data_by_pair


def load_stats_data(base_dir=".", timestamp_filter=TIMESTAMP_FILTER):
    """Load STATS data (global) from folders."""
    stats_data = {}
    result_dirs = glob.glob("results_*_highres")

    for dir_name in result_dirs:
        stats_files = _get_most_recent_files(
            glob.glob(f"{dir_name}/*_stats_*.csv"), timestamp_filter
        )

        for file in stats_files:
            basename = os.path.basename(file).replace('.csv', '')
            parts = basename.split('_')

            if len(parts) >= 6:
                algoritmo = normalize_algo(parts[0])
                rede = parts[2]
                lambdas = parts[3].replace('l', '')
                max_load = parts[4].replace('loads', '')

                df = pd.read_csv(file)

                key = f"{rede}_{lambdas}l"
                stats_data.setdefault(key, {}).setdefault(max_load, {})

                stats_data[key][max_load][algoritmo] = {
                    'type': 'global',
                    'data': df,
                    'file': file
                }

                print(f"  Loaded stats (global): {algoritmo} - {rede} - {lambdas}l - {max_load} loads")

    return stats_data


def load_stats_data_by_pair(base_dir=".", timestamp_filter=TIMESTAMP_FILTER):
    """Load STATS data per OD pair."""
    stats_data_by_pair = {}
    result_dirs = glob.glob("results_*_highres")

    for dir_name in result_dirs:
        for pair_dir in glob.glob(f"{dir_name}/por_par/*"):
            pair_name = os.path.basename(pair_dir)
            stats_files = _get_most_recent_files(
                glob.glob(f"{pair_dir}/*_stats_*.csv"), timestamp_filter
            )

            for file in stats_files:
                basename = os.path.basename(file).replace('.csv', '')
                parts = basename.split('_')

                if len(parts) >= 6:
                    algoritmo = normalize_algo(parts[0])
                    rede = parts[2]
                    lambdas = parts[3].replace('l', '')
                    max_load = parts[4].replace('loads', '')

                    df = pd.read_csv(file)

                    key = f"{rede}_{lambdas}l"
                    stats_data_by_pair.setdefault(key, {}).setdefault(max_load, {}).setdefault(pair_name, {})

                    stats_data_by_pair[key][max_load][pair_name][algoritmo] = df

                    print(f"  Loaded stats (pair {pair_name}): {algoritmo} - {rede} - {lambdas}l - {max_load} loads")

    return stats_data_by_pair


# ============================================
# STATISTICAL TESTS
# ============================================

def calculate_wilcoxon_pairwise(data_algo1, data_algo2, load):
    """Wilcoxon signed-rank test for a pair of algorithms at a given load."""
    load_str = str(int(load)) if load == int(load) else str(load)

    if load_str not in data_algo1.columns or load_str not in data_algo2.columns:
        return 1.0, False

    values1 = data_algo1[load_str].dropna().values
    values2 = data_algo2[load_str].dropna().values

    if len(values1) != len(values2):
        return 1.0, False

    if np.array_equal(values1, values2):
        return 1.0, False

    try:
        stat, p_value = wilcoxon(values1, values2, alternative='two-sided')
        return p_value, p_value < 0.05
    except Exception:
        return 1.0, False


def calculate_friedman_test(data_pso, data_de, data_ga, load):
    """Friedman test for the three algorithms at a given load."""
    load_str = str(int(load)) if load == int(load) else str(load)

    if (load_str not in data_pso.columns or
        load_str not in data_de.columns or
        load_str not in data_ga.columns):
        return 1.0

    values_pso = data_pso[load_str].dropna().values
    values_de = data_de[load_str].dropna().values
    values_ga = data_ga[load_str].dropna().values

    min_len = min(len(values_pso), len(values_de), len(values_ga))
    values_pso = values_pso[:min_len]
    values_de = values_de[:min_len]
    values_ga = values_ga[:min_len]

    try:
        stat, p_value = friedmanchisquare(values_pso, values_de, values_ga)
        return p_value
    except Exception:
        return 1.0


def calculate_effect_size(data_algo1, data_algo2, load):
    """Cohen's d effect size between two algorithms."""
    load_str = str(int(load)) if load == int(load) else str(load)

    if load_str not in data_algo1.columns or load_str not in data_algo2.columns:
        return 0.0, "no data"

    values1 = data_algo1[load_str].dropna().values
    values2 = data_algo2[load_str].dropna().values

    mean1, mean2 = np.mean(values1), np.mean(values2)
    std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)

    n1, n2 = len(values1), len(values2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0, "negligible"

    cohens_d = abs(mean1 - mean2) / pooled_std

    if cohens_d < 0.2:
        interpretation = "negligible"
    elif cohens_d < 0.5:
        interpretation = "small"
    elif cohens_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return cohens_d, interpretation


# ============================================
# GLOBAL STATISTICAL REPORT
# ============================================

def generate_statistical_report(raw_data, output_dir="relatorio_final"):
    """Generate comprehensive statistical report (global data)."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{output_dir}/relatorio_estatistico_completo_{timestamp}.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("COMPLETE STATISTICAL REPORT - GLOBAL DATA\n")
        f.write("Tests: Wilcoxon, Friedman, p-value, Effect Size (Cohen's d)\n")
        f.write("Method: Common Random Numbers (CRN) for pairing\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")

        for config_key in sorted(raw_data.keys()):
            rede, lambdas = config_key.split('_')
            f.write(f"\n{'#' * 80}\n")
            f.write(f"# NETWORK: {rede} | WAVELENGTHS: {lambdas}\n")
            f.write(f"{'#' * 80}\n\n")

            for max_load_str in sorted(raw_data[config_key].keys(), key=int):
                max_load_int = int(max_load_str)
                f.write(f"\n{'=' * 70}\n")
                f.write(f"RESULTS FOR LOADS FROM 1 TO {max_load_int} ERLANGS\n")
                f.write(f"{'=' * 70}\n\n")

                available = [a for a in ALGORITHMS if a in raw_data[config_key][max_load_str]]
                if len(available) < 3:
                    f.write(f"Incomplete data. Available: {available}\n\n")
                    continue

                data_pso = raw_data[config_key][max_load_str]['PSO']['data']
                data_de = raw_data[config_key][max_load_str]['DE']['data']
                data_ga = raw_data[config_key][max_load_str]['GA']['data']

                loads_analysis = [50, 100, 150, 200]
                if max_load_int >= 400:
                    loads_analysis.extend([250, 300, 350, 400])

                # 1. FRIEDMAN
                f.write("FRIEDMAN TEST (Global comparison of the 3 algorithms)\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Load':>10} | {'p-value':<15} | {'Significant':<15}\n")
                f.write("-" * 60 + "\n")

                for load in loads_analysis:
                    if load <= max_load_int:
                        p_valor = calculate_friedman_test(data_pso, data_de, data_ga, load)
                        significativo = p_valor < 0.05
                        p_str = f"{p_valor:.2e}" if p_valor < 0.001 else f"{p_valor:.6f}"
                        f.write(f"{load:>10} | {p_str:<15} | {str(significativo):<15}\n")

                f.write("\n")

                # 2. WILCOXON
                f.write("WILCOXON TEST (Pairwise comparisons)\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Load':>10} | {'Comparison':<15} | {'p-value':<15} | {'Significant':<15} | {'Cohen d':<20}\n")
                f.write("-" * 90 + "\n")

                for load in loads_analysis:
                    if load <= max_load_int:
                        for a1, a2, d1, d2 in [
                            ('PSO', 'DE', data_pso, data_de),
                            ('PSO', 'GA', data_pso, data_ga),
                            ('DE', 'GA', data_de, data_ga),
                        ]:
                            p_val, sig = calculate_wilcoxon_pairwise(d1, d2, load)
                            d_val, interp = calculate_effect_size(d1, d2, load)
                            p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.6f}"
                            f.write(f"{load:>10} | {f'{a1} vs {a2}':<15} | {p_str:<15} | {str(sig):<15} | {d_val:.3f} ({interp})\n")
                        f.write("-" * 90 + "\n")

                f.write("\n")

                # 3. BONFERRONI
                f.write("BONFERRONI CORRECTION\n")
                f.write("-" * 60 + "\n")

                all_p_values = []
                for load in loads_analysis:
                    if load <= max_load_int:
                        for d1, d2 in [(data_pso, data_de), (data_pso, data_ga), (data_de, data_ga)]:
                            p_val, _ = calculate_wilcoxon_pairwise(d1, d2, load)
                            all_p_values.append(p_val)

                alpha = 0.05
                n_tests = len(all_p_values)
                bonferroni_alpha = alpha / n_tests if n_tests > 0 else alpha

                f.write(f"Original alpha: {alpha}\n")
                f.write(f"Number of tests: {n_tests}\n")
                f.write(f"Corrected alpha (Bonferroni): {bonferroni_alpha:.6f}\n\n")

                # 4. INFLECTION POINTS
                f.write("INFLECTION POINTS (1% blocking)\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Algorithm':<12} | {'Load (Erlangs)':<18} | {'BP at point':<15}\n")
                f.write("-" * 60 + "\n")

                for algo_name, data in [('PSO', data_pso), ('DE', data_de), ('GA', data_ga)]:
                    found = False
                    for col in data.columns:
                        try:
                            load_val = float(col)
                            mean_val = data[col].mean()
                            if mean_val >= 0.01:
                                f.write(f"{algo_name:<12} | {load_val:>18.0f} | {mean_val:>15.6f}\n")
                                found = True
                                break
                        except Exception:
                            continue
                    if not found:
                        last_col = data.columns[-1]
                        last_load = float(last_col)
                        last_mean = data[last_col].mean()
                        f.write(f"{algo_name:<12} | >{last_load:>17.0f} | {last_mean:>15.6f} (not reached)\n")

                f.write("\n")

    print(f"✓ Global statistical report: {report_file}")
    return report_file


# ============================================
# PER-PAIR STATISTICAL REPORT
# ============================================

def generate_statistical_report_by_pair(raw_data_by_pair, output_dir="relatorio_final"):
    """Generate statistical report per OD pair."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{output_dir}/relatorio_estatistico_por_par_{timestamp}.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("STATISTICAL REPORT - PER OD PAIR\n")
        f.write("Method: Common Random Numbers (CRN) for pairing\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")

        for config_key in sorted(raw_data_by_pair.keys()):
            rede, lambdas = config_key.split('_')
            f.write(f"\n{'#' * 80}\n")
            f.write(f"# NETWORK: {rede} | WAVELENGTHS: {lambdas}\n")
            f.write(f"{'#' * 80}\n\n")

            for max_load_str in sorted(raw_data_by_pair[config_key].keys(), key=int):
                max_load_int = int(max_load_str)
                f.write(f"\n{'=' * 70}\n")
                f.write(f"RESULTS FOR LOADS FROM 1 TO {max_load_int} ERLANGS\n")
                f.write(f"{'=' * 70}\n\n")

                for pair_name in sorted(raw_data_by_pair[config_key][max_load_str].keys()):
                    pair_data = raw_data_by_pair[config_key][max_load_str][pair_name]

                    available = [a for a in ALGORITHMS if a in pair_data]
                    if len(available) < 3:
                        continue

                    data_pso = pair_data['PSO']
                    data_de = pair_data['DE']
                    data_ga = pair_data['GA']

                    f.write(f"\n--- OD PAIR: {pair_name.replace('_', '->')} ---\n\n")
                    f.write(f"General averages:\n")
                    f.write(f"  PSO: {data_pso.mean().mean():.6f}\n")
                    f.write(f"  DE:  {data_de.mean().mean():.6f}\n")
                    f.write(f"  GA:  {data_ga.mean().mean():.6f}\n\n")

                    loads_analysis = [50, 100, 150, 200]
                    if max_load_int >= 400:
                        loads_analysis.extend([250, 300, 350, 400])

                    f.write("Algorithm comparison:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{'Load':>10} | {'PSO vs DE p':<15} | {'PSO vs GA p':<15} | {'DE vs GA p':<15} | {'Friedman p':<15}\n")
                    f.write("-" * 80 + "\n")

                    for load in loads_analysis:
                        if load <= max_load_int:
                            p_pso_de, _ = calculate_wilcoxon_pairwise(data_pso, data_de, load)
                            p_pso_ga, _ = calculate_wilcoxon_pairwise(data_pso, data_ga, load)
                            p_de_ga, _ = calculate_wilcoxon_pairwise(data_de, data_ga, load)
                            p_friedman = calculate_friedman_test(data_pso, data_de, data_ga, load)

                            f.write(f"{load:>10} | {p_pso_de:<15.6f} | {p_pso_ga:<15.6f} | {p_de_ga:<15.6f} | {p_friedman:<15.6f}\n")

                    f.write("\n")

    print(f"✓ Per-pair statistical report: {report_file}")
    return report_file


# ============================================
# COMPARISON PLOTS
# ============================================

def generate_comparison_plots(stats_data, output_dir="relatorio_final"):
    """Generate comparative plots between algorithms (global data)."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for config_key in sorted(stats_data.keys()):
        parts = config_key.split('_')
        rede, lambdas = parts[0], parts[1].replace('l', '')

        for max_load_str in sorted(stats_data[config_key].keys(), key=lambda x: int(x)):
            max_load_int = int(max_load_str)

            if not all(a in stats_data[config_key][max_load_str] for a in ALGORITHMS):
                continue

            fig, ax = plt.subplots(figsize=(14, 8))

            for algo in ALGORITHMS:
                df = stats_data[config_key][max_load_str][algo]['data']
                ax.plot(df['load'], df['mean'],
                        color=COLORS[algo], marker=MARKERS[algo],
                        linewidth=2, markersize=3, label=f'{algo}', alpha=0.8)
                if 'ci_lower' in df.columns and 'ci_upper' in df.columns:
                    ax.fill_between(df['load'], df['ci_lower'], df['ci_upper'],
                                    alpha=0.15, color=COLORS[algo])

            ax.axhline(y=0.01, color='gray', linestyle=':', linewidth=1, alpha=0.7)
            ax.set_xlabel('Traffic Load (Erlangs)', fontsize=12)
            ax.set_ylabel('Blocking Probability (mean)', fontsize=12)
            ax.set_title(f'Algorithm Comparison - {rede} ({lambdas} wavelengths)\nLoads 1 to {max_load_int} Erlangs', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.set_xlim(0, max_load_int)

            plt.tight_layout()
            filename = f"{output_dir}/comparacao_{rede}_{lambdas}l_{max_load_int}loads_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Global plot: {filename}")


def generate_comparison_plots_by_pair(stats_data_by_pair, output_dir="relatorio_final"):
    """Generate comparative plots per OD pair."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for config_key in sorted(stats_data_by_pair.keys()):
        parts = config_key.split('_')
        rede, lambdas = parts[0], parts[1].replace('l', '')

        for max_load_str in sorted(stats_data_by_pair[config_key].keys(), key=lambda x: int(x)):
            max_load_int = int(max_load_str)

            for pair_name in sorted(stats_data_by_pair[config_key][max_load_str].keys()):
                pair_data = stats_data_by_pair[config_key][max_load_str][pair_name]

                if not all(a in pair_data for a in ALGORITHMS):
                    continue

                fig, ax = plt.subplots(figsize=(14, 8))

                for algo in ALGORITHMS:
                    df = pair_data[algo]
                    ax.plot(df['load'], df['mean'],
                            color=COLORS[algo], marker=MARKERS[algo],
                            linewidth=2, markersize=3, label=f'{algo}', alpha=0.8)
                    if 'ci_lower' in df.columns and 'ci_upper' in df.columns:
                        ax.fill_between(df['load'], df['ci_lower'], df['ci_upper'],
                                        alpha=0.15, color=COLORS[algo])

                ax.axhline(y=0.01, color='gray', linestyle=':', linewidth=1, alpha=0.7)
                ax.set_xlabel('Traffic Load (Erlangs)', fontsize=12)
                ax.set_ylabel('Blocking Probability (mean)', fontsize=12)
                ax.set_title(f'Comparison - Pair {pair_name.replace("_", "->")} - {rede} ({lambdas} wavelengths)', fontsize=14)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                ax.set_xlim(0, max_load_int)

                plt.tight_layout()
                filename = f"{output_dir}/comparacao_{rede}_{lambdas}l_par_{pair_name}_{max_load_int}loads_{timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Pair plot {pair_name}: {filename}")


# ============================================
# SUMMARY TABLES
# ============================================

def generate_summary_table(stats_data, output_dir="relatorio_final"):
    """Generate global summary table."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_data = []

    for config_key in sorted(stats_data.keys()):
        parts = config_key.split('_')
        rede, lambdas_str = parts[0], parts[1].replace('l', '')

        for max_load_str in sorted(stats_data[config_key].keys(), key=lambda x: int(x)):
            max_load_int = int(max_load_str)

            for algo in ALGORITHMS:
                if algo in stats_data[config_key][max_load_str]:
                    df = stats_data[config_key][max_load_str][algo]['data']

                    inflexion = None
                    for _, row in df.iterrows():
                        if row['mean'] >= 0.01:
                            inflexion = row['load']
                            break

                    summary_data.append({
                        'Network': rede,
                        'Wavelengths': int(lambdas_str),
                        'Max_Load': max_load_int,
                        'Algorithm': algo,
                        'BP_Min': df['mean'].min(),
                        'BP_Max': df['mean'].max(),
                        'BP_Mean': df['mean'].mean(),
                        'Std_Mean': df['std'].mean(),
                        'Inflection_1pct': inflexion if inflexion else f">{max_load_int}",
                        'Executions': df['n_executions'].iloc[0] if len(df) > 0 else 0
                    })

    df_summary = pd.DataFrame(summary_data)
    csv_file = f"{output_dir}/resumo_metricas_{timestamp}.csv"
    df_summary.to_csv(csv_file, index=False)
    print(f"✓ Global summary table: {csv_file}")
    return df_summary


def generate_pairs_summary_table(stats_data_by_pair, output_dir="relatorio_final"):
    """Generate per-OD-pair summary table."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_data = []

    for config_key in sorted(stats_data_by_pair.keys()):
        parts = config_key.split('_')
        rede, lambdas_str = parts[0], parts[1].replace('l', '')

        for max_load_str in sorted(stats_data_by_pair[config_key].keys(), key=lambda x: int(x)):
            max_load_int = int(max_load_str)

            for pair_name in sorted(stats_data_by_pair[config_key][max_load_str].keys()):
                pair_data = stats_data_by_pair[config_key][max_load_str][pair_name]

                for algo in ALGORITHMS:
                    if algo in pair_data:
                        df = pair_data[algo]

                        inflexion = None
                        for _, row in df.iterrows():
                            if row['mean'] >= 0.01:
                                inflexion = row['load']
                                break

                        summary_data.append({
                            'Network': rede,
                            'Wavelengths': int(lambdas_str),
                            'Max_Load': max_load_int,
                            'OD_Pair': pair_name.replace('_', '->'),
                            'Algorithm': algo,
                            'BP_Mean': df['mean'].mean(),
                            'Inflection_1pct': inflexion if inflexion else f">{max_load_int}"
                        })

    df_summary = pd.DataFrame(summary_data)

    pivot_df = df_summary.pivot_table(
        index=['Network', 'Wavelengths', 'Max_Load', 'OD_Pair'],
        columns='Algorithm',
        values='BP_Mean'
    ).reset_index()

    csv_file = f"{output_dir}/resumo_metricas_por_par_{timestamp}.csv"
    df_summary.to_csv(csv_file, index=False)

    pivot_file = f"{output_dir}/resumo_metricas_por_par_pivot_{timestamp}.csv"
    pivot_df.to_csv(pivot_file, index=False)

    print(f"✓ Per-pair summary table: {csv_file}")
    print(f"✓ Per-pair pivot table: {pivot_file}")
    return df_summary


# ============================================
# BEST SOLUTIONS SUMMARY (new simulator output)
# ============================================

def generate_best_solutions_summary(output_dir="relatorio_final", timestamp_filter=TIMESTAMP_FILTER):
    """Summarize the route selected by each algorithm (files *_best_solutions_*.csv).

    Shows how often each algorithm reached the minimum possible hops in every
    OD pair and how many distinct solutions it produced across executions.
    """
    import re
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(r'^(?P<algo>[A-Za-z]+)_best_solutions_(?P<net>[^_]+)_(?P<lam>\d+)l_(?P<ml>\d+)loads_')
    latest = {}
    for dir_name in glob.glob("results_*_highres"):
        files = _get_most_recent_files(glob.glob(f"{dir_name}/*_best_solutions_*.csv"), timestamp_filter)
        for file in files:
            m = pattern.match(os.path.basename(file))
            if not m:
                continue
            key = (m['net'], int(m['lam']), int(m['ml']), normalize_algo(m['algo']))
            latest[key] = file  # sorted by mtime: the most recent wins

    rows = []
    for (net, lam, ml, algo), file in sorted(latest.items()):
        df = pd.read_csv(file)
        rows.append({
            'Network': net, 'Wavelengths': lam, 'Max_Load': ml, 'Algorithm': algo,
            'Executions': len(df),
            'Pct_min_hops_all_pairs': 100.0 * df['is_min_hops_everywhere'].mean(),
            'Distinct_solutions': df['best_ind'].nunique(),
            'Mean_algo_time_s': df['algo_time_s'].mean(),
            'Mean_simulation_time_s': df['simulation_time_s'].mean(),
        })
    if not rows:
        print("  (no *_best_solutions_*.csv found)")
        return None
    df_out = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{output_dir}/resumo_solucoes_{ts}.csv"
    df_out.to_csv(out, index=False)
    print(f"✓ Best solutions summary: {out}")
    return df_out


# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Main function for statistical post-processing."""

    print("\n" + "=" * 80)
    print("COMPLETE STATISTICAL POST-PROCESSING")
    print("=" * 80)
    print("\n⚠ Requires pairing via Common Random Numbers (CRN)")
    print("   Guaranteed by sim-high-resolution.py (shared traffic_seed)")
    print("=" * 80)

    print("\n📂 Loading raw data (global)...")
    raw_data = load_raw_data()

    print("\n📂 Loading raw data per OD pair...")
    raw_data_by_pair = load_raw_data_by_pair()

    print("\n📂 Loading stats data (global)...")
    stats_data = load_stats_data()

    print("\n📂 Loading stats data per OD pair...")
    stats_data_by_pair = load_stats_data_by_pair()

    if not raw_data and not stats_data:
        print("\n⚠ No data found!")
        print("   Check that 'results_*_highres' folders exist and contain files.")
        return

    print("\n📊 Generating global statistical report...")
    if raw_data:
        generate_statistical_report(raw_data)

    print("\n📊 Generating per-pair statistical report...")
    if raw_data_by_pair:
        generate_statistical_report_by_pair(raw_data_by_pair)

    print("\n📈 Generating global comparison plots...")
    if stats_data:
        generate_comparison_plots(stats_data)

    print("\n📈 Generating per-pair comparison plots...")
    if stats_data_by_pair:
        generate_comparison_plots_by_pair(stats_data_by_pair)

    print("\n📋 Generating global summary table...")
    if stats_data:
        generate_summary_table(stats_data)

    print("\n📋 Generating per-pair summary table...")
    if stats_data_by_pair:
        generate_pairs_summary_table(stats_data_by_pair)

    print("\n🧭 Generating best-solutions summary...")
    generate_best_solutions_summary()

    print("\n" + "=" * 80)
    print("✅ COMPLETE STATISTICAL ANALYSIS FINISHED!")
    print("   Results saved to: relatorio_final/")
    print("=" * 80)


if __name__ == "__main__":
    main()