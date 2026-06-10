"""
PÓS-PROCESSAMENTO COMPLETO COM TESTES ESTATÍSTICOS
Leitura de arquivos dentro das pastas results_*_highres
Inclui: Wilcoxon, p-value, Friedman, IC, etc.
Suporte para dados GERAIS e por PAR O-D
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from scipy.stats import wilcoxon, friedmanchisquare

# Configuração
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {'PSO': 'blue', 'DE': 'green', 'AG': 'red'}
MARKERS = {'PSO': 'o', 'DE': 's', 'AG': '^'}


# ============================================
# CARREGAMENTO DOS DADOS
# ============================================

def load_raw_data(base_dir="."):
    """
    Carrega os dados BRUTOS (raw_data) de dentro das pastas.
    Procura por: results_*_highres/*_raw_*.csv
    """
    raw_data = {}
    
    # Procura por todas as pastas de resultados
    pattern_dirs = "results_*_highres"
    result_dirs = glob.glob(pattern_dirs)
    
    print(f"Pastas encontradas: {result_dirs}")
    
    for dir_name in result_dirs:
        # Procura arquivos raw dentro da pasta (dados gerais)
        raw_files = glob.glob(f"{dir_name}/*_raw_*.csv")
        
        for file in raw_files:
            # Extrai informações do nome do arquivo
            basename = os.path.basename(file).replace('.csv', '')
            parts = basename.split('_')
            
            if len(parts) >= 6:
                algoritmo = parts[0]      # PSO, DE, AG
                rede = parts[2]           # RedCLARA, JANET6, IPE
                lambdas = parts[3].replace('l', '')  # 40, 80
                max_load = parts[4].replace('loads', '')  # 200, 400
                
                df = pd.read_csv(file)
                
                # Converte as colunas para string (cargas são números)
                df.columns = df.columns.astype(str)
                
                key = f"{rede}_{lambdas}l"
                if key not in raw_data:
                    raw_data[key] = {}
                if max_load not in raw_data[key]:
                    raw_data[key][max_load] = {}
                
                raw_data[key][max_load][algoritmo] = {
                    'type': 'global',
                    'data': df
                }
                
                print(f"  Carregado (global): {algoritmo} - {rede} - {lambdas}l - {max_load} loads")
    
    return raw_data


def load_raw_data_by_pair(base_dir="."):
    """
    Carrega os dados BRUTOS por par O-D.
    Procura por: results_*_highres/por_par/*/*_raw_*.csv
    """
    raw_data_by_pair = {}
    
    # Procura por todas as pastas de resultados
    pattern_dirs = "results_*_highres"
    result_dirs = glob.glob(pattern_dirs)
    
    for dir_name in result_dirs:
        # Procura dentro das subpastas por_par
        pair_dirs = glob.glob(f"{dir_name}/por_par/*")
        
        for pair_dir in pair_dirs:
            pair_name = os.path.basename(pair_dir)  # ex: "0_12"
            
            # Procura arquivos raw dentro desta pasta do par
            raw_files = glob.glob(f"{pair_dir}/*_raw_*.csv")
            
            for file in raw_files:
                basename = os.path.basename(file).replace('.csv', '')
                parts = basename.split('_')
                
                if len(parts) >= 6:
                    algoritmo = parts[0]  # PSO, DE, AG
                    rede = parts[2]       # RedCLARA, JANET6, IPE
                    lambdas = parts[3].replace('l', '')
                    max_load = parts[4].replace('loads', '')
                    
                    df = pd.read_csv(file)
                    df.columns = df.columns.astype(str)
                    
                    key = f"{rede}_{lambdas}l"
                    if key not in raw_data_by_pair:
                        raw_data_by_pair[key] = {}
                    if max_load not in raw_data_by_pair[key]:
                        raw_data_by_pair[key][max_load] = {}
                    if pair_name not in raw_data_by_pair[key][max_load]:
                        raw_data_by_pair[key][max_load][pair_name] = {}
                    
                    raw_data_by_pair[key][max_load][pair_name][algoritmo] = df
                    
                    print(f"  Carregado (par {pair_name}): {algoritmo} - {rede} - {lambdas}l - {max_load} loads")
    
    return raw_data_by_pair


def load_stats_data(base_dir="."):
    """
    Carrega os dados de estatísticas (stats) de dentro das pastas.
    """
    stats_data = {}
    
    pattern_dirs = "results_*_highres"
    result_dirs = glob.glob(pattern_dirs)
    
    for dir_name in result_dirs:
        stats_files = glob.glob(f"{dir_name}/*_stats_*.csv")
        
        for file in stats_files:
            basename = os.path.basename(file).replace('.csv', '')
            parts = basename.split('_')
            
            if len(parts) >= 6:
                algoritmo = parts[0]
                rede = parts[2]
                lambdas = parts[3].replace('l', '')
                max_load = parts[4].replace('loads', '')
                
                df = pd.read_csv(file)
                
                key = f"{rede}_{lambdas}l"
                if key not in stats_data:
                    stats_data[key] = {}
                if max_load not in stats_data[key]:
                    stats_data[key][max_load] = {}
                
                stats_data[key][max_load][algoritmo] = {
                    'type': 'global',
                    'data': df
                }
                
                print(f"  Carregado stats (global): {algoritmo} - {rede} - {lambdas}l - {max_load} loads")
    
    return stats_data


def load_stats_data_by_pair(base_dir="."):
    """
    Carrega os dados de estatísticas por par O-D.
    """
    stats_data_by_pair = {}
    
    pattern_dirs = "results_*_highres"
    result_dirs = glob.glob(pattern_dirs)
    
    for dir_name in result_dirs:
        pair_dirs = glob.glob(f"{dir_name}/por_par/*")
        
        for pair_dir in pair_dirs:
            pair_name = os.path.basename(pair_dir)
            stats_files = glob.glob(f"{pair_dir}/*_stats_*.csv")
            
            for file in stats_files:
                basename = os.path.basename(file).replace('.csv', '')
                parts = basename.split('_')
                
                if len(parts) >= 6:
                    algoritmo = parts[0]
                    rede = parts[2]
                    lambdas = parts[3].replace('l', '')
                    max_load = parts[4].replace('loads', '')
                    
                    df = pd.read_csv(file)
                    
                    key = f"{rede}_{lambdas}l"
                    if key not in stats_data_by_pair:
                        stats_data_by_pair[key] = {}
                    if max_load not in stats_data_by_pair[key]:
                        stats_data_by_pair[key][max_load] = {}
                    if pair_name not in stats_data_by_pair[key][max_load]:
                        stats_data_by_pair[key][max_load][pair_name] = {}
                    
                    stats_data_by_pair[key][max_load][pair_name][algoritmo] = df
                    
                    print(f"  Carregado stats (par {pair_name}): {algoritmo} - {rede} - {lambdas}l - {max_load} loads")
    
    return stats_data_by_pair


# ============================================
# TESTES ESTATÍSTICOS
# ============================================

def calculate_wilcoxon_pairwise(data_algo1, data_algo2, load):
    """
    Calcula teste de Wilcoxon para um par de algoritmos em uma carga específica.
    """
    load_str = str(int(load)) if load == int(load) else str(load)
    
    if load_str not in data_algo1.columns or load_str not in data_algo2.columns:
        return 1.0, False
    
    values1 = data_algo1[load_str].dropna().values
    values2 = data_algo2[load_str].dropna().values
    
    if len(values1) != len(values2):
        return 1.0, False
    
    # Verifica se os dados são todos iguais
    if np.array_equal(values1, values2):
        return 1.0, False
    
    try:
        stat, p_value = wilcoxon(values1, values2)
        significant = p_value < 0.05
        return p_value, significant
    except:
        return 1.0, False


def calculate_friedman_test(data_pso, data_de, data_ag, load):
    """
    Calcula teste de Friedman para os três algoritmos em uma carga.
    """
    load_str = str(int(load)) if load == int(load) else str(load)
    
    if load_str not in data_pso.columns or load_str not in data_de.columns or load_str not in data_ag.columns:
        return 1.0
    
    values_pso = data_pso[load_str].dropna().values
    values_de = data_de[load_str].dropna().values
    values_ag = data_ag[load_str].dropna().values
    
    min_len = min(len(values_pso), len(values_de), len(values_ag))
    values_pso = values_pso[:min_len]
    values_de = values_de[:min_len]
    values_ag = values_ag[:min_len]
    
    try:
        stat, p_value = friedmanchisquare(values_pso, values_de, values_ag)
        return p_value
    except:
        return 1.0


def calculate_effect_size(data_algo1, data_algo2, load):
    """
    Calcula o tamanho do efeito (Cohen's d) entre dois algoritmos.
    """
    load_str = str(int(load)) if load == int(load) else str(load)
    
    if load_str not in data_algo1.columns or load_str not in data_algo2.columns:
        return 0.0, "sem dados"
    
    values1 = data_algo1[load_str].dropna().values
    values2 = data_algo2[load_str].dropna().values
    
    mean1, mean2 = np.mean(values1), np.mean(values2)
    std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)
    
    n1, n2 = len(values1), len(values2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0, "insignificante"
    
    cohens_d = abs(mean1 - mean2) / pooled_std
    
    if cohens_d < 0.2:
        interpretation = "insignificante"
    elif cohens_d < 0.5:
        interpretation = "pequeno"
    elif cohens_d < 0.8:
        interpretation = "medio"
    else:
        interpretation = "grande"
    
    return cohens_d, interpretation


# ============================================
# RELATÓRIO COMPLETO (GERAL)
# ============================================

def generate_statistical_report(raw_data, output_dir="relatorio_final"):
    """Gera relatório completo com todos os testes estatísticos (dados gerais)."""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_file = f"{output_dir}/relatorio_estatistico_completo_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("RELATORIO ESTATISTICO COMPLETO - DADOS GERAIS\n")
        f.write("Testes: Wilcoxon, Friedman, p-valor, Tamanho de Efeito (Cohen's d)\n")
        f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        
        for config_key in sorted(raw_data.keys()):
            rede, lambdas = config_key.split('_')
            f.write(f"\n{'#'*80}\n")
            f.write(f"# REDE: {rede} | LAMBDAS: {lambdas}\n")
            f.write(f"{'#'*80}\n\n")
            
            for max_load_str in sorted(raw_data[config_key].keys(), key=int):
                max_load_int = int(max_load_str)
                f.write(f"\n{'='*70}\n")
                f.write(f"RESULTADOS PARA CARGAS DE 1 A {max_load_int} ERLANGS\n")
                f.write(f"{'='*70}\n\n")
                
                if not all(a in raw_data[config_key][max_load_str] for a in ['PSO', 'DE', 'AG']):
                    f.write("Dados incompletos para esta configuracao\n\n")
                    continue
                
                data_pso = raw_data[config_key][max_load_str]['PSO']['data']
                data_de = raw_data[config_key][max_load_str]['DE']['data']
                data_ag = raw_data[config_key][max_load_str]['AG']['data']
                
                cargas_analise = [50, 100, 150, 200]
                if max_load_int >= 400:
                    cargas_analise.extend([250, 300, 350, 400])
                
                # 1. TESTE DE FRIEDMAN
                f.write("TESTE DE FRIEDMAN (Comparacao global dos 3 algoritmos)\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Carga':>10} | {'p-valor':<15} | {'Significativo':<15}\n")
                f.write("-" * 60 + "\n")
                
                for load in cargas_analise:
                    if load <= max_load_int:
                        p_valor = calculate_friedman_test(data_pso, data_de, data_ag, load)
                        significativo = p_valor < 0.05
                        if p_valor < 0.001:
                            p_str = f"{p_valor:.2e}"
                        else:
                            p_str = f"{p_valor:.6f}"
                        f.write(f"{load:>10} | {p_str:<15} | {str(significativo):<15}\n")
                
                f.write("\n")
                
                # 2. TESTE DE WILCOXON
                f.write("TESTE DE WILCOXON (Comparacoes pareadas)\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Carga':>10} | {'Comparacao':<15} | {'p-valor':<15} | {'Significativo':<15} | {'Cohen d':<20}\n")
                f.write("-" * 90 + "\n")
                
                for load in cargas_analise:
                    if load <= max_load_int:
                        p_pso_de, sig_pso_de = calculate_wilcoxon_pairwise(data_pso, data_de, load)
                        d_pso_de, interp_pso_de = calculate_effect_size(data_pso, data_de, load)
                        
                        p_str = f"{p_pso_de:.2e}" if p_pso_de < 0.001 else f"{p_pso_de:.6f}"
                        f.write(f"{load:>10} | {'PSO vs DE':<15} | {p_str:<15} | {str(sig_pso_de):<15} | {d_pso_de:.3f} ({interp_pso_de})\n")
                        
                        p_pso_ag, sig_pso_ag = calculate_wilcoxon_pairwise(data_pso, data_ag, load)
                        d_pso_ag, interp_pso_ag = calculate_effect_size(data_pso, data_ag, load)
                        
                        p_str = f"{p_pso_ag:.2e}" if p_pso_ag < 0.001 else f"{p_pso_ag:.6f}"
                        f.write(f"{load:>10} | {'PSO vs AG':<15} | {p_str:<15} | {str(sig_pso_ag):<15} | {d_pso_ag:.3f} ({interp_pso_ag})\n")
                        
                        p_de_ag, sig_de_ag = calculate_wilcoxon_pairwise(data_de, data_ag, load)
                        d_de_ag, interp_de_ag = calculate_effect_size(data_de, data_ag, load)
                        
                        p_str = f"{p_de_ag:.2e}" if p_de_ag < 0.001 else f"{p_de_ag:.6f}"
                        f.write(f"{load:>10} | {'DE vs AG':<15} | {p_str:<15} | {str(sig_de_ag):<15} | {d_de_ag:.3f} ({interp_de_ag})\n")
                        f.write("-" * 90 + "\n")
                
                f.write("\n")
                
                # 3. CORRECAO DE BONFERRONI
                f.write("CORRECAO DE BONFERRONI\n")
                f.write("-" * 60 + "\n")
                
                all_p_values = []
                for load in cargas_analise:
                    if load <= max_load_int:
                        p1, _ = calculate_wilcoxon_pairwise(data_pso, data_de, load)
                        p2, _ = calculate_wilcoxon_pairwise(data_pso, data_ag, load)
                        p3, _ = calculate_wilcoxon_pairwise(data_de, data_ag, load)
                        all_p_values.extend([p1, p2, p3])
                
                alpha = 0.05
                n_tests = len(all_p_values)
                bonferroni_alpha = alpha / n_tests if n_tests > 0 else alpha
                
                f.write(f"Alpha original: {alpha}\n")
                f.write(f"Numero de testes: {n_tests}\n")
                f.write(f"Alpha corrigido (Bonferroni): {bonferroni_alpha:.6f}\n")
                f.write(f"p-valor < {bonferroni_alpha:.6f} para ser significativo apos correcao\n\n")
                
                # 4. PONTOS DE INFLEXAO
                f.write("PONTOS DE INFLEXAO (1% de bloqueio)\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Algoritmo':<12} | {'Carga (Erlangs)':<18} | {'BP no ponto':<15}\n")
                f.write("-" * 60 + "\n")
                
                for algo_name, data in [('PSO', data_pso), ('DE', data_de), ('AG', data_ag)]:
                    found = False
                    for col in data.columns:
                        try:
                            load_val = float(col)
                            mean_val = data[col].mean()
                            if mean_val > 0.01:
                                f.write(f"{algo_name:<12} | {load_val:>18.0f} | {mean_val:>15.6f}\n")
                                found = True
                                break
                        except:
                            continue
                    if not found:
                        last_col = data.columns[-1]
                        last_load = float(last_col)
                        last_mean = data[last_col].mean()
                        f.write(f"{algo_name:<12} | >{last_load:>17.0f} | {last_mean:>15.6f} (nao atingiu)\n")
                
                f.write("\n")
    
    print(f"✓ Relatorio estatistico global: {report_file}")
    return report_file


# ============================================
# RELATÓRIO POR PAR O-D
# ============================================

def generate_statistical_report_by_pair(raw_data_by_pair, output_dir="relatorio_final"):
    """Gera relatório com testes estatísticos para cada par O-D."""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_file = f"{output_dir}/relatorio_estatistico_por_par_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("RELATORIO ESTATISTICO - POR PAR O-D\n")
        f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        
        for config_key in sorted(raw_data_by_pair.keys()):
            rede, lambdas = config_key.split('_')
            f.write(f"\n{'#'*80}\n")
            f.write(f"# REDE: {rede} | LAMBDAS: {lambdas}\n")
            f.write(f"{'#'*80}\n\n")
            
            for max_load_str in sorted(raw_data_by_pair[config_key].keys(), key=int):
                max_load_int = int(max_load_str)
                f.write(f"\n{'='*70}\n")
                f.write(f"RESULTADOS PARA CARGAS DE 1 A {max_load_int} ERLANGS\n")
                f.write(f"{'='*70}\n\n")
                
                for pair_name in sorted(raw_data_by_pair[config_key][max_load_str].keys()):
                    pair_data = raw_data_by_pair[config_key][max_load_str][pair_name]
                    
                    if not all(a in pair_data for a in ['PSO', 'DE', 'AG']):
                        continue
                    
                    data_pso = pair_data['PSO']
                    data_de = pair_data['DE']
                    data_ag = pair_data['AG']
                    
                    f.write(f"\n--- PAR O-D: {pair_name.replace('_', '->')} ---\n\n")
                    
                    # Médias por algoritmo
                    mean_pso = data_pso.mean().mean()
                    mean_de = data_de.mean().mean()
                    mean_ag = data_ag.mean().mean()
                    
                    f.write(f"Medias gerais:\n")
                    f.write(f"  PSO: {mean_pso:.6f}\n")
                    f.write(f"  DE:  {mean_de:.6f}\n")
                    f.write(f"  AG:  {mean_ag:.6f}\n\n")
                    
                    # Cargas para análise
                    cargas_analise = [50, 100, 150, 200]
                    if max_load_int >= 400:
                        cargas_analise.extend([250, 300, 350, 400])
                    
                    # Wilcoxon e Friedman para o par
                    f.write("Comparacao entre algoritmos:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{'Carga':>10} | {'PSO vs DE p':<15} | {'PSO vs AG p':<15} | {'DE vs AG p':<15} | {'Friedman p':<15}\n")
                    f.write("-" * 80 + "\n")
                    
                    for load in cargas_analise:
                        if load <= max_load_int:
                            p_pso_de, _ = calculate_wilcoxon_pairwise(data_pso, data_de, load)
                            p_pso_ag, _ = calculate_wilcoxon_pairwise(data_pso, data_ag, load)
                            p_de_ag, _ = calculate_wilcoxon_pairwise(data_de, data_ag, load)
                            p_friedman = calculate_friedman_test(data_pso, data_de, data_ag, load)
                            
                            p_pso_de_str = f"{p_pso_de:.2e}" if p_pso_de < 0.001 else f"{p_pso_de:.6f}"
                            p_pso_ag_str = f"{p_pso_ag:.2e}" if p_pso_ag < 0.001 else f"{p_pso_ag:.6f}"
                            p_de_ag_str = f"{p_de_ag:.2e}" if p_de_ag < 0.001 else f"{p_de_ag:.6f}"
                            p_friedman_str = f"{p_friedman:.2e}" if p_friedman < 0.001 else f"{p_friedman:.6f}"
                            
                            f.write(f"{load:>10} | {p_pso_de_str:<15} | {p_pso_ag_str:<15} | {p_de_ag_str:<15} | {p_friedman_str:<15}\n")
                    
                    f.write("\n")
    
    print(f"✓ Relatorio estatistico por par: {report_file}")
    return report_file


# ============================================
# GRÁFICOS COMPARATIVOS (GERAL)
# ============================================

def generate_comparison_plots(stats_data, output_dir="relatorio_final"):
    """Gera gráficos comparativos entre algoritmos (dados gerais)."""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for config_key in sorted(stats_data.keys()):
        parts = config_key.split('_')
        rede = parts[0]
        lambdas = parts[1].replace('l', '')  # Remove o 'l'
        
        for max_load_str in sorted(stats_data[config_key].keys(), key=lambda x: int(x)):
            max_load_int = int(max_load_str)
            
            if not all(a in stats_data[config_key][max_load_str] for a in ['PSO', 'DE', 'AG']):
                continue
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            for algo in ['PSO', 'DE', 'AG']:
                df = stats_data[config_key][max_load_str][algo]['data']
                
                ax.plot(df['load'], df['mean'], 
                       color=COLORS[algo], marker=MARKERS[algo],
                       linewidth=2, markersize=3, label=f'{algo}', alpha=0.8)
                
                if 'ci_lower' in df.columns and 'ci_upper' in df.columns:
                    ax.fill_between(df['load'], df['ci_lower'], df['ci_upper'],
                                   alpha=0.15, color=COLORS[algo])
            
            ax.axhline(y=0.01, color='gray', linestyle=':', linewidth=1, alpha=0.7)
            ax.text(max_load_int * 0.95, 0.012, '1% de bloqueio', fontsize=9, alpha=0.7)
            
            ax.set_xlabel('Carga (Erlangs)', fontsize=12)
            ax.set_ylabel('Probabilidade de Bloqueio (media)', fontsize=12)
            ax.set_title(f'Comparacao de Algoritmos - {rede} ({lambdas} lambdas)\n'
                        f'Cargas de 1 a {max_load_int} Erlangs', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.set_xlim(0, max_load_int)
            
            plt.tight_layout()
            
            filename = f"{output_dir}/comparacao_{rede}_{lambdas}l_{max_load_int}loads_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Grafico global: {filename}")
            
def generate_comparison_plots_by_pair(stats_data_by_pair, output_dir="relatorio_final"):
    """Gera gráficos comparativos para cada par O-D."""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for config_key in sorted(stats_data_by_pair.keys()):
        parts = config_key.split('_')
        rede = parts[0]
        lambdas = parts[1].replace('l', '')  # Remove o 'l'
        
        for max_load_str in sorted(stats_data_by_pair[config_key].keys(), key=lambda x: int(x)):
            max_load_int = int(max_load_str)
            
            for pair_name in sorted(stats_data_by_pair[config_key][max_load_str].keys()):
                pair_data = stats_data_by_pair[config_key][max_load_str][pair_name]
                
                if not all(a in pair_data for a in ['PSO', 'DE', 'AG']):
                    continue
                
                fig, ax = plt.subplots(figsize=(14, 8))
                
                for algo in ['PSO', 'DE', 'AG']:
                    df = pair_data[algo]
                    
                    ax.plot(df['load'], df['mean'], 
                           color=COLORS[algo], marker=MARKERS[algo],
                           linewidth=2, markersize=3, label=f'{algo}', alpha=0.8)
                    
                    if 'ci_lower' in df.columns and 'ci_upper' in df.columns:
                        ax.fill_between(df['load'], df['ci_lower'], df['ci_upper'],
                                       alpha=0.15, color=COLORS[algo])
                
                ax.axhline(y=0.01, color='gray', linestyle=':', linewidth=1, alpha=0.7)
                ax.set_xlabel('Carga (Erlangs)', fontsize=12)
                ax.set_ylabel('Probabilidade de Bloqueio (media)', fontsize=12)
                ax.set_title(f'Comparacao - Par {pair_name.replace("_","->")} - {rede} ({lambdas} lambdas)', fontsize=14)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                ax.set_xlim(0, max_load_int)
                
                plt.tight_layout()
                
                filename = f"{output_dir}/comparacao_{rede}_{lambdas}l_par_{pair_name}_{max_load_int}loads_{timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✓ Grafico par {pair_name}: {filename}")



# ============================================
# TABELA RESUMO
# ============================================

def generate_summary_table(stats_data, output_dir="relatorio_final"):
    """Gera tabela resumo em CSV (dados gerais)."""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_data = []
    
    for config_key in sorted(stats_data.keys()):
        # Config_key é algo como "RedCLARA_40l"
        parts = config_key.split('_')
        rede = parts[0]
        lambdas_str = parts[1].replace('l', '')  # Remove o 'l' e converte
        
        for max_load_str in sorted(stats_data[config_key].keys(), key=lambda x: int(x)):
            max_load_int = int(max_load_str)
            
            for algo in ['PSO', 'DE', 'AG']:
                if algo in stats_data[config_key][max_load_str]:
                    df = stats_data[config_key][max_load_str][algo]['data']
                    
                    inflexion = None
                    for _, row in df.iterrows():
                        if row['mean'] > 0.01:
                            inflexion = row['load']
                            break
                    
                    summary_data.append({
                        'Rede': rede,
                        'Lambdas': int(lambdas_str),  # Agora está correto
                        'Max_Load': max_load_int,
                        'Algoritmo': algo,
                        'BP_Min': df['mean'].min(),
                        'BP_Max': df['mean'].max(),
                        'BP_Medio': df['mean'].mean(),
                        'Std_Medio': df['std'].mean(),
                        'Inflexao_1pct': inflexion if inflexion else f">{max_load_int}",
                        'Execucoes': df['n_executions'].iloc[0] if len(df) > 0 else 0
                    })
    
    df_summary = pd.DataFrame(summary_data)
    csv_file = f"{output_dir}/resumo_metricas_{timestamp}.csv"
    df_summary.to_csv(csv_file, index=False)
    
    print(f"✓ Tabela resumo global: {csv_file}")
    return df_summary

def generate_pairs_summary_table(stats_data_by_pair, output_dir="relatorio_final"):
    """Gera tabela resumo por par O-D."""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_data = []
    
    for config_key in sorted(stats_data_by_pair.keys()):
        # Config_key é algo como "RedCLARA_40l"
        parts = config_key.split('_')
        rede = parts[0]
        lambdas_str = parts[1].replace('l', '')  # Remove o 'l'
        
        for max_load_str in sorted(stats_data_by_pair[config_key].keys(), key=lambda x: int(x)):
            max_load_int = int(max_load_str)
            
            for pair_name in sorted(stats_data_by_pair[config_key][max_load_str].keys()):
                pair_data = stats_data_by_pair[config_key][max_load_str][pair_name]
                
                for algo in ['PSO', 'DE', 'AG']:
                    if algo in pair_data:
                        df = pair_data[algo]
                        
                        inflexion = None
                        for _, row in df.iterrows():
                            if row['mean'] > 0.01:
                                inflexion = row['load']
                                break
                        
                        summary_data.append({
                            'Rede': rede,
                            'Lambdas': int(lambdas_str),  # Agora está correto
                            'Max_Load': max_load_int,
                            'Par_OD': pair_name.replace('_', '->'),
                            'Algoritmo': algo,
                            'BP_Medio': df['mean'].mean(),
                            'Inflexao_1pct': inflexion if inflexion else f">{max_load_int}"
                        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Cria tabela pivotada
    pivot_df = df_summary.pivot_table(
        index=['Rede', 'Lambdas', 'Max_Load', 'Par_OD'],
        columns='Algoritmo',
        values='BP_Medio'
    ).reset_index()
    
    csv_file = f"{output_dir}/resumo_metricas_por_par_{timestamp}.csv"
    df_summary.to_csv(csv_file, index=False)
    
    pivot_file = f"{output_dir}/resumo_metricas_por_par_pivot_{timestamp}.csv"
    pivot_df.to_csv(pivot_file, index=False)
    
    print(f"✓ Tabela resumo por par: {csv_file}")
    print(f"✓ Tabela pivotada por par: {pivot_file}")
    return df_summary

# ============================================
# FUNCAO PRINCIPAL
# ============================================

def main():
    """Função principal do pós-processamento estatístico."""
    
    print("\n" + "="*80)
    print("POS-PROCESSAMENTO ESTATISTICO COMPLETO")
    print("="*80)
    
    # Carrega dados brutos
    print("\n📂 Carregando dados brutos (global)...")
    raw_data = load_raw_data()
    
    print("\n📂 Carregando dados brutos por par O-D...")
    raw_data_by_pair = load_raw_data_by_pair()
    
    print("\n📂 Carregando dados de estatisticas (global)...")
    stats_data = load_stats_data()
    
    print("\n📂 Carregando dados de estatisticas por par O-D...")
    stats_data_by_pair = load_stats_data_by_pair()
    
    if not raw_data and not stats_data:
        print("\n⚠ Nenhum dado encontrado!")
        print("   Verifique se as pastas 'results_*_highres' existem e contêm arquivos.")
        return
    
    # Gerar relatórios
    print("\n📊 Gerando relatorio estatistico global...")
    if raw_data:
        generate_statistical_report(raw_data)
    
    print("\n📊 Gerando relatorio estatistico por par O-D...")
    if raw_data_by_pair:
        generate_statistical_report_by_pair(raw_data_by_pair)
    
    print("\n📈 Gerando graficos comparativos globais...")
    if stats_data:
        generate_comparison_plots(stats_data)
    
    print("\n📈 Gerando graficos comparativos por par O-D...")
    if stats_data_by_pair:
        generate_comparison_plots_by_pair(stats_data_by_pair)
    
    print("\n📋 Gerando tabela resumo global...")
    if stats_data:
        generate_summary_table(stats_data)
    
    print("\n📋 Gerando tabela resumo por par O-D...")
    if stats_data_by_pair:
        generate_pairs_summary_table(stats_data_by_pair)
    
    print("\n" + "="*80)
    print("✅ ANALISE ESTATISTICA COMPLETA CONCLUIDA!")
    print("   Resultados salvos em: relatorio_final/")
    print("="*80)


if __name__ == "__main__":
    main()