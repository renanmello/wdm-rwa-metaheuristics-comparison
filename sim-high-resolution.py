"""
ANÁLISE ESTATÍSTICA COMPLETA PARA PSO, DE E AG
Problema: RWA em redes WDM com tráfego dinâmico
Alta Resolução: Cargas de 1 a 200 (ou 1 a 400)

Autor: Tese de Doutorado
"""

import random
import os
from itertools import islice
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t as t_dist
from datetime import datetime
import json

# ============================================
# IMPORTAÇÕES PYMOO (para PSO e DE)
# ============================================
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.pso import PSO as PSOAlgorithm
from pymoo.algorithms.soo.nonconvex.de import DE as DEAlgorithm
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize


# ============================================
# CLASSE PROBLEMA PARA PYMOO
# ============================================
class RWAAProblem(ElementwiseProblem):
    """Problema RWA para uso com pymoo (PSO e DE)."""
    
    def __init__(self, gene_size, fitness_func, manual_pairs):
        self.fitness_func = fitness_func
        self.manual_pairs = manual_pairs
        super().__init__(n_var=gene_size,
                         n_obj=1,
                         xl=np.array([0]*gene_size),
                         xu=np.array([gene_size - 1]*gene_size),
                         vtype=int)
    
    def _evaluate(self, x, out, *args, **kwargs):
        x_int = x.astype(int)
        fitness = -self.fitness_func(x_int.tolist(), self.manual_pairs)
        out["F"] = [fitness]


# ============================================
# SIMULADOR WDM COM ANÁLISE ESTATÍSTICA
# ============================================
class WDMSimulatorStatistical:
    """
    Simulador de rede WDM com suporte a PSO, DE e AG,
    incluindo análise estatística completa e alta resolução.
    """

    def __init__(self,
                 graph: nx.Graph,
                 num_wavelengths: int = 40,
                 gene_size: int = 5,
                 manual_pairs: List[Tuple[int, int]] = None,
                 k: int = 150,
                 # Parâmetros comuns
                 population_size: int = 120,
                 n_gen: int = 40,
                 hops_weight: float = 0.55,
                 wavelength_weight: float = 0.45,
                 # Parâmetros PSO
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 # Parâmetros DE
                 CR: float = 0.9,
                 F: float = 0.8,
                 # Parâmetros AG
                 crossover_rate: float = 0.6,
                 mutation_rate: float = 0.02,
                 tournament_size: int = 3,
                 num_generations_ag: int = 40
                 ):
        
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.gene_size = gene_size
        self.manual_pairs = manual_pairs if manual_pairs else [(0, 6), (2, 5), (0, 3), (1, 4), (2, 6)]
        self.k = k
        
        # Parâmetros comuns
        self.population_size = population_size
        self.n_gen = n_gen
        self.hops_weight = hops_weight
        self.wavelength_weight = wavelength_weight
        
        # Parâmetros PSO
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Parâmetros DE
        self.CR = CR
        self.F = F
        
        # Parâmetros AG
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.num_generations_ag = num_generations_ag
        
        # Calcula k-shortest paths
        self.k_shortest_paths = self._get_all_k_shortest_paths()
        self.reset_network()
        
        # Dá nome ao grafo para identificação
        if not hasattr(self.graph, 'name'):
            self.graph.name = "Custom"

    def reset_network(self) -> None:
        """Reseta os canais de comprimento de onda."""
        for u, v in self.graph.edges:
            self.graph[u][v]['wavelengths'] = np.ones(self.num_wavelengths, dtype=bool)
            self.graph[u][v]['current_wavelength'] = -1

    def release_wavelength(self, route: List[int], wavelength: int) -> None:
        """Libera um comprimento de onda."""
        if not (0 <= wavelength < self.num_wavelengths):
            return
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if self.graph.has_edge(u, v):
                self.graph[u][v]['wavelengths'][wavelength] = True
                if self.graph[u][v]['current_wavelength'] == wavelength:
                    self.graph[u][v]['current_wavelength'] = -1

    def allocate_wavelength(self, route: List[int], wavelength: int) -> bool:
        """Aloca um comprimento de onda."""
        if not (0 <= wavelength < self.num_wavelengths):
            return False
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if not self.graph.has_edge(u, v) or not self.graph[u][v]['wavelengths'][wavelength]:
                return False
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            self.graph[u][v]['wavelengths'][wavelength] = False
            self.graph[u][v]['current_wavelength'] = wavelength
        
        return True

    def find_available_wavelength(self, route: List[int]) -> Optional[int]:
        """Encontra o primeiro wavelength disponível."""
        for wavelength in range(self.num_wavelengths):
            available = True
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                if not self.graph.has_edge(u, v) or not self.graph[u][v]['wavelengths'][wavelength]:
                    available = False
                    break
            if available:
                return wavelength
        return None

    def _get_k_shortest_paths(self, source: int, target: int) -> List[List[int]]:
        """Calcula os k menores caminhos."""
        if not nx.has_path(self.graph, source, target):
            return []
        try:
            return list(islice(nx.shortest_simple_paths(self.graph, source, target), self.k))
        except nx.NetworkXNoPath:
            return []

    def _get_all_k_shortest_paths(self) -> Dict[Tuple[int, int], List[List[int]]]:
        """Calcula caminhos para todos os pares."""
        paths = {}
        for source, target in self.manual_pairs:
            paths[(source, target)] = self._get_k_shortest_paths(source, target)
        return paths

    def _fitness_route(self, route: List[int]) -> float:
        """Calcula fitness de uma rota."""
        if len(route) < 2:
            return 0.0
        
        hops = len(route) - 1
        wavelength_changes = 0
        
        for i in range(len(route) - 2):
            u1, v1 = route[i], route[i + 1]
            u2, v2 = route[i + 1], route[i + 2]
            
            if (self.graph.has_edge(u1, v1) and self.graph.has_edge(u2, v2) and
                self.graph[u1][v1].get('current_wavelength', -1) != -1 and
                self.graph[u2][v2].get('current_wavelength', -1) != -1 and
                self.graph[u1][v1]['current_wavelength'] != self.graph[u2][v2]['current_wavelength']):
                wavelength_changes += 1
        
        fitness = (self.hops_weight * (1 / (hops + 1)) +
                   self.wavelength_weight * (1 / (wavelength_changes + 1)))
        
        return fitness

    def _fitness(self, individual: List[int], source_targets: List[Tuple[int, int]]) -> float:
        """Calcula fitness total do indivíduo."""
        total_fitness = 0.0
        
        for i, (source, target) in enumerate(source_targets):
            if i >= len(individual):
                continue
            
            route_idx = individual[i]
            available_routes = self.k_shortest_paths.get((source, target), [])
            
            if not available_routes or route_idx >= len(available_routes):
                continue
            
            route = available_routes[route_idx]
            total_fitness += self._fitness_route(route)
        
        return total_fitness

    # ============================================
    # ALGORITMO PSO (via pymoo)
    # ============================================
    def run_pso(self, seed: int = None) -> Tuple[np.ndarray, float]:
        """Executa PSO."""
        problem = RWAAProblem(self.gene_size, self._fitness, self.manual_pairs)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        algorithm = PSOAlgorithm(
            pop_size=self.population_size,
            w=self.w,
            c1=self.c1,
            c2=self.c2,
            sampling=IntegerRandomSampling(),
        )
        
        termination = get_termination("n_gen", self.n_gen)
        
        res = minimize(problem, algorithm, termination,
                       seed=seed if seed else 1,
                       save_history=False,
                       verbose=False)
        
        X = res.X.astype(int)
        # PSO: o problema minimiza, mas fitness foi definido para maximizar
        # Por isso invertemos o sinal
        F = -res.F if hasattr(res.F, '__len__') else -res.F
        
        return X, F

    # ============================================
    # ALGORITMO DE (via pymoo) - CORRIGIDO
    # ============================================
    def run_de(self, seed: int = None) -> Tuple[np.ndarray, float]:
        """Executa DE."""
        problem = RWAAProblem(self.gene_size, self._fitness, self.manual_pairs)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        algorithm = DEAlgorithm(
            pop_size=self.population_size,
            CR=self.CR,
            F=self.F,
            variant="DE/rand/1/bin",
            sampling=IntegerRandomSampling(),
        )
        
        termination = get_termination("n_gen", self.n_gen)
        
        res = minimize(problem, algorithm, termination,
                       seed=seed if seed else 1,
                       save_history=False,
                       verbose=False)
        
        X = res.X.astype(int)
        # CORRIGIDO: DE não precisa inverter o sinal
        # O problema já está configurado corretamente
        F = res.F if hasattr(res.F, '__len__') else res.F
        
        return X, F

    # ============================================
    # ALGORITMO AG (implementação manual)
    # ============================================
    def _initialize_population_ag(self) -> List[List[int]]:
        """Inicializa população para AG."""
        population = []
        for _ in range(self.population_size):
            individual = []
            for source, target in self.manual_pairs:
                routes = self.k_shortest_paths.get((source, target), [])
                if routes:
                    max_idx = min(len(routes) - 1, self.gene_size - 1)
                    individual.append(random.randint(0, max_idx))
                else:
                    individual.append(0)
            population.append(individual)
        return population

    def _tournament_selection_ag(self, population: List[List[int]], 
                                   fitness_scores: List[float]) -> List[int]:
        """Seleção por torneio para AG."""
        tournament = random.sample(list(zip(population, fitness_scores)), 
                                   min(self.tournament_size, len(population)))
        return max(tournament, key=lambda x: x[1])[0]

    def _crossover_ag(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Crossover de um ponto para AG."""
        if len(parent1) <= 1:
            return parent1[:], parent2[:]
        
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutate_ag(self, individual: List[int]) -> None:
        """Mutação para AG."""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                source, target = self.manual_pairs[i]
                routes = self.k_shortest_paths.get((source, target), [])
                if routes:
                    max_idx = min(len(routes) - 1, self.gene_size - 1)
                    individual[i] = random.randint(0, max_idx)

    def run_ag(self, seed: int = None) -> Tuple[List[int], float]:
        """Executa Algoritmo Genético."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Inicialização
        population = self._initialize_population_ag()
        
        best_individual_overall = None
        best_fitness_overall = -float('inf')
        
        for generation in range(self.num_generations_ag):
            # Avaliação
            fitness_scores = [self._fitness(ind, self.manual_pairs) for ind in population]
            
            # Melhor da geração
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            
            if best_fitness > best_fitness_overall:
                best_fitness_overall = best_fitness
                best_individual_overall = population[best_idx].copy()
            
            # Elitismo
            elite_size = max(1, self.population_size // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            new_population = [population[i] for i in elite_indices]
            
            # Geração da nova população
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection_ag(population, fitness_scores)
                parent2 = self._tournament_selection_ag(population, fitness_scores)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover_ag(parent1, parent2)
                    new_population.extend([child1, child2])
                else:
                    new_population.extend([parent1.copy(), parent2.copy()])
            
            # Mutação (exceto elite)
            for i in range(elite_size, len(new_population)):
                self._mutate_ag(new_population[i])
            
            population = new_population[:self.population_size]
        
        return best_individual_overall, best_fitness_overall

    # ============================================
    # SIMULAÇÃO DE TRÁFEGO DINÂMICO OTIMIZADA
    # ============================================
    def simulate_dynamic_traffic_optimized(self, 
                                            best_individual: List[int],
                                            load: float,
                                            num_requests: int = None) -> Dict[Tuple[int, int], float]:
        """
        Simula tráfego dinâmico real com número adaptativo de requisições.
        
        Retorna:
            Dicionário com BP por par O-D e também a média geral
        """
        # Ajusta número de requisições baseado na carga
        if num_requests is None:
            if load < 50:
                actual_requests = 2000
            elif load < 100:
                actual_requests = 3000
            elif load < 200:
                actual_requests = 5000
            elif load < 300:
                actual_requests = 8000
            else:
                actual_requests = 10000
        else:
            actual_requests = num_requests
        
        hold_time_mean = 1.0
        arrival_rate = load / hold_time_mean
        mean_interarrival = 1.0 / arrival_rate if arrival_rate > 0 else float('inf')
        
        # Carga extremamente baixa: bloqueio zero
        if mean_interarrival > 1e6:
            # Retorna zero para todos os pares
            bp_by_pair = {pair: 0.0 for pair in self.manual_pairs}
            bp_by_pair['global'] = 0.0
            return bp_by_pair
        
        # Contadores por par O-D
        blocked_by_pair = {pair: 0 for pair in self.manual_pairs}
        total_by_pair = {pair: 0 for pair in self.manual_pairs}
        
        active_connections = {}
        next_id = 0
        current_time = 0.0
        
        # Gera eventos em lote (vetorizado)
        interarrival_times = np.random.exponential(mean_interarrival, actual_requests)
        arrival_times = np.cumsum(interarrival_times)
        hold_times = np.random.exponential(hold_time_mean, actual_requests)
        
        for req_idx in range(actual_requests):
            current_time = arrival_times[req_idx]
            release_time = current_time + hold_times[req_idx]
            
            # Libera conexões expiradas
            to_remove = [cid for cid, (_, _, rtime) in active_connections.items() 
                         if rtime <= current_time]
            
            for conn_id in to_remove:
                conn_route, conn_wavelength, _ = active_connections[conn_id]
                self.release_wavelength(conn_route, conn_wavelength)
                del active_connections[conn_id]
            
            # Escolhe par origem-destino aleatório
            source, target = random.choice(self.manual_pairs)
            pair = (source, target)
            total_by_pair[pair] += 1
            
            # Obtém rota
            pair_idx = self.manual_pairs.index((source, target))
            if pair_idx < len(best_individual):
                route_idx = best_individual[pair_idx]
                routes = self.k_shortest_paths.get((source, target), [])
                if route_idx < len(routes):
                    route = routes[route_idx]
                else:
                    blocked_by_pair[pair] += 1
                    continue
            else:
                blocked_by_pair[pair] += 1
                continue
            
            # Tenta alocar
            wavelength = self.find_available_wavelength(route)
            
            if wavelength is not None:
                self.allocate_wavelength(route, wavelength)
                active_connections[next_id] = (route, wavelength, release_time)
                next_id += 1
            else:
                blocked_by_pair[pair] += 1
        
        # Libera conexões restantes
        for conn_route, conn_wavelength, _ in active_connections.values():
            self.release_wavelength(conn_route, conn_wavelength)
        
        # Calcula BP por par
        bp_by_pair = {}
        total_blocked = 0
        total_requests = 0
        
        for pair in self.manual_pairs:
            if total_by_pair[pair] > 0:
                bp = blocked_by_pair[pair] / total_by_pair[pair]
                bp_by_pair[pair] = bp
                total_blocked += blocked_by_pair[pair]
                total_requests += total_by_pair[pair]
            else:
                bp_by_pair[pair] = 0.0
        
        # Calcula média global
        bp_by_pair['global'] = total_blocked / total_requests if total_requests > 0 else 0.0
        
        return bp_by_pair

    # ============================================
    # EXPERIMENTO DE ALTA RESOLUÇÃO
    # ============================================
    def run_high_resolution_experiment(self,
                                        algorithm: str,
                                        max_load: int,
                                        num_executions: int = 10,
                                        save_results: bool = True,
                                        results_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Executa experimento com alta resolução (1 até max_load).
        Salva dados GERAIS e por PAR O-D.
        
        Retorna:
            df_results: DataFrame com médias gerais
            stats_summary: Estatísticas gerais
            results_by_pair: Dicionário com resultados por par O-D
        """
        if results_dir is None:
            results_dir = f"results_{algorithm}_highres"
        os.makedirs(results_dir, exist_ok=True)
        
        # Cria subpastas para dados por par
        pairs_dir = os.path.join(results_dir, "por_par")
        os.makedirs(pairs_dir, exist_ok=True)
        
        loads = list(range(1, max_load + 1))
        
        # Resultados gerais
        results_global = {load: [] for load in loads}
        
        # Resultados por par O-D
        results_by_pair = {pair: {load: [] for load in loads} for pair in self.manual_pairs}
        
        # Medição de tempo total
        start_total_time = time.time()
        
        print("="*80)
        print(f"EXPERIMENTO ALTA RESOLUÇÃO - {algorithm}")
        print(f"Rede: {self.graph.name}")
        print(f"Lambdas: {self.num_wavelengths}")
        print(f"Cargas: 1 a {max_load} ({len(loads)} pontos)")
        print(f"Execuções: {num_executions}")
        print(f"Pares O-D: {self.manual_pairs}")
        print("="*80)
        
        for exec_idx in range(num_executions):
            print(f"\nExecução {exec_idx + 1}/{num_executions}")
            
            # Mede tempo desta execução
            start_exec_time = time.time()
            
            # Sementes diferentes para cada algoritmo na mesma execução
            base_seed = exec_idx * 100 + 42
            
            if algorithm == 'PSO':
                best_ind, _ = self.run_pso(seed=base_seed)
            elif algorithm == 'DE':
                best_ind, _ = self.run_de(seed=base_seed + 1000)  # +1000 para DE
            else:  # AG
                best_ind, _ = self.run_ag(seed=base_seed + 2000)  # +2000 para AG
            
            algo_time = time.time() - start_exec_time
            
            for i, load in enumerate(loads):
                self.reset_network()
                bp_dict = self.simulate_dynamic_traffic_optimized(best_ind, load)
                
                # Armazena média global
                results_global[load].append(bp_dict['global'])
                
                # Armazena por par
                for pair in self.manual_pairs:
                    results_by_pair[pair][load].append(bp_dict[pair])
                
                if exec_idx == 0 and (i + 1) % 50 == 0:
                    print(f"  Progresso: {i+1}/{len(loads)} cargas")
            
            # Estatísticas desta execução
            bp_values = [results_global[load][-1] for load in loads]
            bp_array = np.array(bp_values)
            exec_time = time.time() - start_exec_time
            print(f"  BP global: min={bp_array.min():.6f}, max={bp_array.max():.6f}")
            print(f"  Tempo execução: {exec_time:.2f}s (algoritmo: {algo_time:.2f}s)")
        
        total_time = time.time() - start_total_time
        print(f"\n⏱ Tempo TOTAL para {algorithm}: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
        
        # DataFrame com resultados globais
        df_results_global = pd.DataFrame(results_global)
        stats_summary_global = self._compute_statistics_highres(df_results_global, loads)
        
        # DataFrames por par
        stats_by_pair = {}
        df_by_pair = {}
        
        for pair in self.manual_pairs:
            pair_name = f"{pair[0]}_{pair[1]}"
            df_by_pair[pair_name] = pd.DataFrame(results_by_pair[pair])
            stats_by_pair[pair_name] = self._compute_statistics_highres(df_by_pair[pair_name], loads)
        
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Salva dados GLOBAIS
            raw_filename = f"{algorithm}_raw_{self.graph.name}_{self.num_wavelengths}l_{max_load}loads_{timestamp}.csv"
            df_results_global.to_csv(f"{results_dir}/{raw_filename}", index=False)
            
            stats_filename = f"{algorithm}_stats_{self.graph.name}_{self.num_wavelengths}l_{max_load}loads_{timestamp}.csv"
            stats_summary_global.to_csv(f"{results_dir}/{stats_filename}", index=False)
            
            # Gráfico global
            self._plot_highres_curve(df_results_global, stats_summary_global, loads, algorithm, max_load, results_dir, timestamp)
            
            # Salva dados por PAR O-D
            for pair in self.manual_pairs:
                pair_name = f"{pair[0]}_{pair[1]}"
                pair_dir = os.path.join(pairs_dir, pair_name)
                os.makedirs(pair_dir, exist_ok=True)
                
                # Dados brutos do par
                raw_pair_filename = f"{algorithm}_raw_{self.graph.name}_{self.num_wavelengths}l_{max_load}loads_par_{pair_name}_{timestamp}.csv"
                df_by_pair[pair_name].to_csv(f"{pair_dir}/{raw_pair_filename}", index=False)
                
                # Estatísticas do par
                stats_pair_filename = f"{algorithm}_stats_{self.graph.name}_{self.num_wavelengths}l_{max_load}loads_par_{pair_name}_{timestamp}.csv"
                stats_by_pair[pair_name].to_csv(f"{pair_dir}/{stats_pair_filename}", index=False)
                
                # Gráfico do par
                self._plot_highres_curve_pair(df_by_pair[pair_name], stats_by_pair[pair_name], 
                                              loads, algorithm, max_load, pair_dir, timestamp, pair_name)
            
            # Salva tempo de execução
            self._save_execution_time(algorithm, max_load, num_executions, total_time, 
                                       self.graph.name, self.num_wavelengths, results_dir, timestamp)
            
            # Salva resumo dos pares
            self._save_pairs_summary(stats_by_pair, algorithm, self.graph.name, self.num_wavelengths, 
                                     max_load, results_dir, timestamp)
        
        return df_results_global, stats_summary_global, results_by_pair

    def _compute_statistics_highres(self, df_results: pd.DataFrame, loads: List[float]) -> pd.DataFrame:
        """Calcula estatísticas para alta resolução."""
        from scipy import stats as scipy_stats
        
        stats = []
        
        for load in loads:
            data = df_results[load].dropna()
            
            if len(data) > 0:
                mean_val = np.mean(data)
                std_val = np.std(data, ddof=1)
                median_val = np.median(data)
                min_val = np.min(data)
                max_val = np.max(data)
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                
                # IC 95%
                conf_int = scipy_stats.t.interval(0.95, df=len(data)-1, 
                                                   loc=mean_val, 
                                                   scale=std_val/np.sqrt(len(data)))
                
                stats.append({
                    'load': load,
                    'mean': mean_val,
                    'std': std_val,
                    'median': median_val,
                    'min': min_val,
                    'max': max_val,
                    'q1': q1,
                    'q3': q3,
                    'iqr': q3 - q1,
                    'ci_lower': conf_int[0],
                    'ci_upper': conf_int[1],
                    'n_executions': len(data)
                })
        
        return pd.DataFrame(stats)

    def _plot_highres_curve(self, df_results, stats_summary, loads, algorithm, max_load, results_dir, timestamp):
        """Gráfico de alta resolução (dados globais)."""
        colors = {'PSO': 'blue', 'DE': 'green', 'AG': 'red'}
        color = colors.get(algorithm, 'black')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        means = stats_summary['mean'].values
        ci_lower = stats_summary['ci_lower'].values
        ci_upper = stats_summary['ci_upper'].values
        
        ax.plot(loads, means, color=color, linewidth=1.5, alpha=0.8, label=f'{algorithm} (média)')
        ax.fill_between(loads, ci_lower, ci_upper, alpha=0.2, color=color, label='IC 95%')
        
        # Ponto de inflexão (1% de bloqueio)
        inflexion_load = None
        for i, mean in enumerate(means):
            if mean > 0.01:
                inflexion_load = loads[i]
                ax.axvline(x=inflexion_load, color='red', linestyle='--', linewidth=2,
                          label=f'1% de bloqueio: {inflexion_load:.0f} Erlangs')
                break
        
        # Linha de referência de 1%
        ax.axhline(y=0.01, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='1% de bloqueio')
        
        ax.set_xlabel('Carga (Erlangs)', fontsize=12)
        ax.set_ylabel('Probabilidade de Bloqueio', fontsize=12)
        ax.set_title(f'{algorithm} - {self.graph.name} ({self.num_wavelengths} lambdas)\n'
                     f'Curva de Bloqueio - Cargas 1 a {max_load}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_xlim(0, max_load)
        
        plt.tight_layout()
        filename = f"{results_dir}/{algorithm}_curve_{self.graph.name}_{self.num_wavelengths}l_{max_load}loads_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_highres_curve_pair(self, df_results, stats_summary, loads, algorithm, max_load, results_dir, timestamp, pair_name):
        """Gráfico de alta resolução para um par O-D específico."""
        colors = {'PSO': 'blue', 'DE': 'green', 'AG': 'red'}
        color = colors.get(algorithm, 'black')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        means = stats_summary['mean'].values
        ci_lower = stats_summary['ci_lower'].values
        ci_upper = stats_summary['ci_upper'].values
        
        ax.plot(loads, means, color=color, linewidth=1.5, alpha=0.8, label=f'{algorithm} (média)')
        ax.fill_between(loads, ci_lower, ci_upper, alpha=0.2, color=color, label='IC 95%')
        
        # Ponto de inflexão (1% de bloqueio)
        for i, mean in enumerate(means):
            if mean > 0.01:
                ax.axvline(x=loads[i], color='red', linestyle='--', linewidth=1, alpha=0.5)
                break
        
        ax.axhline(y=0.01, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        
        ax.set_xlabel('Carga (Erlangs)', fontsize=12)
        ax.set_ylabel('Probabilidade de Bloqueio', fontsize=12)
        ax.set_title(f'{algorithm} - Par ({pair_name.replace("_",",")}) - {self.graph.name} ({self.num_wavelengths} lambdas)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_xlim(0, max_load)
        
        plt.tight_layout()
        filename = f"{results_dir}/{algorithm}_curve_{self.graph.name}_{self.num_wavelengths}l_{max_load}loads_par_{pair_name}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_execution_time(self, algorithm: str, max_load: int, num_executions: int,
                              total_time: float, network: str, lambdas: int, 
                              results_dir: str, timestamp: str):
        """Salva o tempo de execução em um arquivo consolidado."""
        import csv
        
        # Arquivo consolidado de tempos
        time_file = "tempos_execucao.csv"
        time_file_path = os.path.join(results_dir, time_file)
        
        # Dados a salvar
        data = {
            'timestamp': timestamp,
            'algorithm': algorithm,
            'network': network,
            'num_wavelengths': lambdas,
            'max_load': max_load,
            'num_executions': num_executions,
            'total_time_seconds': round(total_time, 2),
            'total_time_minutes': round(total_time / 60, 2)
        }
        
        # Verifica se arquivo já existe
        file_exists = os.path.exists(time_file_path)
        
        # Escreve ou cria arquivo
        with open(time_file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        
        print(f"  ⏱ Tempo registrado em: {time_file_path}")
        
        # Também salva um arquivo separado para esta execução
        time_detail_file = f"{results_dir}/{algorithm}_time_{network}_{lambdas}l_{max_load}loads_{timestamp}.json"
        with open(time_detail_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_pairs_summary(self, stats_by_pair, algorithm, network, lambdas, max_load, results_dir, timestamp):
        """Salva um resumo comparativo dos pares O-D."""
        summary_data = []
        
        for pair_name, df_stats in stats_by_pair.items():
            # Encontra ponto de inflexão para este par
            inflexion = None
            for _, row in df_stats.iterrows():
                if row['mean'] > 0.01:
                    inflexion = row['load']
                    break
            
            summary_data.append({
                'par_OD': pair_name.replace('_', '->'),
                'bp_medio': df_stats['mean'].mean(),
                'bp_max': df_stats['mean'].max(),
                'bp_min': df_stats['mean'].min(),
                'inflexao_1pct': inflexion if inflexion else f">{max_load}",
                'carga_max_bp': df_stats.loc[df_stats['mean'].idxmax(), 'load'],
                'valor_max_bp': df_stats['mean'].max()
            })
        
        df_summary = pd.DataFrame(summary_data)
        summary_file = f"{results_dir}/{algorithm}_pairs_summary_{network}_{lambdas}l_{max_load}loads_{timestamp}.csv"
        df_summary.to_csv(summary_file, index=False)
        print(f"  ✓ Resumo dos pares salvo em: {summary_file}")


# ============================================
# CONFIGURAÇÃO DAS REDES
# ============================================
def get_janet6_graph():
    G = nx.Graph()
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 6), (5, 6)]
    G.add_edges_from(edges)
    G.name = "JANET6"
    return G


def get_redclara_graph():
    G = nx.Graph()
    edges = [(0, 1), (0, 5), (0, 8), (0, 11), (1, 2), (2, 3), (3, 4), (4, 5),
             (5, 6), (5, 7), (5, 11), (7, 8), (8, 9), (8, 11), (9, 10), (9, 11), (11, 12)]
    G.add_edges_from(edges)
    G.name = "RedCLARA"
    return G


def get_ipe_graph():
    G = nx.Graph()
    edges = [(0, 1), (1, 3), (1, 4), (2, 4), (3, 4), (3, 7), (3, 17), (3, 19), (3, 25),
             (4, 6), (4, 12), (5, 25), (6, 7), (7, 8), (7, 11), (7, 18), (7, 19),
             (8, 9), (9, 10), (10, 11), (11, 12), (11, 13), (11, 15), (13, 14),
             (14, 15), (15, 16), (15, 19), (16, 17), (17, 18), (18, 19), (18, 20), (18, 22),
             (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (24, 26), (26, 27)]
    G.add_edges_from(edges)
    G.name = "IPE"
    return G


# ============================================
# PARÂMETROS OTIMIZADOS
# ============================================
OPTIMIZED_PARAMS = {
    'PSO': {
        'population_size': 120,
        'n_gen': 35,
        'w': 0.6719,
        'c1': 1.9636,
        'c2': 1.4178,
        'hops_weight': 0.55,
        'wavelength_weight': 0.45,
        'k': 150
    },
    'DE': {
        'population_size': 120,
        'n_gen': 35,
        'CR': 0.5355,
        'F': 0.4713,
        'hops_weight': 0.55,
        'wavelength_weight': 0.45,
        'k': 150
    },
    'AG': {
        'population_size': 120,
        'num_generations_ag': 35,
        'crossover_rate': 0.2710,
        'mutation_rate': 0.2952,
        'tournament_size': 4,
        'hops_weight': 0.55,
        'wavelength_weight': 0.45,
        'k': 150
    }
}

# Pares para cada rede
JANET_PAIRS = [(0, 6), (2, 5), (0, 3), (1, 4), (2, 6)]
REDCLARA_PAIRS = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
IPE_PAIRS = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]


# ============================================
# FUNÇÃO PRINCIPAL
# ============================================
def run_single_experiment(network_name: str, graph_func, pairs, lambdas: int, max_load: int, num_executions: int = 10):
    """
    Executa experimento para um único algoritmo em uma rede específica.
    Roda PSO, DE e AG em sequência.
    """
    graph = graph_func()
    
    print(f"\n{'='*80}")
    print(f"EXECUTANDO EXPERIMENTO: {network_name} | {lambdas} lambdas | Cargas 1 a {max_load}")
    print(f"{'='*80}")
    
    for algo in ['PSO', 'DE', 'AG']:
        print(f"\n{'#'*80}")
        print(f"# INICIANDO {algo} EM {network_name} COM {lambdas} LAMBDAS")
        print(f"{'#'*80}")
        
        params = OPTIMIZED_PARAMS[algo].copy()
        
        simulator = WDMSimulatorStatistical(
            graph=graph,
            num_wavelengths=lambdas,
            gene_size=5,
            manual_pairs=pairs,
            k=params.pop('k', 150),
            population_size=params.pop('population_size', 120),
            n_gen=params.pop('n_gen', 40),
            hops_weight=params.pop('hops_weight', 0.55),
            wavelength_weight=params.pop('wavelength_weight', 0.45),
            **params
        )
        
        simulator.run_high_resolution_experiment(
            algorithm=algo,
            max_load=max_load,
            num_executions=num_executions,
            save_results=True
        )
        
        print(f"\n✓ {algo} concluído para {network_name} ({lambdas} lambdas)")


# ============================================
# FUNÇÃO PARA CONSOLIDAR TEMPOS DE EXECUÇÃO
# ============================================

def consolidate_execution_times(base_dir="."):
    """
    Consolida todos os arquivos de tempo de execução encontrados.
    """
    import glob
    
    all_times = []
    
    # Procura em todas as pastas results_*_highres
    pattern = "results_*_highres/tempos_execucao.csv"
    time_files = glob.glob(pattern)
    
    for file in time_files:
        try:
            df = pd.read_csv(file)
            all_times.append(df)
        except Exception as e:
            print(f"Erro ao ler {file}: {e}")
    
    if not all_times:
        print("Nenhum arquivo de tempo encontrado!")
        return None
    
    # Concatena todos
    df_times = pd.concat(all_times, ignore_index=True)
    
    # Remove duplicatas (mantém o mais recente por algoritmo/rede/lambdas)
    df_times = df_times.sort_values('timestamp').drop_duplicates(
        subset=['algorithm', 'network', 'num_wavelengths', 'max_load'], 
        keep='last'
    )
    
    # Ordena
    df_times = df_times.sort_values(['network', 'num_wavelengths', 'algorithm'])
    
    # Salva consolidado
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"consolidado_tempos_execucao_{timestamp}.csv"
    df_times.to_csv(output_file, index=False)
    
    print(f"\n⏱ Tempos de execução consolidados em: {output_file}")
    print("\nResumo de tempos (minutos):")
    print(df_times[['algorithm', 'network', 'num_wavelengths', 'total_time_minutes']].to_string(index=False))
    
    return df_times


def main():
    """Função principal."""
    
    # Configurações gerais
    NUM_EXECUTIONS = 20  # Número de execuções por algoritmo
    
    # Experimentos: (nome, função_grafo, pares, lambdas, max_load)
    experiments = [
        ('JANET6', get_janet6_graph, JANET_PAIRS, 40, 200),
        ('JANET6', get_janet6_graph, JANET_PAIRS, 80, 400),
        ('RedCLARA', get_redclara_graph, REDCLARA_PAIRS, 40, 200),
        ('RedCLARA', get_redclara_graph, REDCLARA_PAIRS, 80, 400),
        ('IPE', get_ipe_graph, IPE_PAIRS, 40, 200),
        ('IPE', get_ipe_graph, IPE_PAIRS, 80, 400),
    ]
    
    for net_name, graph_func, pairs, lambdas, max_load in experiments:
        run_single_experiment(
            network_name=net_name,
            graph_func=graph_func,
            pairs=pairs,
            lambdas=lambdas,
            max_load=max_load,
            num_executions=NUM_EXECUTIONS
        )
    
    print("\n" + "="*80)
    print("✅ TODOS OS EXPERIMENTOS CONCLUÍDOS!")
    print("="*80)
    
    # Consolida tempos de execução
    print("\n" + "="*80)
    print("CONSOLIDANDO TEMPOS DE EXECUÇÃO...")
    print("="*80)
    consolidate_execution_times()
    
    print("\n" + "="*80)
    print("📁 ARQUIVOS GERADOS:")
    print("="*80)
    print("  - *raw_*.csv : Dados brutos (geral)")
    print("  - *stats_*.csv : Estatísticas (geral)")
    print("  - *curve_*.png : Gráficos (geral)")
    print("  - por_par/ : Dados separados por par O-D")
    print("    - *raw_*.csv : Dados brutos por par")
    print("    - *stats_*.csv : Estatísticas por par")
    print("    - *curve_*.png : Gráficos por par")
    print("  - *pairs_summary_*.csv : Resumo comparativo dos pares")
    print("  - tempos_execucao.csv : Tempos de execução")
    print("="*80)


if __name__ == "__main__":
    main()