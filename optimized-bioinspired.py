"""
OTIMIZAÇÃO DE HIPERPARÂMETROS PARA AG, PSO E DE
Problema: RWA em redes WDM com tráfego dinâmico
Autor: Tese de Doutorado

Este script utiliza Optuna para encontrar os melhores hiperparâmetros
para cada algoritmo (AG, PSO, DE) de forma individual.
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
from datetime import datetime
import json
import optuna

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
# SIMULADOR WDM COMPLETO (VERSÃO OTIMIZADA)
# ============================================
class WDMSimulatorOptimized:
    """
    Simulador de rede WDM com suporte a AG, PSO e DE,
    incluindo análise estatística para otimização de hiperparâmetros.
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
    # ALGORITMO PSO
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
        F = -res.F if hasattr(res.F, '__len__') else -res.F
        
        return X, F

    # ============================================
    # ALGORITMO DE
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
        
        population = self._initialize_population_ag()
        
        best_individual_overall = None
        best_fitness_overall = -float('inf')
        
        for generation in range(self.num_generations_ag):
            fitness_scores = [self._fitness(ind, self.manual_pairs) for ind in population]
            
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            
            if best_fitness > best_fitness_overall:
                best_fitness_overall = best_fitness
                best_individual_overall = population[best_idx].copy()
            
            elite_size = max(1, self.population_size // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            new_population = [population[i] for i in elite_indices]
            
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection_ag(population, fitness_scores)
                parent2 = self._tournament_selection_ag(population, fitness_scores)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover_ag(parent1, parent2)
                    new_population.extend([child1, child2])
                else:
                    new_population.extend([parent1.copy(), parent2.copy()])
            
            for i in range(elite_size, len(new_population)):
                self._mutate_ag(new_population[i])
            
            population = new_population[:self.population_size]
        
        return best_individual_overall, best_fitness_overall

    # ============================================
    # SIMULAÇÃO DE TRÁFEGO (SIMPLIFICADA PARA OTIMIZAÇÃO)
    # ============================================
    def simulate_blocking_probability(self, best_individual: List[int], 
                                       load: float = 100.0,
                                       num_requests: int = 2000) -> float:
        """
        Simula tráfego dinâmico para uma carga específica.
        Versão simplificada para otimização de hiperparâmetros.
        """
        hold_time_mean = 1.0
        arrival_rate = load / hold_time_mean
        mean_interarrival = 1.0 / arrival_rate if arrival_rate > 0 else float('inf')
        
        if mean_interarrival > 1e6:
            return 0.0
        
        blocked_requests = 0
        active_connections = {}
        next_id = 0
        current_time = 0.0
        
        interarrival_times = np.random.exponential(mean_interarrival, num_requests)
        arrival_times = np.cumsum(interarrival_times)
        hold_times = np.random.exponential(hold_time_mean, num_requests)
        
        for req_idx in range(num_requests):
            current_time = arrival_times[req_idx]
            release_time = current_time + hold_times[req_idx]
            
            to_remove = [cid for cid, (_, _, rtime) in active_connections.items() 
                         if rtime <= current_time]
            
            for conn_id in to_remove:
                conn_route, conn_wavelength, _ = active_connections[conn_id]
                self.release_wavelength(conn_route, conn_wavelength)
                del active_connections[conn_id]
            
            source, target = random.choice(self.manual_pairs)
            pair_idx = self.manual_pairs.index((source, target))
            
            if pair_idx < len(best_individual):
                route_idx = best_individual[pair_idx]
                routes = self.k_shortest_paths.get((source, target), [])
                if route_idx < len(routes):
                    route = routes[route_idx]
                else:
                    blocked_requests += 1
                    continue
            else:
                blocked_requests += 1
                continue
            
            wavelength = self.find_available_wavelength(route)
            
            if wavelength is not None:
                self.allocate_wavelength(route, wavelength)
                active_connections[next_id] = (route, wavelength, release_time)
                next_id += 1
            else:
                blocked_requests += 1
        
        for conn_route, conn_wavelength, _ in active_connections.values():
            self.release_wavelength(conn_route, conn_wavelength)
        
        return blocked_requests / num_requests if num_requests > 0 else 1.0


# ============================================
# FUNÇÃO OBJETIVO PARA OPTUNA
# ============================================

def objective_ag(trial: optuna.Trial, graph: nx.Graph, pairs: List, num_wavelengths: int) -> float:
    """Função objetivo para otimização do AG."""
    
    # Espaço de busca para AG
    population_size = trial.suggest_int('population_size', 50, 200)
    num_generations_ag = trial.suggest_int('num_generations_ag', 20, 100)
    crossover_rate = trial.suggest_float('crossover_rate', 0.1, 0.9)
    mutation_rate = trial.suggest_float('mutation_rate', 0.01, 0.5)
    tournament_size = trial.suggest_int('tournament_size', 2, 10)
    hops_weight = trial.suggest_float('hops_weight', 0.3, 0.7)
    wavelength_weight = 1.0 - hops_weight
    
    # Cria simulador
    simulator = WDMSimulatorOptimized(
        graph=graph,
        num_wavelengths=num_wavelengths,
        gene_size=5,
        manual_pairs=pairs,
        k=150,
        population_size=population_size,
        n_gen=num_generations_ag,
        num_generations_ag=num_generations_ag,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        hops_weight=hops_weight,
        wavelength_weight=wavelength_weight
    )
    
    # Executa AG
    best_ind, _ = simulator.run_ag(seed=42)
    
    # Simula tráfego
    simulator.reset_network()
    bp = simulator.simulate_blocking_probability(best_ind, load=100.0, num_requests=2000)
    
    return bp


def objective_pso(trial: optuna.Trial, graph: nx.Graph, pairs: List, num_wavelengths: int) -> float:
    """Função objetivo para otimização do PSO."""
    
    # Espaço de busca para PSO
    population_size = trial.suggest_int('population_size', 50, 200)
    n_gen = trial.suggest_int('n_gen', 20, 100)
    w = trial.suggest_float('w', 0.4, 0.9)
    c1 = trial.suggest_float('c1', 1.0, 2.5)
    c2 = trial.suggest_float('c2', 1.0, 2.5)
    hops_weight = trial.suggest_float('hops_weight', 0.3, 0.7)
    wavelength_weight = 1.0 - hops_weight
    
    simulator = WDMSimulatorOptimized(
        graph=graph,
        num_wavelengths=num_wavelengths,
        gene_size=5,
        manual_pairs=pairs,
        k=150,
        population_size=population_size,
        n_gen=n_gen,
        w=w,
        c1=c1,
        c2=c2,
        hops_weight=hops_weight,
        wavelength_weight=wavelength_weight
    )
    
    best_ind, _ = simulator.run_pso(seed=42)
    
    simulator.reset_network()
    bp = simulator.simulate_blocking_probability(best_ind, load=100.0, num_requests=2000)
    
    return bp


def objective_de(trial: optuna.Trial, graph: nx.Graph, pairs: List, num_wavelengths: int) -> float:
    """Função objetivo para otimização do DE."""
    
    # Espaço de busca para DE
    population_size = trial.suggest_int('population_size', 50, 200)
    n_gen = trial.suggest_int('n_gen', 20, 100)
    CR = trial.suggest_float('CR', 0.3, 0.9)
    F = trial.suggest_float('F', 0.3, 0.9)
    hops_weight = trial.suggest_float('hops_weight', 0.3, 0.7)
    wavelength_weight = 1.0 - hops_weight
    
    simulator = WDMSimulatorOptimized(
        graph=graph,
        num_wavelengths=num_wavelengths,
        gene_size=5,
        manual_pairs=pairs,
        k=150,
        population_size=population_size,
        n_gen=n_gen,
        CR=CR,
        F=F,
        hops_weight=hops_weight,
        wavelength_weight=wavelength_weight
    )
    
    best_ind, _ = simulator.run_de(seed=42)
    
    simulator.reset_network()
    bp = simulator.simulate_blocking_probability(best_ind, load=100.0, num_requests=2000)
    
    return bp


# ============================================
# FUNÇÃO PRINCIPAL DE OTIMIZAÇÃO
# ============================================

def get_redclara_graph():
    G = nx.Graph()
    edges = [(0, 1), (0, 5), (0, 8), (0, 11), (1, 2), (2, 3), (3, 4), (4, 5),
             (5, 6), (5, 7), (5, 11), (7, 8), (8, 9), (8, 11), (9, 10), (9, 11), (11, 12)]
    G.add_edges_from(edges)
    G.name = "RedCLARA"
    return G


REDCLARA_PAIRS = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]


def optimize_algorithm(algorithm_name: str, n_trials: int = 100):
    """
    Otimiza os hiperparâmetros para um algoritmo específico.
    
    Args:
        algorithm_name: 'AG', 'PSO' ou 'DE'
        n_trials: Número de tentativas do Optuna
    
    Returns:
        Dicionário com os melhores parâmetros encontrados
    """
    print("\n" + "="*80)
    print(f"OTIMIZANDO HIPERPARÂMETROS PARA {algorithm_name}")
    print("="*80)
    
    # Configuração fixa para otimização
    graph = get_redclara_graph()
    pairs = REDCLARA_PAIRS
    num_wavelengths = 40
    
    # Seleciona a função objetivo
    if algorithm_name == 'AG':
        objective_func = lambda trial: objective_ag(trial, graph, pairs, num_wavelengths)
    elif algorithm_name == 'PSO':
        objective_func = lambda trial: objective_pso(trial, graph, pairs, num_wavelengths)
    elif algorithm_name == 'DE':
        objective_func = lambda trial: objective_de(trial, graph, pairs, num_wavelengths)
    else:
        raise ValueError(f"Algoritmo desconhecido: {algorithm_name}")
    
    # Cria estudo Optuna
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Executa otimização
    study.optimize(objective_func, n_trials=n_trials, show_progress_bar=True)
    
    # Resultados
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n✅ Melhores parâmetros para {algorithm_name}:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    print(f"   Melhor blocking probability: {best_value:.6f}")
    
    # Salva resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"best_params_{algorithm_name}_{timestamp}.json"
    
    results = {
        "algorithm": algorithm_name,
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": n_trials,
        "datetime": timestamp
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Resultados salvos em: {output_file}")
    
    return best_params, best_value


def main():
    """Função principal - otimiza os 3 algoritmos sequencialmente."""
    
    print("\n" + "="*80)
    print("OTIMIZAÇÃO DE HIPERPARÂMETROS PARA AG, PSO E DE")
    print("="*80)
    print("\nEste script irá otimizar cada algoritmo individualmente.")
    print("Recomenda-se 100-200 trials por algoritmo para resultados robustos.\n")
    
    # Número de trials (reduza para teste rápido, aumente para resultado final)
    N_TRIALS = 100  # Para teste rápido
    # N_TRIALS = 500  # Para resultado final
    
    results = {}
    
    # Otimiza AG
    best_ag, val_ag = optimize_algorithm('AG', n_trials=N_TRIALS)
    results['AG'] = {'params': best_ag, 'value': val_ag}
    
    # Otimiza PSO
    best_pso, val_pso = optimize_algorithm('PSO', n_trials=N_TRIALS)
    results['PSO'] = {'params': best_pso, 'value': val_pso}
    
    # Otimiza DE
    best_de, val_de = optimize_algorithm('DE', n_trials=N_TRIALS)
    results['DE'] = {'params': best_de, 'value': val_de}
    
    # Resumo final
    print("\n" + "="*80)
    print("RESUMO FINAL DA OTIMIZAÇÃO")
    print("="*80)
    
    for algo in ['AG', 'PSO', 'DE']:
        print(f"\n{algo}:")
        print(f"   Melhor BP: {results[algo]['value']:.6f}")
        for param, value in results[algo]['params'].items():
            print(f"   {param}: {value}")
    
    # Salva resumo consolidado
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"optimization_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Resumo consolidado salvo em: {summary_file}")
    print("\n✅ OTIMIZAÇÃO CONCLUÍDA!")


if __name__ == "__main__":
    main()