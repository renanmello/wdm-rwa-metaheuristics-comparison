import random
import os
import time
from itertools import islice
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pymoo.core.problem import ElementwiseProblem


class MyProblem(ElementwiseProblem):
    """Classe do problema para o algoritmo DE usando pymoo (AGORA SUPORTA QUALQUER k)."""
    
    def __init__(self, gene_size, calc_fitness, manual_pairs, num_routes_per_pair):
        """
        Args:
            gene_size: Número de genes (requisições)
            calc_fitness: Função de fitness
            manual_pairs: Lista de pares OD
            num_routes_per_pair: Lista com número de rotas disponíveis para cada par
        """
        self.calc_fitness = calc_fitness
        self.manual_pairs = manual_pairs
        self.num_routes_per_pair = num_routes_per_pair

        # CORREÇÃO: Limites agora são baseados no número de rotas por par
        xl = np.array([0] * gene_size)
        xu = np.array([max(0, n-1) for n in num_routes_per_pair])  # índice máximo = n-1

        super().__init__(
            n_var=gene_size,
            n_obj=1,
            xl=xl,
            xu=xu,
            vtype=int
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        x_int = x.astype(int)
        # Garante que índices estão dentro dos limites
        for i, max_idx in enumerate(self.num_routes_per_pair):
            if max_idx > 0:
                x_int[i] = min(max_idx - 1, max(0, x_int[i]))
        
        # DE minimiza, então invertemos o sinal do fitness
        fitness = -self.calc_fitness(x_int.tolist(), self.manual_pairs)
        out["F"] = [fitness]


class WDMSimulator:
    """
    Simulador de rede WDM (Wavelength Division Multiplexing) usando
    algoritmo DE (Differential Evolution) para resolver o problema 
    RWA (Routing and Wavelength Assignment).
    """

    def __init__(
        self,
        graph: nx.Graph,
        num_wavelengths: int = 4,
        gene_size: int = 5,
        manual_selection: bool = True,
        k: int = 83,  # AGORA PODE SER QUALQUER VALOR
        population_size: int = 120,
        hops_weight: float = 0.7,
        wavelength_weight: float = 0.3,
        # Parâmetros DE
        CR: float = 0.9,
        F: float = 0.8,
        n_gen: int = 40
    ):
        """
        Inicializa o simulador WDM com DE.
        """
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.gene_size = gene_size
        self.k = k  # AGORA PODE SER 2, 83, 150, ETC.
        self.manual_pairs = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
        self.manual_selection = manual_selection
        self.hops_weight = hops_weight
        self.wavelength_weight = wavelength_weight
        
        # Parâmetros DE
        self.population_size = population_size
        self.CR = CR
        self.F = F
        self.n_gen = n_gen

        # Parâmetros da simulação
        self.base_calls = 100  # Chamadas base por load 1
        self.calls_multiplier = 5  # Incremento por load
        
        # Parâmetros de fitness IGUAIS AO AGP
        self.penalty_per_hop = 0.2  # Igual ao AGP: 0.2 por hop
        self.penalty_per_congested_link = 0.5  # Igual ao AGP: 0.5
        
        # Estrutura para rastrear uso de links (para cálculo de congestionamento)
        self.link_usage_count = defaultdict(int)
        
        # Calcula todos os k-shortest paths uma vez
        self.k_shortest_paths = {}
        self.num_routes_per_pair = []  # NOVO: guarda quantas rotas cada par tem
        self._precompute_routes()
        
        # Dicionário para guardar rotas selecionadas pelo melhor indivíduo
        self.selected_routes = {}
        
        self.reset_network()

    def _precompute_routes(self):
        """Pré-computa k rotas para cada requisição."""
        self.num_routes_per_pair = []
        for source, target in self.manual_pairs:
            routes = self._get_k_shortest_paths(source, target, self.k)
            self.k_shortest_paths[(source, target)] = routes
            self.num_routes_per_pair.append(len(routes))
            print(f"  Par [{source},{target}]: {len(routes)} rotas encontradas")

    def _get_k_shortest_paths(self, source: int, target: int, k: int) -> List[List[int]]:
        """Calcula os k menores caminhos entre dois nós."""
        if not nx.has_path(self.graph, source, target):
            return []
        try:
            return list(islice(nx.shortest_simple_paths(self.graph, source, target), k))
        except nx.NetworkXNoPath:
            return []

    def reset_network(self) -> None:
        """Reseta completamente a rede para estado inicial."""
        for u, v in self.graph.edges:
            self.graph[u][v]['available_wavelengths'] = [True] * self.num_wavelengths
        self.link_usage_count.clear()

    def _check_wavelength_availability(self, path: List[int], wavelength: int) -> bool:
        """Verifica se um wavelength específico está disponível em todos os links."""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not self.graph.has_edge(u, v):
                return False
            if not self.graph[u][v]['available_wavelengths'][wavelength]:
                return False
        return True

    def _allocate_wavelength(self, path: List[int], wavelength: int) -> None:
        """Aloca um wavelength específico em todos os links."""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            self.graph[u][v]['available_wavelengths'][wavelength] = False
            self.link_usage_count[(u, v)] += 1

    def _release_wavelength(self, path: List[int], wavelength: int) -> None:
        """Libera um wavelength específico."""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.graph.has_edge(u, v):
                self.graph[u][v]['available_wavelengths'][wavelength] = True
                if (u, v) in self.link_usage_count:
                    self.link_usage_count[(u, v)] = max(0, self.link_usage_count[(u, v)] - 1)

    def _find_available_wavelength(self, path: List[int]) -> Optional[int]:
        """Encontra o primeiro wavelength disponível."""
        for w in range(self.num_wavelengths):
            if self._check_wavelength_availability(path, w):
                return w
        return None

    def _estimate_congestion(self, route: List[int]) -> float:
        """ESTIMA congestionamento da rota baseado no histórico de uso."""
        if len(route) < 2:
            return 1.0
        
        total_congestion = 0.0
        links = 0
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            link_key = (u, v) if (u, v) in self.link_usage_count else (v, u)
            
            if link_key in self.link_usage_count:
                usage = self.link_usage_count[link_key]
                congestion = min(1.0, usage / (self.num_wavelengths * 2))
            else:
                congestion = 0.0
            
            total_congestion += congestion
            links += 1
        
        return total_congestion / links if links > 0 else 1.0

    def _fitness_route(self, route: List[int]) -> float:
        """Fitness IGUAL AO AGP: hops + congestionamento."""
        if len(route) < 2:
            return 0.0

        hops = len(route) - 1
        hops_penalty = hops * self.penalty_per_hop
        congestion = self._estimate_congestion(route)
        congestion_penalty = congestion * self.penalty_per_congested_link
        
        fitness = 1.0 - hops_penalty - congestion_penalty
        
        if hops <= 3:
            fitness *= 1.5
        
        return max(0.01, fitness)

    def _fitness(self, individual: List[int], source_targets: List[Tuple[int, int]]) -> float:
        """Calcula fitness total."""
        total_fitness = 0.0
        valid_routes = 0

        for i, (source, target) in enumerate(source_targets):
            if i >= len(individual):
                continue

            route_idx = individual[i]
            available_routes = self.k_shortest_paths.get((source, target), [])

            if not available_routes or route_idx >= len(available_routes):
                continue

            route = available_routes[route_idx]
            if len(route) >= 2:
                route_fitness = self._fitness_route(route)
                total_fitness += route_fitness
                valid_routes += 1

        return total_fitness / valid_routes if valid_routes > 0 else 0.0

    def _simulate_calls_for_load(self, load: int, individual: List[int]) -> Dict[int, float]:
        """Simula chamadas."""
        self.reset_network()
        
        self.selected_routes = {}
        for i, (source, target) in enumerate(self.manual_pairs):
            if i < len(individual):
                routes = self.k_shortest_paths.get((source, target), [])
                if routes and individual[i] < len(routes):
                    self.selected_routes[i] = routes[individual[i]]
        
        blocked_by_pair = defaultdict(int)
        arrivals_by_pair = defaultdict(int)
        
        total_attempts = self.base_calls + (load * self.calls_multiplier)
        
        active_calls = []
        
        for attempt in range(total_attempts):
            pair_idx = random.randint(0, len(self.manual_pairs) - 1)
            arrivals_by_pair[pair_idx] += 1
            
            if pair_idx in self.selected_routes:
                route = self.selected_routes[pair_idx]
                wavelength = self._find_available_wavelength(route)
                
                if wavelength is not None:
                    self._allocate_wavelength(route, wavelength)
                    active_calls.append((pair_idx, route, wavelength))
                    
                    if len(active_calls) > self.num_wavelengths * 10:
                        if random.random() < 0.1 and active_calls:
                            release_idx = random.randint(0, len(active_calls) - 1)
                            released_pair, released_route, released_wavelength = active_calls.pop(release_idx)
                            self._release_wavelength(released_route, released_wavelength)
                else:
                    blocked_by_pair[pair_idx] += 1
                    
                    if active_calls and random.random() < 0.3:
                        release_idx = random.randint(0, len(active_calls) - 1)
                        released_pair, released_route, released_wavelength = active_calls.pop(release_idx)
                        self._release_wavelength(released_route, released_wavelength)
            else:
                blocked_by_pair[pair_idx] += 1
        
        blocking_probs_by_pair = {}
        for pair_idx in range(len(self.manual_pairs)):
            if arrivals_by_pair[pair_idx] > 0:
                blocking_probs_by_pair[pair_idx] = blocked_by_pair[pair_idx] / arrivals_by_pair[pair_idx]
            else:
                blocking_probs_by_pair[pair_idx] = 0.0
            
        return blocking_probs_by_pair

    def de_algorithm(self) -> Tuple[List[int], float]:
        """Executa DE."""
        print(f"Iniciando DE com {self.population_size} indivíduos, {self.n_gen} gerações...")
        print(f"Parâmetros DE: CR={self.CR}, F={self.F}")
        print(f"k={self.k} rotas por requisição (índices 0 a {max(self.num_routes_per_pair)-1})")
    
        # CORREÇÃO: Passa num_routes_per_pair para o problema
        problem = MyProblem(
            self.gene_size, 
            self._fitness, 
            self.manual_pairs,
            self.num_routes_per_pair
        )

        from pymoo.algorithms.soo.nonconvex.de import DE
        from pymoo.operators.sampling.rnd import IntegerRandomSampling

        algorithm = DE(
            pop_size=self.population_size,
            CR=self.CR,
            F=self.F,
            variant="DE/rand/1/bin",
            sampling=IntegerRandomSampling(),
        )

        from pymoo.termination import get_termination
        termination = get_termination("n_gen", self.n_gen)

        from pymoo.optimize import minimize
        res_de = minimize(
            problem, 
            algorithm,
            termination,
            seed=None,
            save_history=True,
            verbose=False
        )

        X_DE = res_de.X.astype(int)
        F_DE = -res_de.F[0]

        print("-" * 50)
        print(f"RESULTADOS DE (k={self.k})")
        print(f"Melhor Indivíduo: {X_DE}")
        print(f"Fitness: {F_DE:.6f}")
        
        print("Rotas selecionadas:")
        for i, (source, target) in enumerate(self.manual_pairs):
            if i < len(X_DE):
                route_idx = X_DE[i]
                routes = self.k_shortest_paths.get((source, target), [])
                if routes and route_idx < len(routes):
                    route = routes[route_idx]
                    hops = len(route) - 1
                    print(f"  [{source},{target}]: Índice {route_idx}/{len(routes)-1}, {hops} hops")
        
        print("-" * 50)

        return X_DE.tolist(), F_DE

    def simulate_network(
        self, 
        num_simulations: int = 1,
        output_file: str = "blocking_results_de.txt"
    ) -> Tuple[Dict[str, List[float]], List[int], float]:
        """Simula rede."""
        start_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"EXECUTANDO DE COM FITNESS IGUAL AO AGP (k={self.k})")
        print(f"{'='*50}")
        best_individual, best_fitness_value = self.de_algorithm()
        
        all_simulations = {i: [] for i in range(self.gene_size)}
        results_by_route = {}
        
        # Abre arquivos com k no nome
        route_files = {}
        for gene_idx in range(self.gene_size):
            source, target = self.manual_pairs[gene_idx]
            filename = f"route_{source}_{target}_de_k{self.k}.txt"
            route_files[gene_idx] = open(filename, 'w')
            route_files[gene_idx].write(f"# Resultados para rota [{source},{target}] (DE k={self.k})\n")
            route_files[gene_idx].write("# Formato: Load Probabilidade_Media Desvio_Padrao\n")
            results_by_route[gene_idx] = []

        print(f"\n{'='*50}")
        print(f"EXECUTANDO SIMULAÇÕES (DE k={self.k})")
        print(f"Número de simulações: {num_simulations}")
        print(f"Loads: 1-400")
        print(f"{'='*50}")

        for load in range(1, 401):
            print(f"\nProcessando Load {load}/400...")
            
            load_results_by_route = {i: [] for i in range(self.gene_size)}
            
            for sim in range(num_simulations):
                if (sim + 1) % 5 == 0:
                    print(f"  Simulação {sim + 1}/{num_simulations}")
                
                blocking_by_pair = self._simulate_calls_for_load(load, best_individual)
                
                for pair_idx, blocking_prob in blocking_by_pair.items():
                    load_results_by_route[pair_idx].append(blocking_prob)
            
            for gene_idx in range(self.gene_size):
                if load_results_by_route[gene_idx]:
                    mean_prob = np.mean(load_results_by_route[gene_idx])
                    std_prob = np.std(load_results_by_route[gene_idx])
                    
                    results_by_route[gene_idx].append((load, mean_prob, std_prob))
                    route_files[gene_idx].write(f"{load} {mean_prob:.6f} {std_prob:.6f}\n")
                    all_simulations[gene_idx].append(load_results_by_route[gene_idx])
            
            if load % 20 == 0:
                all_probs = []
                for gene_idx in range(self.gene_size):
                    if load_results_by_route[gene_idx]:
                        all_probs.extend(load_results_by_route[gene_idx])
                if all_probs:
                    avg_blocking = np.mean(all_probs)
                    print(f"  Load {load}: Pblock médio = {avg_blocking:.4f}")

        for gene_idx, file in route_files.items():
            file.close()
            source, target = self.manual_pairs[gene_idx]
            print(f"✓ Resultados salvos em route_{source}_{target}_de_k{self.k}.txt")

        end_time = time.time()
        total_time = end_time - start_time
        
        self._save_statistics(all_simulations, num_simulations, f"k{self.k}")
        self._save_execution_time(total_time, num_simulations, f"k{self.k}")

        print(f"\n{'='*50}")
        print(f"Resultados salvos em arquivos individuais por rota (k={self.k})")
        print(f"Tempo total de execução: {total_time:.2f} segundos")
        print(f"{'='*50}")
        
        final_results = {}
        for gene_idx in range(self.gene_size):
            source, target = self.manual_pairs[gene_idx]
            label = f'[{source},{target}]'
            if results_by_route[gene_idx]:
                probs = [item[1] for item in results_by_route[gene_idx]]
                final_results[label] = probs
        
        return final_results, best_individual, best_fitness_value

    def _save_statistics(self, all_simulations: Dict[int, List[List[float]]], 
                        num_sims: int, suffix: str = "") -> None:
        """Salva estatísticas."""
        print(f"\n{'='*50}")
        print("SALVANDO ESTATÍSTICAS CONSOLIDADAS")
        print(f"{'='*50}")
        
        for gene_idx in range(self.gene_size):
            if gene_idx not in all_simulations or not all_simulations[gene_idx]:
                print(f"⚠️  Gene {gene_idx + 1}: Sem dados para processar")
                continue
            
            source, target = self.manual_pairs[gene_idx]
            
            transposed_data = []
            num_loads = 400
            
            for load_idx in range(num_loads):
                load_data = []
                for sim_list in all_simulations[gene_idx]:
                    if load_idx < len(sim_list):
                        load_data.append(sim_list[load_idx])
                transposed_data.append(load_data)
            
            mean_probs = [np.mean(data) if data else 0.0 for data in transposed_data]
            std_probs = [np.std(data) if data else 0.0 for data in transposed_data]
            
            filename_suffix = f"_{suffix}" if suffix else ""
            
            mean_filename = f"gene_{gene_idx + 1}_de_mean{filename_suffix}.txt"
            with open(mean_filename, "w") as f:
                f.write(f"# [{source} -> {target}] - Média de {num_sims} simulações (DE {suffix})\n")
                f.write("# Formato: Load Probabilidade_Media\n")
                for load, prob in enumerate(mean_probs, 1):
                    f.write(f"{load} {prob:.6f}\n")
            print(f"   ✓ Salvo: {mean_filename}")

    def _save_execution_time(self, total_time: float, num_sims: int, 
                            suffix: str = "") -> None:
        """Salva tempo de execução."""
        filename_suffix = f"_{suffix}" if suffix else ""
        filename = f"execution_time_de{filename_suffix}.txt"
        
        with open(filename, "w") as f:
            f.write(f"# Tempo de Execução - DE {suffix}\n")
            f.write(f"# Número de simulações: {num_sims}\n")
            f.write(f"# Número de wavelengths: {self.num_wavelengths}\n")
            f.write(f"# População: {self.population_size}\n")
            f.write(f"# Gerações: {self.n_gen}\n")
            f.write(f"# k (rotas): {self.k}\n")
            f.write(f"# Rotas por requisição: {self.num_routes_per_pair}\n")
            f.write(f"#\n")
            f.write(f"Tempo total (segundos): {total_time:.2f}\n")
            f.write(f"Tempo médio por simulação (segundos): {total_time/num_sims:.2f}\n")
        
        print(f"\n⏱️  Tempo de execução salvo em: {filename}")


def main():
    """Função principal."""
    
    # ESCOLHA A TOPOLOGIA AQUI
    topology = "redclara"  # "nsfnet", "redclara", "janet6", "ipe"
    
    print("\n" + "="*60)
    print(f"DE GENÉRICO (k VARIÁVEL) - TOPOLOGIA: {topology.upper()}")
    print("="*60)
    
    # Criar rede
    graph = nx.Graph()
    
    if topology == "nsfnet":
        nsfnet_edges = [
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
            (4, 6), (4, 5), (5, 8), (5, 12), (6, 7), (7, 9), (8, 9), (9, 11),
            (9, 13), (10, 11), (10, 13), (11, 12)
        ]
        graph.add_edges_from(nsfnet_edges)
    elif topology == "redclara":
        redclara_edges =[
            (0, 1), (0, 5), (0, 8), (0, 11),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6), (5, 7), (5, 11),
            (7, 8),
            (8, 9), (8, 11),
            (9, 10), (9, 11),
            (11, 12)
        ]
        graph.add_edges_from(redclara_edges)
    elif topology == "janet6":
        janet6_edges = [
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4),
            (3, 4), (3, 5), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7), (6, 8),
            (7, 8), (7, 9), (8, 9), (8, 10), (9, 10), (9, 11), (10, 11),
            (10, 12), (11, 12), (11, 13), (12, 13), (12, 14), (13, 14),
            (13, 15), (14, 15), (14, 16), (15, 16), (15, 17), (16, 17),
            (16, 18), (17, 18), (17, 19), (18, 19), (18, 20), (19, 20),
            (19, 21), (20, 21), (20, 22), (21, 22), (21, 23), (22, 23)
        ]
        graph.add_edges_from(janet6_edges)
    elif topology == "ipe":
        ipe_edges = [
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4),
            (3, 4), (3, 5), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7), (6, 8),
            (7, 8), (7, 9), (8, 9), (8, 10), (9, 10), (9, 11), (10, 11),
            (10, 12), (11, 12), (11, 13), (12, 13), (12, 14), (13, 14),
            (13, 15), (14, 15), (14, 16), (15, 16), (15, 17), (16, 17),
            (16, 18), (17, 18), (17, 19), (18, 19), (18, 20), (19, 20),
            (19, 21), (20, 21), (20, 22), (21, 22), (21, 23), (22, 23),
            (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
            (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35),
            (35, 36), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41),
            (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47)
        ]
        graph.add_edges_from(ipe_edges)
    
    # ESCOLHA O VALOR DE k AQUI (PARA COMPARAR COM PSO, USE 83)
    k_value = 83  # PODE SER 2, 83, 150, ETC.
    
    # Configurações
    num_wavelengths = 80
    
    print(f"\nConfigurações:")
    print(f"  k = {k_value} rotas por requisição")
    print(f"  wavelengths = {num_wavelengths}")
    print(f"  população = 120")
    print(f"  gerações = 40")
    print(f"  loads = 1-400\n")

    # Criar simulador
    simulator = WDMSimulator(
        graph=graph,
        num_wavelengths=num_wavelengths,
        gene_size=5,
        manual_selection=True,
        k=k_value,
        population_size=120,
        hops_weight=0.7,
        wavelength_weight=0.3,
        CR=0.9,
        F=0.8,
        n_gen=40
    )

    # Executar simulação
    output_file = f"blocking_results_de_k{k_value}_{topology}.txt"
    results, best_individual, best_fitness_value = simulator.simulate_network(
        num_simulations=20,
        output_file=output_file
    )

    print(f"\n{'='*60}")
    print("SIMULAÇÃO CONCLUÍDA!")
    print(f"Melhor indivíduo: {best_individual}")
    print(f"Fitness: {best_fitness_value:.6f}")
    
    print("\nMÉDIAS (loads 1-400):")
    all_probs = []
    for (source, target) in simulator.manual_pairs:
        req_key = f'[{source},{target}]'
        if results[req_key]:
            avg = np.mean(results[req_key])
            all_probs.extend(results[req_key])
            print(f"  [{source},{target}]: {avg:.6f} ({avg*100:.2f}%)")
    
    if all_probs:
        print(f"\n  MÉDIA GERAL: {np.mean(all_probs):.6f} ({np.mean(all_probs)*100:.2f}%)")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
