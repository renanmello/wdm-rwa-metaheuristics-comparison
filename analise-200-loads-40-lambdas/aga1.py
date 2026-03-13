import heapq
import sys
import time
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from collections import Counter, defaultdict
from itertools import islice
from scipy.stats import poisson
from typing import List, Tuple, Dict, Optional


class WDMSimulatorOldSeparated:
    """
    Versão modificada do AG antigo que armazena resultados por requisição.
    """
    def __init__(self, graph: nx.Graph, num_wavelengths: int = 40, 
                 source_target_pairs: List[Tuple[int, int]] = None):
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.traffic_matrix = np.zeros((len(graph.nodes), len(graph.nodes), self.num_wavelengths))
        self.allocated_routes = {}
        self.event_queue = []
        self.k = 150
        self.population_size = 150
        self.num_generations = 40
        self.crossover_rate = 0.8
        self.mutation_rate = 0.15
        self.k_shortest_paths = self.get_all_k_shortest_paths()
        self.route_cache = {}
        self.route_popularity = Counter()
        
        # Lista de pares origem-destino a serem testados
        if source_target_pairs is None:
            self.source_target_pairs = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
        else:
            self.source_target_pairs = source_target_pairs
        
        # Para armazenar resultados por requisição
        self.requisition_results = {}
        
        # Carrega rotas para todos os pares
        for source, target in self.source_target_pairs:
            self.load_fixed_routes(source, target)

    def add_link(self, node1: int, node2: int, capacity: int):
        self.graph.add_edge(node1, node2, capacity=capacity, 
                           wavelengths=np.ones(self.num_wavelengths, dtype=bool))

    def load_fixed_routes(self, source: int, target: int):
        """Carrega os k caminhos mais curtos entre origem e destino."""
        if (source, target) not in self.k_shortest_paths:
            self.k_shortest_paths[(source, target)] = self.get_k_shortest_paths(source, target, self.k)

    def initialize_traffic_matrix(self, max_calls_per_pair: int = 1):
        self.traffic_matrix = np.zeros((len(self.graph.nodes), len(self.graph.nodes), self.num_wavelengths))
        for _ in range(len(self.graph.nodes) ** 2):
            src = random.choice(list(self.graph.nodes))
            dst = random.choice([node for node in self.graph.nodes if node != src])
            wavelength = random.randint(0, self.num_wavelengths - 1)
            num_calls = poisson.rvs(mu=max_calls_per_pair)
            self.traffic_matrix[src][dst][wavelength] += num_calls

    def test_and_allocate_route(self, route: List[int], allocation_method: str = 'first_fit') -> int:
        if allocation_method == 'first_fit':
            return self.allocate_first_fit(route)
        elif allocation_method == 'random_fit':
            return self.allocate_random_fit(route)
        elif allocation_method == 'genetic':
            return self.allocate_genetic(route)
        return -1

    def allocate_first_fit(self, route: List[int]) -> int:
        for wavelength in range(self.num_wavelengths):
            if self.is_viable_route(route, wavelength):
                self.allocate_route(route, wavelength)
                return wavelength
        return -1

    def allocate_random_fit(self, route: List[int]) -> int:
        wavelength = random.randint(0, 3)
        if self.is_viable_route(route, wavelength):
            self.allocate_route(route, wavelength)
            return wavelength
        return -1

    def allocate_genetic(self, route: List[int]) -> int:
        best_wavelength = self.get_best_wavelength(route)
        if best_wavelength != -1:
            self.allocate_route(route, best_wavelength)
        return best_wavelength

    def get_best_wavelength(self, route: List[int]) -> int:
        for wavelength in range(self.num_wavelengths):
            if self.is_viable_route(route, wavelength):
                return wavelength
        return -1

    def is_viable_route(self, route: List[int], wavelength: int) -> bool:
        for i in range(len(route) - 1):
            if not self.graph[route[i]][route[i + 1]]['wavelengths'][wavelength]:
                return False
        return True

    def allocate_route(self, route: List[int], wavelength: int):
        for i in range(len(route) - 1):
            node1, node2 = route[i], route[i + 1]
            link = self.graph[node1][node2]
            link['wavelengths'][wavelength] = False
            self.traffic_matrix[node1][node2][wavelength] += 1
        self.allocated_routes[tuple(route)] = wavelength

    def release_route(self, route: List[int], wavelength: int):
        for i in range(len(route) - 1):
            node1, node2 = route[i], route[i + 1]
            link = self.graph[node1][node2]
            link['wavelengths'][wavelength] = True
            self.traffic_matrix[node1][node2][wavelength] -= 1
        self.allocated_routes.pop(tuple(route), None)

    def get_all_k_shortest_paths(self) -> Dict[Tuple[int, int], List[List[int]]]:
        k_paths = {}
        for source in self.graph.nodes:
            for target in self.graph.nodes:
                if source != target:
                    k_paths[(source, target)] = self.get_k_shortest_paths(source, target, self.k)
        return k_paths

    def is_valid_route_with_wavelengths(self, route: List[int], wavelengths: List[int]) -> bool:
        for i in range(len(route) - 1):
            node1, node2 = route[i], route[i + 1]
            if not self.graph[node1][node2]['wavelengths'][wavelengths[i]]:
                return False
        return True

    def is_valid_route(self, route: List[int], source: int = None, target: int = None) -> bool:
        if source is not None and target is not None:
            if route[0] != source or route[-1] != target:
                return False
        return all(self.graph.has_edge(route[i], route[i + 1]) for i in range(len(route) - 1))

    def get_k_shortest_paths(self, source: int, target: int, k: int) -> List[List[int]]:
        return list(islice(nx.shortest_simple_paths(self.graph, source, target), k))

    def get_route(self, src: int, dst: int, routing_method: str) -> List[int]:
        if (src, dst, routing_method) in self.route_cache:
            return self.route_cache[(src, dst, routing_method)]

        if routing_method == 'traditional':
            route = nx.shortest_path(self.graph, src, dst)
        elif routing_method == 'dijkstra':
            route = self.get_dijkstra_path(src, dst)
        elif routing_method == 'genetic':
            # Para cada par específico, executamos o AG
            if (src, dst) in self.k_shortest_paths:
                route, _ = self.genetic_algorithm_specific(src, dst)
            else:
                route = nx.shortest_path(self.graph, src, dst)

        self.route_cache[(src, dst, routing_method)] = route
        return route

    def get_dijkstra_path(self, src: int, dst: int) -> List[int]:
        try:
            return nx.dijkstra_path(self.graph, src, dst)
        except:
            return nx.shortest_path(self.graph, src, dst)

    def initialize_population_specific(self, source: int, target: int) -> List[Tuple[List[int], List[int]]]:
        """Inicializa população para um par específico."""
        population = []
        base_routes = self.k_shortest_paths.get((source, target), [])

        # Garante que a população inicial inclui os k caminhos mais curtos fixos
        for base_route in base_routes:
            if len(population) < self.population_size:
                population.append((base_route, []))

        return population[:self.population_size]

    def fitness(self, solution: Tuple[List[int], List[int]]) -> float:
        route, _ = solution
        last_wavelength = None
        switch_penalty = 0
        route_length_penalty = len(route) - 1
        total_availability = 0
        valid_links = 0

        for i in range(len(route) - 1):
            node1, node2 = route[i], route[i + 1]

            if not self.graph.has_edge(node1, node2):
                return 0

            link_availability = sum(self.graph[node1][node2]['wavelengths']) / self.num_wavelengths
            total_availability += link_availability
            valid_links += 1

            if last_wavelength is not None and self.graph[node1][node2]['wavelengths'][last_wavelength]:
                continue
            else:
                for wavelength in range(self.num_wavelengths):
                    if self.graph[node1][node2]['wavelengths'][wavelength]:
                        last_wavelength = wavelength
                        break
                else:
                    return 0
                switch_penalty += 1

        mean_availability = total_availability / valid_links if valid_links > 0 else 0
        fitness_value = mean_availability / (1 + 0.2 * switch_penalty + 0.1 * route_length_penalty)
        return fitness_value

    def selection(self, population: List[Tuple[List[int], List[int]]], 
                  fitnesses: List[float]) -> List[Tuple[List[int], List[int]]]:
        selected = []
        for _ in range(len(population)):
            idx1, idx2 = random.sample(range(len(population)), 2)
            if fitnesses[idx1] > fitnesses[idx2]:
                selected.append(population[idx1])
            else:
                selected.append(population[idx2])
        return selected

    def crossover(self, parent1: Tuple[List[int], List[int]], 
                  parent2: Tuple[List[int], List[int]]) -> Tuple[Tuple[List[int], List[int]], 
                                                                 Tuple[List[int], List[int]]]:
        route1, wavelengths1 = parent1
        route2, wavelengths2 = parent2

        wavelengths1 = wavelengths1 if wavelengths1 else [0] * (len(route1) - 1)
        wavelengths2 = wavelengths2 if wavelengths2 else [0] * (len(route2) - 1)

        if len(route1) > 2 and len(route2) > 2 and random.random() < self.crossover_rate:
            crossover_point = random.randint(1, min(len(route1), len(route2)) - 2)
            child1_route = route1[:crossover_point] + route2[crossover_point:]
            child2_route = route2[:crossover_point] + route1[crossover_point:]

            child1_wavelengths = wavelengths1[:crossover_point] + wavelengths2[crossover_point:]
            child2_wavelengths = wavelengths2[:crossover_point] + wavelengths1[crossover_point:]

            return (child1_route, child1_wavelengths), (child2_route, child2_wavelengths)

        return parent1, parent2

    def mutate(self, solution: Tuple[List[int], List[int]]) -> Tuple[List[int], List[int]]:
        route, wavelengths = solution
        if wavelengths is None or len(wavelengths) == 0:
            wavelengths = [random.randint(0, self.num_wavelengths - 1) for _ in range(len(route) - 1)]

        new_wavelengths = wavelengths[:]

        if len(new_wavelengths) > 0 and random.random() < self.mutation_rate:
            segment_to_mutate = random.randint(0, len(new_wavelengths) - 1)
            new_wavelengths[segment_to_mutate] = random.randint(0, self.num_wavelengths - 1)

        return route, new_wavelengths

    def genetic_algorithm_specific(self, source: int, target: int) -> Tuple[List[int], List[int]]:
        """Executa AG para um par específico."""
        population = self.initialize_population_specific(source, target)

        for generation in range(self.num_generations):
            fitnesses = [self.fitness(ind) for ind in population]
            
            if not any(fitnesses):
                population = self.initialize_population_specific(source, target)
                continue

            population = self.selection(population, fitnesses)
            new_population = []
            for i in range(0, len(population), 2):
                if i + 1 < len(population):
                    parent1, parent2 = population[i], population[i + 1]
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population.extend([self.mutate(child1), self.mutate(child2)])
                else:
                    new_population.append(self.mutate(population[i]))
            population = new_population

        best_solution = max(population, key=lambda ind: self.fitness(ind))
        return best_solution

    def simulate_single_requisition(self, source: int, target: int, 
                                  load_values: List[int], 
                                  num_simulations: int = 1,
                                  calls_per_load: int = 1000,
                                  routing_method: str = 'genetic',
                                  allocation_method: str = 'genetic') -> Dict[int, List[float]]:
        """
        Simula uma única requisição (par O-D) para todos os loads.
        Retorna dicionário: {load: [probabilidades_de_bloqueio]}
        """
        results = {load: [] for load in load_values}
        
        print(f"\nSimulando requisição [{source},{target}]...")
        
        # Executa AG uma vez para esta requisição
        print(f"  Executando AG para encontrar melhor rota...")
        best_route, _ = self.genetic_algorithm_specific(source, target)
        print(f"  Melhor rota encontrada: {best_route}")
        
        # Para cada simulação
        for sim in range(num_simulations):
            # Para cada valor de load
            for load in load_values:
                # Resetar rede
                current_time = 0
                self.event_queue = []
                self.allocated_routes = {}
                for u, v in self.graph.edges():
                    self.graph[u][v]['wavelengths'] = np.ones(self.num_wavelengths, dtype=bool)
                self.route_cache = {}

                blocked_calls = 0
                total_calls = calls_per_load
                
                # Gerar chamadas
                for call_id in range(total_calls):
                    current_time += np.random.exponential(1 / load)
                    
                    # Liberar rotas expiradas
                    while self.event_queue and self.event_queue[0][0] <= current_time:
                        event_time, route, wavelength = heapq.heappop(self.event_queue)
                        self.release_route(route, wavelength)

                    # Tentar alocar na rota encontrada pelo AG
                    allocated_wavelength = self.test_and_allocate_route(best_route, allocation_method)
                    
                    if allocated_wavelength == -1:
                        blocked_calls += 1
                    else:
                        duration = np.random.exponential(1)
                        heapq.heappush(self.event_queue,
                                     (current_time + duration, best_route, allocated_wavelength))

                # Liberar rotas restantes
                while self.event_queue:
                    event_time, route, wavelength = heapq.heappop(self.event_queue)
                    self.release_route(route, wavelength)

                # Calcular probabilidade
                blocking_probability = blocked_calls / total_calls if total_calls > 0 else 0
                results[load].append(blocking_probability)
        
        return results

    def simulate_all_requisitions(self, load_values: List[int] = None,
                                num_simulations: int = 1,
                                calls_per_load: int = 1000,
                                routing_method: str = 'genetic',
                                allocation_method: str = 'genetic',
                                output_prefix: str = "simulation_old") -> Dict[str, Dict[int, List[float]]]:
        """
        Simula todas as requisições separadamente.
        Salva resultados em arquivos individuais.
        """
        if load_values is None:
            load_values = list(range(1, 201))
        
        print(f"\n{'='*60}")
        print("SIMULAÇÃO AG ANTIGO - RESULTADOS POR REQUISIÇÃO")
        print(f"{'='*60}")
        print(f"Configuração:")
        print(f"  • Requisições: {len(self.source_target_pairs)}")
        print(f"  • Loads: {len(load_values)} (de {min(load_values)} a {max(load_values)})")
        print(f"  • Simulações por load: {num_simulations}")
        print(f"  • Chamadas por load: {calls_per_load}")
        print(f"{'='*60}")
        
        # Dicionário para armazenar todos os resultados
        all_results = {}
        
        # Arquivos individuais
        req_files = {}
        
        # Cria arquivos para cada requisição
        for idx, (source, target) in enumerate(self.source_target_pairs):
            filename = f"{output_prefix}_req_{idx+1}.txt"
            req_files[idx] = filename
            
            with open(filename, 'w') as f:
                f.write(f"# Requisição {idx+1}: [{source},{target}]\n")
                f.write(f"# Simulações: {num_simulations}, Chamadas/load: {calls_per_load}\n")
                f.write("# Formato: Load Probabilidade_Bloqueio_Média\n")
                f.write("# " + "="*50 + "\n")
        
        # Arquivo principal
        main_file = f"{output_prefix}_all_results.txt"
        with open(main_file, 'w') as f:
            f.write("# Resultados AG Antigo - Separados por Requisição\n")
            f.write("# " + "="*60 + "\n")
            f.write(f"# Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Configuração:\n")
            f.write(f"#   • Requisições: {self.source_target_pairs}\n")
            f.write(f"#   • Simulações: {num_simulations}\n")
            f.write(f"#   • Loads: {len(load_values)}\n")
            f.write(f"#   • Chamadas/load: {calls_per_load}\n")
            f.write("# " + "="*60 + "\n\n")
        
        # Simula cada requisição
        total_start_time = time.time()
        
        for idx, (source, target) in enumerate(self.source_target_pairs):
            print(f"\nRequisição {idx+1}/{len(self.source_target_pairs)}: [{source},{target}]")
            
            start_time = time.time()
            
            # Simula esta requisição
            results = self.simulate_single_requisition(
                source=source,
                target=target,
                load_values=load_values,
                num_simulations=num_simulations,
                calls_per_load=calls_per_load,
                routing_method=routing_method,
                allocation_method=allocation_method
            )
            
            all_results[f"[{source},{target}]"] = results
            
            # Calcula médias
            avg_by_load = {}
            for load in load_values:
                if load in results and results[load]:
                    avg_by_load[load] = np.mean(results[load])
            
            # Salva no arquivo individual
            with open(req_files[idx], 'a') as f:
                for load in sorted(avg_by_load.keys()):
                    f.write(f"{load} {avg_by_load[load]:.6f}\n")
            
            # Salva no arquivo principal
            with open(main_file, 'a') as f:
                f.write(f"\n# Requisição {idx+1} [{source},{target}]\n")
                for load in load_values:
                    if load in results:
                        probs_str = " ".join([f"{p:.6f}" for p in results[load]])
                        f.write(f"{load} {probs_str}\n")
            
            req_time = time.time() - start_time
            print(f"  Concluído em {req_time:.2f}s")
        
        # Tempo total
        total_time = time.time() - total_start_time
        
        # Salva resumo
        with open(main_file, 'a') as f:
            f.write("\n" + "="*60 + "\n")
            f.write("# RESUMO\n")
            f.write("="*60 + "\n")
            f.write(f"# Tempo total: {total_time:.2f}s\n")
            f.write("# Médias por requisição:\n")
            f.write("# Requisição  Load  Probabilidade_Média\n")
            
            for idx, (source, target) in enumerate(self.source_target_pairs):
                req_key = f"[{source},{target}]"
                if req_key in all_results:
                    for load in load_values:
                        if load in all_results[req_key] and all_results[req_key][load]:
                            mean_prob = np.mean(all_results[req_key][load])
                            f.write(f"{idx+1:2d}  {load:4d}  {mean_prob:.6f}\n")
        
        # Salva tempo de execução
        with open(f"{output_prefix}_tempo.txt", "w") as time_file:
            time_file.write(f"Tempo total: {total_time:.2f}s\n")
            time_file.write(f"Início: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))}\n")
            time_file.write(f"Fim: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}\n")
            time_file.write(f"Requisições: {len(self.source_target_pairs)}\n")
            time_file.write(f"Loads: {len(load_values)}\n")
            time_file.write(f"Simulações: {num_simulations}\n")
            time_file.write(f"Chamadas/load: {calls_per_load}\n")
        
        print(f"\n{'='*60}")
        print("SIMULAÇÃO CONCLUÍDA")
        print(f"{'='*60}")
        print(f"Tempo total: {total_time:.2f}s")
        print(f"Arquivos gerados:")
        print(f"  • {main_file} - Resultados principais")
        print(f"  • {output_prefix}_tempo.txt - Tempos de execução")
        for idx in range(len(self.source_target_pairs)):
            print(f"  • {output_prefix}_req_{idx+1}.txt - Requisição {idx+1}")
        print(f"{'='*60}")
        
        # Gera gráficos
        self.plot_requisition_results(all_results, load_values, num_simulations)
        
        return all_results

    def plot_requisition_results(self, all_results: Dict[str, Dict[int, List[float]]],
                               load_values: List[int], num_simulations: int):
        """Gera gráficos para cada requisição."""
        plt.figure(figsize=(15, 10))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Gráfico 1: Todas as requisições
        plt.subplot(2, 1, 1)
        
        for idx, (req_key, results) in enumerate(all_results.items()):
            means = []
            for load in sorted(load_values):
                if load in results and results[load]:
                    means.append(np.mean(results[load]))
                else:
                    means.append(0.0)
            
            plt.plot(load_values, means,
                    color=colors[idx % len(colors)],
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    label=req_key)
        
        plt.xlabel('Load', fontsize=12)
        plt.ylabel('Probabilidade de Bloqueio', fontsize=12)
        plt.title(f'AG Antigo - Probabilidade de Bloqueio por Requisição\n{num_simulations} simulação(ões)',
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(min(load_values) - 5, max(load_values) + 5)
        plt.ylim(-0.02, 1.02)
        plt.xticks(np.arange(min(load_values), max(load_values) + 1, 20))
        
        # Gráfico 2: Média geral
        plt.subplot(2, 1, 2)
        
        overall_means = []
        for load in sorted(load_values):
            all_probs = []
            for req_key in all_results:
                if load in all_results[req_key]:
                    all_probs.extend(all_results[req_key][load])
            
            if all_probs:
                overall_means.append(np.mean(all_probs))
            else:
                overall_means.append(0.0)
        
        plt.plot(load_values, overall_means,
                'o-', linewidth=2,
                color='darkblue',
                label='Média Geral')
        
        plt.xlabel('Load', fontsize=12)
        plt.ylabel('Probabilidade de Bloqueio', fontsize=12)
        plt.title('Média Geral de Todas as Requisições', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(min(load_values) - 5, max(load_values) + 5)
        plt.ylim(-0.02, 1.02)
        plt.xticks(np.arange(min(load_values), max(load_values) + 1, 20))
        
        plt.tight_layout()
        plt.savefig(f'ag_antigo_results_{time.strftime("%Y%m%d_%H%M%S")}.png', dpi=300)
        plt.show()


def main_old():
    """Função principal para executar o AG antigo com resultados separados."""
    start_time = time.time()
    
    # Criar topologia NSFNet
    graph = nx.Graph()
    nsfnet_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
        (4, 6), (4, 5), (5, 8), (5, 12), (6, 7), (7, 9), (8, 9), (9, 11),
        (9, 13), (10, 11), (10, 13), (11, 12)
    ]
    graph.add_edges_from(nsfnet_edges)
    
    # Pares O-D fixos (mesmos do AG do doutorado)
    source_target_pairs = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
    
    print(f"\n{'='*60}")
    print("AG ANTIGO - SIMULAÇÃO COM RESULTADOS SEPARADOS")
    print(f"{'='*60}")
    print(f"Topologia: NSFNet (14 nós, 21 links)")
    print(f"Wavelengths por link: 40")
    print(f"Requisições: {source_target_pairs}")
    print(f"Loads testados: 1 a 200")
    print(f"Simulações por load: 20")
    print(f"Chamadas por load: 1000")
    print(f"{'='*60}")
    
    # Criar simulador
    simulator = WDMSimulatorOldSeparated(
        graph=graph,
        num_wavelengths=40,
        source_target_pairs=source_target_pairs
    )
    
    # Adicionar links
    for edge in nsfnet_edges:
        simulator.add_link(edge[0], edge[1], capacity=40)
    
    # Executar simulação
    results = simulator.simulate_all_requisitions(
        load_values=list(range(1, 201)),
        num_simulations=20,
        calls_per_load=1000,
        routing_method='genetic',
        allocation_method='genetic',
        output_prefix="ag_antigo"
    )
    
    total_time = time.time() - start_time
    
    print(f"\nTempo total de execução: {total_time:.2f}s ({total_time/60:.2f} minutos)")
    print("Resultados salvos em arquivos 'ag_antigo_*.txt'")


if __name__ == "__main__":
    main_old()