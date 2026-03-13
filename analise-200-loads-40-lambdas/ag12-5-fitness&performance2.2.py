import heapq
import sys
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from collections import Counter
from itertools import islice
from scipy.stats import poisson
from typing import List, Tuple, Dict


class WDMSimulator:
    def __init__(self, graph: nx.Graph, num_wavelengths: int = 40):
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.traffic_matrix = np.zeros((len(graph.nodes), len(graph.nodes), self.num_wavelengths))
        self.allocated_routes = {}
        self.event_queue = []
        self.k = 150  # 25
        self.population_size = 150  # 150-200
        self.num_generations = 40  # 40-100
        self.crossover_rate = 0.8  # 0.8
        self.mutation_rate = 0.15  # 0.15
        self.k_shortest_paths = self.get_all_k_shortest_paths()
        self.route_cache = {}  # Cache for routes
        self.route_popularity = Counter()  # Contador de popularidade de rotas
        self.source = 4  # Define a origem das rotas para o exemplo
        self.target = 11  # Define o destino das rotas para o exemplo

        # Carrega rotas fixas entre origem e destino
        self.load_fixed_routes(self.source, self.target)
        
        #[(3, 8)]

    def add_link(self, node1: int, node2: int, capacity: int):
        self.graph.add_edge(node1, node2, capacity=capacity, wavelengths=np.ones(self.num_wavelengths, dtype=bool))

    def load_fixed_routes(self, source: int, target: int):
        """Carrega os k caminhos mais curtos entre a origem e o destino definidos."""
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
        """Verifica se a rota é válida com os comprimentos de onda alocados."""
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
            route, wavelengths = self.genetic_algorithm(src, dst)  # Passa os argumentos corretos

        self.route_cache[(src, dst, routing_method)] = route

        return route

    def load_fixed_routes(self, source: int, target: int):
        """Carrega os k caminhos mais curtos entre a origem e o destino definidos."""
        if (source, target) not in self.k_shortest_paths:
            # Armazena os caminhos mais curtos apenas uma vez
            self.k_shortest_paths[(source, target)] = self.get_k_shortest_paths(source, target, self.k)

    def initialize_population(self) -> List[Tuple[List[int], List[int]]]:
        """Inicializa uma população de rotas válidas com listas de comprimentos de onda vazias."""
        population = []
        base_routes = self.k_shortest_paths.get((self.source, self.target), [])

        # Garante que a população inicial inclui os k caminhos mais curtos fixos
        for base_route in base_routes:
            if len(population) < self.population_size:
                population.append((base_route, []))  # Inicializa com lista vazia de comprimentos de onda

        '''''
        # Se a população ainda não atingiu o tamanho desejado, repete ou faz pequenas variações
        while len(population) < self.population_size:
            route = random.choice(base_routes)
            if len(route) > 2 and random.random() < 0.5:  # Introduz variações simples
                split_point = random.randint(1, len(route) - 2)
                alternative_route = route[:split_point] + route[split_point:]
                if self.is_valid_route(alternative_route, self.source, self.target):
                    population.append((alternative_route, []))
            else:
                population.append((route, []))  # Adiciona rota com lista vazia
        '''''
        return population[:self.population_size]  # Limita ao tamanho desejado

    def fitness(self, solution: Tuple[List[int], List[int]]) -> float:
        route, _ = solution
        last_wavelength = None
        switch_penalty = 0
        route_length_penalty = len(route) - 1  # Penaliza com base no número de links
        total_availability = 0
        valid_links = 0

        for i in range(len(route) - 1):
            node1, node2 = route[i], route[i + 1]

            # Verifica se o link existe
            if not self.graph.has_edge(node1, node2):
                return 0  # Rota inválida

            # Calcula disponibilidade de comprimento de onda
            link_availability = sum(self.graph[node1][node2]['wavelengths']) / self.num_wavelengths
            total_availability += link_availability
            valid_links += 1

            # Verifica reutilização de comprimento de onda
            if last_wavelength is not None and self.graph[node1][node2]['wavelengths'][last_wavelength]:
                continue
            else:
                # Busca outro comprimento de onda disponível
                for wavelength in range(self.num_wavelengths):
                    if self.graph[node1][node2]['wavelengths'][wavelength]:
                        last_wavelength = wavelength
                        break
                else:
                    return 0  # Sem comprimento de onda disponível

                # Penaliza troca de comprimento de onda
                switch_penalty += 1

        # Calcula disponibilidade média
        mean_availability = total_availability / valid_links if valid_links > 0 else 0

        # Fitness: disponibilidade normalizada, penalizada por trocas e tamanho
        fitness_value = mean_availability / (1 + 0.2 * switch_penalty + 0.1 * route_length_penalty)
        return fitness_value

    def selection(self, population: List[Tuple[List[int], List[int]]], fitnesses: List[float]) -> List[
        Tuple[List[int], List[int]]]:
        """Realiza seleção baseada em torneio."""
        selected = []
        for _ in range(len(population)):
            idx1, idx2 = random.sample(range(len(population)), 2)
            if fitnesses[idx1] > fitnesses[idx2]:
                selected.append(population[idx1])
            else:
                selected.append(population[idx2])
        return selected

    def crossover(self, parent1: Tuple[List[int], List[int]], parent2: Tuple[List[int], List[int]]) -> Tuple[
        Tuple[List[int], List[int]], Tuple[List[int], List[int]]]:
        """Realiza cruzamento entre dois pais."""
        route1, wavelengths1 = parent1
        route2, wavelengths2 = parent2

        # Garante que os comprimentos de onda não sejam None
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
        """Aplica mutação alterando comprimentos de onda."""
        route, wavelengths = solution
        if wavelengths is None or len(wavelengths) == 0:
            # Garante que a lista de comprimentos de onda não está vazia
            wavelengths = [random.randint(0, self.num_wavelengths - 1) for _ in range(len(route) - 1)]

        new_wavelengths = wavelengths[:]

        if len(new_wavelengths) > 0 and random.random() < self.mutation_rate:
            segment_to_mutate = random.randint(0, len(new_wavelengths) - 1)
            new_wavelengths[segment_to_mutate] = random.randint(0, self.num_wavelengths - 1)

        return route, new_wavelengths

    def genetic_algorithm(self, source: int, target: int) -> Tuple[List[int], List[int]]:
        """Executa o algoritmo genético para encontrar a melhor rota e comprimentos de onda."""
        population = self.initialize_population()
        #print("Initial population: \n",population)
        #exit()

        for generation in range(self.num_generations):
            fitnesses = [self.fitness(ind) for ind in population]
            print(f"Generation: {generation} \n")
            print("fitness: \n", fitnesses)
            #exit()
            if not any(fitnesses):  # Se a população inteira for inválida
                population = self.initialize_population()
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


    def reintroduce_diversity(self, population: List[Tuple[List[int], int]]) -> List[Tuple[List[int], int]]:
        """Substitui uma fração da população por novos indivíduos para aumentar a diversidade."""
        num_individuals_to_replace = int(len(population) * 0.3)  # Reintroduz 30% da população
        for _ in range(num_individuals_to_replace):
            new_individual = self.initialize_population()[0]  # Novo indivíduo aleatório usando rotas fixas
            population[random.randint(0, len(population) - 1)] = new_individual
        return population


    def simulate_network(self, load_values: List[int], num_simulations: int, output_file: str,
                         routing_method: str = 'traditional', allocation_method: str = 'first_fit'):
        all_blocking_probabilities = {load: [] for load in load_values}

        for sim in range(num_simulations):
            print(f"\nSimulação {sim + 1}/{num_simulations}")
            
            for load in load_values:
                # RESETAR ESTADO DA REDE PARA CADA LOAD
                current_time = 0
                self.event_queue = []
                self.allocated_routes = {}
                # Resetar todos os comprimentos de onda para disponíveis
                for u, v in self.graph.edges():
                    self.graph[u][v]['wavelengths'] = np.ones(self.num_wavelengths, dtype=bool)
                self.route_cache = {}

                blocked_calls = 0  # Contador de chamadas bloqueadas
                total_calls = 1000  # Agora fixamos em 1000 chamadas por load

                print(f"  Processando load {load} com {total_calls} chamadas...")
                
                # GERAR 1000 CHAMADAS ALEATÓRIAS
                for call_id in range(total_calls):
                    # Selecionar origem e destino aleatórios
                    src = random.choice(list(self.graph.nodes))
                    dst = random.choice([node for node in self.graph.nodes if node != src])
                    
                    current_time += np.random.exponential(1 / load)
                    
                    # Liberar rotas que já expiraram
                    while self.event_queue and self.event_queue[0][0] <= current_time:
                        event_time, route, wavelength = heapq.heappop(self.event_queue)
                        self.release_route(route, wavelength)

                    # Obter rota e tentar alocar
                    route = self.get_route(src, dst, routing_method)
                    allocated_wavelength = self.test_and_allocate_route(route, allocation_method)
                    
                    if allocated_wavelength == -1:
                        blocked_calls += 1  # Incrementa em caso de bloqueio
                    else:
                        duration = np.random.exponential(1)
                        heapq.heappush(self.event_queue,
                                       (current_time + duration, route, allocated_wavelength))

                # Liberar todas as rotas restantes após processar todas as chamadas
                while self.event_queue:
                    event_time, route, wavelength = heapq.heappop(self.event_queue)
                    self.release_route(route, wavelength)

                # Cálculo da probabilidade de bloqueio para a carga atual
                blocking_probability = blocked_calls / total_calls if total_calls > 0 else 0
                all_blocking_probabilities[load].append(blocking_probability)

                # Registro no arquivo
                with open(output_file, 'a') as file:
                    file.write(
                        f"Simulacao {sim + 1} - Load: {load} - Chamadas: {total_calls} - "
                        f"Bloqueadas: {blocked_calls} - Probabilidade de Bloqueio: {blocking_probability:.4f}\n")
                
                print(f"    Load {load}: {blocked_calls}/{total_calls} chamadas bloqueadas "
                      f"({blocking_probability*100:.2f}%)")

        self.plot_results(load_values, all_blocking_probabilities, num_simulations)

    def plot_results(self, load_values: List[int], all_blocking_probabilities: Dict[int, List[float]],
                     num_simulations: int):
        plt.figure(figsize=(10, 8))
        for load in load_values:
            if len(all_blocking_probabilities[load]) == num_simulations:
                plt.plot([load] * num_simulations, all_blocking_probabilities[load], 'o', label=f'Load {load}')
            else:
                print(f"Warning: Load {load} has inconsistent data length.")

        mean_blocking_probabilities = [np.mean(all_blocking_probabilities[load]) for load in load_values]
        plt.plot(load_values, mean_blocking_probabilities, marker='o', linestyle='-', color='b', label='Média')
        plt.title('Probabilidade de Bloqueio vs Load (1000 chamadas por load)')
        plt.xlabel('Load')
        plt.ylabel('Probabilidade de Bloqueio (%)')
        plt.ylim(0, 1)
        plt.yticks(np.linspace(0, 1, 11), [f'{int(i * 100)}%' for i in np.linspace(0, 1, 11)])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('simulation-ag-antigo-req-lambidas.png')
        plt.show()


# Initial Setup
def create_nsfnet_topology() -> Tuple[nx.Graph, List[Tuple[int, int]]]:
    nsfnet_nodes = range(14)
    nsfnet_edges = [(0, 1), (0, 2), (0, 3),
                    (1, 2), (1, 7), (2, 5),
                    (3, 4), (3, 10), (4, 6),
                    (4, 5), (5, 8), (5, 12),
                    (6, 7), (7, 9), (8, 9),
                    (9, 11), (9, 13), (10, 11),
                    (10, 13), (11, 12)]
    nsfnet = nx.Graph()
    nsfnet.add_nodes_from(nsfnet_nodes)
    nsfnet.add_edges_from(nsfnet_edges)
    return nsfnet, nsfnet_edges


# INÍCIO DA MEDIÇÃO DO TEMPO
start_time = time.time()


# Simulation Execution
nsfnet, nsfnet_edges = create_nsfnet_topology()

simulator = WDMSimulator(nsfnet, num_wavelengths=40)

for edge in nsfnet_edges:
    simulator.add_link(edge[0], edge[1], capacity=40)

# Não precisamos mais inicializar a matriz de tráfego para a simulação
# mas mantemos para compatibilidade
simulator.initialize_traffic_matrix()

load_values = range(1, 201)
num_simulations = 1
routing_method = 'genetic'  # 'traditional', 'dijkstra', 'genetic'
allocation_method = 'genetic'  # 'first_fit', 'random_fit', 'genetic'

simulator.simulate_network(load_values, num_simulations, 'simulation-ag-antigo-req4-11.txt', routing_method,
                           allocation_method)

# FIM DA MEDIÇÃO DO TEMPO
end_time = time.time()
execution_time = end_time - start_time

# SALVAR TEMPO DE EXECUÇÃO EM ARQUIVO TXT
with open('tempo_execucao.txt', 'w') as time_file:
    time_file.write(f"Tempo total de execução: {execution_time:.2f} segundos\n")
    time_file.write(f"Tempo total de execução: {execution_time/60:.2f} minutos\n")
    time_file.write(f"Início: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
    time_file.write(f"Fim: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
    time_file.write(f"Configuração: {num_simulations} simulação(ões), 1000 chamadas por load\n")
    time_file.write(f"Loads testados: {list(load_values)}\n")

print(f"\nTempo de execução salvo em 'tempo_execucao-4-11.txt'")
print(f"Tempo total: {execution_time:.2f} segundos ({execution_time/60:.2f} minutos)")
print(f"Configuração: {num_simulations} simulação(ões), 1000 chamadas por load")