import heapq
import random
import os
import time
from itertools import islice
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class WDMSimulatorPhD_Completo:
    """
    AG do Doutorado COMPLETO com opções separado/conjunto.
    
    Versões disponíveis:
    1. SEPARADO: Cada requisição otimizada e simulada isoladamente
    2. CONJUNTO: 5 requisições otimizadas e simuladas juntas (competição)
    """
    
    def __init__(self,
                 graph: nx.Graph,
                 num_wavelengths: int = 40,
                 population_size: int = 120,
                 num_generations: int = 40,
                 k: int = 150,
                 mode: str = "conjunto"):  # "separado" ou "conjunto"
        """
        Inicializa o simulador.
        """
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.population_size = population_size
        self.num_generations = num_generations
        self.k = k
        self.mode = mode
        
        # Pares O-D fixos
        self.manual_pairs = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
        
        # Parâmetros para fitness
        self.penalty_per_hop = 0.2
        self.penalty_per_wavelength_change = 0.3
        self.reward_per_wavelength_reuse = 0.25
        self.penalty_per_congested_link = 0.5
        
        # Parâmetros de simulação
        self.simulation_time_units = 1000  # Unidades de tempo de simulação
        self.mean_call_duration = 10.0  # Duração média das chamadas (exponencial)
        
        # INICIALIZA OS ATRIBUTOS DO GRAFO
        self._initialize_graph_attributes()
        
        # Calcula todos os k-shortest paths
        print("Calculando k-shortest paths...")
        self.k_shortest_paths = self._get_all_k_shortest_paths()
        
        print(f"Simulador inicializado no modo: {mode.upper()}")
        print(f"  Requisições: {self.manual_pairs}")
        print(f"  Wavelengths: {num_wavelengths}")
        print(f"  k-paths: {k}")
        print(f"  População: {population_size}")
        print(f"  Gerações: {num_generations}")
        print(f"  Tempo simulação: {self.simulation_time_units} unidades")
        print(f"  Duração média chamadas: {self.mean_call_duration} unidades")
    
    def _initialize_graph_attributes(self):
        """Inicializa os atributos necessários no grafo."""
        for u, v in self.graph.edges():
            self.graph[u][v]['wavelengths'] = [True] * self.num_wavelengths
            self.graph[u][v]['usage_count'] = 0
            self.graph[u][v]['blocked_count'] = 0
            self.graph[u][v]['current_allocations'] = defaultdict(list)  # {wavelength: [call_ids]}
    
    def reset_network(self):
        """Reseta a rede para estado inicial."""
        for u, v in self.graph.edges():
            self.graph[u][v]['wavelengths'] = [True] * self.num_wavelengths
            self.graph[u][v]['usage_count'] = 0
            self.graph[u][v]['blocked_count'] = 0
            self.graph[u][v]['current_allocations'] = defaultdict(list)
    
    # ========== MÉTODOS DE ALOCAÇÃO REALISTA ==========
    
    def allocate_route_with_first_fit(self, route: List[int], call_id: int) -> Optional[int]:
        """
        Aloca uma rota usando algoritmo First-Fit.
        Retorna o wavelength alocado ou None se bloqueado.
        """
        if len(route) < 2:
            return None
        
        # Procura primeiro wavelength disponível em todos os enlaces
        for wl in range(self.num_wavelengths):
            available = True
            
            # Verifica disponibilidade em todos os enlaces
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                if not self.graph.has_edge(u, v) or not self.graph[u][v]['wavelengths'][wl]:
                    available = False
                    break
            
            # Se encontrou wavelength disponível, aloca
            if available:
                for i in range(len(route) - 1):
                    u, v = route[i], route[i + 1]
                    self.graph[u][v]['wavelengths'][wl] = False
                    self.graph[u][v]['usage_count'] += 1
                    self.graph[u][v]['current_allocations'][wl].append(call_id)
                return wl
        
        # Se não encontrou wavelength disponível
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            self.graph[u][v]['blocked_count'] += 1
        
        return None
    
    def release_route(self, route: List[int], wavelength: int, call_id: int):
        """Libera um wavelength alocado em uma rota para uma chamada específica."""
        if wavelength is None:
            return
            
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if self.graph.has_edge(u, v):
                # Remove o call_id da lista de alocações
                if call_id in self.graph[u][v]['current_allocations'][wavelength]:
                    self.graph[u][v]['current_allocations'][wavelength].remove(call_id)
                
                # Se não há mais chamadas usando este wavelength, libera-o
                if not self.graph[u][v]['current_allocations'][wavelength]:
                    self.graph[u][v]['wavelengths'][wavelength] = True
    
    def get_route_congestion(self, route: List[int]) -> float:
        """Retorna o nível de congestionamento médio da rota."""
        if len(route) < 2:
            return 1.0
        
        total_congestion = 0.0
        links = 0
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if self.graph.has_edge(u, v):
                used = self.num_wavelengths - sum(self.graph[u][v]['wavelengths'])
                congestion = used / self.num_wavelengths
                total_congestion += congestion
                links += 1
        
        return total_congestion / links if links > 0 else 1.0
    
    # ========== MÉTODOS DO ALGORITMO GENÉTICO ==========
    
    def _get_k_shortest_paths(self, source: int, target: int, k: int) -> List[List[int]]:
        """Calcula os k menores caminhos entre dois nós."""
        if not nx.has_path(self.graph, source, target):
            return []
        try:
            return list(islice(nx.shortest_simple_paths(self.graph, source, target), k))
        except nx.NetworkXNoPath:
            return []
    
    def _get_all_k_shortest_paths(self) -> Dict[Tuple[int, int], List[List[int]]]:
        """Calcula k-shortest paths para todos os pares O-D."""
        paths = {}
        for source, target in self.manual_pairs:
            paths[(source, target)] = self._get_k_shortest_paths(source, target, self.k)
        return paths
    
    def _fitness_route(self, route: List[int]) -> float:
        """
        Calcula a aptidão de uma rota específica.
        """
        if len(route) < 2:
            return 0.0
        
        # Penalidade por número de hops (saltos)
        hops = len(route) - 1
        hops_penalty = hops * self.penalty_per_hop
        
        # Penalidade por congestionamento
        congestion = self.get_route_congestion(route)
        congestion_penalty = congestion * self.penalty_per_congested_link
        
        # Fitness base (quanto maior, melhor)
        fitness = 1.0
        
        # Aplica penalidades
        fitness -= hops_penalty
        fitness -= congestion_penalty
        
        # Garante que fitness não seja negativo
        return max(0.01, fitness)
    
    def _initialize_population_single(self, source: int, target: int) -> List[Tuple[int, List[int]]]:
        """Inicializa população para uma ÚNICA requisição."""
        population = []
        routes = self.k_shortest_paths.get((source, target), [])
        
        if not routes:
            return []
        
        for _ in range(self.population_size):
            # Escolhe rota aleatória
            route_idx = random.randint(0, len(routes) - 1)
            route = routes[route_idx]
            
            population.append((route_idx, route))
        
        return population
    
    def _initialize_population_conjunto(self) -> List[List[Tuple[int, List[int]]]]:
        """Inicializa população para TODAS as requisições (conjunto)."""
        population = []
        
        for _ in range(self.population_size):
            individual = []
            
            for source, target in self.manual_pairs:
                routes = self.k_shortest_paths.get((source, target), [])
                
                if routes:
                    route_idx = random.randint(0, len(routes) - 1)
                    route = routes[route_idx]
                else:
                    route_idx = 0
                    route = []
                
                individual.append((route_idx, route))
            
            population.append(individual)
        
        return population
    
    def _evaluate_individual_single(self, individual: Tuple[int, List[int]]) -> float:
        """Avalia um indivíduo no modo SEPARADO."""
        route_idx, route = individual
        
        if len(route) < 2:
            return 0.0
        
        # Usa a função de fitness da rota
        fitness = self._fitness_route(route)
        
        # Bônus adicional se a rota for mais curta
        hops = len(route) - 1
        if hops <= 3:  # Rotas muito curtas recebem bônus
            fitness *= 1.5
        
        return fitness
    
    def _evaluate_individual_conjunto(self, individual: List[Tuple[int, List[int]]]) -> float:
        """Avalia um indivíduo no modo CONJUNTO."""
        if len(individual) != len(self.manual_pairs):
            return 0.0
        
        total_fitness = 0.0
        valid_routes = 0
        
        for route_idx, route in individual:
            if len(route) >= 2:
                fitness = self._fitness_route(route)
                total_fitness += fitness
                valid_routes += 1
        
        # Penalidade por sobreposição de rotas (conflito de recursos)
        conflict_penalty = self._calculate_conflict_penalty(individual)
        total_fitness -= conflict_penalty
        
        # Retorna média das fitness válidas
        return total_fitness / valid_routes if valid_routes > 0 else 0.0
    
    def _calculate_conflict_penalty(self, individual: List[Tuple[int, List[int]]]) -> float:
        """Calcula penalidade por conflitos entre rotas."""
        link_usage = defaultdict(int)
        
        # Conta uso de enlaces por todas as rotas
        for _, route in individual:
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                link_usage[(u, v)] += 1
        
        # Penaliza enlaces muito utilizados
        penalty = 0.0
        for count in link_usage.values():
            if count > 1:  # Se mais de uma rota usa o mesmo enlace
                penalty += (count - 1) * 0.1
        
        return penalty
    
    def _selection_tournament(self, population: List, fitness_scores: List[float], 
                             tournament_size: int = 3) -> Any:
        """Seleção por torneio com 3 indivíduos."""
        # Escolhe aleatoriamente tournament_size indivíduos
        tournament_indices = random.sample(range(len(population)), 
                                          min(tournament_size, len(population)))
        
        # Encontra o melhor entre eles
        best_idx = tournament_indices[0]
        best_fitness = fitness_scores[best_idx]
        
        for idx in tournament_indices[1:]:
            if fitness_scores[idx] > best_fitness:
                best_idx = idx
                best_fitness = fitness_scores[idx]
        
        return population[best_idx]
    
    def _crossover_single(self, parent1: Tuple[int, List[int]], parent2: Tuple[int, List[int]], 
                         source: int, target: int) -> Tuple[Tuple[int, List[int]], Tuple[int, List[int]]]:
        """Crossover para uma única requisição."""
        routes = self.k_shortest_paths.get((source, target), [])
        
        # 70% chance de crossover
        if random.random() < 0.7 and routes:
            # Crossover uniforme: mistura índices de rota
            child1_idx = (parent1[0] + parent2[0]) // 2
            child2_idx = (parent1[0] + parent2[0] + 1) // 2
            
            # Garante que os índices estão dentro dos limites
            child1_idx = max(0, min(child1_idx, len(routes) - 1))
            child2_idx = max(0, min(child2_idx, len(routes) - 1))
            
            child1 = (child1_idx, routes[child1_idx])
            child2 = (child2_idx, routes[child2_idx])
        else:
            child1, child2 = parent1, parent2
        
        return child1, child2
    
    def _crossover_conjunto(self, parent1: List[Tuple[int, List[int]]], 
                           parent2: List[Tuple[int, List[int]]]) -> Tuple[List[Tuple[int, List[int]]], 
                                                                         List[Tuple[int, List[int]]]]:
        """Crossover para múltiplas requisições."""
        child1 = []
        child2 = []
        
        # Crossover uniforme por requisição
        for idx in range(len(parent1)):
            if random.random() < 0.5:
                child1.append(parent1[idx])
                child2.append(parent2[idx])
            else:
                child1.append(parent2[idx])
                child2.append(parent1[idx])
        
        return child1, child2
    
    def _mutate_single(self, individual: Tuple[int, List[int]], 
                      source: int, target: int) -> Tuple[int, List[int]]:
        """Mutação para uma única requisição."""
        route_idx, route = individual
        routes = self.k_shortest_paths.get((source, target), [])
        
        if not routes:
            return individual
        
        # 20% chance de mutação
        if random.random() < 0.2:
            # Mutação: troca para rota vizinha
            new_idx = route_idx + random.choice([-2, -1, 1, 2])
            new_idx = max(0, min(new_idx, len(routes) - 1))
            
            return (new_idx, routes[new_idx])
        
        return individual
    
    def _mutate_conjunto(self, individual: List[Tuple[int, List[int]]]) -> List[Tuple[int, List[int]]]:
        """Mutação para múltiplas requisições."""
        mutated = []
        
        for idx, (route_idx, route) in enumerate(individual):
            source, target = self.manual_pairs[idx]
            if random.random() < 0.1:  # 10% chance por requisição
                mutated.append(self._mutate_single((route_idx, route), source, target))
            else:
                mutated.append((route_idx, route))
        
        return mutated
    
    def genetic_algorithm_single(self, source: int, target: int) -> Optional[List[int]]:
        """Executa AG para uma ÚNICA requisição."""
        print(f"  Executando AG para [{source},{target}]...")
        start_time = time.time()
        
        # Inicializa população
        population = self._initialize_population_single(source, target)
        if not population:
            return None
        
        # Executa AG
        for gen in range(self.num_generations):
            # Avalia fitness
            fitness_scores = [self._evaluate_individual_single(ind) for ind in population]
            
            # Nova população
            new_population = []
            
            # Mantém elite (10% melhores)
            elite_size = max(1, len(population) // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # Gera resto da população com seleção por torneio
            while len(new_population) < self.population_size:
                # Seleciona pais por torneio
                parent1 = self._selection_tournament(population, fitness_scores)
                parent2 = self._selection_tournament(population, fitness_scores)
                
                # Crossover
                child1, child2 = self._crossover_single(parent1, parent2, source, target)
                
                # Mutação
                child1 = self._mutate_single(child1, source, target)
                child2 = self._mutate_single(child2, source, target)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            # Progresso
            if (gen + 1) % 10 == 0:
                best_fitness = max(fitness_scores)
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                print(f"    Geração {gen+1}/{self.num_generations}, Melhor: {best_fitness:.3f}, Média: {avg_fitness:.3f}")
        
        # Escolhe o melhor
        fitness_scores = [self._evaluate_individual_single(ind) for ind in population]
        best_idx = np.argmax(fitness_scores)
        best_individual = population[best_idx]
        best_route = best_individual[1]
        
        ga_time = time.time() - start_time
        print(f"    ✓ Concluído em {ga_time:.2f}s: {len(best_route)-1} hops")
        
        return best_route
    
    def genetic_algorithm_conjunto(self) -> Dict[Tuple[int, int], List[int]]:
        """Executa AG para TODAS as requisições (conjunto)."""
        print(f"  Executando AG para {len(self.manual_pairs)} requisições conjuntas...")
        start_time = time.time()
        
        # Inicializa população
        population = self._initialize_population_conjunto()
        
        # Executa AG
        for gen in range(self.num_generations):
            # Avalia fitness
            fitness_scores = [self._evaluate_individual_conjunto(ind) for ind in population]
            
            # Nova população
            new_population = []
            
            # Mantém elite (10% melhores)
            elite_size = max(1, len(population) // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # Gera resto da população com seleção por torneio
            while len(new_population) < self.population_size:
                # Seleciona pais por torneio
                parent1 = self._selection_tournament(population, fitness_scores)
                parent2 = self._selection_tournament(population, fitness_scores)
                
                # Crossover
                child1, child2 = self._crossover_conjunto(parent1, parent2)
                
                # Mutação
                child1 = self._mutate_conjunto(child1)
                child2 = self._mutate_conjunto(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            # Progresso
            if (gen + 1) % 10 == 0:
                best_fitness = max(fitness_scores)
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                print(f"    Geração {gen+1}/{self.num_generations}, Melhor: {best_fitness:.3f}, Média: {avg_fitness:.3f}")
        
        # Escolhe o melhor
        fitness_scores = [self._evaluate_individual_conjunto(ind) for ind in population]
        best_idx = np.argmax(fitness_scores)
        best_individual = population[best_idx]
        
        # Converte para dicionário de soluções
        best_solutions = {}
        for idx, (source, target) in enumerate(self.manual_pairs):
            if idx < len(best_individual):
                route = best_individual[idx][1]
                best_solutions[(source, target)] = route
        
        ga_time = time.time() - start_time
        print(f"    ✓ AG conjunto concluído em {ga_time:.2f}s")
        
        # Mostra resultados
        print(f"    Rotas encontradas:")
        for (source, target), route in best_solutions.items():
            hops = len(route) - 1
            print(f"      [{source},{target}]: {hops} hops")
        
        return best_solutions
    
    # ========== MÉTODOS DE SIMULAÇÃO COM POISSON ==========
    
    def _generate_poisson_arrivals(self, arrival_rate: float, total_time: int) -> List[float]:
        """
        Gera tempos de chegada de Poisson.
        
        Args:
            arrival_rate: Taxa de chegada (chamadas por unidade de tempo)
            total_time: Tempo total de simulação
            
        Returns:
            Lista de tempos de chegada ordenados
        """
        arrivals = []
        current_time = 0.0
        
        while current_time < total_time:
            # Tempo entre chegadas é exponencial com média 1/arrival_rate
            inter_arrival = random.expovariate(arrival_rate)
            current_time += inter_arrival
            
            if current_time < total_time:
                arrivals.append(current_time)
        
        return arrivals
    
    def simulate_single_requisition(self, source: int, target: int,
                                    best_route: List[int],
                                    load_values: List[int],
                                    num_simulations: int = 20) -> Dict[int, List[float]]:
        """
        Simula uma ÚNICA requisição de forma isolada.
        
        Simulação realista com chegadas Poisson e durações exponenciais.
        """
        if not best_route or len(best_route) < 2:
            return {load: [1.0] * num_simulations for load in load_values}
        
        results = {load: [] for load in load_values}
        
        print(f"    Simulando requisição [{source},{target}]...")
        start_sim_time = time.time()
        
        for sim in range(num_simulations):
            print(f"      Simulação {sim+1}/{num_simulations}", end='\r')
            
            for load_idx, load in enumerate(load_values):
                # Reset da rede
                self.reset_network()
                
                # Taxa de chegada: load representa intensidade de tráfego (Erlangs)
                arrival_rate = load / self.mean_call_duration
                
                # Gera chegadas Poisson
                arrivals = self._generate_poisson_arrivals(arrival_rate, self.simulation_time_units)
                num_arrivals = len(arrivals)
                
                if num_arrivals == 0:
                    results[load].append(0.0)  # Nenhuma chegada = 0 bloqueio
                    continue
                
                # Lista de eventos: (tempo, tipo, call_id, [dados])
                # tipos: 'arrival' ou 'departure'
                events = []
                
                # Adiciona eventos de chegada
                call_id_counter = 0
                for arrival_time in arrivals:
                    events.append((arrival_time, 'arrival', call_id_counter, {
                        'route': best_route,
                        'source': source,
                        'target': target
                    }))
                    call_id_counter += 1
                
                # Ordena eventos por tempo
                events.sort(key=lambda x: x[0])
                
                # Dicionário para rastrear chamadas ativas
                active_calls = {}  # call_id: {'wavelength': wl, 'route': route, 'departure_time': time}
                
                # Processa eventos
                blocked_calls = 0
                total_arrivals = 0
                
                while events:
                    current_time, event_type, call_id, event_data = heapq.heappop(events)
                    
                    if event_type == 'arrival':
                        total_arrivals += 1
                        
                        # Tenta alocar rota
                        wavelength = self.allocate_route_with_first_fit(event_data['route'], call_id)
                        
                        if wavelength is not None:
                            # Gera duração exponencial
                            duration = random.expovariate(1.0 / self.mean_call_duration)
                            departure_time = current_time + duration
                            
                            # Armazena informações da chamada
                            active_calls[call_id] = {
                                'wavelength': wavelength,
                                'route': event_data['route'],
                                'departure_time': departure_time
                            }
                            
                            # Agenda evento de saída
                            heapq.heappush(events, (departure_time, 'departure', call_id, {}))
                        else:
                            # Chamada bloqueada
                            blocked_calls += 1
                    
                    elif event_type == 'departure':
                        if call_id in active_calls:
                            call_info = active_calls[call_id]
                            self.release_route(call_info['route'], call_info['wavelength'], call_id)
                            del active_calls[call_id]
                
                # Calcula probabilidade de bloqueio
                if total_arrivals > 0:
                    blocking_prob = blocked_calls / total_arrivals
                else:
                    blocking_prob = 0.0
                    
                results[load].append(blocking_prob)
        
        sim_time = time.time() - start_sim_time
        print(f"    ✓ Simulação concluída em {sim_time:.2f}s")
        
        return results
    
    def simulate_conjunto(self, best_solutions: Dict[Tuple[int, int], List[int]],
                          load_values: List[int],
                          num_simulations: int = 20) -> Dict[int, Dict[int, List[float]]]:
        """
        Simula TODAS as requisições conjuntamente.
        
        Simulação realista com múltiplas requisições e chegadas Poisson.
        """
        results = {idx: {load: [] for load in load_values} 
                  for idx in range(len(self.manual_pairs))}
        
        print(f"    Simulando {len(self.manual_pairs)} requisições conjuntamente...")
        start_sim_time = time.time()
        
        for sim in range(num_simulations):
            print(f"      Simulação {sim+1}/{num_simulations}", end='\r')
            
            for load_idx, load in enumerate(load_values):
                # Reset da rede
                self.reset_network()
                
                # Lista de eventos
                events = []
                call_id_counter = 0
                
                # Para cada requisição, gera chegadas Poisson
                for req_idx, (source, target) in enumerate(self.manual_pairs):
                    if (source, target) not in best_solutions:
                        continue
                    
                    route = best_solutions[(source, target)]
                    
                    # Taxa de chegada para esta requisição (distribuída igualmente)
                    arrival_rate = (load / self.mean_call_duration) / len(self.manual_pairs)
                    
                    # Gera chegadas Poisson
                    arrivals = self._generate_poisson_arrivals(arrival_rate, self.simulation_time_units)
                    
                    # Adiciona eventos de chegada
                    for arrival_time in arrivals:
                        events.append((arrival_time, 'arrival', call_id_counter, {
                            'req_idx': req_idx,
                            'route': route,
                            'source': source,
                            'target': target
                        }))
                        call_id_counter += 1
                
                # Ordena eventos por tempo
                events.sort(key=lambda x: x[0])
                events_heap = [(time, event_type, call_id, data) for time, event_type, call_id, data in events]
                heapq.heapify(events_heap)
                
                # Dicionário para rastrear chamadas ativas
                active_calls = {}
                
                # Contadores por requisição
                arrivals_by_req = [0] * len(self.manual_pairs)
                blocked_by_req = [0] * len(self.manual_pairs)
                
                # Processa eventos
                while events_heap:
                    current_time, event_type, call_id, event_data = heapq.heappop(events_heap)
                    
                    if event_type == 'arrival':
                        req_idx = event_data['req_idx']
                        arrivals_by_req[req_idx] += 1
                        
                        # Tenta alocar rota
                        wavelength = self.allocate_route_with_first_fit(event_data['route'], call_id)
                        
                        if wavelength is not None:
                            # Gera duração exponencial
                            duration = random.expovariate(1.0 / self.mean_call_duration)
                            departure_time = current_time + duration
                            
                            # Armazena informações da chamada
                            active_calls[call_id] = {
                                'wavelength': wavelength,
                                'route': event_data['route'],
                                'req_idx': req_idx,
                                'departure_time': departure_time
                            }
                            
                            # Agenda evento de saída
                            heapq.heappush(events_heap, (departure_time, 'departure', call_id, {}))
                        else:
                            # Chamada bloqueada
                            blocked_by_req[req_idx] += 1
                    
                    elif event_type == 'departure':
                        if call_id in active_calls:
                            call_info = active_calls[call_id]
                            self.release_route(call_info['route'], call_info['wavelength'], call_id)
                            del active_calls[call_id]
                
                # Calcula probabilidades de bloqueio por requisição
                for req_idx in range(len(self.manual_pairs)):
                    if arrivals_by_req[req_idx] > 0:
                        blocking_prob = blocked_by_req[req_idx] / arrivals_by_req[req_idx]
                    else:
                        blocking_prob = 0.0
                        
                    results[req_idx][load].append(blocking_prob)
        
        sim_time = time.time() - start_sim_time
        print(f"    ✓ Simulação conjunta concluída em {sim_time:.2f}s")
        
        return results
    
    # ========== MÉTODOS PRINCIPAIS ==========
    
    def run_simulation(self, load_values: List[int] = None,
                       num_simulations: int = 20,
                       output_dir: str = "resultados_phd") -> Dict:
        """
        Executa simulação completa no modo configurado.
        """
        if load_values is None:
            load_values = list(range(1, 201))  # Loads 1-200 (sem pular)
        
        print(f"\n{'='*70}")
        print(f"SIMULAÇÃO AG DOUTORADO - MODO: {self.mode.upper()}")
        print(f"{'='*70}")
        print(f"Parâmetros REALISTAS COM POISSON:")
        print(f"  • Requisições: {len(self.manual_pairs)}")
        print(f"  • Loads: {len(load_values)} (de {min(load_values)} a {max(load_values)}, SEM PULAR)")
        print(f"  • Simulações por load: {num_simulations}")
        print(f"  • Unidades de tempo por simulação: {self.simulation_time_units}")
        print(f"  • Duração média das chamadas: {self.mean_call_duration}")
        print(f"  • Wavelengths: {self.num_wavelengths}")
        print(f"  • População AG: {self.population_size}")
        print(f"  • Gerações AG: {self.num_generations}")
        print(f"  • k-paths: {self.k}")
        print(f"{'='*70}")
        
        start_total = time.time()
        
        if self.mode == "separado":
            results = self._run_mode_separado(load_values, num_simulations)
        else:  # conjunto
            results = self._run_mode_conjunto(load_values, num_simulations)
        
        total_time = time.time() - start_total
        
        # Salva resultados detalhados
        self._save_detailed_results(results, load_values, num_simulations, 
                                   total_time, output_dir)
        
        # Gera gráficos
        self._plot_results(results, load_values, output_dir)
        
        print(f"\n{'='*70}")
        print(f"SIMULAÇÃO CONCLUÍDA!")
        print(f"Tempo total de execução: {total_time:.2f}s ({total_time/60:.2f} minutos)")
        print(f"Resultados salvos em: {output_dir}/")
        print(f"{'='*70}")
        
        return results
    
    def _run_mode_separado(self, load_values: List[int],
                          num_simulations: int) -> Dict:
        """Executa no modo SEPARADO."""
        print(f"\nMODO SEPARADO")
        
        all_results = {}
        
        for idx, (source, target) in enumerate(self.manual_pairs):
            print(f"\n[{idx+1}/{len(self.manual_pairs)}] Requisição: [{source},{target}]")
            
            # Executa AG para esta requisição
            best_route = self.genetic_algorithm_single(source, target)
            
            if not best_route:
                print(f"  ⚠ Usando caminho mais curto como fallback")
                try:
                    best_route = nx.shortest_path(self.graph, source, target)
                except:
                    print(f"  ⚠ ERRO: Não foi possível encontrar rota")
                    all_results[idx] = {load: [1.0] * num_simulations for load in load_values}
                    continue
            
            # Simula esta requisição
            results = self.simulate_single_requisition(
                source, target, best_route, 
                load_values, num_simulations
            )
            
            all_results[idx] = results
            
            # Calcula e mostra estatísticas
            self._print_simulation_stats(idx, results, load_values)
        
        return all_results
    
    def _run_mode_conjunto(self, load_values: List[int],
                          num_simulations: int) -> Dict:
        """Executa no modo CONJUNTO."""
        print(f"\nMODO CONJUNTO")
        
        # Executa AG conjunto
        best_solutions = self.genetic_algorithm_conjunto()
        
        if not best_solutions:
            print(f"  ⚠ Usando caminhos mais curtos como fallback")
            best_solutions = {}
            for source, target in self.manual_pairs:
                try:
                    best_solutions[(source, target)] = nx.shortest_path(self.graph, source, target)
                except:
                    print(f"  ⚠ Não encontrou rota para [{source},{target}]")
        
        # Simula todas as requisições conjuntamente
        all_results = self.simulate_conjunto(
            best_solutions, load_values, num_simulations
        )
        
        # Calcula e mostra estatísticas para cada requisição
        print(f"\n  📊 ESTATÍSTICAS POR REQUISIÇÃO:")
        for idx in range(len(self.manual_pairs)):
            source, target = self.manual_pairs[idx]
            print(f"\n    Requisição [{source},{target}]:")
            self._print_simulation_stats(idx, all_results[idx], load_values)
        
        return all_results
    
    def _print_simulation_stats(self, idx: int, results: Dict[int, List[float]], 
                               load_values: List[int]):
        """Imprime estatísticas de simulação para uma requisição."""
        if not results:
            return
        
        # Calcula estatísticas para alguns loads representativos
        sample_loads = [1, 50, 100, 150, 200]
        
        for load in sample_loads:
            if load in load_values and load in results and results[load]:
                probs = results[load]
                mean_prob = np.mean(probs)
                std_prob = np.std(probs) if len(probs) > 1 else 0.0
                min_prob = min(probs)
                max_prob = max(probs)
                
                print(f"      Load {load:3d}: {mean_prob:.6f} ± {std_prob:.6f} "
                      f"(min: {min_prob:.6f}, max: {max_prob:.6f})")
    
    def _save_detailed_results(self, results: Dict, load_values: List[int],
                              num_simulations: int, total_time: float, 
                              output_dir: str):
        """Salva resultados detalhados em arquivos."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Cria subdiretórios para organizar melhor
        subdirs = {
            'probabilidades_completas': f"{output_dir}/probabilidades_completas",
            'medias_estatisticas': f"{output_dir}/medias_estatisticas",
            'resumos': f"{output_dir}/resumos"
        }
        
        for subdir_name, subdir_path in subdirs.items():
            os.makedirs(subdir_path, exist_ok=True)
        
        # 1. Salva TODAS as probabilidades por requisição (20 simulações cada)
        for idx in range(len(self.manual_pairs)):
            source, target = self.manual_pairs[idx]
            
            # Arquivo com todas as probabilidades
            filename_all = f"{subdirs['probabilidades_completas']}/todas_probs_req_{idx+1}_{source}_{target}_{self.mode}.txt"
            
            with open(filename_all, 'w') as f:
                f.write(f"# TODAS AS PROBABILIDADES (20 SIMULAÇÕES)\n")
                f.write(f"# Requisição {idx+1}: [{source},{target}]\n")
                f.write(f"# Modo: {self.mode.upper()}\n")
                f.write(f"# Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# " + "="*80 + "\n")
                f.write("# Formato: Load | Sim1 | Sim2 | ... | Sim20 | Média | DesvioPadrão\n")
                f.write("# " + "="*80 + "\n\n")
                
                for load in load_values:
                    if idx in results and load in results[idx] and results[idx][load]:
                        probs = results[idx][load]
                        if len(probs) == num_simulations:
                            mean_prob = np.mean(probs)
                            std_prob = np.std(probs) if len(probs) > 1 else 0.0
                            
                            # Escreve todas as probabilidades individuais
                            prob_str = " | ".join([f"{p:.8f}" for p in probs])
                            f.write(f"Load {load:3d} | {prob_str} | {mean_prob:.8f} | {std_prob:.8f}\n")
        
        # 2. Salva MÉDIAS e estatísticas por requisição
        for idx in range(len(self.manual_pairs)):
            source, target = self.manual_pairs[idx]
            
            filename_medias = f"{subdirs['medias_estatisticas']}/medias_req_{idx+1}_{source}_{target}_{self.mode}.txt"
            
            with open(filename_medias, 'w') as f:
                f.write(f"# MÉDIAS E ESTATÍSTICAS\n")
                f.write(f"# Requisição {idx+1}: [{source},{target}]\n")
                f.write(f"# Modo: {self.mode.upper()}\n")
                f.write("# " + "="*60 + "\n")
                f.write("# Load | Média | DesvioPadrão | Mínimo | Máximo | IC 95% (inferior) | IC 95% (superior)\n")
                f.write("# " + "="*60 + "\n")
                
                for load in load_values:
                    if idx in results and load in results[idx] and results[idx][load]:
                        probs = results[idx][load]
                        mean_prob = np.mean(probs)
                        std_prob = np.std(probs) if len(probs) > 1 else 0.0
                        min_prob = min(probs)
                        max_prob = max(probs)
                        
                        # Intervalo de confiança 95%
                        n = len(probs)
                        if n > 1 and std_prob > 0:
                            t_value = 2.086  # Para n=20, graus de liberdade=19
                            se = std_prob / np.sqrt(n)
                            ci_lower = mean_prob - t_value * se
                            ci_upper = mean_prob + t_value * se
                        else:
                            ci_lower = ci_upper = mean_prob
                        
                        f.write(f"{load:4d} | {mean_prob:.8f} | {std_prob:.8f} | "
                               f"{min_prob:.8f} | {max_prob:.8f} | "
                               f"{ci_lower:.8f} | {ci_upper:.8f}\n")
        
        # 3. Salva arquivo de RESUMO GERAL para este modo
        summary_file = f"{subdirs['resumos']}/resumo_geral_{self.mode}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"RESUMO GERAL DA SIMULAÇÃO - MODO {self.mode.upper()}\n")
            f.write("="*80 + "\n")
            f.write(f"Data e hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tempo total de execução: {total_time:.2f}s ({total_time/60:.2f} minutos)\n")
            f.write(f"Número de requisições: {len(self.manual_pairs)}\n")
            f.write(f"Loads simulados: {len(load_values)} (de {min(load_values)} a {max(load_values)})\n")
            f.write(f"Simulações por load: {num_simulations}\n")
            f.write(f"Unidades de tempo por simulação: {self.simulation_time_units}\n")
            f.write(f"Duração média das chamadas: {self.mean_call_duration}\n")
            f.write(f"Total de simulações realizadas: {len(load_values) * num_simulations:,}\n")
            f.write(f"Parâmetros AG:\n")
            f.write(f"  • População: {self.population_size}\n")
            f.write(f"  • Gerações: {self.num_generations}\n")
            f.write(f"  • k-paths: {self.k}\n")
            f.write(f"  • Wavelengths: {self.num_wavelengths}\n")
            f.write("\n" + "="*80 + "\n\n")
            
            # Estatísticas por requisição
            for idx in range(len(self.manual_pairs)):
                source, target = self.manual_pairs[idx]
                f.write(f"REQUISIÇÃO {idx+1}: [{source},{target}]\n")
                f.write("-"*40 + "\n")
                
                if idx in results:
                    # Calcula estatísticas gerais
                    all_probs = []
                    for load in load_values:
                        if load in results[idx] and results[idx][load]:
                            all_probs.extend(results[idx][load])
                    
                    if all_probs:
                        overall_mean = np.mean(all_probs)
                        overall_std = np.std(all_probs) if len(all_probs) > 1 else 0.0
                        overall_min = min(all_probs)
                        overall_max = max(all_probs)
                        
                        f.write(f"Probabilidade média geral: {overall_mean:.8f}\n")
                        f.write(f"Desvio padrão geral: {overall_std:.8f}\n")
                        f.write(f"Mínimo geral: {overall_min:.8f}\n")
                        f.write(f"Máximo geral: {overall_max:.8f}\n")
                        
                        # Probabilidade para loads específicos
                        for load in [1, 50, 100, 150, 200]:
                            if load in load_values and load in results[idx] and results[idx][load]:
                                probs_load = results[idx][load]
                                mean_load = np.mean(probs_load)
                                f.write(f"Load {load:3d}: {mean_load:.8f}\n")
                    
                    f.write("\n")
            
            # Tempo de execução detalhado
            f.write("\n" + "="*80 + "\n")
            f.write("TEMPO DE EXECUÇÃO DETALHADO\n")
            f.write("-"*40 + "\n")
            f.write(f"Tempo total: {total_time:.2f} segundos\n")
            f.write(f"Tempo total: {total_time/60:.2f} minutos\n")
            f.write(f"Tempo total: {total_time/3600:.2f} horas\n")
            f.write(f"Simulações realizadas: {len(load_values) * num_simulations}\n")
            f.write(f"Unidades de tempo simuladas: {len(load_values) * num_simulations * self.simulation_time_units:,}\n")
            f.write(f"Taxa de simulação: {total_time/(len(load_values) * num_simulations):.2f} segundos/simulação\n")
        
        # 4. Salva arquivo de TEMPO DE EXECUÇÃO separado
        time_file = f"{output_dir}/tempo_execucao_{self.mode}.txt"
        with open(time_file, 'w') as f:
            f.write(f"TEMPO DE EXECUÇÃO - MODO {self.mode.upper()}\n")
            f.write("="*60 + "\n")
            f.write(f"Data e hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tempo total: {total_time:.2f} segundos\n")
            f.write(f"Tempo total: {total_time/60:.2f} minutos\n")
            f.write(f"Tempo total: {total_time/3600:.2f} horas\n")
            f.write(f"Loads simulados: {len(load_values)}\n")
            f.write(f"Simulações por load: {num_simulations}\n")
            f.write(f"Total simulações: {len(load_values) * num_simulations}\n")
            f.write(f"Unidades tempo/simulação: {self.simulation_time_units}\n")
            f.write(f"Duração média chamadas: {self.mean_call_duration}\n")
            f.write(f"Wavelengths: {self.num_wavelengths}\n")
        
        print(f"  📁 Estrutura de resultados criada em {output_dir}/:")
        print(f"     • probabilidades_completas/ - Todas as probabilidades individuais (20 simulações)")
        print(f"     • medias_estatisticas/ - Médias e estatísticas por load")
        print(f"     • resumos/ - Resumos gerais da simulação")
        print(f"     • tempo_execucao_{self.mode}.txt - Tempo de execução")
        print(f"     • grafico_detalhado_{self.mode}.png - Gráfico das probabilidades")
    
    def _plot_results(self, results: Dict, load_values: List[int], 
                     output_dir: str):
        """Gera gráficos dos resultados."""
        plt.figure(figsize=(14, 10))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        markers = ['o', 's', '^', 'D', 'v']
        
        for idx in range(len(self.manual_pairs)):
            source, target = self.manual_pairs[idx]
            
            means = []
            stds = []
            valid_loads = []
            
            for load in load_values:
                if idx in results and load in results[idx] and results[idx][load]:
                    probs = results[idx][load]
                    if probs:
                        means.append(np.mean(probs) * 100)  # Em %
                        stds.append(np.std(probs) * 100 if len(probs) > 1 else 0.0)
                        valid_loads.append(load)
            
            if valid_loads and means:
                plt.plot(valid_loads, means,
                        color=colors[idx % len(colors)],
                        marker=markers[idx % len(markers)],
                        markersize=4,
                        linewidth=2,
                        label=f'[{source},{target}]',
                        alpha=0.8)
                
                # Adiciona banda de desvio padrão
                plt.fill_between(valid_loads, 
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                color=colors[idx % len(colors)],
                                alpha=0.2)
        
        plt.xlabel('Carga (Load) [Erlangs]', fontsize=14)
        plt.ylabel('Probabilidade de Bloqueio (%)', fontsize=14)
        plt.title(f'Simulação com Poisson - Modo {self.mode.upper()}\n'
                 f'AG: População={self.population_size}, Gerações={self.num_generations}, k={self.k}',
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim(-1, 101)
        plt.xlim(0, max(load_values) + 10)
        
        # Adiciona grade secundária
        plt.minorticks_on()
        plt.grid(which='minor', alpha=0.1, linestyle=':')
        
        plt.tight_layout()
        
        # Salva gráfico em alta resolução
        plot_file = f"{output_dir}/grafico_detalhado_{self.mode}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        # Salva também em formato PDF
        plot_pdf = f"{output_dir}/grafico_detalhado_{self.mode}.pdf"
        plt.savefig(plot_pdf, bbox_inches='tight')
        
        plt.show()
        
        print(f"  📊 Gráficos salvos em:")
        print(f"     • {plot_file}")
        print(f"     • {plot_pdf}")


# ========== FUNÇÕES AUXILIARES ==========

def create_nsfnet_graph() -> nx.Graph:
    """Cria e retorna grafo NSFNet."""
    graph = nx.Graph()
    nsfnet_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
        (4, 6), (4, 5), (5, 8), (5, 12), (6, 7), (7, 9), (8, 9), (9, 11),
        (9, 13), (10, 11), (10, 13), (11, 12)
    ]
    graph.add_edges_from(nsfnet_edges)
    return graph


def main():
    """Função principal."""
    print(f"\n{'='*80}")
    print("SIMULAÇÃO AG DOUTORADO - OTIMIZAÇÃO DE RWA (ROUTING AND WAVELENGTH ASSIGNMENT)")
    print(f"{'='*80}")
    
    print("\nConfiguração da simulação REALISTA COM POISSON:")
    print("  • Chegadas com distribuição de Poisson")
    print("  • Durações com distribuição exponencial")
    print("  • 20 simulações completas por load")
    print("  • Loads de 1 a 200 Erlangs (sem pular)")
    print("  • 1000 unidades de tempo por simulação")
    print("  • 5 requisições simultâneas")
    print("  • Seleção por torneio (3 indivíduos)")
    print("  • Alocação First-Fit")
    
    print("\nEscolha o modo de operação:")
    print("1. Modo SEPARADO - Cada requisição otimizada e simulada individualmente")
    print("2. Modo CONJUNTO - Todas requisições otimizadas e simuladas conjuntamente")
    print("3. Sair")
    
    choice = input("\nDigite sua escolha (1-3): ").strip()
    
    if choice == "1":
        print(f"\n>>> INICIANDO MODO SEPARADO (REALISTA COM POISSON)")
        print(">>> Esta execução pode levar algum tempo devido às 20 simulações por load...")
        
        graph = create_nsfnet_graph()
        
        simulator = WDMSimulatorPhD_Completo(
            graph=graph,
            num_wavelengths=40,
            population_size=120,
            num_generations=40,
            k=20,
            mode="separado"
        )
        
        simulator.run_simulation(
            load_values=list(range(1, 201)),  # 1-200 Erlangs (sem pular)
            num_simulations=20,  # 20 simulações
            output_dir="resultados_separado_poisson"
        )
        
    elif choice == "2":
        print(f"\n>>> INICIANDO MODO CONJUNTO (REALISTA COM POISSON)")
        print(">>> Esta execução pode levar algum tempo devido às 20 simulações por load...")
        
        graph = create_nsfnet_graph()
        
        simulator = WDMSimulatorPhD_Completo(
            graph=graph,
            num_wavelengths=40,
            population_size=120,
            num_generations=40,
            k=83,
            mode="conjunto"
        )
        
        simulator.run_simulation(
            load_values=list(range(1, 201)),  # 1-200 Erlangs (sem pular)
            num_simulations=20,  # 20 simulações
            output_dir="resultados_conjunto_poisson"
        )
        
    elif choice == "3":
        print(f"\nSaindo...")
        return
    
    else:
        print(f"\nOpção inválida!")
        return
    
    print(f"\n{'='*80}")
    print("EXECUÇÃO CONCLUÍDA COM SUCESSO!")
    print("Todos os resultados foram salvos nos diretórios especificados.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()