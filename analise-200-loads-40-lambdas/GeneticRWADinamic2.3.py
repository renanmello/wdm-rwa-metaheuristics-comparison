import heapq
import random
import os
import time
from itertools import islice
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any
import pickle

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class WDMSimulatorPhD_Completo:
    """
    AG do Doutorado COMPLETO com opções separado/conjunto.
    
    Versões disponíveis:
    1. SEPARADO: Cada requisição otimizada e simulada isoladamente
    2. CONJUNTO: 5 requisições otimizadas e simuladas juntas (competição)
    
    Características:
    - Permite conversão de wavelengths (diferentes λ na mesma rota)
    - Penaliza mudanças de wavelength (mas permite)
    - Fitness function inteligente que considera congestionamento
    """
    
    def __init__(self,
                 graph: nx.Graph,
                 num_wavelengths: int = 40,
                 population_size: int = 120,
                 num_generations: int = 40,
                 k: int = 20,
                 mode: str = "conjunto"):  # "separado" ou "conjunto"
        """
        Inicializa o simulador.
        
        Args:
            graph: Grafo da rede
            num_wavelengths: Número de wavelengths por enlace
            population_size: Tamanho da população do AG
            num_generations: Número de gerações do AG
            k: Número de k-shortest paths a considerar
            mode: "separado" ou "conjunto"
        """
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.population_size = population_size
        self.num_generations = num_generations
        self.k = k
        self.mode = mode
        
        # Pares O-D fixos (mesmos do AG antigo)
        self.manual_pairs = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
        
        # Parâmetros de penalidade/recompensa
        self.penalty_per_hop = 0.15
        self.penalty_per_wavelength_change = 0.25
        self.reward_per_wavelength_reuse = 0.20
        self.penalty_per_congested_link = 0.40
        
        # Calcula todos os k-shortest paths
        print("Calculando k-shortest paths...")
        self.k_shortest_paths = self._get_all_k_shortest_paths()
        
        # Para cache de fitness
        self.fitness_cache = {}
        
        print(f"Simulador inicializado no modo: {mode.upper()}")
        print(f"  Requisições: {self.manual_pairs}")
        print(f"  Wavelengths: {num_wavelengths}")
        print(f"  k-paths: {k}")
    
    def reset_network(self):
        """Reseta a rede para estado inicial."""
        for u, v in self.graph.edges:
            self.graph[u][v]['wavelengths'] = np.ones(self.num_wavelengths, dtype=bool)
            self.graph[u][v]['usage_count'] = 0
            self.graph[u][v]['blocked_count'] = 0
    
    # ========== MÉTODOS DE ALOCAÇÃO COM CONVERSÃO ==========
    
    def find_best_allocation_for_route(self, route: List[int]) -> Optional[Dict[int, int]]:
        """
        Encontra a MELHOR alocação de wavelengths para uma rota.
        Considera disponibilidade e minimiza mudanças de wavelength.
        
        Retorna: {índice_enlace: wavelength} ou None
        """
        if len(route) < 2:
            return None
        
        # Encontra todas as alocações possíveis
        all_allocations = self._find_all_allocations_dfs(route, 0, {})
        
        if not all_allocations:
            return None
        
        # Avalia cada alocação e escolhe a melhor
        best_allocation = None
        best_score = -float('inf')
        
        for allocation in all_allocations:
            score = self._evaluate_allocation_score(route, allocation)
            if score > best_score:
                best_score = score
                best_allocation = allocation
        
        return best_allocation
    
    def _find_all_allocations_dfs(self, route: List[int], idx: int, 
                                 current: Dict[int, int]) -> List[Dict[int, int]]:
        """DFS recursivo para encontrar todas as alocações possíveis."""
        if idx == len(route) - 1:
            return [current.copy()]
        
        u, v = route[idx], route[idx + 1]
        allocations = []
        
        # Wavelengths disponíveis neste enlace
        available_wavelengths = []
        if self.graph.has_edge(u, v):
            for wl in range(self.num_wavelengths):
                if self.graph[u][v]['wavelengths'][wl]:
                    available_wavelengths.append(wl)
        
        # Tenta cada wavelength disponível
        for wl in available_wavelengths:
            current[idx] = wl
            allocations.extend(self._find_all_allocations_dfs(route, idx + 1, current))
        
        # Backtracking
        if idx in current:
            del current[idx]
        
        return allocations
    
    def _evaluate_allocation_score(self, route: List[int], 
                                  allocation: Dict[int, int]) -> float:
        """Avalia a qualidade de uma alocação (maior = melhor)."""
        if len(allocation) != len(route) - 1:
            return 0.0
        
        score = 0.0
        last_wavelength = None
        
        for i in range(len(route) - 1):
            current_wl = allocation[i]
            
            # Recompensa por reutilizar mesmo wavelength
            if last_wavelength is not None and current_wl == last_wavelength:
                score += self.reward_per_wavelength_reuse
            elif last_wavelength is not None:
                score -= self.penalty_per_wavelength_change
            
            # Penalidade por enlace congestionado
            u, v = route[i], route[i + 1]
            if self.graph.has_edge(u, v):
                used = self.num_wavelengths - np.sum(self.graph[u][v]['wavelengths'])
                congestion = used / self.num_wavelengths
                score -= congestion * self.penalty_per_congested_link
            
            last_wavelength = current_wl
        
        return score
    
    def allocate_wavelengths(self, route: List[int], 
                            allocation: Dict[int, int]) -> bool:
        """Aloca wavelengths específicos para cada enlace."""
        # Primeiro verifica se tudo está disponível
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            wavelength = allocation.get(i)
            
            if wavelength is None or not (0 <= wavelength < self.num_wavelengths):
                return False
            if not self.graph.has_edge(u, v):
                return False
            if not self.graph[u][v]['wavelengths'][wavelength]:
                return False
        
        # Se tudo OK, aloca
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            wavelength = allocation[i]
            self.graph[u][v]['wavelengths'][wavelength] = False
            self.graph[u][v]['usage_count'] += 1
        
        return True
    
    def release_wavelengths(self, route: List[int], 
                           allocation: Dict[int, int]):
        """Libera wavelengths alocados."""
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            wavelength = allocation.get(i)
            
            if (wavelength is not None and 
                0 <= wavelength < self.num_wavelengths and 
                self.graph.has_edge(u, v)):
                self.graph[u][v]['wavelengths'][wavelength] = True
                self.graph[u][v]['usage_count'] = max(0, self.graph[u][v]['usage_count'] - 1)
    
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
    
    def _initialize_population_single(self, source: int, target: int) -> List[Tuple[List[int], Dict]]:
        """Inicializa população para uma ÚNICA requisição."""
        population = []
        available_routes = self.k_shortest_paths.get((source, target), [])
        
        if not available_routes:
            return []
        
        for _ in range(self.population_size):
            # Escolhe rota aleatória
            route_idx = random.randint(0, len(available_routes) - 1)
            route = available_routes[route_idx]
            
            # Encontra alocação para esta rota
            allocation = self.find_best_allocation_for_route(route)
            
            if allocation:
                population.append((route, allocation))
            else:
                # Se não encontrar alocação, tenta outra rota
                for alt_route in available_routes:
                    alt_allocation = self.find_best_allocation_for_route(alt_route)
                    if alt_allocation:
                        population.append((alt_route, alt_allocation))
                        break
        
        return population
    
    def _initialize_population_conjunto(self) -> List[Dict]:
        """Inicializa população para TODAS as requisições (conjunto)."""
        population = []
        
        for _ in range(self.population_size):
            individual = {}
            
            for source, target in self.manual_pairs:
                available_routes = self.k_shortest_paths.get((source, target), [])
                if not available_routes:
                    continue
                
                # Escolhe rota aleatória
                route_idx = random.randint(0, len(available_routes) - 1)
                route = available_routes[route_idx]
                
                # Encontra alocação
                allocation = self.find_best_allocation_for_route(route)
                if allocation:
                    individual[(source, target)] = (route, allocation)
            
            if individual:
                population.append(individual)
        
        return population
    
    def _fitness_single(self, individual: Tuple[List[int], Dict]) -> float:
        """Fitness para uma única requisição."""
        route, allocation = individual
        
        # Penalidade por hops
        hops_penalty = (len(route) - 1) * self.penalty_per_hop
        
        # Penalidade por mudanças de wavelength
        changes = self._count_wavelength_changes(allocation)
        changes_penalty = changes * self.penalty_per_wavelength_change
        
        # Recompensa por reutilização
        reuse = max(0, (len(route) - 1) - changes - 1)
        reuse_reward = reuse * self.reward_per_wavelength_reuse
        
        # Penalidade por congestionamento
        congestion_penalty = 0
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if self.graph.has_edge(u, v):
                used = self.num_wavelengths - np.sum(self.graph[u][v]['wavelengths'])
                congestion = used / self.num_wavelengths
                congestion_penalty += congestion
        
        congestion_penalty = congestion_penalty / (len(route) - 1) if len(route) > 1 else 0
        
        # Score final
        score = 1.0
        score -= hops_penalty * 0.2
        score -= changes_penalty * 0.3
        score += reuse_reward * 0.2
        score -= congestion_penalty * 0.3
        
        return max(0.0, score)
    
    def _fitness_conjunto(self, individual: Dict) -> float:
        """Fitness para todas as requisições (conjunto)."""
        if not individual:
            return 0.0
        
        total_score = 0.0
        valid_requisitions = 0
        
        for (source, target), (route, allocation) in individual.items():
            score = self._fitness_single((route, allocation))
            total_score += score
            valid_requisitions += 1
        
        # Penalidade adicional por interferência entre rotas
        interference_penalty = self._calculate_interference_penalty(individual)
        total_score -= interference_penalty
        
        return total_score / valid_requisitions if valid_requisitions > 0 else 0.0
    
    def _calculate_interference_penalty(self, individual: Dict) -> float:
        """Calcula penalidade por interferência entre rotas."""
        link_usage = defaultdict(int)
        
        # Conta quantas rotas usam cada enlace
        for (source, target), (route, allocation) in individual.items():
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                link = tuple(sorted((u, v)))
                link_usage[link] += 1
        
        # Penaliza enlaces muito usados
        penalty = 0
        for link, count in link_usage.items():
            if count > 1:
                penalty += (count - 1) * 0.1  # 0.1 por rota extra
        
        return penalty
    
    def _count_wavelength_changes(self, allocation: Dict[int, int]) -> int:
        """Conta número de mudanças de wavelength."""
        changes = 0
        last_wl = None
        
        for i in sorted(allocation.keys()):
            if last_wl is not None and allocation[i] != last_wl:
                changes += 1
            last_wl = allocation[i]
        
        return changes
    
    def _selection_tournament(self, population: List, fitness_scores: List[float], 
                             tournament_size: int = 3) -> Any:
        """Seleção por torneio."""
        tournament_indices = random.sample(range(len(population)), 
                                          min(tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover_single(self, parent1: Tuple, parent2: Tuple) -> Tuple[Tuple, Tuple]:
        """Crossover para uma única requisição."""
        route1, alloc1 = parent1
        route2, alloc2 = parent2
        
        # Crossover simples: troca as rotas
        if random.random() < 0.8 and len(route1) > 2 and len(route2) > 2:
            # Encontra ponto de crossover comum
            common_nodes = set(route1) & set(route2)
            if len(common_nodes) >= 2:
                # Escolhe dois nós comuns
                crossover_nodes = random.sample(sorted(common_nodes), 2)
                start_node, end_node = crossover_nodes
                
                # Encontra índices nos caminhos
                if (start_node in route1 and end_node in route1 and
                    start_node in route2 and end_node in route2):
                    
                    idx1_start = route1.index(start_node)
                    idx1_end = route1.index(end_node)
                    idx2_start = route2.index(start_node)
                    idx2_end = route2.index(end_node)
                    
                    if idx1_start < idx1_end and idx2_start < idx2_end:
                        # Cria filhos
                        child1_route = route1[:idx1_start] + route2[idx2_start:idx2_end] + route1[idx1_end:]
                        child2_route = route2[:idx2_start] + route1[idx1_start:idx1_end] + route2[idx2_end:]
                        
                        # Encontra novas alocações
                        child1_alloc = self.find_best_allocation_for_route(child1_route)
                        child2_alloc = self.find_best_allocation_for_route(child2_route)
                        
                        if child1_alloc and child2_alloc:
                            return ((child1_route, child1_alloc), 
                                    (child2_route, child2_alloc))
        
        # Se não fez crossover, retorna os pais
        return parent1, parent2
    
    def _crossover_conjunto(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Crossover para múltiplas requisições."""
        child1 = {}
        child2 = {}
        
        for req in self.manual_pairs:
            if random.random() < 0.5:
                # Herda do parent1
                if req in parent1:
                    child1[req] = parent1[req]
                if req in parent2:
                    child2[req] = parent2[req]
            else:
                # Herda do parent2
                if req in parent2:
                    child1[req] = parent2[req]
                if req in parent1:
                    child2[req] = parent1[req]
        
        return child1, child2
    
    def _mutate_single(self, individual: Tuple[List[int], Dict]) -> Tuple[List[int], Dict]:
        """Mutação para uma única requisição."""
        route, allocation = individual
        new_route = route.copy()
        
        if random.random() < 0.1:  # 10% chance de mutação
            source, target = route[0], route[-1]
            available_routes = self.k_shortest_paths.get((source, target), [])
            
            if available_routes and len(available_routes) > 1:
                # Escolhe nova rota diferente da atual
                other_routes = [r for r in available_routes if r != route]
                if other_routes:
                    new_route = random.choice(other_routes)
        
        # Encontra nova alocação para a rota (mutada ou não)
        new_allocation = self.find_best_allocation_for_route(new_route)
        if new_allocation:
            return (new_route, new_allocation)
        else:
            return individual  # Mantém original se não encontrar alocação
    
    def _mutate_conjunto(self, individual: Dict) -> Dict:
        """Mutação para múltiplas requisições."""
        mutated = individual.copy()
        
        for req in list(mutated.keys()):
            if random.random() < 0.05:  # 5% chance por requisição
                route, allocation = mutated[req]
                source, target = req
                
                available_routes = self.k_shortest_paths.get((source, target), [])
                if available_routes and len(available_routes) > 1:
                    # Escolhe nova rota
                    other_routes = [r for r in available_routes if r != route]
                    if other_routes:
                        new_route = random.choice(other_routes)
                        new_allocation = self.find_best_allocation_for_route(new_route)
                        if new_allocation:
                            mutated[req] = (new_route, new_allocation)
        
        return mutated
    
    def genetic_algorithm_single(self, source: int, target: int) -> Tuple[List[int], Dict]:
        """Executa AG para uma ÚNICA requisição."""
        print(f"  Executando AG para [{source},{target}]...")
        start_time = time.time()
        
        # Inicializa população
        population = self._initialize_population_single(source, target)
        if not population:
            print(f"    ⚠ Nenhuma rota válida encontrada!")
            return None
        
        best_fitness_history = []
        
        for gen in range(self.num_generations):
            # Avalia fitness
            fitness_scores = [self._fitness_single(ind) for ind in population]
            
            # Ordena por fitness
            sorted_pop = [x for _, x in sorted(zip(fitness_scores, population), 
                                             key=lambda pair: pair[0], reverse=True)]
            
            # Mantém elite
            elite_size = max(2, self.population_size // 5)
            new_population = sorted_pop[:elite_size]
            
            # Melhor indivíduo
            best_idx = np.argmax(fitness_scores)
            best_individual = population[best_idx]
            best_fitness = fitness_scores[best_idx]
            best_fitness_history.append(best_fitness)
            
            # Verifica estagnação
            if len(best_fitness_history) > 10:
                recent_avg = np.mean(best_fitness_history[-5:])
                if abs(best_fitness - recent_avg) < 0.001:
                    if gen > self.num_generations // 2:
                        break
            
            # Gera nova população
            while len(new_population) < self.population_size:
                # Seleção
                parent1 = self._selection_tournament(population, fitness_scores)
                parent2 = self._selection_tournament(population, fitness_scores)
                
                # Crossover
                if random.random() < 0.8:
                    child1, child2 = self._crossover_single(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # Mutação
                child1 = self._mutate_single(child1)
                child2 = self._mutate_single(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # Encontra melhor solução final
        fitness_scores = [self._fitness_single(ind) for ind in population]
        best_idx = np.argmax(fitness_scores)
        best_solution = population[best_idx]
        
        ga_time = time.time() - start_time
        route, allocation = best_solution
        print(f"    ✓ Concluído em {ga_time:.2f}s: {len(route)-1} hops, {self._count_wavelength_changes(allocation)} mudanças de λ")
        
        return best_solution
    
    def genetic_algorithm_conjunto(self) -> Dict:
        """Executa AG para TODAS as requisições (conjunto)."""
        print(f"  Executando AG para {len(self.manual_pairs)} requisições conjuntas...")
        start_time = time.time()
        
        # Inicializa população
        population = self._initialize_population_conjunto()
        
        best_fitness_history = []
        
        for gen in range(self.num_generations):
            # Avalia fitness
            fitness_scores = [self._fitness_conjunto(ind) for ind in population]
            
            # Ordena por fitness
            sorted_pop = [x for _, x in sorted(zip(fitness_scores, population), 
                                             key=lambda pair: pair[0], reverse=True)]
            
            # Mantém elite
            elite_size = max(2, self.population_size // 5)
            new_population = sorted_pop[:elite_size]
            
            # Melhor indivíduo
            best_idx = np.argmax(fitness_scores)
            best_individual = population[best_idx]
            best_fitness = fitness_scores[best_idx]
            best_fitness_history.append(best_fitness)
            
            # Verifica estagnação
            if len(best_fitness_history) > 10:
                recent_avg = np.mean(best_fitness_history[-5:])
                if abs(best_fitness - recent_avg) < 0.001:
                    if gen > self.num_generations // 2:
                        break
            
            # Gera nova população
            while len(new_population) < self.population_size:
                # Seleção
                parent1 = self._selection_tournament(population, fitness_scores)
                parent2 = self._selection_tournament(population, fitness_scores)
                
                # Crossover
                if random.random() < 0.8:
                    child1, child2 = self._crossover_conjunto(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # Mutação
                child1 = self._mutate_conjunto(child1)
                child2 = self._mutate_conjunto(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # Encontra melhor solução final
        fitness_scores = [self._fitness_conjunto(ind) for ind in population]
        best_idx = np.argmax(fitness_scores)
        best_solution = population[best_idx]
        
        ga_time = time.time() - start_time
        print(f"    ✓ AG conjunto concluído em {ga_time:.2f}s")
        
        # Mostra resultados
        print(f"    Rotas encontradas:")
        for (source, target), (route, allocation) in best_solution.items():
            hops = len(route) - 1
            changes = self._count_wavelength_changes(allocation)
            print(f"      [{source},{target}]: {hops} hops, {changes} mudanças de λ")
        
        return best_solution
    
    # ========== MÉTODOS DE SIMULAÇÃO ==========
    
    def simulate_single_requisition(self, source: int, target: int,
                                   best_solution: Tuple[List[int], Dict],
                                   load_values: List[int],
                                   num_simulations: int = 10,
                                   calls_per_load: int = 1000) -> Dict[int, List[float]]:
        """Simula uma ÚNICA requisição de forma isolada."""
        if not best_solution:
            return {load: [1.0] * num_simulations for load in load_values}
        
        route, allocation = best_solution
        results = {load: [] for load in load_values}
        
        print(f"    Simulando requisição [{source},{target}] isoladamente...")
        
        for sim in range(num_simulations):
            for load in load_values:
                # Reset da rede (isolada para esta requisição)
                self.reset_network()
                
                blocked_calls = 0
                total_calls = calls_per_load
                
                current_time = 0
                event_queue = []
                
                for _ in range(total_calls):
                    current_time += np.random.exponential(1.0 / max(load, 0.1))
                    
                    # Libera chamadas expiradas
                    while event_queue and event_queue[0][0] <= current_time:
                        event_time, route_to_free, alloc_to_free = heapq.heappop(event_queue)
                        self.release_wavelengths(route_to_free, alloc_to_free)
                    
                    # Tenta alocar
                    can_allocate = True
                    for i in range(len(route) - 1):
                        u, v = route[i], route[i + 1]
                        wavelength = allocation.get(i)
                        if (wavelength is None or 
                            not self.graph.has_edge(u, v) or
                            not self.graph[u][v]['wavelengths'][wavelength]):
                            can_allocate = False
                            break
                    
                    if can_allocate:
                        self.allocate_wavelengths(route, allocation)
                        duration = np.random.exponential(1.0)
                        heapq.heappush(event_queue, 
                                     (current_time + duration, route, allocation))
                    else:
                        blocked_calls += 1
                
                # Libera chamadas restantes
                while event_queue:
                    event_time, route_to_free, alloc_to_free = heapq.heappop(event_queue)
                    self.release_wavelengths(route_to_free, alloc_to_free)
                
                blocking_prob = blocked_calls / total_calls
                results[load].append(blocking_prob)
        
        return results
    
    def simulate_conjunto(self, best_solutions: Dict,
                         load_values: List[int],
                         num_simulations: int = 10,
                         calls_per_load: int = 1000) -> Dict[int, Dict[int, List[float]]]:
        """Simula TODAS as requisições conjuntamente (competição)."""
        results = {idx: {load: [] for load in load_values} 
                  for idx in range(len(self.manual_pairs))}
        
        print(f"    Simulando {len(self.manual_pairs)} requisições conjuntamente...")
        
        for sim in range(num_simulations):
            for load in load_values:
                # Reset da rede (compartilhada entre todas requisições)
                self.reset_network()
                
                blocked_calls = [0] * len(self.manual_pairs)
                total_calls = calls_per_load
                
                current_time = 0
                event_queue = []
                
                for _ in range(total_calls):
                    # Escolhe requisição aleatória
                    req_idx = random.randint(0, len(self.manual_pairs) - 1)
                    source, target = self.manual_pairs[req_idx]
                    
                    if (source, target) not in best_solutions:
                        blocked_calls[req_idx] += 1
                        continue
                    
                    route, allocation = best_solutions[(source, target)]
                    
                    current_time += np.random.exponential(1.0 / max(load, 0.1))
                    
                    # Libera chamadas expiradas
                    while event_queue and event_queue[0][0] <= current_time:
                        event_time, req_idx_free, route_to_free, alloc_to_free = heapq.heappop(event_queue)
                        self.release_wavelengths(route_to_free, alloc_to_free)
                    
                    # Tenta alocar
                    can_allocate = True
                    for i in range(len(route) - 1):
                        u, v = route[i], route[i + 1]
                        wavelength = allocation.get(i)
                        if (wavelength is None or 
                            not self.graph.has_edge(u, v) or
                            not self.graph[u][v]['wavelengths'][wavelength]):
                            can_allocate = False
                            break
                    
                    if can_allocate:
                        self.allocate_wavelengths(route, allocation)
                        duration = np.random.exponential(1.0)
                        heapq.heappush(event_queue, 
                                     (current_time + duration, req_idx, route, allocation))
                    else:
                        blocked_calls[req_idx] += 1
                
                # Libera chamadas restantes
                while event_queue:
                    event_time, req_idx_free, route_to_free, alloc_to_free = heapq.heappop(event_queue)
                    self.release_wavelengths(route_to_free, alloc_to_free)
                
                # Armazena resultados
                for req_idx in range(len(self.manual_pairs)):
                    blocking_prob = blocked_calls[req_idx] / total_calls
                    results[req_idx][load].append(blocking_prob)
        
        return results
    
    # ========== MÉTODOS PRINCIPAIS ==========
    
    def run_simulation(self, load_values: List[int] = None,
                      num_simulations: int = 20,
                      calls_per_load: int = 1000,
                      output_dir: str = "resultados_phd") -> Dict:
        """
        Executa simulação completa no modo configurado.
        
        Returns:
            Dicionário com resultados por requisição
        """
        if load_values is None:
            load_values = list(range(1, 201))
        
        print(f"\n{'='*70}")
        print(f"SIMULAÇÃO AG DOUTORADO - MODO: {self.mode.upper()}")
        print(f"{'='*70}")
        print(f"Parâmetros:")
        print(f"  • Requisições: {len(self.manual_pairs)}")
        print(f"  • Loads: {len(load_values)} (de {min(load_values)} a {max(load_values)})")
        print(f"  • Simulações por load: {num_simulations}")
        print(f"  • Chamadas por load: {calls_per_load}")
        print(f"  • Wavelengths: {self.num_wavelengths}")
        print(f"{'='*70}")
        
        start_total = time.time()
        
        if self.mode == "separado":
            results = self._run_mode_separado(load_values, num_simulations, calls_per_load)
        else:  # conjunto
            results = self._run_mode_conjunto(load_values, num_simulations, calls_per_load)
        
        total_time = time.time() - start_total
        
        # Salva resultados
        self._save_results(results, load_values, num_simulations, 
                          calls_per_load, total_time, output_dir)
        
        # Gera gráficos
        self._plot_results(results, load_values, output_dir)
        
        print(f"\n{'='*70}")
        print(f"SIMULAÇÃO CONCLUÍDA!")
        print(f"Tempo total: {total_time:.2f}s ({total_time/60:.2f} minutos)")
        print(f"Resultados em: {output_dir}/")
        print(f"{'='*70}")
        
        return results
    
    def _run_mode_separado(self, load_values: List[int],
                          num_simulations: int,
                          calls_per_load: int) -> Dict:
        """Executa no modo SEPARADO."""
        print(f"\nMODO SEPARADO: Cada requisição otimizada e simulada isoladamente")
        
        all_results = {}
        
        for idx, (source, target) in enumerate(self.manual_pairs):
            print(f"\n[{idx+1}/{len(self.manual_pairs)}] Requisição: [{source},{target}]")
            
            # Executa AG para esta requisição
            best_solution = self.genetic_algorithm_single(source, target)
            
            if not best_solution:
                print(f"  ⚠ Não foi possível encontrar solução válida")
                all_results[idx] = {load: [1.0] * num_simulations for load in load_values}
                continue
            
            # Simula esta requisição
            results = self.simulate_single_requisition(
                source, target, best_solution, 
                load_values, num_simulations, calls_per_load
            )
            
            all_results[idx] = results
            
            # Mostra estatísticas rápidas
            avg_blocking = np.mean([np.mean(results[load]) for load in load_values[:5]])
            print(f"  📊 Probabilidade média (loads 1-5): {avg_blocking:.6f}")
        
        return all_results
    
    def _run_mode_conjunto(self, load_values: List[int],
                          num_simulations: int,
                          calls_per_load: int) -> Dict:
        """Executa no modo CONJUNTO."""
        print(f"\nMODO CONJUNTO: Todas requisições otimizadas e simuladas juntas")
        
        # Executa AG conjunto
        best_solutions = self.genetic_algorithm_conjunto()
        
        if not best_solutions:
            print(f"  ⚠ Não foi possível encontrar soluções válidas")
            return {}
        
        # Simula todas as requisições conjuntamente
        all_results = self.simulate_conjunto(
            best_solutions, load_values, num_simulations, calls_per_load
        )
        
        # Mostra estatísticas rápidas
        print(f"\n📊 Estatísticas rápidas (loads 1-5):")
        for idx, (source, target) in enumerate(self.manual_pairs):
            if idx in all_results:
                avg_blocking = np.mean([np.mean(all_results[idx][load]) for load in load_values[:5]])
                print(f"  [{source},{target}]: {avg_blocking:.6f}")
        
        return all_results
    
    # ========== MÉTODOS DE SAÍDA ==========
    
    def _save_results(self, results: Dict, load_values: List[int],
                     num_simulations: int, calls_per_load: int,
                     total_time: float, output_dir: str):
        """Salva resultados em arquivos."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Salva por requisição
        for idx in results:
            source, target = self.manual_pairs[idx]
            filename = f"{output_dir}/req_{idx+1}_{source}_{target}.txt"
            
            with open(filename, 'w') as f:
                f.write(f"# Requisição {idx+1}: [{source},{target}]\n")
                f.write(f"# Modo: {self.mode}\n")
                f.write(f"# Simulações: {num_simulations}, Chamadas/load: {calls_per_load}\n")
                f.write(f"# Wavelengths: {self.num_wavelengths}\n")
                f.write("# " + "="*50 + "\n")
                f.write("# Load  Probabilidade_Média  Desvio_Padrao\n")
                
                for load in load_values:
                    if load in results[idx] and results[idx][load]:
                        probs = results[idx][load]
                        mean_prob = np.mean(probs)
                        std_prob = np.std(probs) if len(probs) > 1 else 0.0
                        f.write(f"{load:4d}  {mean_prob:.6f}  {std_prob:.6f}\n")
        
        # Salva arquivo de comparação
        comp_file = f"{output_dir}/comparacao_{self.mode}.txt"
        with open(comp_file, 'w') as f:
            f.write(f"# COMPARAÇÃO ENTRE REQUISIÇÕES - MODO: {self.mode.upper()}\n")
            f.write("# " + "="*70 + "\n")
            f.write(f"# Tempo total: {total_time:.2f}s\n")
            f.write(f"# Simulações: {num_simulations}\n")
            f.write(f"# Chamadas/load: {calls_per_load}\n")
            f.write("# " + "="*70 + "\n")
            f.write("# Req  O-D    Load  Prob_Média  Dif_vs_Melhor  Ranking\n")
            
            # Calcula ranking por load
            for load in load_values[:50]:  # Primeiros 50 loads apenas
                load_probs = []
                for idx in results:
                    if load in results[idx] and results[idx][load]:
                        mean_prob = np.mean(results[idx][load])
                        load_probs.append((idx, mean_prob))
                
                if load_probs:
                    # Ordena por probabilidade (menor é melhor)
                    load_probs.sort(key=lambda x: x[1])
                    
                    for rank, (idx, mean_prob) in enumerate(load_probs, 1):
                        source, target = self.manual_pairs[idx]
                        best_prob = load_probs[0][1]
                        diff = mean_prob - best_prob
                        
                        f.write(f"{idx+1:3d}  {source:2d}-{target:2d}  {load:4d}  {mean_prob:.6f}  {diff:+.6f}  {rank:2d}\n")
        
        # Salva arquivo de tempo
        time_file = f"{output_dir}/tempo_execucao.txt"
        with open(time_file, 'w') as f:
            f.write(f"Tempo total: {total_time:.2f}s\n")
            f.write(f"Modo: {self.mode}\n")
            f.write(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Requisições: {len(self.manual_pairs)}\n")
            f.write(f"Loads: {len(load_values)}\n")
            f.write(f"Simulações: {num_simulations}\n")
    
    def _plot_results(self, results: Dict, load_values: List[int], 
                     output_dir: str):
        """Gera gráficos dos resultados."""
        plt.figure(figsize=(14, 10))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        markers = ['o', 's', '^', 'D', 'v']
        
        # Gráfico 1: Todas as requisições
        ax1 = plt.subplot(2, 2, 1)
        
        for idx in range(len(self.manual_pairs)):
            source, target = self.manual_pairs[idx]
            
            means = []
            for load in load_values[:100]:  # Primeiros 100 loads
                if idx in results and load in results[idx] and results[idx][load]:
                    means.append(np.mean(results[idx][load]) * 100)  # Em %
                else:
                    means.append(0)
            
            ax1.plot(load_values[:len(means)], means,
                    color=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)],
                    markersize=4,
                    linewidth=1.5,
                    label=f'[{source},{target}]',
                    alpha=0.8)
        
        ax1.set_xlabel('Carga (Load)', fontsize=11)
        ax1.set_ylabel('Prob. Bloqueio (%)', fontsize=11)
        ax1.set_title(f'AG Doutorado ({self.mode}) - Todas Requisições', 
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper left')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(0, 105)
        ax1.set_ylim(-2, 102)
        
        # Gráfico 2: Média geral
        ax2 = plt.subplot(2, 2, 2)
        
        overall_means = []
        overall_stds = []
        
        for load in load_values[:100]:
            all_probs = []
            for idx in results:
                if load in results[idx] and results[idx][load]:
                    all_probs.extend(results[idx][load])
            
            if all_probs:
                overall_means.append(np.mean(all_probs) * 100)
                overall_stds.append(np.std(all_probs) * 100 if len(all_probs) > 1 else 0)
            else:
                overall_means.append(0)
                overall_stds.append(0)
        
        ax2.errorbar(load_values[:len(overall_means)], overall_means, yerr=overall_stds,
                    fmt='o-', linewidth=2, capsize=3, capthick=1,
                    color='darkblue', ecolor='lightblue', markersize=4,
                    label='Média Geral ± Desvio')
        
        ax2.set_xlabel('Carga (Load)', fontsize=11)
        ax2.set_ylabel('Prob. Bloqueio (%)', fontsize=11)
        ax2.set_title('Média Geral de Todas as Requisições', 
                     fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(0, 105)
        ax2.set_ylim(-2, 102)
        
        # Gráfico 3: Diferenças entre requisições (load específico)
        ax3 = plt.subplot(2, 2, 3)
        
        load_to_analyze = 50  # Analisa load 50
        if load_to_analyze in load_values:
            req_probs = []
            req_labels = []
            
            for idx in results:
                if load_to_analyze in results[idx] and results[idx][load_to_analyze]:
                    source, target = self.manual_pairs[idx]
                    mean_prob = np.mean(results[idx][load_to_analyze]) * 100
                    req_probs.append(mean_prob)
                    req_labels.append(f'[{source},{target}]')
            
            if req_probs:
                bars = ax3.bar(range(len(req_probs)), req_probs, 
                              color=colors[:len(req_probs)], alpha=0.7)
                ax3.set_xticks(range(len(req_probs)))
                ax3.set_xticklabels(req_labels, rotation=45, fontsize=9)
                ax3.set_ylabel('Prob. Bloqueio (%)', fontsize=11)
                ax3.set_title(f'Comparação no Load {load_to_analyze}', 
                            fontsize=13, fontweight='bold')
                ax3.grid(True, alpha=0.3, axis='y')
                
                # Adiciona valores nas barras
                for bar, prob in zip(bars, req_probs):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{prob:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Gráfico 4: Evolução do blocking com carga
        ax4 = plt.subplot(2, 2, 4)
        
        # Escolhe 3 requisições para mostrar evolução
        reqs_to_show = min(3, len(self.manual_pairs))
        
        for idx in range(reqs_to_show):
            source, target = self.manual_pairs[idx]
            
            means = []
            for load in load_values[:50]:
                if idx in results and load in results[idx] and results[idx][load]:
                    means.append(np.mean(results[idx][load]) * 100)
                else:
                    means.append(0)
            
            ax4.plot(load_values[:len(means)], means,
                    color=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)],
                    markersize=3,
                    linewidth=1.2,
                    label=f'[{source},{target}]')
        
        ax4.set_xlabel('Carga (Load)', fontsize=11)
        ax4.set_ylabel('Prob. Bloqueio (%)', fontsize=11)
        ax4.set_title('Evolução com Carga (Primeiras 3 Requisições)', 
                     fontsize=13, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_xlim(0, 55)
        ax4.set_ylim(-2, 102)
        
        plt.suptitle(f'Resultados AG Doutorado - Modo: {self.mode.upper()}', 
                    fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Salva gráfico
        plot_file = f"{output_dir}/grafico_resultados_{self.mode}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  Gráfico salvo em: {plot_file}")


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


def compare_modes_separado_vs_conjunto():
    """Compara resultados dos modos separado e conjunto."""
    print(f"\n{'='*70}")
    print("COMPARAÇÃO: MODO SEPARADO vs MODO CONJUNTO")
    print(f"{'='*70}")
    
    # Configurações
    load_values = list(range(1, 101))  # 1 a 100 para ser mais rápido
    num_simulations = 10  # Menos simulações para teste rápido
    
    # Cria grafo
    graph = create_nsfnet_graph()
    
    # Executa modo SEPARADO
    print(f"\n>>> EXECUTANDO MODO SEPARADO...")
    start_separado = time.time()
    
    simulator_separado = WDMSimulatorPhD_Completo(
        graph=graph.copy(),
        num_wavelengths=40,
        population_size=80,
        num_generations=30,
        k=15,
        mode="separado"
    )
    
    results_separado = simulator_separado.run_simulation(
        load_values=load_values,
        num_simulations=num_simulations,
        calls_per_load=1000,
        output_dir="resultados_separado"
    )
    
    time_separado = time.time() - start_separado
    
    # Executa modo CONJUNTO
    print(f"\n>>> EXECUTANDO MODO CONJUNTO...")
    start_conjunto = time.time()
    
    simulator_conjunto = WDMSimulatorPhD_Completo(
        graph=graph.copy(),
        num_wavelengths=40,
        population_size=80,
        num_generations=30,
        k=15,
        mode="conjunto"
    )
    
    results_conjunto = simulator_conjunto.run_simulation(
        load_values=load_values,
        num_simulations=num_simulations,
        calls_per_load=1000,
        output_dir="resultados_conjunto"
    )
    
    time_conjunto = time.time() - start_conjunto
    
    # Compara resultados
    print(f"\n{'='*70}")
    print("RESUMO DA COMPARAÇÃO")
    print(f"{'='*70}")
    print(f"Tempos de execução:")
    print(f"  • Separado: {time_separado:.2f}s ({time_separado/60:.2f} min)")
    print(f"  • Conjunto: {time_conjunto:.2f}s ({time_conjunto/60:.2f} min)")
    print(f"  • Diferença: {time_conjunto - time_separado:+.2f}s")
    
    # Calcula médias
    if results_separado and results_conjunto:
        print(f"\n📊 Probabilidades médias (loads 1-50):")
        
        for idx in range(len(simulator_separado.manual_pairs)):
            source, target = simulator_separado.manual_pairs[idx]
            
            if idx in results_separado and idx in results_conjunto:
                # Média do modo separado
                sep_probs = []
                for load in load_values[:50]:
                    if load in results_separado[idx]:
                        sep_probs.extend(results_separado[idx][load])
                sep_mean = np.mean(sep_probs) if sep_probs else 0
                
                # Média do modo conjunto
                conj_probs = []
                for load in load_values[:50]:
                    if load in results_conjunto[idx]:
                        conj_probs.extend(results_conjunto[idx][load])
                conj_mean = np.mean(conj_probs) if conj_probs else 0
                
                diff = conj_mean - sep_mean
                diff_pct = (diff / sep_mean * 100) if sep_mean > 0 else 0
                
                melhor = "SEPARADO" if sep_mean < conj_mean else "CONJUNTO"
                
                print(f"  [{source},{target}]:")
                print(f"    • Separado: {sep_mean:.6f}")
                print(f"    • Conjunto: {conj_mean:.6f}")
                print(f"    • Diferença: {diff:+.6f} ({diff_pct:+.1f}%)")
                print(f"    • Melhor: {melhor}")
    
    print(f"\n{'='*70}")
    print("COMPARAÇÃO CONCLUÍDA!")
    print(f"{'='*70}")


def main():
    """Função principal com menu de opções."""
    print(f"\n{'='*70}")
    print("SISTEMA AG DOUTORADO - ROTEAMENTO E ALOCAÇÃO DE WAVELENGTHS")
    print(f"{'='*70}")
    
    print("\nEscolha uma opção:")
    print("1. Executar modo SEPARADO (cada requisição isolada)")
    print("2. Executar modo CONJUNTO (todas juntas, competição)")
    print("3. Comparar ambos os modos")
    print("4. Teste rápido (loads 1-50, 5 simulações)")
    print("5. Sair")
    
    choice = input("\nDigite sua escolha (1-5): ").strip()
    
    if choice == "1":
        print(f"\n>>> INICIANDO MODO SEPARADO")
        
        graph = create_nsfnet_graph()
        
        simulator = WDMSimulatorPhD_Completo(
            graph=graph,
            num_wavelengths=40,
            population_size=120,
            num_generations=40,
            k=83,
            mode="separado"
        )
        
        simulator.run_simulation(
            load_values=list(range(1, 201)),
            num_simulations=20,
            calls_per_load=1000,
            output_dir="resultados_final_separado"
        )
        
    elif choice == "2":
        print(f"\n>>> INICIANDO MODO CONJUNTO")
        
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
            load_values=list(range(1, 201)),
            num_simulations=20,
            calls_per_load=1000,
            output_dir="resultados_final_conjunto"
        )
        
    elif choice == "3":
        compare_modes_separado_vs_conjunto()
        
    elif choice == "4":
        print(f"\n>>> TESTE RÁPIDO (loads 1-50, 5 simulações)")
        
        graph = create_nsfnet_graph()
        
        simulator = WDMSimulatorPhD_Completo(
            graph=graph,
            num_wavelengths=40,
            population_size=50,  # Menor para teste rápido
            num_generations=20,  # Menor para teste rápido
            k=10,  # Menor para teste rápido
            mode="conjunto"  # Testa o conjunto que é mais interessante
        )
        
        simulator.run_simulation(
            load_values=list(range(1, 201)),
            num_simulations=5,
            calls_per_load=1000,  # Menos chamadas para teste rápido
            output_dir="teste_rapido"
        )
        
    elif choice == "5":
        print(f"\nSaindo...")
        return
    
    else:
        print(f"\nOpção inválida!")
        return
    
    print(f"\n{'='*70}")
    print("EXECUÇÃO CONCLUÍDA!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()