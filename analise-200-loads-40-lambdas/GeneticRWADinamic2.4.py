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
    """
    
    def __init__(self,
                 graph: nx.Graph,
                 num_wavelengths: int = 40,
                 population_size: int = 120,
                 num_generations: int = 40,
                 k: int = 20,  # Reduzido para ser mais eficiente
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
        
        # Parâmetros para fitness (ajustados para minimizar bloqueio)
        self.penalty_per_hop = 0.2  # Penalidade por hop adicional
        self.penalty_per_wavelength_change = 0.3  # Penalidade por mudança de λ
        self.reward_per_wavelength_reuse = 0.25  # Recompensa por reutilizar λ
        self.penalty_per_congested_link = 0.5  # Penalidade por enlace congestionado
        
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
    
    def _initialize_graph_attributes(self):
        """Inicializa os atributos necessários no grafo."""
        for u, v in self.graph.edges():
            self.graph[u][v]['wavelengths'] = [True] * self.num_wavelengths
            self.graph[u][v]['usage_count'] = 0
            self.graph[u][v]['blocked_count'] = 0
            self.graph[u][v]['current_allocations'] = []
    
    def reset_network(self):
        """Reseta a rede para estado inicial."""
        for u, v in self.graph.edges():
            self.graph[u][v]['wavelengths'] = [True] * self.num_wavelengths
            self.graph[u][v]['usage_count'] = 0
            self.graph[u][v]['blocked_count'] = 0
            self.graph[u][v]['current_allocations'] = []
    
    # ========== MÉTODOS DE ALOCAÇÃO REALISTA ==========
    
    def allocate_route_with_first_fit(self, route: List[int]) -> Optional[int]:
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
                    self.graph[u][v]['current_allocations'].append(wl)
                return wl
        
        # Se não encontrou wavelength disponível
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            self.graph[u][v]['blocked_count'] += 1
        
        return None
    
    def release_route(self, route: List[int], wavelength: int):
        """Libera um wavelength alocado em uma rota."""
        if wavelength is None:
            return
            
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if self.graph.has_edge(u, v):
                self.graph[u][v]['wavelengths'][wavelength] = True
                
                # Remove da lista de alocações atuais
                if wavelength in self.graph[u][v]['current_allocations']:
                    self.graph[u][v]['current_allocations'].remove(wavelength)
    
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
        Calcula a aptidão de uma rota específica (otimizada para minimizar bloqueio).
        
        Args:
            route: Lista de nós representando a rota
            
        Returns:
            Valor de fitness da rota (maior = melhor)
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
            
            # Seleção por roleta viciada
            total_fitness = sum(fitness_scores)
            if total_fitness > 0:
                probabilities = [score / total_fitness for score in fitness_scores]
            else:
                probabilities = [1.0 / len(population)] * len(population)
            
            # Nova população
            new_population = []
            
            # Mantém elite (10% melhores)
            elite_size = max(1, len(population) // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # Gera resto da população
            while len(new_population) < self.population_size:
                # Seleciona pais
                parent1_idx = np.random.choice(len(population), p=probabilities)
                parent2_idx = np.random.choice(len(population), p=probabilities)
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
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
            
            # Seleção por roleta viciada
            total_fitness = sum(fitness_scores)
            if total_fitness > 0:
                probabilities = [score / total_fitness for score in fitness_scores]
            else:
                probabilities = [1.0 / len(population)] * len(population)
            
            # Nova população
            new_population = []
            
            # Mantém elite (10% melhores)
            elite_size = max(1, len(population) // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # Gera resto da população
            while len(new_population) < self.population_size:
                # Seleciona pais
                parent1_idx = np.random.choice(len(population), p=probabilities)
                parent2_idx = np.random.choice(len(population), p=probabilities)
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
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
    
    # ========== MÉTODOS DE SIMULAÇÃO REALISTA ==========
    
    def simulate_single_requisition(self, source: int, target: int,
                                    best_route: List[int],
                                    load_values: List[int],
                                    num_simulations: int = 20,
                                    calls_per_load: int = 1000) -> Dict[int, List[float]]:
        """
        Simula uma ÚNICA requisição de forma isolada.
        
        Simulação realista com alocação First-Fit e liberação após tempo aleatório.
        """
        if not best_route or len(best_route) < 2:
            return {load: [1.0] * num_simulations for load in load_values}
        
        results = {load: [] for load in load_values}
        
        print(f"    Simulando requisição [{source},{target}]...")
        start_sim_time = time.time()
        
        for sim in range(num_simulations):
            print(f"      Simulação {sim+1}/{num_simulations}", end='\r')
            
            for load in load_values:
                # Reset da rede
                self.reset_network()
                
                blocked_calls = 0
                total_calls = calls_per_load
                
                # Lista para controlar alocações ativas (simulando duração de chamadas)
                active_allocations = []
                
                # Taxa de chegada baseada no load
                arrival_rate = load / 100.0  # Convertendo load para probabilidade
                
                for call_num in range(total_calls):
                    # Simula chegada de chamada baseada no load
                    if random.random() < arrival_rate:
                        # Tenta alocar a rota
                        allocated_wl = self.allocate_route_with_first_fit(best_route)
                        
                        if allocated_wl is not None:
                            # Chamada bem-sucedida - adiciona à lista de alocações ativas
                            # Duração aleatória entre 1 e 10 unidades de tempo
                            duration = random.randint(1, 10)
                            active_allocations.append({
                                'route': best_route,
                                'wavelength': allocated_wl,
                                'remaining_time': duration
                            })
                        else:
                            # Chamada bloqueada
                            blocked_calls += 1
                    
                    # Processa alocações ativas (simulando passagem do tempo)
                    completed_allocations = []
                    for i, allocation in enumerate(active_allocations):
                        allocation['remaining_time'] -= 1
                        if allocation['remaining_time'] <= 0:
                            # Libera recursos
                            self.release_route(allocation['route'], allocation['wavelength'])
                            completed_allocations.append(i)
                    
                    # Remove alocações completadas
                    for i in sorted(completed_allocations, reverse=True):
                        active_allocations.pop(i)
                
                # Libera quaisquer alocações restantes
                for allocation in active_allocations:
                    self.release_route(allocation['route'], allocation['wavelength'])
                
                # Calcula probabilidade de bloqueio
                if total_calls > 0:
                    blocking_prob = blocked_calls / total_calls
                else:
                    blocking_prob = 1.0
                    
                results[load].append(blocking_prob)
        
        sim_time = time.time() - start_sim_time
        print(f"    ✓ Simulação concluída em {sim_time:.2f}s")
        
        return results
    
    def simulate_conjunto(self, best_solutions: Dict[Tuple[int, int], List[int]],
                          load_values: List[int],
                          num_simulations: int = 20,
                          calls_per_load: int = 1000) -> Dict[int, Dict[int, List[float]]]:
        """
        Simula TODAS as requisições conjuntamente.
        
        Simulação realista com múltiplas requisições competindo por recursos.
        """
        results = {idx: {load: [] for load in load_values} 
                  for idx in range(len(self.manual_pairs))}
        
        print(f"    Simulando {len(self.manual_pairs)} requisições conjuntamente...")
        start_sim_time = time.time()
        
        for sim in range(num_simulations):
            print(f"      Simulação {sim+1}/{num_simulations}", end='\r')
            
            for load in load_values:
                # Reset da rede
                self.reset_network()
                
                blocked_calls = [0] * len(self.manual_pairs)
                total_calls = calls_per_load
                
                # Lista para controlar alocações ativas
                active_allocations = []
                
                # Taxa de chegada baseada no load
                arrival_rate = load / 100.0
                
                for call_num in range(total_calls):
                    # Simula chegada de chamada baseada no load
                    if random.random() < arrival_rate:
                        # Escolhe requisição aleatória
                        req_idx = random.randint(0, len(self.manual_pairs) - 1)
                        source, target = self.manual_pairs[req_idx]
                        
                        if (source, target) not in best_solutions:
                            blocked_calls[req_idx] += 1
                            continue
                        
                        route = best_solutions[(source, target)]
                        
                        # Tenta alocar a rota
                        allocated_wl = self.allocate_route_with_first_fit(route)
                        
                        if allocated_wl is not None:
                            # Chamada bem-sucedida
                            duration = random.randint(1, 10)
                            active_allocations.append({
                                'req_idx': req_idx,
                                'route': route,
                                'wavelength': allocated_wl,
                                'remaining_time': duration
                            })
                        else:
                            # Chamada bloqueada
                            blocked_calls[req_idx] += 1
                    
                    # Processa alocações ativas
                    completed_allocations = []
                    for i, allocation in enumerate(active_allocations):
                        allocation['remaining_time'] -= 1
                        if allocation['remaining_time'] <= 0:
                            self.release_route(allocation['route'], allocation['wavelength'])
                            completed_allocations.append(i)
                    
                    # Remove alocações completadas
                    for i in sorted(completed_allocations, reverse=True):
                        active_allocations.pop(i)
                
                # Libera alocações restantes
                for allocation in active_allocations:
                    self.release_route(allocation['route'], allocation['wavelength'])
                
                # Armazena resultados
                for req_idx in range(len(self.manual_pairs)):
                    if total_calls > 0:
                        blocking_prob = blocked_calls[req_idx] / total_calls
                    else:
                        blocking_prob = 1.0
                        
                    results[req_idx][load].append(blocking_prob)
        
        sim_time = time.time() - start_sim_time
        print(f"    ✓ Simulação conjunta concluída em {sim_time:.2f}s")
        
        return results
    
    # ========== MÉTODOS PRINCIPAIS ==========
    
    def run_simulation(self, load_values: List[int] = None,
                       num_simulations: int = 20,
                       calls_per_load: int = 1000,
                       output_dir: str = "resultados_phd") -> Dict:
        """
        Executa simulação completa no modo configurado.
        """
        if load_values is None:
            load_values = list(range(1, 201))  # Loads 1-200 (sem pular)
        
        print(f"\n{'='*70}")
        print(f"SIMULAÇÃO AG DOUTORADO - MODO: {self.mode.upper()}")
        print(f"{'='*70}")
        print(f"Parâmetros REALISTAS:")
        print(f"  • Requisições: {len(self.manual_pairs)}")
        print(f"  • Loads: {len(load_values)} (de {min(load_values)} a {max(load_values)}, SEM PULAR)")
        print(f"  • Simulações por load: {num_simulations}")
        print(f"  • Chamadas por load: {calls_per_load}")
        print(f"  • Wavelengths: {self.num_wavelengths}")
        print(f"  • População AG: {self.population_size}")
        print(f"  • Gerações AG: {self.num_generations}")
        print(f"  • k-paths: {self.k}")
        print(f"{'='*70}")
        
        start_total = time.time()
        
        if self.mode == "separado":
            results = self._run_mode_separado(load_values, num_simulations, calls_per_load)
        else:  # conjunto
            results = self._run_mode_conjunto(load_values, num_simulations, calls_per_load)
        
        total_time = time.time() - start_total
        
        # Salva resultados detalhados
        self._save_detailed_results(results, load_values, num_simulations, 
                                   calls_per_load, total_time, output_dir)
        
        # Gera gráficos
        self._plot_results(results, load_values, output_dir)
        
        print(f"\n{'='*70}")
        print(f"SIMULAÇÃO CONCLUÍDA!")
        print(f"Tempo total de execução: {total_time:.2f}s ({total_time/60:.2f} minutos)")
        print(f"Resultados salvos em: {output_dir}/")
        print(f"{'='*70}")
        
        return results
    
    def _run_mode_separado(self, load_values: List[int],
                          num_simulations: int,
                          calls_per_load: int) -> Dict:
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
                load_values, num_simulations, calls_per_load
            )
            
            all_results[idx] = results
            
            # Calcula e mostra estatísticas
            self._print_simulation_stats(idx, results, load_values)
        
        return all_results
    
    def _run_mode_conjunto(self, load_values: List[int],
                          num_simulations: int,
                          calls_per_load: int) -> Dict:
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
            best_solutions, load_values, num_simulations, calls_per_load
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
                              num_simulations: int, calls_per_load: int,
                              total_time: float, output_dir: str):
        """Salva resultados detalhados em arquivos."""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Salva arquivo com TODAS as probabilidades por requisição
        for idx in range(len(self.manual_pairs)):
            source, target = self.manual_pairs[idx]
            filename = f"{output_dir}/todas_probabilidades_req_{idx+1}_{source}_{target}.txt"
            
            with open(filename, 'w') as f:
                f.write(f"# TODAS AS PROBABILIDADES - Requisição {idx+1}: [{source},{target}]\n")
                f.write(f"# Modo: {self.mode}\n")
                f.write(f"# População AG: {self.population_size}\n")
                f.write(f"# Gerações AG: {self.num_generations}\n")
                f.write(f"# k-paths: {self.k}\n")
                f.write(f"# Simulações: {num_simulations}\n")
                f.write(f"# Chamadas por load: {calls_per_load}\n")
                f.write(f"# Wavelengths: {self.num_wavelengths}\n")
                f.write("# " + "="*80 + "\n")
                f.write("# Formato: Load Simulação1 Simulação2 ... SimulaçãoN Média DesvioPadrão\n")
                
                for load in load_values:
                    if idx in results and load in results[idx] and results[idx][load]:
                        probs = results[idx][load]
                        if len(probs) == num_simulations:
                            mean_prob = np.mean(probs)
                            std_prob = np.std(probs) if len(probs) > 1 else 0.0
                            
                            # Escreve todas as probabilidades individuais
                            prob_str = " ".join([f"{p:.8f}" for p in probs])
                            f.write(f"{load:4d} {prob_str} {mean_prob:.8f} {std_prob:.8f}\n")
        
        # 2. Salva arquivo com MÉDIAS por requisição
        for idx in range(len(self.manual_pairs)):
            source, target = self.manual_pairs[idx]
            filename = f"{output_dir}/medias_req_{idx+1}_{source}_{target}.txt"
            
            with open(filename, 'w') as f:
                f.write(f"# MÉDIAS - Requisição {idx+1}: [{source},{target}]\n")
                f.write(f"# Modo: {self.mode}\n")
                f.write("# " + "="*50 + "\n")
                f.write("# Load  Média  DesvioPadrão  Mínimo  Máximo\n")
                
                for load in load_values:
                    if idx in results and load in results[idx] and results[idx][load]:
                        probs = results[idx][load]
                        mean_prob = np.mean(probs)
                        std_prob = np.std(probs) if len(probs) > 1 else 0.0
                        min_prob = min(probs)
                        max_prob = max(probs)
                        
                        f.write(f"{load:4d}  {mean_prob:.8f}  {std_prob:.8f}  "
                               f"{min_prob:.8f}  {max_prob:.8f}\n")
        
        # 3. Salva arquivo de RESUMO GERAL
        summary_file = f"{output_dir}/resumo_geral.txt"
        with open(summary_file, 'w') as f:
            f.write(f"RESUMO GERAL DA SIMULAÇÃO\n")
            f.write("="*80 + "\n")
            f.write(f"Data e hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Modo de operação: {self.mode.upper()}\n")
            f.write(f"Tempo total de execução: {total_time:.2f}s ({total_time/60:.2f} minutos)\n")
            f.write(f"Número de requisições: {len(self.manual_pairs)}\n")
            f.write(f"Loads simulados: {len(load_values)} (de {min(load_values)} a {max(load_values)})\n")
            f.write(f"Simulações por load: {num_simulations}\n")
            f.write(f"Chamadas por load: {calls_per_load}\n")
            f.write(f"Total de chamadas simuladas: {len(load_values) * num_simulations * calls_per_load:,}\n")
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
                        
                        # Probabilidade para load 200 (mais crítico)
                        if 200 in load_values and 200 in results[idx] and results[idx][200]:
                            probs_200 = results[idx][200]
                            mean_200 = np.mean(probs_200)
                            f.write(f"Probabilidade para Load 200: {mean_200:.8f}\n")
                    
                    f.write("\n")
        
        # 4. Salva arquivo de TEMPO DE EXECUÇÃO
        time_file = f"{output_dir}/tempo_execucao_detalhado.txt"
        with open(time_file, 'w') as f:
            f.write(f"TEMPO DE EXECUÇÃO DETALHADO\n")
            f.write("="*60 + "\n")
            f.write(f"Data e hora de início: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tempo total: {total_time:.2f} segundos\n")
            f.write(f"Tempo total: {total_time/60:.2f} minutos\n")
            f.write(f"Tempo total: {total_time/3600:.2f} horas\n")
            f.write(f"Modo: {self.mode}\n")
            f.write(f"Simulações realizadas: {len(load_values) * num_simulations}\n")
            f.write(f"Chamadas totais simuladas: {len(load_values) * num_simulations * calls_per_load:,}\n")
        
        print(f"  📁 Resultados salvos em {output_dir}/")
        print(f"     • todas_probabilidades_req_X.txt - Todas as probabilidades individuais")
        print(f"     • medias_req_X.txt - Médias por requisição")
        print(f"     • resumo_geral.txt - Resumo geral da simulação")
        print(f"     • tempo_execucao_detalhado.txt - Tempo de execução")
    
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
        
        plt.xlabel('Carga (Load)', fontsize=14)
        plt.ylabel('Probabilidade de Bloqueio (%)', fontsize=14)
        plt.title(f'Simulação Realista - Modo {self.mode.upper()}\n'
                 f'{len(self.manual_pairs)} requisições, {len(load_values)} loads, {len(load_values)*20} simulações',
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
    
    print("\nConfiguração da simulação REALISTA:")
    print("  • 20 simulações completas por load")
    print("  • Loads de 1 a 200 (sem pular)")
    print("  • 1000 chamadas por load")
    print("  • 5 requisições simultâneas")
    print("  • Alocação First-Fit com duração aleatória de chamadas")
    
    print("\nEscolha o modo de operação:")
    print("1. Modo SEPARADO - Cada requisição otimizada e simulada individualmente")
    print("2. Modo CONJUNTO - Todas requisições otimizadas e simuladas conjuntamente")
    print("3. Sair")
    
    choice = input("\nDigite sua escolha (1-3): ").strip()
    
    if choice == "1":
        print(f"\n>>> INICIANDO MODO SEPARADO (REALISTA)")
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
            load_values=list(range(1, 201)),  # 1-200 (sem pular)
            num_simulations=20,  # 20 simulações
            calls_per_load=1000,
            output_dir="resultados_separado_realista"
        )
        
    elif choice == "2":
        print(f"\n>>> INICIANDO MODO CONJUNTO (REALISTA)")
        print(">>> Esta execução pode levar algum tempo devido às 20 simulações por load...")
        
        graph = create_nsfnet_graph()
        
        simulator = WDMSimulatorPhD_Completo(
            graph=graph,
            num_wavelengths=40,
            population_size=120,
            num_generations=40,
            k=20,
            mode="conjunto"
        )
        
        simulator.run_simulation(
            load_values=list(range(1, 201)),  # 1-200 (sem pular)
            num_simulations=20,  # 20 simulações
            calls_per_load=1000,
            output_dir="resultados_conjunto_realista"
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