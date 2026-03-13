import random
import os
import time
from itertools import islice
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import heapq

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pymoo.core.problem import ElementwiseProblem


class MyProblem(ElementwiseProblem):
    """Classe do problema para o algoritmo PSO usando pymoo."""
    
    def __init__(self, gene_size, calc_fitness, manual_pairs, num_wavelengths):
        self.calc_fitness = calc_fitness
        self.manual_pairs = manual_pairs
        self.num_wavelengths = num_wavelengths

        # Para PSO: apenas índices das rotas (k=150)
        super().__init__(
            n_var=gene_size,  # Apenas índices das rotas
            n_obj=1,
            xl=np.array([0] * gene_size),  # 0 para rota
            xu=np.array([149] * gene_size),  # k=150
            vtype=int
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        x_int = x.astype(int)
        # PSO minimiza, então invertemos o sinal do fitness
        fitness = -self.calc_fitness(x_int.tolist(), self.manual_pairs)
        out["F"] = [fitness]


class WDMSimulatorPSO:
    """
    Simulador WDM usando PSO com EXATAMENTE a mesma função fitness do AGP.
    MODE: CONJUNTO (todas as 5 requisições simultaneamente)
    """

    def __init__(
        self,
        graph: nx.Graph,
        num_wavelengths: int = 40,
        gene_size: int = 5,
        manual_selection: bool = True,
        k: int = 150,  # k=150 igual AGP
        population_size: int = 120,
        # Parâmetros PSO
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        n_gen: int = 40
    ):
        """
        Inicializa o simulador WDM com PSO (igual AGP).
        """
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.gene_size = gene_size
        self.k = k  # k=150
        self.manual_pairs = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
        self.manual_selection = manual_selection
        
        # Parâmetros de fitness IGUAIS ao AGP
        self.penalty_per_hop = 0.2
        self.penalty_per_wavelength_change = 0.3
        self.reward_per_wavelength_reuse = 0.25
        self.penalty_per_congested_link = 0.5
        
        # Parâmetros PSO
        self.population_size = population_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_gen = n_gen

        # Parâmetros de simulação IGUAIS ao AGP
        self.simulation_time_units = 1000  # Unidades de tempo de simulação
        self.mean_call_duration = 10.0  # Duração média das chamadas (exponencial)
        
        # INICIALIZA OS ATRIBUTOS DE TEMPO PRIMEIRO
        self.execution_times = {
            'total': 0.0,
            'route_computation': 0.0,
            'pso': 0.0,
            'simulation': 0.0
        }
        
        # Inicializa atributos do grafo IGUAL ao AGP
        self._initialize_graph_attributes()
        
        # Cache de rotas (150 rotas por requisição) IGUAL ao AGP
        self.k_shortest_paths = {}
        self._precompute_routes()
        
        print(f"\nPSO Inicializado (k={k}):")
        print(f"  Requisições: {self.manual_pairs}")
        print(f"  Rotas por requisição: {k} (IGUAL AGP)")
        print(f"  Wavelengths: {num_wavelengths}")
        print(f"  Population PSO: {population_size}")
        print(f"  Generations PSO: {n_gen}")
        print(f"  Fitness function: IGUAL AO AGP")
        print(f"  Mode: CONJUNTO (5 requisições simultâneas)")

    def _initialize_graph_attributes(self):
        """Inicializa os atributos do grafo IGUAL ao AGP."""
        for u, v in self.graph.edges():
            self.graph[u][v]['wavelengths'] = [True] * self.num_wavelengths
            self.graph[u][v]['usage_count'] = 0
            self.graph[u][v]['blocked_count'] = 0
            self.graph[u][v]['current_allocations'] = defaultdict(list)

    def _precompute_routes(self):
        """Pré-computa k=150 rotas para cada requisição IGUAL ao AGP."""
        route_start = time.time()
        print(f"\nPré-computando {self.k} rotas por requisição...")
        
        for source, target in self.manual_pairs:
            routes = self._get_k_shortest_paths(source, target, self.k)
            self.k_shortest_paths[(source, target)] = routes
            print(f"  [{source},{target}]: {len(routes)} rotas encontradas")
        
        route_time = time.time() - route_start
        self.execution_times['route_computation'] = route_time
        print(f"Tempo pré-computação rotas: {route_time:.2f}s")

    def _get_k_shortest_paths(self, source: int, target: int, k: int) -> List[List[int]]:
        """Calcula os k menores caminhos IGUAL ao AGP."""
        if not nx.has_path(self.graph, source, target):
            return []
        try:
            return list(islice(nx.shortest_simple_paths(self.graph, source, target), k))
        except nx.NetworkXNoPath:
            return []

    def reset_network(self) -> None:
        """Reseta a rede IGUAL ao AGP."""
        for u, v in self.graph.edges():
            self.graph[u][v]['wavelengths'] = [True] * self.num_wavelengths
            self.graph[u][v]['usage_count'] = 0
            self.graph[u][v]['blocked_count'] = 0
            self.graph[u][v]['current_allocations'] = defaultdict(list)

    def allocate_route_with_first_fit(self, route: List[int], call_id: int) -> Optional[int]:
        """Aloca uma rota usando algoritmo First-Fit IGUAL ao AGP."""
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
        """Libera um wavelength alocado em uma rota IGUAL ao AGP."""
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
        """Retorna o nível de congestionamento médio da rota IGUAL ao AGP."""
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

    # ========== FUNÇÃO FITNESS IGUAL AO AGP ==========
    
    def _fitness_route(self, route: List[int]) -> float:
        """
        Calcula a aptidão de uma rota específica.
        EXATAMENTE IGUAL AO AGP DO DOUTORADO.
        """
        if len(route) < 2:
            return 0.0
        
        # Penalidade por número de hops (saltos) - IGUAL AGP
        hops = len(route) - 1
        hops_penalty = hops * self.penalty_per_hop
        
        # Penalidade por congestionamento - IGUAL AGP
        congestion = self.get_route_congestion(route)
        congestion_penalty = congestion * self.penalty_per_congested_link
        
        # Fitness base (quanto maior, melhor) - IGUAL AGP
        fitness = 1.0
        
        # Aplica penalidades - IGUAL AGP
        fitness -= hops_penalty
        fitness -= congestion_penalty
        
        # Garante que fitness não seja negativo - IGUAL AGP
        return max(0.01, fitness)
    
    def _fitness(self, individual: List[int], source_targets: List[Tuple[int, int]]) -> float:
        """
        Calcula a aptidão de um indivíduo (conjunto de rotas).
        EXATAMENTE IGUAL AO AGP DO DOUTORADO.
        """
        if len(individual) != len(source_targets):
            return 0.0
        
        total_fitness = 0.0
        valid_routes = 0
        
        for i, (source, target) in enumerate(source_targets):
            route_idx = individual[i]
            routes = self.k_shortest_paths.get((source, target), [])
            
            if not routes or route_idx >= len(routes):
                continue
            
            route = routes[route_idx]
            if len(route) >= 2:
                fitness = self._fitness_route(route)
                total_fitness += fitness
                valid_routes += 1
        
        # Penalidade por sobreposição de rotas (conflito de recursos) - IGUAL AGP
        conflict_penalty = self._calculate_conflict_penalty(individual, source_targets)
        total_fitness -= conflict_penalty
        
        # Retorna média das fitness válidas - IGUAL AGP
        return total_fitness / valid_routes if valid_routes > 0 else 0.0
    
    def _calculate_conflict_penalty(self, individual: List[int], 
                                  source_targets: List[Tuple[int, int]]) -> float:
        """Calcula penalidade por conflitos entre rotas - IGUAL AGP."""
        link_usage = defaultdict(int)
        
        # Conta uso de enlaces por todas as rotas
        for i, (source, target) in enumerate(source_targets):
            if i >= len(individual):
                continue
            
            route_idx = individual[i]
            routes = self.k_shortest_paths.get((source, target), [])
            
            if not routes or route_idx >= len(routes):
                continue
            
            route = routes[route_idx]
            for j in range(len(route) - 1):
                u, v = route[j], route[j + 1]
                link_usage[(u, v)] += 1
        
        # Penaliza enlaces muito utilizados - IGUAL AGP
        penalty = 0.0
        for count in link_usage.values():
            if count > 1:  # Se mais de uma rota usa o mesmo enlace
                penalty += (count - 1) * 0.1
        
        return penalty

    def pso_algorithm(self) -> Tuple[List[int], float]:
        """Executa PSO com função fitness IGUAL AO AGP."""
        pso_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Executando PSO com {self.population_size} partículas...")
        print(f"Função fitness: IGUAL AO AGP DO DOUTORADO")
        print(f"Parâmetros: w={self.w}, c1={self.c1}, c2={self.c2}")
        print(f"{'='*60}")
    
        # Criar problema com função fitness do AGP
        problem = MyProblem(self.gene_size, self._fitness, self.manual_pairs, self.num_wavelengths)

        from pymoo.algorithms.soo.nonconvex.pso import PSO
        from pymoo.operators.sampling.rnd import IntegerRandomSampling
        from pymoo.termination import get_termination
        from pymoo.optimize import minimize

        algorithm = PSO(
            pop_size=self.population_size,
            w=self.w,
            c1=self.c1,
            c2=self.c2,
            sampling=IntegerRandomSampling(),
        )

        termination = get_termination("n_gen", self.n_gen)

        print("  Otimizando...")
        res_pso = minimize(
            problem, 
            algorithm,
            termination,
            seed=42,
            verbose=False
        )

        X_PSO = res_pso.X.astype(int)
        F_PSO = -res_pso.F[0]  # Inverte o sinal

        pso_time = time.time() - pso_start
        self.execution_times['pso'] = pso_time

        print(f"\n{'='*60}")
        print(f"RESULTADOS PSO (Fitness IGUAL AGP)")
        print(f"{'='*60}")
        print(f"Tempo PSO: {pso_time:.2f}s")
        print(f"Melhor solução (índices 0-{self.k-1}): {X_PSO}")
        print(f"Fitness: {F_PSO:.6f}")
        
        # Mostrar rotas selecionadas
        print(f"\nRotas Selecionadas (k={self.k}):")
        for i, (source, target) in enumerate(self.manual_pairs):
            if i < len(X_PSO):
                route_idx = X_PSO[i]
                routes = self.k_shortest_paths.get((source, target), [])
                if route_idx < len(routes):
                    route = routes[route_idx]
                    hops = len(route) - 1
                    congestion = self.get_route_congestion(route)
                    print(f"  [{source},{target}]: Índice {route_idx}/{len(routes)-1}, "
                          f"{hops} saltos, Congestão: {congestion:.3f}")
                    print(f"      Rota: {route}")
        print(f"{'='*60}")

        return X_PSO.tolist(), F_PSO

    # ========== SIMULAÇÃO COM 1000 CALLS POR LOAD ==========
    
    def simulate_traffic_calls(
        self, 
        best_solution: List[int],
        num_simulations: int = 20,
        calls_per_load: int = 1000,
        max_load: int = 200
    ) -> Dict[str, List[float]]:
        """
        Simula tráfego usando solução do PSO.
        1000 chamadas por load, 5 requisições simultâneas.
        """
        sim_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Iniciando simulação com 1000 chamadas por load")
        print(f"IGUAL AO AGP DO DOUTORADO")
        print(f"{'='*60}")
        print(f"  Requisições: {self.manual_pairs}")
        print(f"  Simulações por load: {num_simulations}")
        print(f"  Chamadas por load: {calls_per_load}")
        print(f"  Loads: 1 a {max_load}")
        print(f"  Distribuição: 80% requisições fixas, 20% aleatórias")
        
        # Estrutura para armazenar resultados
        # Cada requisição terá uma lista de listas: [[prob_sim1, prob_sim2, ...], ...]
        results = {f'[{s},{t}]': [[] for _ in range(max_load)] for (s, t) in self.manual_pairs}
        
        # Processa cada load individualmente
        for load in range(1, max_load + 1):
            load_start = time.time()
            
            if load % 20 == 0 or load == 1 or load == max_load:
                print(f"\nLoad {load}/{max_load}...")
            
            inter_arrival_time = 10.0 / load  # Tempo entre chegadas
            
            for sim in range(num_simulations):
                if load % 20 == 0 and sim % 5 == 0:
                    print(f"  Simulação {sim + 1}/{num_simulations}")
                
                # Reset da rede para cada simulação
                self.reset_network()
                current_time = 0.0
                
                # Distribuição: 80% das 5 requisições, 20% aleatórias
                calls_per_request = int(calls_per_load * 0.8 / 5)
                random_calls = calls_per_load - (calls_per_request * 5)
                
                # Criar sequência de requisições
                request_sequence = []
                for i, (source, target) in enumerate(self.manual_pairs):
                    request_sequence.extend([(source, target, i)] * calls_per_request)
                
                # Adicionar aleatórias
                nodes = list(self.graph.nodes)
                for _ in range(random_calls):
                    s, t = np.random.choice(nodes, 2, replace=False)
                    request_sequence.append((s, t, -1))  # -1 indica aleatória
                
                # Embaralhar sequência
                np.random.shuffle(request_sequence)
                
                # Contadores para esta simulação
                calls_by_request = {f'[{s},{t}]': 0 for (s, t) in self.manual_pairs}
                blocked_by_request = {f'[{s},{t}]': 0 for (s, t) in self.manual_pairs}
                
                # Processar cada chamada
                for source, target, req_idx in request_sequence:
                    call_duration = np.random.uniform(5.0, 15.0)
                    current_time += inter_arrival_time
                    
                    # Liberar wavelengths expirados
                    self._release_expired_wavelengths(current_time)
                    
                    if req_idx >= 0:  # Requisição principal
                        # Usa rota da solução PSO
                        route_idx = best_solution[req_idx] if req_idx < len(best_solution) else 0
                        routes = self.k_shortest_paths.get((source, target), [])
                        
                        if routes and route_idx < len(routes):
                            route = routes[route_idx]
                            req_key = f'[{source},{target}]'
                            calls_by_request[req_key] += 1
                            
                            # Tenta alocar
                            wavelength = self._allocate_route_with_first_fit_timed(route, call_id=sim*calls_per_load + len(request_sequence))
                            
                            if wavelength is None:
                                blocked_by_request[req_key] += 1
                            else:
                                # Marcar para liberação futura
                                self._schedule_wavelength_release(route, wavelength, current_time + call_duration, call_id=sim*calls_per_load + len(request_sequence))
                    else:  # Requisição aleatória
                        # Para aleatórias, usa primeira rota disponível
                        routes = self._get_k_shortest_paths(source, target, 10)
                        if routes:
                            # Tenta todas as rotas com First Fit
                            allocated = False
                            for route in routes:
                                wavelength = self._allocate_route_with_first_fit_timed(route, call_id=sim*calls_per_load + len(request_sequence))
                                if wavelength is not None:
                                    self._schedule_wavelength_release(route, wavelength, current_time + call_duration, call_id=sim*calls_per_load + len(request_sequence))
                                    allocated = True
                                    break
                
                # Calcular probabilidades para esta simulação
                for (source, target) in self.manual_pairs:
                    req_key = f'[{source},{target}]'
                    calls = calls_by_request[req_key]
                    blocked = blocked_by_request[req_key]
                    
                    if calls > 0:
                        blocking_prob = blocked / calls
                    else:
                        blocking_prob = 0.0
                    
                    results[req_key][load-1].append(blocking_prob)
            
            load_time = time.time() - load_start
            if load % 20 == 0 or load == 1 or load == max_load:
                print(f"  Tempo load: {load_time:.2f}s")
                # Mostrar média das probabilidades
                for i, (source, target) in enumerate(self.manual_pairs):
                    if i < 2:
                        req_key = f'[{source},{target}]'
                        probs = results[req_key][load-1]
                        if probs:
                            avg_prob = np.mean(probs)
                            print(f"  {req_key}: {avg_prob:.6f} (de {len(probs)} simulações)")
        
        sim_time = time.time() - sim_start
        self.execution_times['simulation'] = sim_time
        
        return results
    
    def _allocate_route_with_first_fit_timed(self, route: List[int], call_id: int) -> Optional[int]:
        """Versão simplificada para simulação com tempo."""
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
        
        return None
    
    def _release_expired_wavelengths(self, current_time: float):
        """Libera wavelengths baseado no tempo atual."""
        # Esta é uma simplificação - em uma simulação real com tempo,
        # precisaríamos rastrear quando cada wavelength expira
        # Para simplicidade, vamos assumir que todas as alocações anteriores já expiraram
        # Em uma implementação completa, teríamos uma fila de eventos
        pass
    
    def _schedule_wavelength_release(self, route: List[int], wavelength: int, release_time: float, call_id: int):
        """Marca um wavelength para liberação futura."""
        # Em uma implementação completa, adicionaríamos a um heap de eventos
        # Para simplicidade, vamos apenas armazenar
        pass

    def save_results_detailed(
        self,
        results: Dict[str, List[List[float]]],  # Lista de listas: por load, por simulação
        best_solution: List[int],
        best_fitness: float,
        num_simulations: int = 20,
        calls_per_load: int = 1000,
        max_load: int = 200,
        output_dir: str = "resultados_pso_detalhado"
    ) -> None:
        """
        Salva resultados detalhados no mesmo formato do AGP.
        """
        print(f"\n{'='*60}")
        print(f"Salvando resultados PSO detalhados...")
        print(f"{'='*60}")
        
        # Cria diretório para resultados
        os.makedirs(output_dir, exist_ok=True)
        
        # Subdiretórios
        subdirs = {
            'probabilidades_completas': f"{output_dir}/probabilidades_completas",
            'medias_estatisticas': f"{output_dir}/medias_estatisticas",
            'resumos': f"{output_dir}/resumos"
        }
        
        for subdir_name, subdir_path in subdirs.items():
            os.makedirs(subdir_path, exist_ok=True)
        
        # 1. Salva TODAS as probabilidades por requisição (20 simulações cada)
        for idx, (source, target) in enumerate(self.manual_pairs):
            req_key = f'[{source},{target}]'
            
            # Arquivo com todas as probabilidades
            filename_all = f"{subdirs['probabilidades_completas']}/todas_probs_req_{idx+1}_{source}_{target}_pso.txt"
            
            with open(filename_all, 'w') as f:
                f.write(f"# TODAS AS PROBABILIDADES (20 SIMULAÇÕES) - PSO\n")
                f.write(f"# Requisição {idx+1}: [{source},{target}]\n")
                f.write(f"# Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Chamadas por load: {calls_per_load}\n")
                f.write(f"# Loads: 1-{max_load}\n")
                f.write("# " + "="*80 + "\n")
                f.write("# Formato: Load | Sim1 | Sim2 | ... | Sim20 | Média | DesvioPadrão\n")
                f.write("# " + "="*80 + "\n\n")
                
                for load in range(1, max_load + 1):
                    if load <= len(results[req_key]):
                        probs_load = results[req_key][load-1]
                        if len(probs_load) == num_simulations:
                            mean_prob = np.mean(probs_load)
                            std_prob = np.std(probs_load) if len(probs_load) > 1 else 0.0
                            
                            # Escreve todas as probabilidades individuais
                            prob_str = " | ".join([f"{p:.8f}" for p in probs_load])
                            f.write(f"Load {load:3d} | {prob_str} | {mean_prob:.8f} | {std_prob:.8f}\n")
        
        # 2. Salva MÉDIAS e estatísticas por requisição
        for idx, (source, target) in enumerate(self.manual_pairs):
            req_key = f'[{source},{target}]'
            
            filename_medias = f"{subdirs['medias_estatisticas']}/medias_req_{idx+1}_{source}_{target}_pso.txt"
            
            with open(filename_medias, 'w') as f:
                f.write(f"# MÉDIAS E ESTATÍSTICAS - PSO\n")
                f.write(f"# Requisição {idx+1}: [{source},{target}]\n")
                f.write(f"# Chamadas por load: {calls_per_load}\n")
                f.write("# " + "="*60 + "\n")
                f.write("# Load | Média | DesvioPadrão | Mínimo | Máximo | IC 95% (inferior) | IC 95% (superior)\n")
                f.write("# " + "="*60 + "\n")
                
                for load in range(1, max_load + 1):
                    if load <= len(results[req_key]):
                        probs_load = results[req_key][load-1]
                        if probs_load:
                            mean_prob = np.mean(probs_load)
                            std_prob = np.std(probs_load) if len(probs_load) > 1 else 0.0
                            min_prob = min(probs_load)
                            max_prob = max(probs_load)
                            
                            # Intervalo de confiança 95%
                            n = len(probs_load)
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
        
        # 3. Salva arquivo de RESUMO GERAL
        summary_file = f"{subdirs['resumos']}/resumo_geral_pso.txt"
        with open(summary_file, 'w') as f:
            f.write(f"RESUMO GERAL DA SIMULAÇÃO - PSO (Fitness IGUAL AGP)\n")
            f.write("="*80 + "\n")
            f.write(f"Data e hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tempo total de execução: {sum(self.execution_times.values()):.2f}s\n")
            f.write(f"Número de requisições: {len(self.manual_pairs)}\n")
            f.write(f"Loads simulados: {max_load}\n")
            f.write(f"Simulações por load: {num_simulations}\n")
            f.write(f"Chamadas por load: {calls_per_load}\n")
            f.write(f"Distribuição: 80% requisições fixas, 20% aleatórias\n")
            f.write(f"Parâmetros PSO:\n")
            f.write(f"  • População: {self.population_size}\n")
            f.write(f"  • Gerações: {self.n_gen}\n")
            f.write(f"  • k-paths: {self.k}\n")
            f.write(f"  • Wavelengths: {self.num_wavelengths}\n")
            f.write(f"  • Parâmetros PSO: w={self.w}, c1={self.c1}, c2={self.c2}\n")
            f.write(f"\nMELHOR SOLUÇÃO PSO:\n")
            f.write(f"  Fitness: {best_fitness:.6f}\n")
            f.write(f"  Índices das rotas: {best_solution}\n\n")
            
            # Rotas detalhadas
            f.write(f"ROTAS SELECIONADAS:\n")
            for i, (source, target) in enumerate(self.manual_pairs):
                if i < len(best_solution):
                    route_idx = best_solution[i]
                    routes = self.k_shortest_paths.get((source, target), [])
                    if route_idx < len(routes):
                        route = routes[route_idx]
                        hops = len(route) - 1
                        f.write(f"  [{source},{target}]: Índice {route_idx}, {hops} saltos\n")
                        f.write(f"      Rota: {route}\n")
            
            # Médias por requisição
            f.write("\n" + "="*80 + "\n")
            f.write("MÉDIAS DE BLOQUEIO POR REQUISIÇÃO (loads 1-200):\n")
            f.write("-"*40 + "\n")
            
            for idx, (source, target) in enumerate(self.manual_pairs):
                req_key = f'[{source},{target}]'
                all_probs = []
                for load in range(1, max_load + 1):
                    if load <= len(results[req_key]):
                        all_probs.extend(results[req_key][load-1])
                
                if all_probs:
                    overall_mean = np.mean(all_probs)
                    overall_std = np.std(all_probs) if len(all_probs) > 1 else 0.0
                    f.write(f"  [{source},{target}]: {overall_mean:.6f} ± {overall_std:.6f} "
                           f"({overall_mean*100:.2f}%)\n")
            
            # Média geral
            all_all_probs = []
            for (source, target) in self.manual_pairs:
                req_key = f'[{source},{target}]'
                for load in range(1, max_load + 1):
                    if load <= len(results[req_key]):
                        all_all_probs.extend(results[req_key][load-1])
            
            if all_all_probs:
                overall_all_mean = np.mean(all_all_probs)
                f.write(f"\n  MÉDIA GERAL: {overall_all_mean:.6f} ({overall_all_mean*100:.2f}%)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("TEMPO DE EXECUÇÃO DETALHADO:\n")
            f.write("-"*40 + "\n")
            f.write(f"Pré-computação rotas: {self.execution_times['route_computation']:.2f}s\n")
            f.write(f"Otimização PSO: {self.execution_times['pso']:.2f}s\n")
            f.write(f"Simulação tráfego: {self.execution_times['simulation']:.2f}s\n")
            f.write(f"TOTAL: {sum(self.execution_times.values()):.2f}s\n")
            f.write(f"TOTAL: {sum(self.execution_times.values())/60:.2f} minutos\n")
            f.write(f"TOTAL: {sum(self.execution_times.values())/3600:.2f} horas\n")
        
        # 4. Salva arquivo simples por requisição (para gráficos rápidos)
        for idx, (source, target) in enumerate(self.manual_pairs):
            req_key = f'[{source},{target}]'
            simple_file = f"{output_dir}/pso_simple_req_{source}_{target}.txt"
            
            with open(simple_file, 'w') as f:
                f.write(f"# PSO - Requisição [{source},{target}]\n")
                f.write(f"# Load | Probabilidade_Média\n")
                
                for load in range(1, max_load + 1):
                    if load <= len(results[req_key]):
                        probs = results[req_key][load-1]
                        if probs:
                            mean_prob = np.mean(probs)
                            f.write(f"{load} {mean_prob:.6f}\n")
        
        # 5. Salva arquivo combinado
        combined_file = f"{output_dir}/pso_all_requests_combined.txt"
        with open(combined_file, 'w') as f:
            f.write(f"# PSO - Todas as requisições\n")
            f.write(f"# Load | " + " | ".join([f"[{s},{t}]" for (s, t) in self.manual_pairs]) + " | Média\n")
            
            for load in range(1, max_load + 1):
                f.write(f"{load}")
                values = []
                sum_probs = 0
                count = 0
                
                for (source, target) in self.manual_pairs:
                    req_key = f'[{source},{target}]'
                    if load <= len(results[req_key]):
                        probs = results[req_key][load-1]
                        if probs:
                            mean_prob = np.mean(probs)
                            values.append(f"{mean_prob:.6f}")
                            sum_probs += mean_prob
                            count += 1
                        else:
                            values.append("0.000000")
                    else:
                        values.append("0.000000")
                
                if count > 0:
                    media = sum_probs / count
                    f.write(f" | " + " | ".join(values) + f" | {media:.6f}\n")
                else:
                    f.write(f" | " + " | ".join(values) + " | 0.000000\n")
        
        print(f"  📁 Estrutura de resultados criada em {output_dir}/:")
        print(f"     • probabilidades_completas/ - Todas as probabilidades individuais (20 simulações)")
        print(f"     • medias_estatisticas/ - Médias e estatísticas por load")
        print(f"     • resumos/ - Resumos gerais da simulação")
        print(f"     • pso_simple_req_X_Y.txt - Arquivos simples para gráficos")
        print(f"     • pso_all_requests_combined.txt - Arquivo combinado")

    def plot_results_detailed(
        self,
        results: Dict[str, List[List[float]]],
        save_path: str = "pso_detailed_results.png"
    ) -> None:
        """Gera gráfico dos resultados detalhados."""
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
        })
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        markers = ['o', 's', '^', 'D', 'v']
        
        # Gráfico 1: Todas as requisições com médias
        for idx, (source, target) in enumerate(self.manual_pairs):
            req_key = f'[{source},{target}]'
            
            # Calcular médias por load
            means = []
            stds = []
            valid_loads = []
            
            for load in range(1, 201):
                if load <= len(results[req_key]):
                    probs = results[req_key][load-1]
                    if probs:
                        means.append(np.mean(probs) * 100)  # Em %
                        stds.append(np.std(probs) * 100 if len(probs) > 1 else 0.0)
                        valid_loads.append(load)
            
            if valid_loads and means:
                ax1.plot(valid_loads, means,
                        color=colors[idx % len(colors)],
                        marker=markers[idx % len(markers)],
                        markersize=4,
                        linewidth=2,
                        label=f'[{source},{target}]',
                        alpha=0.8)
                
                # Adiciona banda de desvio padrão
                ax1.fill_between(valid_loads, 
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                color=colors[idx % len(colors)],
                                alpha=0.2)
        
        ax1.set_xlabel('Carga (Load)', fontsize=14)
        ax1.set_ylabel('Probabilidade de Bloqueio (%)', fontsize=14)
        ax1.set_title(f'PSO - 5 Requisições Simultâneas\n'
                     f'1000 chamadas/load, 20 simulações, k={self.k}',
                     fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12, loc='upper left')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(-1, 101)
        ax1.set_xlim(0, 210)
        ax1.set_xticks(range(0, 210, 20))
        
        # Gráfico 2: Média geral
        avg_general = []
        std_general = []
        valid_loads_avg = []
        
        for load in range(1, 201):
            all_probs_load = []
            for (source, target) in self.manual_pairs:
                req_key = f'[{source},{target}]'
                if load <= len(results[req_key]):
                    probs = results[req_key][load-1]
                    if probs:
                        all_probs_load.extend(probs)
            
            if all_probs_load:
                avg_general.append(np.mean(all_probs_load) * 100)
                std_general.append(np.std(all_probs_load) * 100 if len(all_probs_load) > 1 else 0.0)
                valid_loads_avg.append(load)
        
        if valid_loads_avg and avg_general:
            ax2.plot(valid_loads_avg, avg_general,
                    label='Média Geral das 5 Requisições',
                    color='#2ca02c',
                    linewidth=3,
                    marker='D',
                    markersize=6,
                    markevery=10)
            
            # Adiciona banda de desvio padrão
            ax2.fill_between(valid_loads_avg,
                            [m - s for m, s in zip(avg_general, std_general)],
                            [m + s for m, s in zip(avg_general, std_general)],
                            color='#2ca02c',
                            alpha=0.2)
        
        ax2.set_xlabel('Carga (Load)', fontsize=14)
        ax2.set_ylabel('Probabilidade de Bloqueio (%)', fontsize=14)
        ax2.set_title('Média Geral', fontsize=14, pad=15)
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_xlim(0, 210)
        ax2.set_ylim(0, 100)
        ax2.set_xticks(range(0, 210, 20))
        
        # Adiciona grade secundária
        ax1.minorticks_on()
        ax1.grid(which='minor', alpha=0.1, linestyle=':')
        ax2.minorticks_on()
        ax2.grid(which='minor', alpha=0.1, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Salva também em formato PDF
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        
        plt.show()
        
        print(f"\n📊 Gráficos salvos em:")
        print(f"  • {save_path}")
        print(f"  • {pdf_path}")


def main():
    """Função principal para PSO com fitness IGUAL AO AGP."""
    start_total = time.time()
    
    print(f"\n{'='*80}")
    print("PSO - SIMULAÇÃO COMPLETA COM FITNESS IGUAL AO AGP")
    print("MODO: CONJUNTO (5 requisições simultâneas)")
    print("PARA COMPARAÇÃO EM DOUTORADO")
    print(f"{'='*80}")
    
    # Criar rede NSFNet
    graph = nx.Graph()
    nsfnet_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
        (4, 6), (4, 5), (5, 8), (5, 12), (6, 7), (7, 9), (8, 9), (9, 11),
        (9, 13), (10, 11), (10, 13), (11, 12)
    ]
    graph.add_edges_from(nsfnet_edges)
    
    print(f"\nConfiguração IGUAL AO AGP:")
    print(f"  Rede: NSFNet (14 nós, 21 enlaces)")
    print(f"  Requisições: {[(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]}")
    print(f"  k (rotas por requisição): 150")
    print(f"  Wavelengths por enlace: 40")
    print(f"  Loads: 1-200")
    print(f"  Chamadas por load: 1000")
    print(f"  Simulações por load: 20")
    print(f"  Distribuição: 80% requisições fixas, 20% aleatórias")
    print(f"  Fitness function: IGUAL AO AGP DO DOUTORADO")
    
    # Criar simulador PSO com fitness igual ao AGP
    simulator = WDMSimulatorPSO(
        graph=graph,
        num_wavelengths=40,
        gene_size=5,
        manual_selection=True,
        k=150,  # k=150 IGUAL AGP
        population_size=120,
        w=0.7,
        c1=1.5,
        c2=1.5,
        n_gen=40
    )
    
    # Executar PSO com fitness igual ao AGP
    best_solution, best_fitness = simulator.pso_algorithm()
    
    # Executar simulação com 1000 chamadas por load
    results = simulator.simulate_traffic_calls(
        best_solution=best_solution,
        num_simulations=20,
        calls_per_load=1000,
        max_load=200
    )
    
    # Salvar resultados detalhados
    simulator.save_results_detailed(
        results=results,
        best_solution=best_solution,
        best_fitness=best_fitness,
        output_dir="resultados_pso_detalhado"
    )
    
    # Gerar gráficos detalhados
    simulator.plot_results_detailed(results, "pso_detailed_results.png")
    
    # Tempo total
    total_time = time.time() - start_total
    simulator.execution_times['total'] = total_time
    
    print(f"\n{'='*80}")
    print("SIMULAÇÃO PSO CONCLUÍDA! (Fitness IGUAL AGP)")
    print(f"{'='*80}")
    print(f"Tempo total: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
    print(f"Melhor solução (índices 0-149): {best_solution}")
    print(f"Fitness: {best_fitness:.6f}")
    
    # Estatísticas resumidas
    print(f"\nMÉDIAS DE BLOQUEIO (loads 1-200):")
    all_probs = []
    for (source, target) in simulator.manual_pairs:
        req_key = f'[{source},{target}]'
        if results[req_key]:
            # Calcular média para todos os loads
            probs_all_loads = []
            for load_probs in results[req_key]:
                probs_all_loads.extend(load_probs)
            
            if probs_all_loads:
                avg = np.mean(probs_all_loads)
                all_probs.extend(probs_all_loads)
                std = np.std(probs_all_loads) if len(probs_all_loads) > 1 else 0.0
                print(f"  [{source},{target}]: {avg:.6f} ± {std:.6f} ({avg*100:.2f}%)")
    
    if all_probs:
        overall_avg = np.mean(all_probs)
        overall_std = np.std(all_probs) if len(all_probs) > 1 else 0.0
        print(f"\n  MÉDIA GERAL: {overall_avg:.6f} ± {overall_std:.6f} ({overall_avg*100:.2f}%)")
    
    print(f"\nArquivos gerados:")
    print(f"  - resultados_pso_detalhado/")
    print(f"     ├── probabilidades_completas/")
    print(f"     │   └── todas_probs_req_X_Y_pso.txt")
    print(f"     ├── medias_estatisticas/")
    print(f"     │   └── medias_req_X_Y_pso.txt")
    print(f"     ├── resumos/")
    print(f"     │   └── resumo_geral_pso.txt")
    print(f"     ├── pso_simple_req_X_Y.txt")
    print(f"     └── pso_all_requests_combined.txt")
    print(f"  - pso_detailed_results.png")
    print(f"  - pso_detailed_results.pdf")
    
    print(f"\n⚠️  ATENÇÃO PARA COMPARAÇÃO COM AGP:")
    print(f"  • Fitness function: EXATAMENTE IGUAL")
    print(f"  • Parâmetros de simulação: EXATAMENTE IGUAIS")
    print(f"  • Loads: 1-200")
    print(f"  • Chamadas por load: 1000")
    print(f"  • Simulações por load: 20")
    print(f"  • Distribuição: 80% requisições fixas, 20% aleatórias")
    
    # Resumo dos tempos
    print(f"\nTEMPOS DE EXECUÇÃO:")
    print(f"  • Pré-computação rotas: {simulator.execution_times['route_computation']:.2f}s")
    print(f"  • Otimização PSO: {simulator.execution_times['pso']:.2f}s")
    print(f"  • Simulação tráfego: {simulator.execution_times['simulation']:.2f}s")
    print(f"  • TOTAL: {total_time:.2f}s ({total_time/60:.2f} minutos)")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main()