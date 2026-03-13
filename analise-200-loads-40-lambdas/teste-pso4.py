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
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize


class WDMPsoGlobalOptimizer:
    """
    PSO para otimização global de RWA (alinhado com AGP/DE).
    Encontra a melhor combinação de rotas para as 5 requisições.
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        num_wavelengths: int = 40,
        requests: List[Tuple[int, int]] = None,
        k: int = 2,  # 2 rotas por requisição como AGP/DE
        population_size: int = 120,
        n_gen: int = 40,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5
    ):
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.requests = requests if requests else [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
        self.k = k
        self.population_size = population_size
        self.n_gen = n_gen
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Cache de rotas
        self.route_cache = {}
        self._precompute_routes()
        
        # Melhor solução encontrada
        self.best_solution = None
        self.best_fitness = 0.0
        
        print(f"\nPSO Global Optimizer Inicializado:")
        print(f"  Requisições: {self.requests}")
        print(f"  Rotas por requisição: {k}")
        print(f"  Population: {population_size}, Generations: {n_gen}")
    
    def _precompute_routes(self):
        """Pré-computa k rotas para cada requisição."""
        print("\nPré-computando rotas...")
        for source, target in self.requests:
            routes = self._get_k_shortest_paths(source, target, self.k)
            self.route_cache[(source, target)] = routes
            print(f"  [{source},{target}]: {len(routes)} rotas encontradas")
    
    def _get_k_shortest_paths(self, source: int, target: int, k: int) -> List[List[int]]:
        """Calcula os k menores caminhos."""
        if not nx.has_path(self.graph, source, target):
            return []
        try:
            return list(islice(nx.shortest_simple_paths(self.graph, source, target), k))
        except nx.NetworkXNoPath:
            return []
    
    def get_route_from_solution(self, solution: List[int]) -> Dict[Tuple[int, int], List[int]]:
        """
        Converte solução (lista de índices) em dicionário de rotas.
        """
        routes = {}
        for i, (source, target) in enumerate(self.requests):
            if i >= len(solution):
                continue
            route_idx = solution[i]
            available_routes = self.route_cache.get((source, target), [])
            if available_routes and route_idx < len(available_routes):
                routes[(source, target)] = available_routes[route_idx]
            elif available_routes:
                routes[(source, target)] = available_routes[0]  # Fallback
        return routes
    
    def evaluate_solution(self, solution: List[int]) -> float:
        """
        Avalia uma solução (combinação de rotas) para as 5 requisições.
        Fitness baseado em:
        1. Número de saltos total (menor é melhor)
        2. Balanceamento de carga (menos sobreposição é melhor)
        """
        if len(solution) != len(self.requests):
            return 0.0
        
        total_hops = 0
        edge_usage = defaultdict(int)
        
        for i, (source, target) in enumerate(self.requests):
            route_idx = solution[i]
            available_routes = self.route_cache.get((source, target), [])
            
            if not available_routes or route_idx >= len(available_routes):
                return 0.0  # Solução inválida
            
            route = available_routes[route_idx]
            total_hops += len(route) - 1
            
            # Conta uso de enlaces
            for j in range(len(route) - 1):
                u, v = route[j], route[j + 1]
                edge = (min(u, v), max(u, v))
                edge_usage[edge] += 1
        
        # Penaliza enlaces muito congestionados
        congestion_penalty = 0
        for count in edge_usage.values():
            if count > 1:  # Enlace usado por múltiplas rotas
                congestion_penalty += (count - 1) * 10
        
        # Fitness = 1/(total_hops + congestion_penalty + 1)
        fitness = 1.0 / (total_hops + congestion_penalty + 1)
        
        return fitness
    
    def run_optimization(self) -> Tuple[List[int], float]:
        """
        Executa PSO para encontrar a melhor combinação de rotas.
        """
        print("\n" + "="*60)
        print("EXECUTANDO PSO - OTIMIZAÇÃO GLOBAL")
        print("="*60)
        
        class PsoRwaProblem(ElementwiseProblem):
            def __init__(self, optimizer):
                self.optimizer = optimizer
                super().__init__(
                    n_var=len(optimizer.requests),  # 5 variáveis (uma por requisição)
                    n_obj=1,
                    xl=np.array([0] * len(optimizer.requests)),
                    xu=np.array([optimizer.k - 1] * len(optimizer.requests)),
                    vtype=int
                )
            
            def _evaluate(self, x, out, *args, **kwargs):
                x_int = x.astype(int)
                fitness = self.optimizer.evaluate_solution(x_int.tolist())
                out["F"] = [-fitness]  # PSO minimiza, invertemos
        
        # Criar e configurar problema
        problem = PsoRwaProblem(self)
        
        algorithm = PSO(
            pop_size=self.population_size,
            sampling=IntegerRandomSampling(),
            w=self.w,
            c1=self.c1,
            c2=self.c2,
            seed=42
        )
        
        termination = get_termination("n_gen", self.n_gen)
        
        # Executar otimização
        print(f"\nIniciando PSO com {self.population_size} partículas, {self.n_gen} gerações...")
        start_time = time.time()
        
        res = minimize(
            problem,
            algorithm,
            termination,
            verbose=False
        )
        
        exec_time = time.time() - start_time
        
        # Extrair resultados
        self.best_solution = res.X.astype(int).tolist()
        self.best_fitness = -res.F[0]  # Inverter de volta
        
        print(f"\n{'='*50}")
        print("RESULTADOS PSO")
        print(f"{'='*50}")
        print(f"Tempo de execução: {exec_time:.2f}s")
        print(f"Melhor solução: {self.best_solution}")
        print(f"Fitness: {self.best_fitness:.6f}")
        
        # Mostrar rotas selecionadas
        print("\nRotas selecionadas:")
        routes_dict = self.get_route_from_solution(self.best_solution)
        for (source, target), route in routes_dict.items():
            hops = len(route) - 1
            print(f"  [{source},{target}]: {route} (saltos: {hops})")
        
        print(f"{'='*50}")
        
        return self.best_solution, self.best_fitness


class WDMPsoSimulator:
    """
    Simulador WDM que usa a solução do PSO (otimização global).
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        num_wavelengths: int = 40,
        requests: List[Tuple[int, int]] = None,
        pso_solution: List[int] = None
    ):
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.requests = requests if requests else [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
        
        # Cache de rotas (2 por requisição)
        self.route_cache = {}
        self._precompute_routes()
        
        # Solução do PSO
        self.pso_solution = pso_solution
        
        # Estrutura de alocação
        self.wavelength_allocation = {}
        self.call_records = []
        self.current_time = 0.0
        
        self.reset_network()
    
    def _precompute_routes(self):
        """Pré-computa 2 rotas para cada requisição."""
        for source, target in self.requests:
            self.route_cache[(source, target)] = self._get_k_shortest_paths(source, target, 2)
    
    def _get_k_shortest_paths(self, source: int, target: int, k: int) -> List[List[int]]:
        """Calcula os k menores caminhos."""
        if not nx.has_path(self.graph, source, target):
            return []
        try:
            return list(islice(nx.shortest_simple_paths(self.graph, source, target), k))
        except nx.NetworkXNoPath:
            return []
    
    def reset_network(self) -> None:
        """Reseta o estado da rede."""
        for u, v in self.graph.edges:
            edge = (min(u, v), max(u, v))
            self.wavelength_allocation[edge] = {}
        self.current_time = 0.0
        self.call_records = []
    
    def _release_expired_wavelengths(self) -> None:
        """Libera wavelengths expirados."""
        for edge in self.wavelength_allocation:
            expired = [wl for wl, end_time in self.wavelength_allocation[edge].items() 
                      if end_time <= self.current_time]
            for wl in expired:
                del self.wavelength_allocation[edge][wl]
    
    def get_route_for_request(self, source: int, target: int) -> Optional[List[int]]:
        """
        Retorna a rota para uma requisição baseada na solução do PSO.
        Se não tiver solução do PSO ou for requisição aleatória, escolhe aleatoriamente.
        """
        # Verifica se é uma das 5 requisições principais
        req_idx = -1
        for i, (s, t) in enumerate(self.requests):
            if s == source and t == target:
                req_idx = i
                break
        
        # Se não for requisição principal ou não tem solução PSO, escolhe aleatoriamente
        if req_idx == -1 or self.pso_solution is None or req_idx >= len(self.pso_solution):
            routes = self.route_cache.get((source, target), [])
            return random.choice(routes) if routes else None
        
        # Usa a rota da solução PSO
        route_idx = self.pso_solution[req_idx]
        routes = self.route_cache.get((source, target), [])
        
        if not routes or route_idx >= len(routes):
            return routes[0] if routes else None
        
        return routes[route_idx]
    
    def first_fit_allocation(self, route: List[int], call_duration: float) -> bool:
        """
        Tenta alocar usando First Fit.
        Retorna True se conseguir, False se bloquear.
        """
        if not route:
            return False
        
        end_time = self.current_time + call_duration
        
        # Primeiro verifica disponibilidade
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            edge = (min(u, v), max(u, v))
            
            if len(self.wavelength_allocation.get(edge, {})) >= self.num_wavelengths:
                return False
        
        # Se todos têm espaço, aloca
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            edge = (min(u, v), max(u, v))
            
            # Encontra primeiro wavelength disponível
            for wl in range(self.num_wavelengths):
                if wl not in self.wavelength_allocation[edge]:
                    self.wavelength_allocation[edge][wl] = end_time
                    break
        
        return True
    
    def process_call(self, source: int, target: int, call_duration: float) -> bool:
        """
        Processa uma chamada.
        Retorna True se bloqueado, False se aceito.
        """
        self._release_expired_wavelengths()
        
        # Obtém rota baseada na solução PSO
        route = self.get_route_for_request(source, target)
        
        if not route:
            return True  # Bloqueado (sem rota)
        
        # Tenta alocar
        success = self.first_fit_allocation(route, call_duration)
        
        # Registra chamada (para estatísticas)
        self.call_records.append({
            'source': source,
            'target': target,
            'route': route,
            'blocked': not success,
            'time': self.current_time,
            'duration': call_duration
        })
        
        return not success
    
    def simulate_traffic(self, num_runs: int = 20,
                        calls_per_load: int = 1000,
                        call_duration_range: Tuple[float, float] = (5.0, 15.0),
                        max_load: int = 200) -> Dict[str, List[float]]:
        """
        Simula tráfego usando a solução do PSO.
        """
        print("\n" + "="*60)
        print("SIMULAÇÃO DE TRÁFEGO COM SOLUÇÃO PSO")
        print(f"Loads: 1 a {max_load}")
        print(f"Chamadas por load: {calls_per_load}")
        print(f"Simulações por load: {num_runs}")
        print("="*60)
        
        results = {f'[{s},{t}]': [] for (s, t) in self.requests}
        
        for load in range(1, max_load + 1):
            load_start = time.time()
            
            if load % 20 == 0 or load == 1 or load == max_load:
                print(f"\nLoad {load}/{max_load}...")
            
            inter_arrival_time = 10.0 / load
            load_results = {f'[{s},{t}]': {'calls': 0, 'blocked': 0} for (s, t) in self.requests}
            
            for run in range(num_runs):
                self.reset_network()
                self.current_time = 0.0
                
                # Preparar distribuição de chamadas (80% das 5 requisições, 20% aleatórias)
                calls_per_request = int(calls_per_load * 0.8 / 5)  # 16% cada
                random_calls = calls_per_load - (calls_per_request * 5)
                
                # Criar sequência de requisições
                request_sequence = []
                for i, (source, target) in enumerate(self.requests):
                    request_sequence.extend([(source, target, i)] * calls_per_request)
                
                # Adicionar aleatórias
                nodes = list(self.graph.nodes)
                for _ in range(random_calls):
                    s, t = np.random.choice(nodes, 2, replace=False)
                    request_sequence.append((s, t, -1))
                
                np.random.shuffle(request_sequence)
                
                # Processar chamadas
                for source, target, req_idx in request_sequence:
                    call_duration = np.random.uniform(call_duration_range[0], call_duration_range[1])
                    self.current_time += inter_arrival_time
                    
                    blocked = self.process_call(source, target, call_duration)
                    
                    if req_idx >= 0:
                        req_key = f'[{source},{target}]'
                        load_results[req_key]['calls'] += 1
                        if blocked:
                            load_results[req_key]['blocked'] += 1
            
            # Calcular probabilidades para este load
            for (source, target) in self.requests:
                req_key = f'[{source},{target}]'
                calls = load_results[req_key]['calls']
                blocked = load_results[req_key]['blocked']
                
                if calls > 0:
                    blocking_prob = blocked / calls
                else:
                    blocking_prob = 0.0
                
                results[req_key].append(blocking_prob)
            
            load_duration = time.time() - load_start
            if load % 20 == 0 or load == 1 or load == max_load:
                print(f"  Tempo: {load_duration:.2f}s")
                # Mostrar algumas estatísticas
                for i, (source, target) in enumerate(self.requests):
                    if i < 2:  # Mostrar apenas 2
                        req_key = f'[{source},{target}]'
                        print(f"  {req_key}: {results[req_key][-1]:.6f}")
        
        return results
    
    def save_results(self, results: Dict[str, List[float]], 
                    output_prefix: str = "pso_global"):
        """
        Salva resultados em arquivos.
        """
        max_load = len(results[next(iter(results.keys()))])
        
        # Salvar por requisição
        for (source, target) in self.requests:
            req_key = f'[{source},{target}]'
            output_file = f"{output_prefix}_req_{source}_{target}.txt"
            
            with open(output_file, "w") as f:
                f.write(f"=== PSO GLOBAL - REQUISIÇÃO [{source},{target}] ===\n\n")
                f.write("PARÂMETROS:\n")
                f.write(f"  Lambdas: {self.num_wavelengths}\n")
                f.write(f"  Loads: 1 a {max_load}\n")
                f.write(f"  Chamadas por load: 1000\n")
                f.write(f"  Simulações por load: 20\n\n")
                
                f.write("RESULTADOS (Probabilidade de Bloqueio):\n")
                f.write("Load\tProbabilidade\n")
                
                for load in range(1, max_load + 1):
                    if load <= len(results[req_key]):
                        f.write(f"{load}\t{results[req_key][load-1]:.6f}\n")
        
        # Salvar todas as requisições
        output_all = f"{output_prefix}_all_requests.txt"
        with open(output_all, "w") as f:
            f.write(f"=== PSO GLOBAL - TODAS AS REQUISIÇÕES ===\n\n")
            f.write("Load\t" + "\t".join([f"[{s},{t}]" for (s, t) in self.requests]) + "\tMédia\n")
            
            for load in range(1, max_load + 1):
                f.write(f"{load}\t")
                values = []
                sum_probs = 0
                count = 0
                
                for (source, target) in self.requests:
                    req_key = f'[{source},{target}]'
                    if load <= len(results[req_key]):
                        prob = results[req_key][load-1]
                        values.append(f"{prob:.6f}")
                        sum_probs += prob
                        count += 1
                    else:
                        values.append("0.000000")
                
                if count > 0:
                    media = sum_probs / count
                    f.write("\t".join(values) + f"\t{media:.6f}\n")
                else:
                    f.write("\t".join(values) + "\t0.000000\n")
        
        print(f"\nResultados salvos com prefixo: {output_prefix}_*")
    
    def plot_results(self, results: Dict[str, List[float]], 
                    save_path: str = "pso_global_results.png"):
        """
        Gera gráfico dos resultados.
        """
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
        })
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Gráfico 1: Todas as requisições
        for idx, (source, target) in enumerate(self.requests):
            req_key = f'[{source},{target}]'
            blocking_probs = results[req_key]
            loads = np.arange(1, len(blocking_probs) + 1)
            
            ax1.plot(loads, [p * 100 for p in blocking_probs],
                    label=f'[{source},{target}]',
                    color=colors[idx % len(colors)],
                    linewidth=2,
                    marker='o' if idx < 3 else 's',
                    markersize=5,
                    markevery=10)
        
        ax1.set_xlabel('Load de Tráfego', fontsize=14)
        ax1.set_ylabel('Probabilidade de Bloqueio (%)', fontsize=14)
        ax1.set_title('PSO Global - Probabilidade de Bloqueio por Requisição\n'
                     f'(40 lambdas, 1000 calls/load, 20 simulações)',
                     fontsize=16, pad=20)
        ax1.legend(loc='upper left', ncol=2)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_xlim(1, 200)
        ax1.set_ylim(0, 100)
        ax1.set_xticks(range(0, 201, 20))
        
        # Gráfico 2: Média geral
        avg_general = []
        for load in range(1, 201):
            sum_probs = 0
            count = 0
            for (source, target) in self.requests:
                req_key = f'[{source},{target}]'
                if load <= len(results[req_key]):
                    sum_probs += results[req_key][load-1]
                    count += 1
            if count > 0:
                avg_general.append(sum_probs / count * 100)
        
        loads_avg = np.arange(1, len(avg_general) + 1)
        
        ax2.plot(loads_avg, avg_general,
                label='Média Geral das 5 Requisições',
                color='#2ca02c',
                linewidth=3,
                marker='D',
                markersize=6,
                markevery=10)
        
        ax2.set_xlabel('Load de Tráfego', fontsize=14)
        ax2.set_ylabel('Probabilidade de Bloqueio (%)', fontsize=14)
        ax2.set_title('Média Geral', fontsize=14, pad=15)
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_xlim(1, 200)
        ax2.set_ylim(0, 100)
        ax2.set_xticks(range(0, 201, 20))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nGráfico salvo em: {save_path}")


def main():
    """Função principal para execução completa do PSO."""
    start_time_total = time.time()
    
    print("\n" + "="*60)
    print("PSO GLOBAL - OTIMIZAÇÃO E SIMULAÇÃO COMPLETA")
    print("="*60)
    
    # Criar rede NSFNet
    graph = nx.Graph()
    nsfnet_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
        (4, 6), (4, 5), (5, 8), (5, 12), (6, 7), (7, 9), (8, 9), (9, 11),
        (9, 13), (10, 11), (10, 13), (11, 12)
    ]
    graph.add_edges_from(nsfnet_edges)
    
    # Requisições
    requests = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
    
    # ETAPA 1: Otimização Global com PSO
    print("\n>>> ETAPA 1: OTIMIZAÇÃO GLOBAL PSO")
    optimizer = WDMPsoGlobalOptimizer(
        graph=graph,
        num_wavelengths=40,
        requests=requests,
        k=2,
        population_size=120,
        n_gen=40,
        w=0.7,
        c1=1.5,
        c2=1.5
    )
    
    best_solution, best_fitness = optimizer.run_optimization()
    
    # ETAPA 2: Simulação com solução do PSO
    print("\n>>> ETAPA 2: SIMULAÇÃO DE TRÁFEGO")
    simulator = WDMPsoSimulator(
        graph=graph,
        num_wavelengths=40,
        requests=requests,
        pso_solution=best_solution
    )
    
    # Executar simulação
    results = simulator.simulate_traffic(
        num_runs=20,
        calls_per_load=1000,
        max_load=200
    )
    
    # ETAPA 3: Salvar resultados
    print("\n>>> ETAPA 3: SALVANDO RESULTADOS")
    simulator.save_results(results, "pso_global")
    
    # ETAPA 4: Gerar gráficos
    print("\n>>> ETAPA 4: GERANDO GRÁFICOS")
    simulator.plot_results(results, "pso_global_results.png")
    
    # ETAPA 5: Estatísticas finais
    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    
    print(f"\n" + "="*60)
    print("EXECUÇÃO COMPLETA CONCLUÍDA!")
    print(f"{'='*60}")
    print(f"Tempo total: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
    print(f"Melhor solução PSO: {best_solution}")
    print(f"Fitness da solução: {best_fitness:.6f}")
    
    # Estatísticas das simulações
    print(f"\nESTATÍSTICAS DAS SIMULAÇÕES (médias loads 1-200):")
    all_probs = []
    for (source, target) in requests:
        req_key = f'[{source},{target}]'
        if results[req_key]:
            avg = np.mean(results[req_key])
            all_probs.extend(results[req_key])
            print(f"  [{source},{target}]: {avg:.6f} ({avg*100:.4f}%)")
    
    if all_probs:
        print(f"\n  MÉDIA GERAL: {np.mean(all_probs):.6f} ({np.mean(all_probs)*100:.4f}%)")
    
    print(f"{'='*60}")
    
    # Salvar tempo total
    with open("execution_time_pso_global.txt", "w") as f:
        f.write("=== TEMPO DE EXECUÇÃO - PSO GLOBAL ===\n")
        f.write(f"Tempo total: {total_time:.2f} segundos\n")
        f.write(f"Tempo em minutos: {total_time/60:.2f}\n")
        f.write(f"\nPARÂMETROS:\n")
        f.write(f"  Número de lambdas: 40\n")
        f.write(f"  Requisições: {requests}\n")
        f.write(f"  Rotas por requisição: 2\n")
        f.write(f"  Population PSO: 120\n")
        f.write(f"  Generations PSO: 40\n")
        f.write(f"  Loads simulados: 1-200\n")
        f.write(f"  Chamadas por load: 1000\n")
        f.write(f"  Simulações por load: 20\n")
        f.write(f"\nRESULTADO PSO:\n")
        f.write(f"  Melhor solução: {best_solution}\n")
        f.write(f"  Fitness: {best_fitness:.6f}\n")
    
    print("Tempo de execução salvo em: execution_time_pso_global.txt")


if __name__ == "__main__":
    main()