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


class WDMPsoSimulator:
    """
    Simulador WDM usando PSO alinhado com YEN para comparação justa.
    - Processa as 5 requisições conjuntamente
    - Usa First Fit para alocação
    - Simula chamadas uma a uma com controle de tempo
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        num_wavelengths: int = 40,
        requests: List[Tuple[int, int]] = None,
        k: int = 2,  # Usa apenas 2 rotas como YEN
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
        
        # Estrutura de alocação
        self.wavelength_allocation = {}
        self.call_records = []
        self.current_time = 0.0
        
        self.reset_network()
    
    def _precompute_routes(self):
        """Pré-computa as 2 rotas para cada requisição."""
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
    
    def first_fit_allocation(self, route: List[int], call_duration: float) -> bool:
        """
        Tenta alocar usando First Fit.
        Retorna True se conseguir alocar, False caso contrário.
        """
        end_time = self.current_time + call_duration
        
        # Primeiro verifica se todos os enlaces têm wavelengths disponíveis
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            edge = (min(u, v), max(u, v))
            
            # Conta wavelengths ocupados
            if len(self.wavelength_allocation.get(edge, {})) >= self.num_wavelengths:
                return False
        
        # Se todos têm espaço, faz a alocação
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            edge = (min(u, v), max(u, v))
            
            # Encontra primeiro wavelength disponível
            wavelength_found = None
            for wl in range(self.num_wavelengths):
                if wl not in self.wavelength_allocation[edge]:
                    wavelength_found = wl
                    break
            
            if wavelength_found is None:
                return False
            
            self.wavelength_allocation[edge][wavelength_found] = end_time
        
        return True
    
    def evaluate_solution(self, solution: List[int]) -> float:
        """
        Avalia uma solução do PSO (lista de 5 índices de rota).
        Retorna o fitness baseado no número de chamadas aceitas.
        """
        # Fitness = número de requisições que conseguem ser alocadas simultaneamente
        successful_allocations = 0
        
        # Tenta alocar todas as 5 requisições simultaneamente
        temp_allocation = {}
        
        for i, (source, target) in enumerate(self.requests):
            if i >= len(solution):
                continue
            
            route_idx = solution[i]
            routes = self.route_cache.get((source, target), [])
            
            if not routes or route_idx >= len(routes):
                continue
            
            route = routes[route_idx]
            
            # Verifica disponibilidade
            can_allocate = True
            for j in range(len(route) - 1):
                u, v = route[j], route[j + 1]
                edge = (min(u, v), max(u, v))
                
                # Simula alocação
                if edge not in temp_allocation:
                    temp_allocation[edge] = set()
                
                # Encontra wavelength disponível
                found = False
                for wl in range(self.num_wavelengths):
                    if wl not in temp_allocation[edge]:
                        temp_allocation[edge].add(wl)
                        found = True
                        break
                
                if not found:
                    can_allocate = False
                    break
            
            if can_allocate:
                successful_allocations += 1
        
        return successful_allocations / len(self.requests)  # Fitness normalizado
    
    def process_call_with_solution(self, source: int, target: int, solution_idx: int, 
                                 call_duration: float) -> bool:
        """
        Processa uma chamada usando a solução do PSO.
        """
        self._release_expired_wavelengths()
        
        # Encontra a rota baseada na solução do PSO
        # Para isso, precisamos mapear qual índice da solução usar
        # Assumimos que a solução é uma lista de 5 índices na ordem das requisições
        
        # Encontra o índice da requisição
        req_idx = -1
        for i, (s, t) in enumerate(self.requests):
            if s == source and t == target:
                req_idx = i
                break
        
        if req_idx == -1 or req_idx >= len(solution_idx):
            # Requisição não encontrada ou índice inválido, usa aleatório
            routes = self.route_cache.get((source, target), [])
            if not routes:
                return True
            
            selected_route = random.choice(routes)
        else:
            # Usa a rota da solução PSO
            route_idx = solution_idx[req_idx]
            routes = self.route_cache.get((source, target), [])
            
            if not routes or route_idx >= len(routes):
                # Índice inválido, usa primeira rota
                selected_route = routes[0] if routes else None
            else:
                selected_route = routes[route_idx]
        
        if not selected_route:
            return True
        
        # Tenta alocar
        return not self.first_fit_allocation(selected_route, call_duration)
    
    def simulate_traffic_with_pso(self, num_runs: int = 20,
                                 calls_per_load: int = 1000,
                                 call_duration_range: Tuple[float, float] = (5.0, 15.0),
                                 max_load: int = 200) -> Dict[str, List[float]]:
        """
        Simula tráfego usando a solução do PSO.
        """
        print("\n" + "="*60)
        print("PSO - EXECUTANDO SIMULAÇÃO DE TRÁFEGO")
        print(f"Loads: 1 a {max_load}")
        print(f"Chamadas por load: {calls_per_load}")
        print(f"Simulações por load: {num_runs}")
        print("="*60)
        
        # Primeiro executa o PSO para encontrar a melhor solução
        print("\n>>> EXECUTANDO ALGORITMO PSO...")
        best_solution = self.run_pso()
        print(f"Melhor solução encontrada: {best_solution}")
        
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
                
                # Preparar distribuição de chamadas
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
                for call_idx, (source, target, req_idx) in enumerate(request_sequence):
                    call_duration = np.random.uniform(call_duration_range[0], call_duration_range[1])
                    self.current_time += inter_arrival_time
                    
                    blocked = self.process_call_with_solution(source, target, best_solution, call_duration)
                    
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
                    if i < 2:  # Mostrar apenas 2 para não poluir
                        req_key = f'[{source},{target}]'
                        print(f"  {req_key}: {results[req_key][-1]:.6f}")
        
        return results
    
    def run_pso(self) -> List[int]:
        """
        Executa algoritmo PSO para encontrar a melhor combinação de rotas.
        Retorna lista de 5 índices (0 ou 1) para cada requisição.
        """
        from pymoo.algorithms.soo.nonconvex.pso import PSO
        from pymoo.operators.sampling.rnd import IntegerRandomSampling
        from pymoo.termination import get_termination
        from pymoo.optimize import minimize
        
        class PsoProblem(ElementwiseProblem):
            def __init__(self, simulator):
                self.simulator = simulator
                super().__init__(
                    n_var=5,  # 5 requisições
                    n_obj=1,
                    xl=np.array([0, 0, 0, 0, 0]),
                    xu=np.array([1, 1, 1, 1, 1]),  # 0 ou 1 (primeira ou segunda rota)
                    vtype=int
                )
            
            def _evaluate(self, x, out, *args, **kwargs):
                x_int = x.astype(int)
                fitness = self.simulator.evaluate_solution(x_int.tolist())
                out["F"] = [-fitness]  # PSO minimiza, então invertemos
        
        problem = PsoProblem(self)
        
        algorithm = PSO(
            pop_size=self.population_size,
            w=self.w,
            c1=self.c1,
            c2=self.c2,
            sampling=IntegerRandomSampling(),
        )
        
        termination = get_termination("n_gen", self.n_gen)
        
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            verbose=False
        )
        
        return res.X.astype(int).tolist()
    
    def save_results(self, results: Dict[str, List[float]], 
                    output_prefix: str = "pso_aligned"):
        """
        Salva resultados em arquivos.
        """
        max_load = len(results[next(iter(results.keys()))])
        
        # Salvar por requisição
        for (source, target) in self.requests:
            req_key = f'[{source},{target}]'
            output_file = f"{output_prefix}_req_{source}_{target}.txt"
            
            with open(output_file, "w") as f:
                f.write(f"=== PSO ALINHADO - REQUISIÇÃO [{source},{target}] ===\n\n")
                f.write("PARÂMETROS:\n")
                f.write(f"  Lambdas: {self.num_wavelengths}\n")
                f.write(f"  Loads: 1 a {max_load}\n")
                f.write(f"  Chamadas por load: 1000\n")
                f.write(f"  Simulações por load: 20\n\n")
                
                f.write("RESULTADOS:\n")
                f.write("Load\tProbabilidade\n")
                
                for load in range(1, max_load + 1):
                    if load <= len(results[req_key]):
                        f.write(f"{load}\t{results[req_key][load-1]:.6f}\n")
        
        # Salvar todas as requisições
        output_all = f"{output_prefix}_all_requests.txt"
        with open(output_all, "w") as f:
            f.write(f"=== PSO ALINHADO - TODAS AS REQUISIÇÕES ===\n\n")
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
                    save_path: str = "pso_aligned_results.png"):
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
        ax1.set_title('PSO Alinhado - Probabilidade de Bloqueio\n'
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
        
        print(f"Gráfico salvo em: {save_path}")


def main():
    """Função principal."""
    start_time = time.time()
    
    print("\n" + "="*60)
    print("PSO ALINHADO - SIMULAÇÃO WDM")
    print("(5 REQUISIÇÕES CONJUNTAS)")
    print("="*60)
    
    # Criar rede
    graph = nx.Graph()
    nsfnet_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
        (4, 6), (4, 5), (5, 8), (5, 12), (6, 7), (7, 9), (8, 9), (9, 11),
        (9, 13), (10, 11), (10, 13), (11, 12)
    ]
    graph.add_edges_from(nsfnet_edges)
    
    # Requisições
    requests = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
    
    # Criar simulador
    simulator = WDMPsoSimulator(
        graph=graph,
        num_wavelengths=40,
        requests=requests,
        k=2,
        population_size=120,
        n_gen=40
    )
    
    # Executar simulação
    results = simulator.simulate_traffic_with_pso(
        num_runs=20,
        calls_per_load=1000,
        max_load=200
    )
    
    # Salvar resultados
    simulator.save_results(results, "pso_aligned")
    
    # Gerar gráfico
    simulator.plot_results(results, "pso_aligned_results.png")
    
    # Tempo total
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n" + "="*60)
    print("SIMULAÇÃO CONCLUÍDA!")
    print(f"Tempo total: {total_time:.2f} segundos")
    print(f"  ({total_time/60:.2f} minutos)")
    print("="*60)
    
    # Estatísticas resumidas
    print(f"\nMÉDIAS (loads 1-200):")
    all_probs = []
    for (source, target) in requests:
        req_key = f'[{source},{target}]'
        if results[req_key]:
            avg = np.mean(results[req_key])
            all_probs.extend(results[req_key])
            print(f"  [{source},{target}]: {avg:.6f} ({avg*100:.4f}%)")
    
    if all_probs:
        print(f"\n  MÉDIA GERAL: {np.mean(all_probs):.6f} ({np.mean(all_probs)*100:.4f}%)")


if __name__ == "__main__":
    main()