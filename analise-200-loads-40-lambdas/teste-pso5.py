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
    """Classe do problema para o algoritmo PSO usando pymoo."""
    
    def __init__(self, gene_size, calc_fitness, manual_pairs):
        self.calc_fitness = calc_fitness
        self.manual_pairs = manual_pairs

        super().__init__(
            n_var=gene_size,
            n_obj=1,
            xl=np.array([0] * gene_size),  # 0 para primeira rota
            xu=np.array([1] * gene_size),   # 1 para segunda rota (k=2)
            vtype=int
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        x_int = x.astype(int)
        # PSO minimiza, então invertemos o sinal do fitness
        fitness = -self.calc_fitness(x_int.tolist(), self.manual_pairs)
        out["F"] = [fitness]


class WDMSimulatorPSO:
    """
    Simulador WDM usando PSO alinhado com AGP/DE (k=2).
    """

    def __init__(
        self,
        graph: nx.Graph,
        num_wavelengths: int = 40,
        gene_size: int = 5,
        manual_selection: bool = True,
        k: int = 2,  # MUDADO: k=2 como AGP/DE
        population_size: int = 120,
        hops_weight: float = 0.7,
        wavelength_weight: float = 0.3,
        # Parâmetros PSO
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        n_gen: int = 40
    ):
        """
        Inicializa o simulador WDM com PSO alinhado.
        """
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.gene_size = gene_size
        self.k = k  # APENAS 2 ROTAS
        self.manual_pairs = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
        self.manual_selection = manual_selection
        
        # Parâmetros PSO
        self.population_size = population_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_gen = n_gen

        # Estrutura de simulação
        self.wavelength_allocation = {}
        self.call_records = []
        self.current_time = 0.0
        
        # Cache de rotas
        self.k_shortest_paths = {}
        self._precompute_routes()
        
        # Tempo de execução
        self.execution_times = {
            'total': 0.0,
            'pso': 0.0,
            'simulation': 0.0
        }
        
        print(f"\nPSO Alinhado Inicializado:")
        print(f"  Requisições: {self.manual_pairs}")
        print(f"  Rotas por requisição: {k} (alinhado com AGP/DE)")
        print(f"  Wavelengths: {num_wavelengths}")
        print(f"  Population PSO: {population_size}")
        print(f"  Generations PSO: {n_gen}")

    def _precompute_routes(self):
        """Pré-computa k rotas para cada requisição."""
        for source, target in self.manual_pairs:
            routes = self._get_k_shortest_paths(source, target, self.k)
            self.k_shortest_paths[(source, target)] = routes

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
        """Tenta alocar usando First Fit."""
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
            
            for wl in range(self.num_wavelengths):
                if wl not in self.wavelength_allocation[edge]:
                    self.wavelength_allocation[edge][wl] = end_time
                    break
        
        return True

    def _fitness_route(self, route: List[int]) -> float:
        """
        Fitness igual ao AGP/DE.
        Menos saltos = melhor, penaliza enlaces congestionados.
        """
        if len(route) < 2:
            return 0.0

        hops = len(route) - 1
        fitness = 1.0 / (hops + 1)  # Quanto menos saltos, melhor
        
        # Penaliza se a rota for muito longa
        if hops > 6:
            fitness *= 0.5
        
        return fitness

    def _fitness(self, individual: List[int], source_targets: List[Tuple[int, int]]) -> float:
        """
        Calcula fitness total considerando todas as rotas.
        Penaliza soluções que usam os mesmos enlaces.
        """
        total_fitness = 0.0
        edge_usage = defaultdict(int)
        
        for i, (source, target) in enumerate(source_targets):
            if i >= len(individual):
                continue

            route_idx = individual[i]
            routes = self.k_shortest_paths.get((source, target), [])

            if not routes or route_idx >= len(routes):
                continue

            route = routes[route_idx]
            route_fitness = self._fitness_route(route)
            
            # Penaliza enlaces compartilhados
            for j in range(len(route) - 1):
                u, v = route[j], route[j + 1]
                edge = (min(u, v), max(u, v))
                edge_usage[edge] += 1
            
            total_fitness += route_fitness
        
        # Penalidade adicional por enlaces muito usados
        for count in edge_usage.values():
            if count > 1:
                total_fitness *= 0.8  # 20% de penalidade por enlace compartilhado
        
        return total_fitness

    def pso_algorithm(self) -> Tuple[List[int], float]:
        """Executa PSO para encontrar melhor solução."""
        pso_start = time.time()
        
        print(f"\nExecutando PSO com {self.population_size} partículas...")
        print(f"Parâmetros: w={self.w}, c1={self.c1}, c2={self.c2}")
    
        problem = MyProblem(self.gene_size, self._fitness, self.manual_pairs)

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

        res_pso = minimize(
            problem, 
            algorithm,
            termination,
            seed=42,
            verbose=False
        )

        X_PSO = res_pso.X.astype(int)
        F_PSO = -res_pso.F[0]

        pso_time = time.time() - pso_start
        self.execution_times['pso'] = pso_time

        print(f"\n{'='*50}")
        print("RESULTADOS PSO (k=2)")
        print(f"{'='*50}")
        print(f"Tempo PSO: {pso_time:.2f}s")
        print(f"Melhor solução: {X_PSO}")
        print(f"Fitness: {F_PSO:.6f}")
        
        # Mostrar rotas selecionadas
        print("\nRotas Selecionadas:")
        for i, (source, target) in enumerate(self.manual_pairs):
            if i < len(X_PSO):
                route_idx = X_PSO[i]
                routes = self.k_shortest_paths.get((source, target), [])
                if route_idx < len(routes):
                    route = routes[route_idx]
                    hops = len(route) - 1
                    print(f"  [{source},{target}]: Rota {route_idx+1}, {hops} saltos")
        print(f"{'='*50}")

        return X_PSO.tolist(), F_PSO

    def simulate_traffic(self, best_solution: List[int],
                        num_runs: int = 20,
                        calls_per_load: int = 1000,
                        max_load: int = 200) -> Dict[str, List[float]]:
        """
        Simula tráfego usando solução do PSO.
        """
        sim_start = time.time()
        
        print(f"\nSimulação com {num_runs} rodadas, {calls_per_load} calls/load...")
        
        results = {f'[{s},{t}]': [] for (s, t) in self.manual_pairs}
        
        for load in range(1, max_load + 1):
            load_start = time.time()
            
            if load % 20 == 0:
                print(f"Load {load}/{max_load}...")
            
            inter_arrival_time = 10.0 / load
            load_results = {f'[{s},{t}]': {'calls': 0, 'blocked': 0} for (s, t) in self.manual_pairs}
            
            for run in range(num_runs):
                self.reset_network()
                self.current_time = 0.0
                
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
                    request_sequence.append((s, t, -1))
                
                np.random.shuffle(request_sequence)
                
                # Processar chamadas
                for source, target, req_idx in request_sequence:
                    call_duration = np.random.uniform(5.0, 15.0)
                    self.current_time += inter_arrival_time
                    
                    self._release_expired_wavelengths()
                    
                    if req_idx >= 0:  # Requisição principal
                        # Usa rota da solução PSO
                        route_idx = best_solution[req_idx] if req_idx < len(best_solution) else 0
                        routes = self.k_shortest_paths.get((source, target), [])
                        
                        if routes and route_idx < len(routes):
                            route = routes[route_idx]
                            success = self.first_fit_allocation(route, call_duration)
                            
                            req_key = f'[{source},{target}]'
                            load_results[req_key]['calls'] += 1
                            if not success:
                                load_results[req_key]['blocked'] += 1
                    else:  # Requisição aleatória
                        # Escolhe rota aleatória entre as 2
                        routes = self.k_shortest_paths.get((source, target), [])
                        if routes:
                            route_idx = random.randint(0, len(routes) - 1)
                            route = routes[route_idx]
                            self.first_fit_allocation(route, call_duration)
            
            # Calcular probabilidades
            for (source, target) in self.manual_pairs:
                req_key = f'[{source},{target}]'
                calls = load_results[req_key]['calls']
                blocked = load_results[req_key]['blocked']
                
                if calls > 0:
                    blocking_prob = blocked / calls
                else:
                    blocking_prob = 0.0
                
                results[req_key].append(blocking_prob)
            
            load_time = time.time() - load_start
            if load % 20 == 0:
                print(f"  Tempo: {load_time:.2f}s")
        
        sim_time = time.time() - sim_start
        self.execution_times['simulation'] = sim_time
        
        return results

    def save_results(self, results: Dict[str, List[float]], 
                    best_solution: List[int], best_fitness: float,
                    output_prefix: str = "pso_k2"):
        """
        Salva resultados e tempo de execução.
        """
        max_load = len(results[next(iter(results.keys()))])
        
        # 1. Salvar por requisição
        for i, (source, target) in enumerate(self.manual_pairs):
            req_key = f'[{source},{target}]'
            output_file = f"{output_prefix}_req_{source}_{target}.txt"
            
            with open(output_file, "w") as f:
                f.write(f"=== PSO k=2 - REQUISIÇÃO [{source},{target}] ===\n\n")
                f.write("PARÂMETROS:\n")
                f.write(f"  Wavelengths: {self.num_wavelengths}\n")
                f.write(f"  Rota selecionada: {best_solution[i] if i < len(best_solution) else 0}\n")
                f.write(f"  Loads: 1 a {max_load}\n")
                f.write(f"  Calls/load: 1000\n")
                f.write(f"  Rodadas/load: 20\n\n")
                
                f.write("RESULTADOS:\n")
                f.write("Load\tProbabilidade\n")
                
                for load in range(1, max_load + 1):
                    if load <= len(results[req_key]):
                        f.write(f"{load}\t{results[req_key][load-1]:.6f}\n")
        
        # 2. Salvar todas as requisições
        output_all = f"{output_prefix}_all_requests.txt"
        with open(output_all, "w") as f:
            f.write(f"=== PSO k=2 - TODAS AS REQUISIÇÕES ===\n\n")
            f.write("RESULTADOS PSO:\n")
            f.write(f"  Solução: {best_solution}\n")
            f.write(f"  Fitness: {best_fitness:.6f}\n\n")
            
            f.write("RESULTADOS SIMULAÇÃO:\n")
            f.write("Load\t" + "\t".join([f"[{s},{t}]" for (s, t) in self.manual_pairs]) + "\tMédia\n")
            
            for load in range(1, max_load + 1):
                f.write(f"{load}\t")
                values = []
                sum_probs = 0
                count = 0
                
                for (source, target) in self.manual_pairs:
                    req_key = f'[{source},{target}]'
                    if load <= len(results[req_key]):
                        prob = results[req_key][load-1]
                        values.append(f"{prob:.6f}")
                        sum_probs += prob
                        count += 1
                
                if count > 0:
                    media = sum_probs / count
                    f.write("\t".join(values) + f"\t{media:.6f}\n")
        
        # 3. Salvar tempo de execução
        total_time = self.execution_times['pso'] + self.execution_times['simulation']
        time_file = f"execution_time_{output_prefix}.txt"
        with open(time_file, "w") as f:
            f.write("=== TEMPO DE EXECUÇÃO - PSO k=2 ===\n\n")
            f.write("TEMPOS:\n")
            f.write(f"  Total: {total_time:.2f} segundos\n")
            f.write(f"  PSO: {self.execution_times['pso']:.2f} segundos\n")
            f.write(f"  Simulação: {self.execution_times['simulation']:.2f} segundos\n")
            f.write(f"  Total minutos: {total_time/60:.2f}\n\n")
            
            f.write("PARÂMETROS:\n")
            f.write(f"  k (rotas): {self.k}\n")
            f.write(f"  Wavelengths: {self.num_wavelengths}\n")
            f.write(f"  Requisições: {self.manual_pairs}\n")
            f.write(f"  Population: {self.population_size}\n")
            f.write(f"  Generations: {self.n_gen}\n")
            f.write(f"  Loads: 1-200\n")
            f.write(f"  Calls/load: 1000\n")
            f.write(f"  Rodadas/load: 20\n\n")
            
            f.write("RESULTADO PSO:\n")
            f.write(f"  Solução: {best_solution}\n")
            f.write(f"  Fitness: {best_fitness:.6f}\n")
        
        print(f"\nResultados salvos com prefixo: {output_prefix}_*")
        print(f"Tempo de execução salvo em: {time_file}")

    def plot_results(self, results: Dict[str, List[float]], 
                    save_path: str = "pso_k2_results.png"):
        """Gera gráfico dos resultados."""
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
        for idx, (source, target) in enumerate(self.manual_pairs):
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
        ax1.set_title('PSO k=2 - Probabilidade de Bloqueio por Requisição\n'
                     f'(40 wavelengths, 1000 calls/load, 20 simulações)',
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
            for (source, target) in self.manual_pairs:
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
    start_total = time.time()
    
    print("\n" + "="*60)
    print("PSO ALINHADO (k=2) - SIMULAÇÃO COMPLETA")
    print("="*60)
    
    # Criar rede
    graph = nx.Graph()
    nsfnet_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
        (4, 6), (4, 5), (5, 8), (5, 12), (6, 7), (7, 9), (8, 9), (9, 11),
        (9, 13), (10, 11), (10, 13), (11, 12)
    ]
    graph.add_edges_from(nsfnet_edges)
    
    # Criar simulador
    simulator = WDMSimulatorPSO(
        graph=graph,
        num_wavelengths=40,
        gene_size=5,
        manual_selection=True,
        k=2,  # k=2 ALINHADO
        population_size=120,
        hops_weight=0.7,
        wavelength_weight=0.3,
        w=0.7,
        c1=1.5,
        c2=1.5,
        n_gen=40
    )
    
    # Executar PSO
    best_solution, best_fitness = simulator.pso_algorithm()
    
    # Executar simulação
    results = simulator.simulate_traffic(
        best_solution=best_solution,
        num_runs=20,
        calls_per_load=1000,
        max_load=200
    )
    
    # Salvar resultados
    simulator.save_results(
        results=results,
        best_solution=best_solution,
        best_fitness=best_fitness,
        output_prefix="pso_k2_aligned"
    )
    
    # Gerar gráfico
    simulator.plot_results(results, "pso_k2_aligned_results.png")
    
    # Tempo total
    total_time = time.time() - start_total
    simulator.execution_times['total'] = total_time
    
    print(f"\n" + "="*60)
    print("SIMULAÇÃO CONCLUÍDA!")
    print(f"Tempo total: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
    print(f"Melhor solução: {best_solution}")
    print(f"Fitness: {best_fitness:.6f}")
    
    # Estatísticas
    print(f"\nMÉDIAS (loads 1-200):")
    all_probs = []
    for (source, target) in simulator.manual_pairs:
        req_key = f'[{source},{target}]'
        if results[req_key]:
            avg = np.mean(results[req_key])
            all_probs.extend(results[req_key])
            print(f"  [{source},{target}]: {avg:.6f} ({avg*100:.4f}%)")
    
    if all_probs:
        print(f"\n  MÉDIA GERAL: {np.mean(all_probs):.6f} ({np.mean(all_probs)*100:.4f}%)")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()