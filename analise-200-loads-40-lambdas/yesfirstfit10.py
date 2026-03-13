import os
from itertools import islice
from typing import List, Tuple, Dict, Optional
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random


class WDMYenBase:
    """
    Classe base para simulação WDM usando YEN.
    """
    def __init__(self,
                 graph: nx.Graph,
                 num_wavelengths: int = 40,  # 40 wavelengths
                 k: int = 5,
                 requests: List[Tuple[int, int]] = None):
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.k = k
        self.requests = requests if requests else [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
        
        # Estrutura para armazenar alocações com tempo
        self.wavelength_allocation = {}
        self.call_records = []
        self.current_time = 0.0
        
        self.reset_network()
    
    def reset_network(self) -> None:
        """Reseta o estado da rede."""
        for u, v in self.graph.edges:
            edge = (min(u, v), max(u, v))
            self.wavelength_allocation[edge] = {}
        self.current_time = 0.0
        self.call_records = []
    
    def _get_k_shortest_paths(self, source: int, target: int, k: int) -> List[List[int]]:
        """
        Calcula os k menores caminhos usando YEN.
        """
        if not nx.has_path(self.graph, source, target):
            return []
        try:
            return list(islice(nx.shortest_simple_paths(self.graph, source, target), k))
        except nx.NetworkXNoPath:
            return []
    
    def _release_expired_wavelengths(self) -> None:
        """Libera wavelengths cuja duração expirou."""
        for edge in self.wavelength_allocation:
            expired_wavelengths = [
                wl for wl, end_time in self.wavelength_allocation[edge].items()
                if end_time <= self.current_time
            ]
            
            for wl in expired_wavelengths:
                del self.wavelength_allocation[edge][wl]
    
    def first_fit_allocation_with_conversion(self, route: List[int],
                                             call_duration: float) -> Optional[Dict[int, int]]:
        """
        Aloca wavelengths usando First Fit com conversão permitida.
        Cada enlace pode usar um wavelength diferente.
        """
        wavelength_allocation_route = {}
        end_time = self.current_time + call_duration
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            edge = (min(u, v), max(u, v))
            
            wavelength_found = None
            for wavelength in range(self.num_wavelengths):
                if wavelength not in self.wavelength_allocation[edge]:
                    wavelength_found = wavelength
                    break
            
            if wavelength_found is None:
                return None
            
            self.wavelength_allocation[edge][wavelength_found] = end_time
            wavelength_allocation_route[i] = wavelength_found
        
        return wavelength_allocation_route


class WDMYenTwoRoutesRandom(WDMYenBase):
    """
    Simulador WDM usando YEN com 2 rotas aleatórias por requisição.
    - Processa as 5 requisições conjuntamente
    - Para cada chamada, escolhe aleatoriamente entre as 2 menores rotas
    - Usa First Fit para alocação de wavelengths
    """
    
    def process_call(self, source: int, target: int, call_id: int,
                     call_duration: float) -> Tuple[bool, Optional[List[int]], Optional[Dict[int, int]]]:
        """
        Processa uma chamada escolhendo aleatoriamente entre as 2 menores rotas.
        """
        self._release_expired_wavelengths()
        
        # Obtém as 2 menores rotas do YEN
        paths = self._get_k_shortest_paths(source, target, 2)
        
        if not paths:
            return (True, None, None)
        
        # Se só tem 1 rota, usa ela
        if len(paths) == 1:
            selected_route = paths[0]
        else:
            # Escolhe aleatoriamente entre as 2 rotas
            selected_route = random.choice(paths)
        
        # Tenta alocar usando First Fit
        wavelength_allocation = self.first_fit_allocation_with_conversion(selected_route, call_duration)
        
        if wavelength_allocation is None:
            return (True, None, None)
        
        # Registra qual rota foi usada (1 = primeira, 2 = segunda)
        route_index = paths.index(selected_route) + 1
        
        self.call_records.append({
            'call_id': call_id,
            'source': source,
            'target': target,
            'route': selected_route,
            'wavelength_allocation': wavelength_allocation,
            'blocked': False,
            'hops': len(selected_route) - 1,
            'start_time': self.current_time,
            'end_time': self.current_time + call_duration,
            'duration': call_duration,
            'path_used': route_index,
            'strategy': 'YenTwoRoutesRandom'
        })
        
        return (False, selected_route, wavelength_allocation)
    
    def simulate_traffic(self, num_runs: int = 20,  # 20 simulações
                        calls_per_load: int = 1000,  # 1000 calls por load
                        call_duration_range: Tuple[float, float] = (5.0, 15.0),
                        max_load: int = 200) -> Dict[str, List[float]]:
        """
        Simula tráfego com as 5 requisições conjuntas.
        - 200 loads (1 a 200)
        - 1000 chamadas por load
        - 20 simulações por load
        - 5 requisições competindo pelos mesmos recursos
        """
        results = {}
        
        # Inicializar resultados para cada requisição
        for req_idx, (source, target) in enumerate(self.requests):
            results[f'[{source},{target}]'] = []
        
        print("\n" + "="*60)
        print("INICIANDO SIMULAÇÃO - YEN 2 ROTAS ALEATÓRIAS")
        print(f"Parâmetros:")
        print(f"  Número de lambdas: {self.num_wavelengths}")
        print(f"  Loads: 1 a {max_load}")
        print(f"  Chamadas por load: {calls_per_load}")
        print(f"  Simulações por load: {num_runs}")
        print(f"  Requisições: {self.requests}")
        print("="*60)
        
        start_time_total = time.time()
        
        for load in range(1, max_load + 1):
            load_start_time = time.time()
            
            if load % 20 == 0 or load == 1 or load == max_load:
                print(f"\nLoad {load}/{max_load}...")
            
            inter_arrival_time = 10.0 / load
            load_blocking_probs = {f'[{s},{t}]': [] for (s, t) in self.requests}
            
            for run in range(num_runs):
                self.reset_network()
                self.current_time = 0.0
                
                # Resetar contadores para esta rodada
                run_blocked_counts = {f'[{s},{t}]': 0 for (s, t) in self.requests}
                run_total_counts = {f'[{s},{t}]': 0 for (s, t) in self.requests}
                
                # Preparar sequência de requisições para esta rodada
                # 80% das chamadas são das 5 requisições principais (16% cada)
                # 20% são requisições aleatórias
                
                # Número de chamadas para cada requisição principal
                calls_per_request = int(calls_per_load * 0.8 / 5)  # 16% cada
                random_calls = calls_per_load - (calls_per_request * 5)
                
                # Criar lista de requisições
                request_sequence = []
                
                # Adicionar requisições principais
                for req_idx, (source, target) in enumerate(self.requests):
                    request_sequence.extend([(source, target, req_idx)] * calls_per_request)
                
                # Adicionar requisições aleatórias
                nodes = list(self.graph.nodes)
                for _ in range(random_calls):
                    s, t = np.random.choice(nodes, 2, replace=False)
                    request_sequence.append((s, t, -1))  # -1 indica requisição aleatória
                
                # Embaralhar a sequência
                np.random.shuffle(request_sequence)
                
                # Processar todas as chamadas
                for call_idx, (source, target, req_idx) in enumerate(request_sequence):
                    call_duration = np.random.uniform(call_duration_range[0], call_duration_range[1])
                    self.current_time += inter_arrival_time
                    
                    blocked, _, _ = self.process_call(source, target, call_idx, call_duration)
                    
                    # Contar apenas para as requisições principais
                    if req_idx >= 0:
                        req_key = f'[{source},{target}]'
                        run_total_counts[req_key] += 1
                        if blocked:
                            run_blocked_counts[req_key] += 1
                
                # Calcular probabilidade de bloqueio para esta rodada
                for req_key in run_total_counts.keys():
                    if run_total_counts[req_key] > 0:
                        blocking_prob = run_blocked_counts[req_key] / run_total_counts[req_key]
                        load_blocking_probs[req_key].append(blocking_prob)
                    else:
                        load_blocking_probs[req_key].append(0.0)
            
            # Calcular média para este load
            for (source, target) in self.requests:
                req_key = f'[{source},{target}]'
                if load_blocking_probs[req_key]:
                    avg_blocking = np.mean(load_blocking_probs[req_key])
                    results[req_key].append(avg_blocking)
                else:
                    results[req_key].append(0.0)
            
            load_end_time = time.time()
            load_duration = load_end_time - load_start_time
            
            if load % 20 == 0 or load == 1 or load == max_load:
                print(f"  Tempo: {load_duration:.2f}s")
                for req_idx, (source, target) in enumerate(self.requests):
                    req_key = f'[{source},{target}]'
                    if req_idx < 3:  # Mostrar apenas 3 para não poluir
                        print(f"  {req_key}: {results[req_key][-1]:.6f}")
        
        end_time_total = time.time()
        total_duration = end_time_total - start_time_total
        
        print(f"\n" + "="*60)
        print(f"SIMULAÇÃO CONCLUÍDA!")
        print(f"Tempo total: {total_duration:.2f} segundos")
        print(f"Tempo médio por load: {total_duration/max_load:.2f} segundos")
        print("="*60)
        
        return results
    
    def get_path_statistics(self) -> Dict[str, float]:
        """
        Retorna estatísticas sobre quais caminhos foram usados.
        """
        if not self.call_records:
            return {}
        
        successful_calls = [r for r in self.call_records if not r['blocked']]
        total_calls = len(successful_calls)
        
        if total_calls == 0:
            return {}
        
        path_usage = {1: 0, 2: 0}
        
        for record in successful_calls:
            path_num = record.get('path_used', 1)
            if path_num in path_usage:
                path_usage[path_num] += 1
            else:
                path_usage[path_num] = 1
        
        return {
            'total_calls': total_calls,
            'path_distribution': {k: v/total_calls for k, v in sorted(path_usage.items())},
            'avg_path_used': np.mean([r.get('path_used', 1) for r in successful_calls]),
            'first_route_usage': path_usage.get(1, 0) / total_calls,
            'second_route_usage': path_usage.get(2, 0) / total_calls
        }
    
    def save_results_per_request(self, results: Dict[str, List[float]], 
                                output_prefix: str = "yen_two_routes_random"):
        """
        Salva resultados separados por requisição.
        """
        max_load = len(results[next(iter(results.keys()))])
        
        # Salvar cada requisição individualmente
        for (source, target) in self.requests:
            req_key = f'[{source},{target}]'
            output_file = f"{output_prefix}_req_{source}_{target}.txt"
            
            with open(output_file, "w") as f:
                f.write(f"=== YEN 2 ROTAS ALEATÓRIAS - REQUISIÇÃO [{source},{target}] ===\n\n")
                f.write("PARÂMETROS DA SIMULAÇÃO:\n")
                f.write(f"  Rede: NSFNet (14 nós, 21 enlaces)\n")
                f.write(f"  Número de lambdas: {self.num_wavelengths}\n")
                f.write(f"  Rotas consideradas: 2 (menores rotas do YEN)\n")
                f.write(f"  Seleção de rota: Aleatória entre as 2\n")
                f.write(f"  Alocação de wavelength: First Fit\n")
                f.write(f"  Loads testados: 1 a {max_load}\n")
                f.write(f"  Chamadas por load: 1000\n")
                f.write(f"  Simulações por load: 20\n")
                f.write(f"  Duração das chamadas: 5.0 a 15.0 unidades\n\n")
                
                f.write("RESULTADOS (Probabilidade de Bloqueio):\n")
                f.write("Load\tProbabilidade\n")
                
                for load in range(1, max_load + 1):
                    if load <= len(results[req_key]):
                        f.write(f"{load}\t{results[req_key][load-1]:.6f}\n")
        
        # Salvar arquivo com todas as requisições
        output_file_all = f"{output_prefix}_all_requests.txt"
        with open(output_file_all, "w") as f:
            f.write(f"=== YEN 2 ROTAS ALEATÓRIAS - TODAS AS REQUISIÇÕES ===\n\n")
            f.write("PARÂMETROS:\n")
            f.write(f"  Lambdas: {self.num_wavelengths}\n")
            f.write(f"  Loads: 1 a {max_load}\n")
            f.write(f"  Chamadas por load: 1000\n")
            f.write(f"  Simulações por load: 20\n")
            f.write(f"  Requisições: {self.requests}\n\n")
            
            f.write("RESULTADOS:\n")
            f.write("Load\t" + "\t".join([f"[{s},{t}]" for (s, t) in self.requests]) + "\tMédia\n")
            
            for load in range(1, max_load + 1):
                f.write(f"{load}\t")
                values = []
                for (source, target) in self.requests:
                    req_key = f'[{source},{target}]'
                    if load <= len(results[req_key]):
                        values.append(f"{results[req_key][load-1]:.6f}")
                    else:
                        values.append("0.000000")
                
                # Calcular média
                avg_values = []
                for (source, target) in self.requests:
                    req_key = f'[{source},{target}]'
                    if load <= len(results[req_key]):
                        avg_values.append(results[req_key][load-1])
                
                if avg_values:
                    media = np.mean(avg_values)
                    f.write("\t".join(values) + f"\t{media:.6f}\n")
                else:
                    f.write("\t".join(values) + "\t0.000000\n")
        
        print(f"\nResultados salvos em:")
        for (source, target) in self.requests:
            print(f"  - {output_prefix}_req_{source}_{target}.txt")
        print(f"  - {output_file_all}")
    
    def plot_results(self, results: Dict[str, List[float]], 
                    save_path: str = "yen_two_routes_random_results.png"):
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
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
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
        ax1.set_title('YEN 2 Rotas Aleatórias - Probabilidade de Bloqueio por Requisição\n'
                     f'(40 lambdas, 1000 calls/load, 20 simulações)',
                     fontsize=16, pad=20)
        ax1.legend(loc='upper left', fontsize=11, ncol=2)
        ax1.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
        ax1.set_xlim(1, 200)
        ax1.set_ylim(0, 100)
        ax1.set_xticks(range(0, 201, 20))
        
        # Gráfico 2: Média geral
        # Calcular média geral por load
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
        ax2.set_title('Média Geral das 5 Requisições', fontsize=14, pad=15)
        ax2.legend(loc='upper left', fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
        ax2.set_xlim(1, 200)
        ax2.set_ylim(0, 100)
        ax2.set_xticks(range(0, 201, 20))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Gráfico salvo em: {save_path}")
        
        # Gráfico detalhado (0-50)
        self.plot_detailed_results(results, max_load=50)
    
    def plot_detailed_results(self, results: Dict[str, List[float]], 
                            max_load: int = 50,
                            save_path: str = "yen_two_routes_random_detailed.png"):
        """
        Gera gráfico detalhado das primeiras cargas.
        """
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        })
        
        plt.figure(figsize=(12, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, (source, target) in enumerate(self.requests):
            req_key = f'[{source},{target}]'
            blocking_probs = results[req_key]
            # Limitar ao max_load
            max_idx = min(max_load, len(blocking_probs))
            loads = np.arange(1, max_idx + 1)
            
            plt.plot(loads, [p * 100 for p in blocking_probs[:max_idx]],
                    label=f'[{source},{target}]',
                    color=colors[idx % len(colors)],
                    linewidth=2.5,
                    marker='o',
                    markersize=6)
        
        plt.xlabel('Load de Tráfego', fontsize=14)
        plt.ylabel('Probabilidade de Bloqueio (%)', fontsize=14)
        plt.title(f'YEN 2 Rotas Aleatórias - Detalhe Loads 1-{max_load}\n'
                 f'(40 lambdas, First Fit, Seleção Aleatória entre 2 Rotas)',
                 fontsize=16, pad=20)
        plt.legend(loc='upper left', fontsize=11, ncol=2)
        plt.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
        plt.xlim(1, max_load)
        plt.ylim(0, 100)
        plt.xticks(range(1, max_load + 1, 2 if max_load <= 30 else 5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Gráfico detalhado salvo em: {save_path}")


def main():
    """Função principal para executar simulação YEN com 2 rotas aleatórias."""
    start_time = time.time()
    
    print("\n" + "="*60)
    print("SIMULAÇÃO WDM - YEN COM 2 ROTAS ALEATÓRIAS")
    print("(5 REQUISIÇÕES CONJUNTAS)")
    print("="*60)
    
    # Criação do grafo NSFNet
    graph = nx.Graph()
    nsfnet_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
        (4, 6), (4, 5), (5, 8), (5, 12), (6, 7), (7, 9), (8, 9), (9, 11),
        (9, 13), (10, 11), (10, 13), (11, 12)
    ]
    graph.add_edges_from(nsfnet_edges)
    
    # Requisições customizadas na ordem correta
    custom_requests = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
    
    # Criar simulador
    simulator = WDMYenTwoRoutesRandom(
        graph=graph,
        num_wavelengths=40,  # 40 lambdas
        k=2,  # Usa apenas 2 rotas
        requests=custom_requests
    )
    
    # Executar simulação
    print("\n>>> EXECUTANDO SIMULAÇÃO...")
    results = simulator.simulate_traffic(
        num_runs=20,        # 20 simulações por load
        calls_per_load=1000, # 1000 calls por load
        max_load=200        # 200 loads
    )
    
    # Tempo de execução
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Salvar resultados
    print("\n>>> SALVANDO RESULTADOS...")
    simulator.save_results_per_request(
        results=results,
        output_prefix="yen_two_routes_random"
    )
    
    # Gerar gráficos
    print("\n>>> GERANDO GRÁFICOS...")
    simulator.plot_results(results)
    
    # Salvar tempo de execução
    with open("execution_time_yen_random.txt", "w") as f:
        f.write("=== TEMPO DE EXECUÇÃO - YEN 2 ROTAS ALEATÓRIAS ===\n")
        f.write(f"Tempo total: {execution_time:.2f} segundos\n")
        f.write(f"  ({execution_time/60:.2f} minutos)\n")
        f.write(f"Início: {time.ctime(start_time)}\n")
        f.write(f"Término: {time.ctime(end_time)}\n")
        f.write(f"\nPARÂMETROS:\n")
        f.write(f"  Número de lambdas: 40\n")
        f.write(f"  Loads: 1 a 200\n")
        f.write(f"  Chamadas por load: 1000\n")
        f.write(f"  Simulações por load: 20\n")
        f.write(f"  Requisições: {custom_requests}\n")
    
    print(f"\n" + "="*60)
    print("SIMULAÇÃO CONCLUÍDA!")
    print(f"Tempo total de execução: {execution_time:.2f} segundos")
    print(f"  ({execution_time/60:.2f} minutos)")
    print("="*60)
    
    # Mostrar estatísticas resumidas
    print(f"\nESTATÍSTICAS RESUMIDAS (médias loads 1-200):")
    for (source, target) in custom_requests:
        req_key = f'[{source},{target}]'
        if results[req_key]:
            avg = np.mean(results[req_key])
            print(f"  [{source},{target}]: {avg:.6f} ({avg*100:.4f}%)")
    
    # Calcular média geral
    all_probs = []
    for (source, target) in custom_requests:
        req_key = f'[{source},{target}]'
        all_probs.extend(results[req_key])
    
    if all_probs:
        print(f"\n  MÉDIA GERAL: {np.mean(all_probs):.6f} ({np.mean(all_probs)*100:.4f}%)")


if __name__ == "__main__":
    main()