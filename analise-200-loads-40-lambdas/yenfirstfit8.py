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
                 num_wavelengths: int = 16,
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
    - Para cada requisição, obtém as 2 menores rotas do YEN
    - Escolhe aleatoriamente uma das 2 rotas para tentar alocação
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
    
    def simulate_traffic_multiple_runs(self, num_runs: int = 20,
                                       total_calls_per_load: int = 1000,
                                       call_duration_range: Tuple[float, float] = (5.0, 15.0),
                                       max_load: int = 200) -> Dict[str, List[float]]:
        """
        Simula tráfego com múltiplas execuções para cada carga (1 a max_load).
        """
        results = {}
        
        for req_idx, (source, target) in enumerate(self.requests):
            results[f'[{source},{target}]'] = [0.0]
        
        for load in range(1, max_load + 1):  # 1 a max_load
            if load % 20 == 0:
                print(f"\n=== CARGA {load}/{max_load} (YenTwoRoutesRandom) ===")
            
            inter_arrival_time = 10.0 / load
            load_results = {}
            
            for req_idx, (source, target) in enumerate(self.requests):
                load_results[f'[{source},{target}]'] = []
            
            for run in range(num_runs):
                if load % 20 == 0 and run == 0:
                    print(f"  Rodada {run + 1}/{num_runs}...", end='\r')
                self.reset_network()
                self.current_time = 0.0
                
                for req_idx, (source, target) in enumerate(self.requests):
                    blocked_count = 0
                    
                    for call_idx in range(total_calls_per_load):
                        call_duration = np.random.uniform(call_duration_range[0], call_duration_range[1])
                        self.current_time += inter_arrival_time
                        
                        if np.random.random() < 0.8:
                            s, t = source, target
                        else:
                            nodes = list(self.graph.nodes)
                            s, t = np.random.choice(nodes, 2, replace=False)
                        
                        blocked, _, _ = self.process_call(s, t, call_idx, call_duration)
                        
                        if blocked:
                            blocked_count += 1
                    
                    blocking_prob = blocked_count / total_calls_per_load
                    load_results[f'[{source},{target}]'].append(blocking_prob)
            
            for req_idx, (source, target) in enumerate(self.requests):
                req_key = f'[{source},{target}]'
                avg_blocking = np.mean(load_results[req_key])
                results[req_key].append(avg_blocking)
                
                if load % 20 == 0:
                    print(f"  {req_key}: {avg_blocking:.6f}")
        
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
    
    def save_results_to_file(self, results: Dict[str, List[float]], 
                            output_file: str = "yen_two_routes_random_results.txt",
                            strategy_name: str = "YEN 2 ROTAS ALEATÓRIAS"):
        """
        Salva resultados em arquivo.
        """
        with open(output_file, "w") as f:
            f.write(f"=== {strategy_name} ===\n\n")
            f.write("PARÂMETROS DA SIMULAÇÃO:\n")
            f.write(f"  Rede: NSFNet (14 nós, 21 enlaces)\n")
            f.write(f"  Wavelengths por enlace: 40\n")
            f.write(f"  Número de rotas consideradas por requisição: 2\n")
            f.write(f"  Requisições: {self.requests}\n")
            f.write(f"  Rodadas por carga: 20\n")
            f.write(f"  Chamadas por rodada: 1000\n")
            f.write(f"  Cargas testadas: 1 a 200\n")
            f.write(f"  Seleção de rota: Aleatória entre as 2 menores\n")
            f.write(f"  Alocação de wavelength: First Fit\n\n")
            
            f.write("RESULTADOS (Blocking Probability):\n")
            f.write("Carga\t" + "\t".join([f"Req{i+1}" for i in range(len(self.requests))]) + "\n")
            
            num_loads = len(next(iter(results.values())))
            
            for load_idx in range(num_loads):
                if load_idx == 0:
                    f.write(f"0\t")
                else:
                    f.write(f"{load_idx}\t")
                
                values = []
                for req_key in sorted(results.keys()):
                    if load_idx < len(results[req_key]):
                        values.append(f"{results[req_key][load_idx]:.6f}")
                    else:
                        values.append("N/A")
                f.write("\t".join(values) + "\n")
            
            # Estatísticas das rotas usadas
            stats = self.get_path_statistics()
            if stats:
                f.write(f"\n=== ESTATÍSTICAS DAS ROTAS USADAS ===\n")
                f.write(f"Total de chamadas bem-sucedidas: {stats['total_calls']}\n")
                f.write(f"Rota média usada: {stats['avg_path_used']:.3f}\n")
                f.write(f"Distribuição de uso das rotas:\n")
                if 'path_distribution' in stats:
                    for path_num, prob in sorted(stats['path_distribution'].items()):
                        f.write(f"  Rota {path_num}: {prob:.3%}\n")
                f.write(f"  Uso da primeira rota: {stats.get('first_route_usage', 0):.3%}\n")
                f.write(f"  Uso da segunda rota: {stats.get('second_route_usage', 0):.3%}\n")
            
            # Estatísticas gerais por requisição
            f.write(f"\n=== ESTATÍSTICAS POR REQUISIÇÃO (Cargas 1-200) ===\n")
            for req_key in sorted(results.keys()):
                blocking_probs = results[req_key][1:]  # Ignora carga 0
                if blocking_probs:
                    f.write(f"\n{req_key}:\n")
                    f.write(f"  Média: {np.mean(blocking_probs):.6f}\n")
                    f.write(f"  Máximo: {np.max(blocking_probs):.6f}\n")
                    f.write(f"  Mínimo: {np.min(blocking_probs):.6f}\n")
                    f.write(f"  Desvio Padrão: {np.std(blocking_probs):.6f}\n")
                    f.write(f"  Média em %: {np.mean(blocking_probs)*100:.4f}%\n")
        
        print(f"\nResultados salvos em: {output_file}")
    
    def plot_results(self, results: Dict[str, List[float]], 
                    save_path: str = "yen_two_routes_random_plot.png",
                    xlim_max: int = 200):
        """
        Gera gráfico dos resultados.
        """
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        })
        
        plt.figure(figsize=(12, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for idx, (req_key, blocking_probs) in enumerate(sorted(results.items())):
            loads = np.arange(0, len(blocking_probs))
            plt.plot(loads[:xlim_max+1], 
                    [p * 100 for p in blocking_probs][:xlim_max+1],
                    label=f'Requisição {req_key}',
                    color=colors[idx % len(colors)],
                    linewidth=2,
                    marker='o' if idx < 3 else 's',
                    markersize=5,
                    markevery=10)
        
        plt.xlabel('Carga de Tráfego', fontsize=14)
        plt.ylabel('Probabilidade de Bloqueio (%)', fontsize=14)
        plt.title('YEN 2 Rotas Aleatórias - Probabilidade de Bloqueio vs Carga de Tráfego\n'
                 f'(Cargas 0-{xlim_max}, First Fit, Seleção Aleatória entre 2 Rotas)',
                 fontsize=16, pad=20)
        plt.legend(loc='upper left', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
        plt.xlim(0, xlim_max)
        plt.ylim(0, 100)
        
        # Adicionar ticks a cada 20 unidades
        plt.xticks(range(0, xlim_max + 1, 20 if xlim_max > 50 else 10))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Gráfico salvo em: {save_path}")


def main():
    """Função principal para executar simulação YEN com 2 rotas aleatórias."""
    start_time = time.time()
    
    print("\n" + "="*60)
    print("SIMULAÇÃO WDM - YEN COM 2 ROTAS ALEATÓRIAS")
    print("="*60)
    
    # Criação do grafo NSFNet
    graph = nx.Graph()
    nsfnet_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
        (4, 6), (4, 5), (5, 8), (5, 12), (6, 7), (7, 9), (8, 9), (9, 11),
        (9, 13), (10, 11), (10, 13), (11, 12)
    ]
    graph.add_edges_from(nsfnet_edges)
    
    # Requisições customizadas
    custom_requests = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
    
    print(f"\nParâmetros da simulação:")
    print(f"  Rede: NSFNet (14 nós, 21 enlaces)")
    print(f"  Wavelengths por enlace: 40")
    print(f"  Rotas por requisição: 2 (menores rotas do YEN)")
    print(f"  Seleção de rota: Aleatória entre as 2")
    print(f"  Alocação de wavelength: First Fit")
    print(f"  Requisições: {custom_requests}")
    print(f"  Rodadas por carga: 20")
    print(f"  Chamadas por rodada: 1000")
    print(f"  Cargas testadas: 1 a 200")
    print(f"  Duração das chamadas: 5.0 a 15.0 unidades de tempo")
    
    # Criar simulador
    simulator = WDMYenTwoRoutesRandom(
        graph=graph,
        num_wavelengths=40,
        k=2,  # Usa apenas 2 rotas
        requests=custom_requests
    )
    
    # Executar simulação
    print("\n>>> INICIANDO SIMULAÇÃO...")
    results = simulator.simulate_traffic_multiple_runs(
        num_runs=20,
        total_calls_per_load=1000,
        max_load=200
    )
    
    # Tempo de execução
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Salvar resultados
    simulator.save_results_to_file(
        results=results,
        output_file="yen_two_routes_random_results.txt",
        strategy_name="YEN 2 ROTAS ALEATÓRIAS (First Fit)"
    )
    
    # Gerar gráficos
    print("\n>>> GERANDO GRÁFICOS...")
    
    # Gráfico completo 0-200
    simulator.plot_results(
        results=results,
        save_path="yen_two_routes_random_full.png",
        xlim_max=200
    )
    
    # Gráfico focado 0-30 (para visualização detalhada)
    simulator.plot_results(
        results=results,
        save_path="yen_two_routes_random_0_30.png",
        xlim_max=30
    )
    
    # Salvar tempo de execução
    with open("execution_time_yen_random.txt", "w") as f:
        f.write("=== TEMPO DE EXECUÇÃO - YEN 2 ROTAS ALEATÓRIAS ===\n")
        f.write(f"Tempo total: {execution_time:.2f} segundos\n")
        f.write(f"  ({execution_time/60:.2f} minutos)\n")
        f.write(f"Início: {time.ctime(start_time)}\n")
        f.write(f"Término: {time.ctime(end_time)}\n")
        f.write(f"\nParâmetros:\n")
        f.write(f"  Rede: NSFNet (14 nós, 21 enlaces)\n")
        f.write(f"  Wavelengths por enlace: 40\n")
        f.write(f"  Rotas por requisição: 2\n")
        f.write(f"  Requisições: {custom_requests}\n")
        f.write(f"  Rodadas por carga: 20\n")
        f.write(f"  Chamadas por rodada: 1000\n")
        f.write(f"  Cargas testadas: 1 a 200\n")
    
    print(f"\n" + "="*60)
    print("SIMULAÇÃO CONCLUÍDA!")
    print(f"Tempo total de execução: {execution_time:.2f} segundos")
    print(f"  ({execution_time/60:.2f} minutos)")
    print("="*60)
    
    print("\nArquivos gerados:")
    print("  - yen_two_routes_random_results.txt: Resultados da simulação")
    print("  - yen_two_routes_random_full.png: Gráfico completo 0-200")
    print("  - yen_two_routes_random_0_30.png: Gráfico detalhado 0-30")
    print("  - execution_time_yen_random.txt: Tempo de execução")
    
    # Mostrar estatísticas
    stats = simulator.get_path_statistics()
    if stats:
        print(f"\nEstatísticas das rotas usadas:")
        print(f"  Total de chamadas bem-sucedidas: {stats['total_calls']}")
        print(f"  Rota média usada: {stats['avg_path_used']:.3f}")
        print(f"  Distribuição de uso das rotas:")
        if 'path_distribution' in stats:
            for path_num, prob in sorted(stats['path_distribution'].items()):
                print(f"    Rota {path_num}: {prob:.3%}")
        print(f"  Uso da primeira rota: {stats.get('first_route_usage', 0):.3%}")
        print(f"  Uso da segunda rota: {stats.get('second_route_usage', 0):.3%}")


if __name__ == "__main__":
    main()