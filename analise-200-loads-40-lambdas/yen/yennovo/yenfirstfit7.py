import os
from itertools import islice
from typing import List, Tuple, Dict, Optional
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


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


class WDMShortestPathOnly(WDMYenBase):
    """
    Simulador WDM usando apenas o menor caminho (como no código original).
    - Sempre usa o primeiro caminho do YEN (k[0])
    - Não tenta caminhos alternativos
    """
    
    def process_call(self, source: int, target: int, call_id: int,
                     call_duration: float) -> Tuple[bool, Optional[List[int]], Optional[Dict[int, int]]]:
        """
        Processa uma chamada usando apenas o menor caminho.
        """
        self._release_expired_wavelengths()
        
        paths = self._get_k_shortest_paths(source, target, self.k)
        
        if not paths:
            return (True, None, None)
        
        route = paths[0]
        wavelength_allocation = self.first_fit_allocation_with_conversion(route, call_duration)
        
        if wavelength_allocation is None:
            return (True, None, None)
        
        self.call_records.append({
            'call_id': call_id,
            'source': source,
            'target': target,
            'route': route,
            'wavelength_allocation': wavelength_allocation,
            'blocked': False,
            'hops': len(route) - 1,
            'start_time': self.current_time,
            'end_time': self.current_time + call_duration,
            'duration': call_duration,
            'path_used': 1,
            'strategy': 'ShortestPathOnly'
        })
        
        return (False, route, wavelength_allocation)
    
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
                print(f"\n=== CARGA {load}/{max_load} (ShortestPathOnly) ===")
            
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


class WDMTraditionalYen(WDMYenBase):
    """
    Simulador WDM usando YEN tradicional.
    - Tenta todos os k caminhos em ordem
    - Usa o primeiro caminho disponível
    """
    
    def process_call(self, source: int, target: int, call_id: int,
                     call_duration: float) -> Tuple[bool, Optional[List[int]], Optional[Dict[int, int]]]:
        """
        Processa uma chamada usando YEN tradicional: tenta todos os caminhos em ordem.
        """
        self._release_expired_wavelengths()
        
        paths = self._get_k_shortest_paths(source, target, self.k)
        
        if not paths:
            return (True, None, None)
        
        for path_idx, route in enumerate(paths):
            wavelength_allocation = self.first_fit_allocation_with_conversion(route, call_duration)
            
            if wavelength_allocation is not None:
                self.call_records.append({
                    'call_id': call_id,
                    'source': source,
                    'target': target,
                    'route': route,
                    'wavelength_allocation': wavelength_allocation,
                    'blocked': False,
                    'hops': len(route) - 1,
                    'start_time': self.current_time,
                    'end_time': self.current_time + call_duration,
                    'duration': call_duration,
                    'path_used': path_idx + 1,
                    'strategy': 'TraditionalYen'
                })
                
                return (False, route, wavelength_allocation)
        
        return (True, None, None)
    
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
                print(f"\n=== CARGA {load}/{max_load} (TraditionalYen) ===")
            
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
        
        total_calls = len([r for r in self.call_records if not r['blocked']])
        path_usage = {}
        
        for record in self.call_records:
            if not record['blocked']:
                path_num = record.get('path_used', 1)
                path_usage[path_num] = path_usage.get(path_num, 0) + 1
        
        return {
            'total_calls': total_calls,
            'path_distribution': {k: v/total_calls for k, v in sorted(path_usage.items())},
            'avg_path_attempts': np.mean([r.get('path_used', 1) for r in self.call_records if not r.get('blocked', False)])
        }


class SimulationComparator:
    """
    Compara os dois métodos de roteamento.
    """
    
    def __init__(self, graph: nx.Graph, requests: List[Tuple[int, int]]):
        self.graph = graph
        self.requests = requests
    
    def run_comparison(self, num_runs: int = 20, 
                      total_calls_per_load: int = 1000,
                      num_wavelengths: int = 40,
                      k: int = 3,
                      max_load: int = 200) -> Tuple[Dict, Dict, Dict]:
        """
        Executa ambas as simulações e retorna resultados.
        """
        print("\n" + "="*60)
        print("COMPARAÇÃO: SHORTEST PATH ONLY vs TRADITIONAL YEN")
        print(f"Cargas: 1 a {max_load}")
        print("="*60)
        
        # Executar ShortestPathOnly
        print("\n>>> EXECUTANDO SHORTEST PATH ONLY...")
        start_time_shortest = time.time()
        
        shortest_sim = WDMShortestPathOnly(
            graph=self.graph,
            num_wavelengths=num_wavelengths,
            k=k,
            requests=self.requests
        )
        
        results_shortest = shortest_sim.simulate_traffic_multiple_runs(
            num_runs=num_runs,
            total_calls_per_load=total_calls_per_load,
            max_load=max_load
        )
        
        end_time_shortest = time.time()
        shortest_time = end_time_shortest - start_time_shortest
        
        # Executar TraditionalYen
        print("\n>>> EXECUTANDO TRADITIONAL YEN...")
        start_time_yen = time.time()
        
        yen_sim = WDMTraditionalYen(
            graph=self.graph,
            num_wavelengths=num_wavelengths,
            k=k,
            requests=self.requests
        )
        
        results_yen = yen_sim.simulate_traffic_multiple_runs(
            num_runs=num_runs,
            total_calls_per_load=total_calls_per_load,
            max_load=max_load
        )
        
        end_time_yen = time.time()
        yen_time = end_time_yen - start_time_yen
        
        # Estatísticas do TraditionalYen
        yen_stats = yen_sim.get_path_statistics()
        
        return results_shortest, results_yen, {
            'execution_time_shortest': shortest_time,
            'execution_time_yen': yen_time,
            'yen_path_stats': yen_stats,
            'max_load': max_load
        }
    
    def save_comparison_results(self, results_shortest: Dict, results_yen: Dict, 
                               stats: Dict, output_prefix: str = "comparison"):
        """
        Salva resultados da comparação.
        """
        max_load = stats.get('max_load', 200)
        
        # Salvar resultados individuais
        self._save_single_results(results_shortest, f"{output_prefix}_shortest.txt", 
                                 "SHORTEST PATH ONLY", max_load)
        self._save_single_results(results_yen, f"{output_prefix}_yen.txt", 
                                 "TRADITIONAL YEN", max_load)
        
        # Salvar comparação combinada
        comparison_file = f"{output_prefix}_summary.txt"
        with open(comparison_file, "w") as f:
            f.write("="*60 + "\n")
            f.write("COMPARAÇÃO: SHORTEST PATH ONLY vs TRADITIONAL YEN\n")
            f.write(f"Cargas: 1 a {max_load}\n")
            f.write("="*60 + "\n\n")
            
            f.write("PARÂMETROS DA SIMULAÇÃO:\n")
            f.write(f"  Rede: NSFNet (14 nós, 21 enlaces)\n")
            f.write(f"  Wavelengths por enlace: 40\n")
            f.write(f"  Caminhos YEN (k): 3\n")
            f.write(f"  Requisições: {self.requests}\n")
            f.write(f"  Rodadas por carga: 20\n")
            f.write(f"  Chamadas por rodada: 1000\n")
            f.write(f"  Cargas testadas: 1 a {max_load}\n\n")
            
            f.write("TEMPOS DE EXECUÇÃO:\n")
            f.write(f"  Shortest Path Only: {stats['execution_time_shortest']:.2f} segundos\n")
            f.write(f"  Traditional YEN: {stats['execution_time_yen']:.2f} segundos\n")
            f.write(f"  Diferença: {stats['execution_time_yen'] - stats['execution_time_shortest']:.2f} segundos\n\n")
            
            f.write("ESTATÍSTICAS DO TRADITIONAL YEN:\n")
            if stats['yen_path_stats']:
                f.write(f"  Total de chamadas bem-sucedidas analisadas: {stats['yen_path_stats'].get('total_calls', 0)}\n")
                f.write(f"  Tentativas médias por chamada: {stats['yen_path_stats'].get('avg_path_attempts', 1):.3f}\n")
                f.write(f"  Distribuição de caminhos usados:\n")
                if 'path_distribution' in stats['yen_path_stats']:
                    for path_num, prob in stats['yen_path_stats']['path_distribution'].items():
                        f.write(f"    Caminho {path_num}: {prob:.3%}\n")
            f.write("\n")
            
            f.write("RESULTADOS POR REQUISIÇÃO (Média 1-{}):\n".format(max_load))
            f.write("Req\tShortestPathOnly\tTraditionalYEN\tDiferença\tMelhoria\n")
            
            for req_key in sorted(results_shortest.keys()):
                # Média das cargas 1-max_load (ignorando carga 0)
                avg_shortest = np.mean(results_shortest[req_key][1:max_load+1]) * 100
                avg_yen = np.mean(results_yen[req_key][1:max_load+1]) * 100
                diff = avg_yen - avg_shortest
                improvement = ((avg_shortest - avg_yen) / avg_shortest * 100) if avg_shortest > 0 else 0
                
                f.write(f"{req_key}\t{avg_shortest:.4f}%\t{avg_yen:.4f}%\t{diff:+.4f}%\t{improvement:+.2f}%\n")
        
        print(f"\nResultados salvos em:")
        print(f"  - {output_prefix}_shortest.txt")
        print(f"  - {output_prefix}_yen.txt")
        print(f"  - {comparison_file}")
    
    def _save_single_results(self, results: Dict[str, List[float]], 
                            output_file: str, strategy_name: str, max_load: int):
        """
        Salva resultados de uma única estratégia.
        """
        with open(output_file, "w") as f:
            f.write(f"=== {strategy_name} ===\n\n")
            f.write("Blocking Probability Results:\n")
            f.write("Load\t" + "\t".join([f"Req{i+1}" for i in range(len(self.requests))]) + "\n")
            
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
            
            # Estatísticas
            f.write(f"\n=== ESTATÍSTICAS (Load 1-{max_load}) ===\n")
            for req_key in sorted(results.keys()):
                blocking_probs = results[req_key][1:max_load+1]  # Apenas cargas 1-max_load
                if blocking_probs:
                    f.write(f"\n{req_key}:\n")
                    f.write(f"  Média: {np.mean(blocking_probs):.6f}\n")
                    f.write(f"  Máx: {np.max(blocking_probs):.6f}\n")
                    f.write(f"  Mín: {np.min(blocking_probs):.6f}\n")
                    f.write(f"  Desvio Padrão: {np.std(blocking_probs):.6f}\n")
    
    def plot_comparison(self, results_shortest: Dict, results_yen: Dict,
                       save_path: str = "comparison_results.png", 
                       xlim_max: int = 30):
        """
        Gera gráfico comparativo dos dois métodos.
        Parâmetro xlim_max permite escolher até qual carga mostrar no gráfico.
        """
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })
        
        # Criar figura maior para incluir todos os gráficos
        num_requests = len(self.requests)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors_shortest = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        colors_yen = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
        
        for idx, (req_key, _) in enumerate(sorted(results_shortest.items())):
            if idx >= len(axes) - 1:  # -1 para reservar o último para o gráfico geral
                break
            
            ax = axes[idx]
            
            # Shortest Path Only
            loads = np.arange(0, len(results_shortest[req_key]))
            blocking_shortest = [p * 100 for p in results_shortest[req_key]]
            
            # Traditional YEN
            blocking_yen = [p * 100 for p in results_yen[req_key]]
            
            ax.plot(loads[:xlim_max+1], blocking_shortest[:xlim_max+1], 
                   label='Shortest Path Only',
                   color=colors_shortest[idx % len(colors_shortest)],
                   linewidth=1.5,
                   marker='o',
                   markersize=4)
            
            ax.plot(loads[:xlim_max+1], blocking_yen[:xlim_max+1],
                   label='Traditional YEN',
                   color=colors_yen[idx % len(colors_yen)],
                   linewidth=1.5,
                   marker='s',
                   markersize=4,
                   linestyle='--')
            
            ax.set_xlabel('Load')
            ax.set_ylabel('Blocking Probability (%)')
            ax.set_title(f'Requisição {req_key}')
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
            ax.set_xlim(0, xlim_max)
            ax.set_ylim(0, 100)
        
        # Gráfico comparativo geral (último subplot)
        ax = axes[-1]
        
        # Calcular médias gerais por carga
        avg_shortest = []
        avg_yen = []
        
        num_loads = len(next(iter(results_shortest.values())))
        for load_idx in range(min(num_loads, xlim_max + 1)):
            sum_shortest = 0
            sum_yen = 0
            count = 0
            
            for req_key in results_shortest.keys():
                if load_idx < len(results_shortest[req_key]):
                    sum_shortest += results_shortest[req_key][load_idx]
                    sum_yen += results_yen[req_key][load_idx]
                    count += 1
            
            if count > 0:
                avg_shortest.append(sum_shortest / count * 100)
                avg_yen.append(sum_yen / count * 100)
        
        loads = np.arange(0, len(avg_shortest))
        
        ax.plot(loads, avg_shortest,
               label='Shortest Path Only (média)',
               color='#1f77b4',
               linewidth=2,
               marker='o',
               markersize=5)
        
        ax.plot(loads, avg_yen,
               label='Traditional YEN (média)',
               color='#ff7f0e',
               linewidth=2,
               marker='s',
               markersize=5,
               linestyle='--')
        
        ax.set_xlabel('Load')
        ax.set_ylabel('Blocking Probability (%)')
        ax.set_title('Média Geral de Todas as Requisições')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
        ax.set_xlim(0, xlim_max)
        ax.set_ylim(0, 100)
        
        plt.suptitle(f'Comparação: Shortest Path Only vs Traditional YEN\nBlocking Probability vs Traffic Load (0-{xlim_max})', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Gráfico comparativo salvo em: {save_path}")
    
    def plot_full_comparison(self, results_shortest: Dict, results_yen: Dict,
                           save_path: str = "comparison_full.png"):
        """
        Gera gráfico com todas as cargas (0-200).
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
        
        # Calcular médias gerais por carga
        avg_shortest = []
        avg_yen = []
        
        num_loads = len(next(iter(results_shortest.values())))
        for load_idx in range(num_loads):
            sum_shortest = 0
            sum_yen = 0
            count = 0
            
            for req_key in results_shortest.keys():
                if load_idx < len(results_shortest[req_key]):
                    sum_shortest += results_shortest[req_key][load_idx]
                    sum_yen += results_yen[req_key][load_idx]
                    count += 1
            
            if count > 0:
                avg_shortest.append(sum_shortest / count * 100)
                avg_yen.append(sum_yen / count * 100)
        
        loads = np.arange(0, len(avg_shortest))
        
        plt.plot(loads, avg_shortest,
                label='Shortest Path Only (média)',
                color='#1f77b4',
                linewidth=2.5,
                marker='o',
                markersize=6,
                markevery=10)
        
        plt.plot(loads, avg_yen,
                label='Traditional YEN (média)',
                color='#ff7f0e',
                linewidth=2.5,
                marker='s',
                markersize=6,
                linestyle='--',
                markevery=10)
        
        plt.xlabel('Load', fontsize=14)
        plt.ylabel('Blocking Probability (%)', fontsize=14)
        plt.title('Comparação: Shortest Path Only vs Traditional YEN\nMédia Geral de Todas as Requisições (Cargas 0-200)',
                 fontsize=16, pad=20)
        plt.legend(loc='upper left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
        plt.xlim(0, 200)
        plt.ylim(0, 100)
        
        # Adicionar ticks a cada 20 unidades
        plt.xticks(range(0, 201, 20))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Gráfico completo (0-200) salvo em: {save_path}")


def main():
    """Função principal para executar comparação."""
    start_time = time.time()
    
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
    
    print("\n" + "="*60)
    print("SIMULAÇÃO WDM - COMPARAÇÃO DE ESTRATÉGIAS DE ROTEAMENTO")
    print("="*60)
    print(f"\nParâmetros:")
    print(f"  Rede: NSFNet (14 nós, 21 enlaces)")
    print(f"  Wavelengths por enlace: 40")
    print(f"  Caminhos YEN (k): 3")
    print(f"  Requisições: {custom_requests}")
    print(f"  Rodadas por carga: 20")
    print(f"  Chamadas por rodada: 1000")
    print(f"  Cargas testadas: 1 a 200")
    
    # Executar comparação
    comparator = SimulationComparator(graph, custom_requests)
    
    results_shortest, results_yen, stats = comparator.run_comparison(
        num_runs=20,
        total_calls_per_load=1000,
        num_wavelengths=40,
        k=3,
        max_load=200  # Mantendo 200 cargas como no original
    )
    
    # Salvar resultados
    comparator.save_comparison_results(results_shortest, results_yen, stats, "wdm_comparison")
    
    # Gerar gráficos comparativos
    # Gráfico focado em 0-30 (para comparação visual)
    comparator.plot_comparison(results_shortest, results_yen, 
                              "comparison_0_30.png", xlim_max=30)
    
    # Gráfico completo 0-200
    comparator.plot_full_comparison(results_shortest, results_yen, 
                                   "comparison_full_0_200.png")
    
    # Tempo total
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n" + "="*60)
    print("COMPARAÇÃO CONCLUÍDA!")
    print(f"Tempo total de execução: {total_time:.2f} segundos")
    print(f"  Shortest Path Only: {stats['execution_time_shortest']:.2f} segundos")
    print(f"  Traditional YEN: {stats['execution_time_yen']:.2f} segundos")
    print("="*60)
    
    # Salvar tempo total
    with open("execution_time_total.txt", "w") as f:
        f.write("=== TEMPO TOTAL DE EXECUÇÃO ===\n")
        f.write(f"Tempo total: {total_time:.2f} segundos\n")
        f.write(f"  Shortest Path Only: {stats['execution_time_shortest']:.2f} segundos\n")
        f.write(f"  Traditional YEN: {stats['execution_time_yen']:.2f} segundos\n")
        f.write(f"  Diferença: {stats['execution_time_yen'] - stats['execution_time_shortest']:.2f} segundos\n")
        f.write(f"Início: {time.ctime(start_time)}\n")
        f.write(f"Término: {time.ctime(end_time)}\n")
    
    print("\nArquivos gerados:")
    print("  - wdm_comparison_shortest.txt: Resultados Shortest Path Only")
    print("  - wdm_comparison_yen.txt: Resultados Traditional YEN")
    print("  - wdm_comparison_summary.txt: Resumo comparativo")
    print("  - comparison_0_30.png: Gráfico 0-30")
    print("  - comparison_full_0_200.png: Gráfico completo 0-200")
    print("  - execution_time_total.txt: Tempos de execução")


if __name__ == "__main__":
    main()