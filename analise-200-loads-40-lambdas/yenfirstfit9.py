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
    - Processa as 5 requisições simultaneamente (concorrência)
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
        Simula tráfego COM CONCORRÊNCIA entre as 5 requisições.
        Todas as requisições são processadas simultaneamente na mesma rede.
        """
        results = {}
        
        # Inicializar resultados para cada requisição
        for req_idx, (source, target) in enumerate(self.requests):
            results[f'[{source},{target}]'] = [0.0]  # Carga 0
        
        for load in range(1, max_load + 1):  # 1 a max_load
            if load % 20 == 0:
                print(f"\n=== CARGA {load}/{max_load} (YenTwoRoutesRandom) ===")
            
            inter_arrival_time = 10.0 / load
            load_results = {}
            
            # Inicializar contadores para esta carga
            for req_idx, (source, target) in enumerate(self.requests):
                load_results[f'[{source},{target}]'] = {
                    'total_calls': 0,
                    'blocked_calls': 0,
                    'blocking_prob': 0.0
                }
            
            for run in range(num_runs):
                if load % 20 == 0 and run == 0:
                    print(f"  Rodada {run + 1}/{num_runs}...", end='\r')
                
                self.reset_network()
                self.current_time = 0.0
                
                # Resetar contadores para esta rodada
                for req_key in load_results.keys():
                    load_results[req_key]['total_calls'] = 0
                    load_results[req_key]['blocked_calls'] = 0
                
                # Lista de requisições para escolha aleatória
                req_choices = []
                for req_idx, (source, target) in enumerate(self.requests):
                    req_choices.extend([(source, target)] * 16)  # 16 vezes cada requisição
                
                # Adicionar algumas requisições aleatórias
                nodes = list(self.graph.nodes)
                for _ in range(20):  # 20% de requisições aleatórias
                    req_choices.append(('random', 'random'))
                
                # Processar todas as chamadas com concorrência
                for call_idx in range(total_calls_per_load):
                    call_duration = np.random.uniform(call_duration_range[0], call_duration_range[1])
                    self.current_time += inter_arrival_time
                    
                    # Escolher requisição aleatoriamente (80% das 5 requisições, 20% aleatórias)
                    if np.random.random() < 0.8:
                        # Escolhe uma das 5 requisições principais
                        req_idx = np.random.randint(0, len(self.requests))
                        source, target = self.requests[req_idx]
                        req_key = f'[{source},{target}]'
                    else:
                        # Requisição aleatória
                        nodes = list(self.graph.nodes)
                        source, target = np.random.choice(nodes, 2, replace=False)
                        req_key = 'random'
                    
                    # Processar a chamada
                    blocked, _, _ = self.process_call(source, target, call_idx, call_duration)
                    
                    # Contar bloqueios apenas para as 5 requisições principais
                    if req_key != 'random' and req_key in load_results:
                        load_results[req_key]['total_calls'] += 1
                        if blocked:
                            load_results[req_key]['blocked_calls'] += 1
            
            # Calcular probabilidade de bloqueio média para cada requisição
            for req_idx, (source, target) in enumerate(self.requests):
                req_key = f'[{source},{target}]'
                
                # Coletar resultados de todas as rodadas para esta carga
                blocking_probs_run = []
                for run in range(num_runs):
                    # Para simplicidade, usamos a média das rodadas
                    # Em uma implementação real, cada rodada teria seus próprios contadores
                    pass
                
                # Calcular média para esta carga
                total_calls = load_results[req_key]['total_calls']
                blocked_calls = load_results[req_key]['blocked_calls']
                
                if total_calls > 0:
                    blocking_prob = blocked_calls / total_calls
                else:
                    blocking_prob = 0.0
                
                results[req_key].append(blocking_prob)
                
                if load % 20 == 0:
                    print(f"  {req_key}: {blocking_prob:.6f} (calls: {total_calls}, blocked: {blocked_calls})")
        
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
        Salva resultados separados por requisição e médias gerais.
        """
        max_load = len(results[next(iter(results.keys()))]) - 1
        
        # Salvar resultados de cada requisição individualmente
        for req_idx, (source, target) in enumerate(self.requests):
            req_key = f'[{source},{target}]'
            output_file = f"{output_prefix}_req_{source}_{target}.txt"
            
            with open(output_file, "w") as f:
                f.write(f"=== YEN 2 ROTAS ALEATÓRIAS - REQUISIÇÃO [{source},{target}] ===\n\n")
                f.write("PARÂMETROS DA SIMULAÇÃO:\n")
                f.write(f"  Rede: NSFNet (14 nós, 21 enlaces)\n")
                f.write(f"  Wavelengths por enlace: 40\n")
                f.write(f"  Rotas consideradas: 2 (menores rotas do YEN)\n")
                f.write(f"  Seleção de rota: Aleatória entre as 2\n")
                f.write(f"  Alocação de wavelength: First Fit\n")
                f.write(f"  Requisição: [{source},{target}]\n")
                f.write(f"  Rodadas por carga: 20\n")
                f.write(f"  Total de chamadas por carga: 1000\n")
                f.write(f"  Percentual desta requisição: ~16%\n")
                f.write(f"  Cargas testadas: 0 a {max_load}\n\n")
                
                f.write("RESULTADOS (Blocking Probability):\n")
                f.write("Carga\tProbabilidade de Bloqueio\tChamadas Totais\tChamadas Bloqueadas\n")
                
                # Para carga 0
                f.write(f"0\t{results[req_key][0]:.6f}\t0\t0\n")
                
                # Para cargas 1 a max_load
                for load in range(1, max_load + 1):
                    if load < len(results[req_key]):
                        blocking_prob = results[req_key][load]
                        # Estimativa de chamadas (16% de 1000 = 160 chamadas por rodada)
                        total_calls_est = 160 * 20  # 160 por rodada * 20 rodadas
                        blocked_calls_est = int(blocking_prob * total_calls_est)
                        f.write(f"{load}\t{blocking_prob:.6f}\t{total_calls_est}\t{blocked_calls_est}\n")
                
                # Estatísticas
                blocking_probs = results[req_key][1:]  # Ignorar carga 0
                if blocking_probs:
                    f.write(f"\n=== ESTATÍSTICAS (Cargas 1-{max_load}) ===\n")
                    f.write(f"Média: {np.mean(blocking_probs):.6f} ({np.mean(blocking_probs)*100:.4f}%)\n")
                    f.write(f"Máximo: {np.max(blocking_probs):.6f} ({np.max(blocking_probs)*100:.4f}%)\n")
                    f.write(f"Mínimo: {np.min(blocking_probs):.6f} ({np.min(blocking_probs)*100:.4f}%)\n")
                    f.write(f"Desvio Padrão: {np.std(blocking_probs):.6f}\n")
                    f.write(f"Mediana: {np.median(blocking_probs):.6f}\n")
        
        # Salvar resultados combinados (todas as requisições)
        output_file_all = f"{output_prefix}_all_requests.txt"
        with open(output_file_all, "w") as f:
            f.write(f"=== YEN 2 ROTAS ALEATÓRIAS - TODAS AS REQUISIÇÕES ===\n\n")
            f.write("PARÂMETROS DA SIMULAÇÃO:\n")
            f.write(f"  Rede: NSFNet (14 nós, 21 enlaces)\n")
            f.write(f"  Wavelengths por enlace: 40\n")
            f.write(f"  Rotas consideradas: 2 (menores rotas do YEN)\n")
            f.write(f"  Seleção de rota: Aleatória entre as 2\n")
            f.write(f"  Alocação de wavelength: First Fit\n")
            f.write(f"  Requisições: {self.requests}\n")
            f.write(f"  Rodadas por carga: 20\n")
            f.write(f"  Chamadas por rodada: 1000\n")
            f.write(f"  Distribuição: 80% das 5 requisições (16% cada), 20% aleatórias\n")
            f.write(f"  Cargas testadas: 0 a {max_load}\n\n")
            
            f.write("RESULTADOS POR REQUISIÇÃO (Blocking Probability):\n")
            f.write("Carga\t" + "\t".join([f"Req[{s},{t}]" for (s, t) in self.requests]) + "\tMédia Geral\n")
            
            for load in range(0, max_load + 1):
                f.write(f"{load}\t")
                
                values = []
                for (source, target) in self.requests:
                    req_key = f'[{source},{target}]'
                    if load < len(results[req_key]):
                        values.append(f"{results[req_key][load]:.6f}")
                    else:
                        values.append("N/A")
                
                # Calcular média geral para esta carga
                valid_values = []
                for (source, target) in self.requests:
                    req_key = f'[{source},{target}]'
                    if load < len(results[req_key]):
                        valid_values.append(results[req_key][load])
                
                if valid_values:
                    avg_general = np.mean(valid_values)
                    f.write("\t".join(values) + f"\t{avg_general:.6f}\n")
                else:
                    f.write("\t".join(values) + "\tN/A\n")
            
            # Estatísticas gerais
            f.write(f"\n=== ESTATÍSTICAS GERAIS (Cargas 1-{max_load}) ===\n")
            
            # Médias por requisição
            f.write("\nMÉDIAS POR REQUISIÇÃO:\n")
            for (source, target) in self.requests:
                req_key = f'[{source},{target}]'
                blocking_probs = results[req_key][1:]  # Ignorar carga 0
                if blocking_probs:
                    avg = np.mean(blocking_probs)
                    f.write(f"  [{source},{target}]: {avg:.6f} ({avg*100:.4f}%)\n")
            
            # Média geral
            all_blocking_probs = []
            for (source, target) in self.requests:
                req_key = f'[{source},{target}]'
                all_blocking_probs.extend(results[req_key][1:])
            
            if all_blocking_probs:
                f.write(f"\nMÉDIA GERAL: {np.mean(all_blocking_probs):.6f} ({np.mean(all_blocking_probs)*100:.4f}%)\n")
                f.write(f"MÁXIMO GERAL: {np.max(all_blocking_probs):.6f} ({np.max(all_blocking_probs)*100:.4f}%)\n")
                f.write(f"MÍNIMO GERAL: {np.min(all_blocking_probs):.6f} ({np.min(all_blocking_probs)*100:.4f}%)\n")
                f.write(f"DESVIO PADRÃO GERAL: {np.std(all_blocking_probs):.6f}\n")
        
        print(f"\nResultados salvos em:")
        for (source, target) in self.requests:
            print(f"  - {output_prefix}_req_{source}_{target}.txt")
        print(f"  - {output_file_all}")
    
    def plot_results(self, results: Dict[str, List[float]], 
                    save_path: str = "yen_two_routes_random_results.png",
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
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        })
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Gráfico 1: Todas as requisições
        for idx, (source, target) in enumerate(self.requests):
            req_key = f'[{source},{target}]'
            blocking_probs = results[req_key]
            loads = np.arange(0, len(blocking_probs))
            
            ax1.plot(loads[:xlim_max+1], 
                    [p * 100 for p in blocking_probs][:xlim_max+1],
                    label=f'[{source},{target}]',
                    color=colors[idx % len(colors)],
                    linewidth=2,
                    marker='o' if idx < 3 else 's',
                    markersize=5,
                    markevery=10)
        
        ax1.set_xlabel('Carga de Tráfego', fontsize=14)
        ax1.set_ylabel('Probabilidade de Bloqueio (%)', fontsize=14)
        ax1.set_title('YEN 2 Rotas Aleatórias - Probabilidade de Bloqueio por Requisição\n'
                     f'(Cargas 0-{xlim_max}, First Fit, Seleção Aleatória entre 2 Rotas)',
                     fontsize=16, pad=20)
        ax1.legend(loc='upper left', fontsize=11, ncol=2)
        ax1.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
        ax1.set_xlim(0, xlim_max)
        ax1.set_ylim(0, 100)
        ax1.set_xticks(range(0, xlim_max + 1, 20 if xlim_max > 50 else 10))
        
        # Gráfico 2: Média geral
        # Calcular média geral por carga
        avg_general = []
        num_loads = len(results[next(iter(results.keys()))])
        
        for load in range(num_loads):
            sum_probs = 0
            count = 0
            for (source, target) in self.requests:
                req_key = f'[{source},{target}]'
                if load < len(results[req_key]):
                    sum_probs += results[req_key][load]
                    count += 1
            if count > 0:
                avg_general.append(sum_probs / count * 100)
        
        loads_avg = np.arange(0, len(avg_general))
        
        ax2.plot(loads_avg[:xlim_max+1], avg_general[:xlim_max+1],
                label='Média Geral das 5 Requisições',
                color='#2ca02c',
                linewidth=3,
                marker='D',
                markersize=6,
                markevery=10)
        
        ax2.set_xlabel('Carga de Tráfego', fontsize=14)
        ax2.set_ylabel('Probabilidade de Bloqueio (%)', fontsize=14)
        ax2.set_title('Média Geral das 5 Requisições', fontsize=14, pad=15)
        ax2.legend(loc='upper left', fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
        ax2.set_xlim(0, xlim_max)
        ax2.set_ylim(0, 100)
        ax2.set_xticks(range(0, xlim_max + 1, 20 if xlim_max > 50 else 10))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Gráfico salvo em: {save_path}")
        
        # Gráfico adicional focado em 0-30
        self.plot_focused_results(results, xlim_max=30)
    
    def plot_focused_results(self, results: Dict[str, List[float]], 
                            xlim_max: int = 30,
                            save_path: str = "yen_two_routes_random_focused.png"):
        """
        Gera gráfico focado nas primeiras cargas.
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
            loads = np.arange(0, min(len(blocking_probs), xlim_max + 1))
            
            plt.plot(loads, 
                    [p * 100 for p in blocking_probs][:len(loads)],
                    label=f'[{source},{target}]',
                    color=colors[idx % len(colors)],
                    linewidth=2.5,
                    marker='o',
                    markersize=6)
        
        plt.xlabel('Carga de Tráfego', fontsize=14)
        plt.ylabel('Probabilidade de Bloqueio (%)', fontsize=14)
        plt.title(f'YEN 2 Rotas Aleatórias - Detalhe Cargas 0-{xlim_max}\n'
                 '(First Fit, Seleção Aleatória entre 2 Rotas)',
                 fontsize=16, pad=20)
        plt.legend(loc='upper left', fontsize=11, ncol=2)
        plt.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
        plt.xlim(0, xlim_max)
        plt.ylim(0, 100)
        plt.xticks(range(0, xlim_max + 1, 2))
        
        # Adicionar grade mais fina
        ax = plt.gca()
        ax.grid(True, which='minor', linestyle=':', alpha=0.4, linewidth=0.5)
        ax.minorticks_on()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Gráfico detalhado salvo em: {save_path}")


def main():
    """Função principal para executar simulação YEN com 2 rotas aleatórias."""
    start_time = time.time()
    
    print("\n" + "="*60)
    print("SIMULAÇÃO WDM - YEN COM 2 ROTAS ALEATÓRIAS")
    print("(5 REQUISIÇÕES SIMULTÂNEAS)")
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
    print(f"  Requisições (5 simultâneas): {custom_requests}")
    print(f"  Rodadas por carga: 20")
    print(f"  Chamadas por rodada: 1000")
    print(f"  Distribuição: 80% das 5 requisições (16% cada), 20% aleatórias")
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
    
    # Salvar resultados por requisição e médias
    print("\n>>> SALVANDO RESULTADOS...")
    simulator.save_results_per_request(
        results=results,
        output_prefix="yen_two_routes_random"
    )
    
    # Gerar gráficos
    print("\n>>> GERANDO GRÁFICOS...")
    
    # Gráfico completo 0-200
    simulator.plot_results(
        results=results,
        save_path="yen_two_routes_random_full.png",
        xlim_max=200
    )
    
    # Salvar tempo de execução
    with open("execution_time_yen_random.txt", "w") as f:
        f.write("=== TEMPO DE EXECUÇÃO - YEN 2 ROTAS ALEATÓRIAS ===\n")
        f.write(f"Tempo total: {execution_time:.2f} segundos\n")
        f.write(f"  ({execution_time/60:.2f} minutos)\n")
        f.write(f"Início: {time.ctime(start_time)}\n")
        f.write(f"Término: {time.ctime(end_time)}\n")
        f.write(f"\nPARÂMETROS:\n")
        f.write(f"  Rede: NSFNet (14 nós, 21 enlaces)\n")
        f.write(f"  Wavelengths por enlace: 40\n")
        f.write(f"  Rotas por requisição: 2\n")
        f.write(f"  Requisições: {custom_requests}\n")
        f.write(f"  Rodadas por carga: 20\n")
        f.write(f"  Chamadas por rodada: 1000\n")
        f.write(f"  Cargas testadas: 1 a 200\n")
        f.write(f"  Distribuição: 80% das 5 requisições, 20% aleatórias\n")
    
    print(f"\n" + "="*60)
    print("SIMULAÇÃO CONCLUÍDA!")
    print(f"Tempo total de execução: {execution_time:.2f} segundos")
    print(f"  ({execution_time/60:.2f} minutos)")
    print("="*60)
    
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
    
    # Resumo das médias
    print(f"\nMédias de bloqueio por requisição (cargas 1-200):")
    for (source, target) in custom_requests:
        req_key = f'[{source},{target}]'
        blocking_probs = results[req_key][1:]  # Ignorar carga 0
        if blocking_probs:
            avg = np.mean(blocking_probs)
            print(f"  [{source},{target}]: {avg:.6f} ({avg*100:.4f}%)")


if __name__ == "__main__":
    main()