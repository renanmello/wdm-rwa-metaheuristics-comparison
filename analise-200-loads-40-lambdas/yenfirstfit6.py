import os
from itertools import islice
from typing import List, Tuple, Dict, Optional

import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class WDMYenFirstFit:
    """
    Simulador WDM usando abordagem clássica:
    - Roteamento: YEN (sempre pega o menor caminho k[0])
    - Alocação de Wavelength: First Fit com conversão
    - Modelo Temporal: Chamadas ocupam wavelengths por tempo determinado
    """

    def __init__(self,
                 graph: nx.Graph,
                 num_wavelengths: int = 16,
                 k: int = 5,
                 requests: List[Tuple[int, int]] = None):
        """
        Inicializa o simulador com abordagem clássica.

        Args:
            graph: Grafo da rede
            num_wavelengths: Número de comprimentos de onda
            k: Número de k-shortest paths para YEN
            requests: Lista de requisições (origem, destino)
        """
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.k = k

        self.requests = requests if requests else [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]

        # Estrutura para armazenar alocações com tempo
        # (u, v): {wavelength: end_time}
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

        Args:
            source: Nó de origem
            target: Nó de destino
            k: Número de caminhos

        Returns:
            Lista com k menores caminhos
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
            # Encontra wavelengths expirados
            expired_wavelengths = [
                wl for wl, end_time in self.wavelength_allocation[edge].items()
                if end_time <= self.current_time
            ]

            # Remove wavelengths expirados
            for wl in expired_wavelengths:
                del self.wavelength_allocation[edge][wl]

    def first_fit_allocation_with_conversion(self, route: List[int],
                                             call_duration: float) -> Optional[Dict[int, int]]:
        """
        Aloca wavelengths usando First Fit com conversão permitida.
        Cada enlace pode usar um wavelength diferente.

        Args:
            route: Rota (lista de nós)
            call_duration: Duração da chamada (unidades de tempo)

        Returns:
            Dicionário {índice_enlace: wavelength} ou None se não puder alocar
        """
        wavelength_allocation_route = {}
        end_time = self.current_time + call_duration

        # Para cada enlace da rota
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            edge = (min(u, v), max(u, v))

            # Encontra o primeiro wavelength disponível neste enlace
            wavelength_found = None
            for wavelength in range(self.num_wavelengths):
                if wavelength not in self.wavelength_allocation[edge]:
                    wavelength_found = wavelength
                    break

            if wavelength_found is None:
                # Nenhum wavelength disponível neste enlace
                return None

            # Aloca este wavelength no enlace com tempo de expiração
            self.wavelength_allocation[edge][wavelength_found] = end_time
            wavelength_allocation_route[i] = wavelength_found

        return wavelength_allocation_route

    def route_with_yen_shortest(self, source: int, target: int) -> Optional[List[int]]:
        """
        Roteamento usando YEN: sempre pega o menor caminho (k[0]).

        Args:
            source: Nó origem
            target: Nó destino

        Returns:
            Rota selecionada (menor caminho) ou None se caminho não existe
        """
        paths = self._get_k_shortest_paths(source, target, self.k)

        if not paths:
            return None

        # YEN clássico: sempre retorna o primeiro (menor) caminho
        # Não verifica wavelength aqui - deixa para first_fit decidir
        return paths[0]

    def process_call(self, source: int, target: int, call_id: int,
                     call_duration: float) -> Tuple[bool, Optional[List[int]], Optional[Dict[int, int]]]:
        """
        Processa uma chamada usando YEN + First Fit com conversão.

        Args:
            source: Nó origem
            target: Nó destino
            call_id: ID da chamada
            call_duration: Duração da chamada

        Returns:
            (bloqueado, rota, wavelength_allocation)
        """
        # Libera wavelengths expirados
        self._release_expired_wavelengths()

        # Roteamento com YEN (sempre o menor caminho)
        route = self.route_with_yen_shortest(source, target)

        if route is None:
            return (True, None, None)

        # Alocação com First Fit (com conversão de wavelength permitida)
        wavelength_allocation = self.first_fit_allocation_with_conversion(route, call_duration)

        if wavelength_allocation is None:
            return (True, None, None)

        # Sucesso
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
            'duration': call_duration
        })

        return (False, route, wavelength_allocation)

    def simulate_traffic_multiple_runs(self, num_runs: int = 20,
                                       total_calls_per_load: int = 1000,
                                       call_duration_range: Tuple[float, float] = (5.0, 15.0)) -> Dict[
        str, List[float]]:
        """
        Simula tráfego com múltiplas execuções para cada carga (1 a 30).
        Carga controla o intervalo entre chegadas de chamadas.

        Args:
            num_runs: Número de rodadas por carga
            total_calls_per_load: Chamadas por rodada
            call_duration_range: Tupla (min_duration, max_duration)

        Returns:
            Dicionário com probabilidades de bloqueio médias por requisição
        """
        results = {}

        for req_idx, (source, target) in enumerate(self.requests):
            results[f'[{source},{target}]'] = [0.0]

        for load in range(1, 201):
            print(f"\n=== CARGA {load}/30 ===")

            # Carga controla o intervalo entre chegadas de chamadas
            # Carga alta = chamadas chegam mais frequentemente = menos tempo para liberar wavelengths
            inter_arrival_time = 10.0 / load

            load_results = {}
            for req_idx, (source, target) in enumerate(self.requests):
                load_results[f'[{source},{target}]'] = []

            for run in range(num_runs):
                print(f"Rodada {run + 1}/{num_runs} (inter-arrival: {inter_arrival_time:.2f})...")
                self.reset_network()
                self.current_time = 0.0

                for req_idx, (source, target) in enumerate(self.requests):
                    blocked_count = 0

                    for call_idx in range(total_calls_per_load):
                        # Duração aleatória da chamada
                        call_duration = np.random.uniform(call_duration_range[0], call_duration_range[1])

                        # Incremento de tempo = intervalo entre chegadas
                        self.current_time += inter_arrival_time

                        # 80% para requisição específica, 20% aleatório
                        if np.random.random() < 0.8:
                            s, t = source, target
                        else:
                            nodes = list(self.graph.nodes)
                            s, t = np.random.choice(nodes, 2, replace=False)

                        blocked, route, wavelength_alloc = self.process_call(s, t, call_idx, call_duration)

                        if blocked:
                            blocked_count += 1

                    blocking_prob = blocked_count / total_calls_per_load
                    load_results[f'[{source},{target}]'].append(blocking_prob)

            for req_idx, (source, target) in enumerate(self.requests):
                req_key = f'[{source},{target}]'
                avg_blocking = np.mean(load_results[req_key])
                results[req_key].append(avg_blocking)
                print(f"  {req_key}: {avg_blocking:.6f}")

        return results

    def save_results(self, results: Dict[str, List[float]],
                     output_file: str = "yen_firstfit_results.txt") -> None:
        """
        Salva resultados em arquivo de texto.

        Args:
            results: Resultados da simulação
            output_file: Arquivo de saída
        """
        with open(output_file, "w") as f:
            f.write("=== YEN ROUTING + FIRST FIT WAVELENGTH ALLOCATION ===\n\n")
            f.write(f"Network Parameters:\n")
            f.write(f"  Wavelengths: {self.num_wavelengths}\n")
            f.write(f"  YEN Strategy: Always shortest path (k[0])\n")
            f.write(f"  Wavelength Strategy: First Fit (WITH WAVELENGTH CONVERSION)\n")
            f.write(f"  Temporal Model: Calls have duration and release wavelengths when expire\n")
            f.write(f"  Requests tested: {self.requests}\n\n")

            f.write("Blocking Probability Results:\n")
            f.write("Load\t" + "\t".join([f"Req{i + 1}" for i in range(len(self.requests))]) + "\n")

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

            f.write(f"\n=== STATISTICS (Load 1-30) ===\n")
            for req_key in sorted(results.keys()):
                blocking_probs = results[req_key][1:]
                if blocking_probs:
                    f.write(f"\n{req_key}:\n")
                    f.write(f"  Average: {np.mean(blocking_probs):.6f}\n")
                    f.write(f"  Max: {np.max(blocking_probs):.6f}\n")
                    f.write(f"  Min: {np.min(blocking_probs):.6f}\n")
                    f.write(f"  Std Dev: {np.std(blocking_probs):.6f}\n")

        print(f"\nResultados salvos em: {output_file}")

    def plot_results(self, results: Dict[str, List[float]],
                     save_path: str = "yen_firstfit_results.png") -> None:
        """
        Gera gráfico com resultados (de carga 0 a 30).

        Args:
            results: Resultados da simulação
            save_path: Caminho para salvar o gráfico
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
        plt.figure(figsize=(6, 4))

        colors = ['green','blue','red','purple', 'gold']

        for idx, (req_key, blocking_probs) in enumerate(sorted(results.items())):
            loads = np.arange(0, len(blocking_probs))
            blocking_probs_percent = [p * 100 for p in blocking_probs]
            plt.plot(loads, blocking_probs_percent,
                     label=req_key,
                     color=colors[idx % len(colors)],
                     linewidth=1.2,
                     marker='o',
                     markersize=3)

        plt.xlabel('Load', fontsize=12)
        plt.ylabel('Blocking Probability (%)', fontsize=12)
        #plt.title('YEN (Shortest Path) + First Fit Wavelength Allocation\nBlocking Probability vs Traffic Load',
        #          fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
        #plt.xticks(range(0, 31, 5))
        plt.xlim(0,30)
        plt.ylim(0, 100)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Gráfico salvo em: {save_path}")


def main():
    """Função principal para executar simulação com YEN + First Fit."""
    
    # >>>>> INÍCIO DA MEDIÇÃO DE TEMPO <<<<<
    start_time = time.time()
    

    # Criação do grafo NSFNet
    graph = nx.Graph()
    nsfnet_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
        (4, 6), (4, 5), (5, 8), (5, 12), (6, 7), (7, 9), (8, 9), (9, 11),
        (9, 13), (10, 11), (10, 13), (11, 12)
    ]
    graph.add_edges_from(nsfnet_edges)

    # ========== CAMPO CUSTOMIZÁVEL ==========
    custom_requests = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
    # =======================================

    print("=== YEN ROUTING (SHORTEST PATH) + FIRST FIT WAVELENGTH ALLOCATION ===\n")
    simulator = WDMYenFirstFit(
        graph=graph,
        num_wavelengths=40,
        k=3,
        requests=custom_requests
    )

    print(f"Requisições testadas: {custom_requests}")
    print(f"Cargas testadas: 0 a 30")
    print(f"Rodadas por carga: 10")
    print(f"Chamadas por rodada: 1.000")
    print(f"Duração das chamadas: 5.0 a 15.0 unidades de tempo")
    print(f"Modelo de Carga: Intervalo entre chegadas = 10.0 / load\n")

    print("Iniciando simulação de tráfego...\n")
    results = simulator.simulate_traffic_multiple_runs(
        num_runs=20,
        total_calls_per_load=1000,
        call_duration_range=(5.0, 15.0)
    )
    end_time = time.time()
    simulator.save_results(results, "yen_firstfit_results.txt")
    simulator.plot_results(results, "yen_firstfit_results.png")
    execution_time = end_time - start_time

    print("\nSimulação concluída com sucesso!")
    # Exibir no console
    print(f"\nTempo total de execução: {execution_time:.2f} segundos")

    # Salvar em arquivo separado
    with open("execution_time.txt", "w") as time_file:
        time_file.write("=== Tempo de Execução da Simulação ===\n")
        time_file.write(f"Tempo total: {execution_time:.2f} segundos\n")
        time_file.write(f"Início: {time.ctime(start_time)}\n")
        time_file.write(f"Término: {time.ctime(end_time)}\n")

    print("Tempo de execução salvo em: execution_time.txt")
    print("\nSimulação concluída com sucesso!")


if __name__ == "__main__":
    main()