import random
import os
import time
from itertools import islice
from typing import List, Tuple, Dict, Optional

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
            xl=np.array([0] * gene_size),
            xu=np.array([gene_size - 1] * gene_size),
            vtype=int
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        x_int = x.astype(int)
        # PSO minimiza, então invertemos o sinal do fitness
        fitness = -self.calc_fitness(x_int.tolist(), self.manual_pairs)
        out["F"] = [fitness]


class WDMSimulator:
    """
    Simulador de rede WDM (Wavelength Division Multiplexing) usando
    algoritmo PSO (Particle Swarm Optimization) para resolver o problema 
    RWA (Routing and Wavelength Assignment).
    """

    def __init__(
        self,
        graph: nx.Graph,
        num_wavelengths: int = 4,
        gene_size: int = 5,
        manual_selection: bool = True,
        k: int = 150,
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
        Inicializa o simulador WDM com PSO.

        Args:
            graph: Grafo da rede
            num_wavelengths: Número de comprimentos de onda disponíveis
            gene_size: Tamanho do gene (número de pares origem-destino)
            manual_selection: Se True, usa pares manuais predefinidos
            k: Número máximo de caminhos mais curtos a considerar
            population_size: Tamanho da população (swarm)
            hops_weight: Peso para o número de saltos no fitness
            wavelength_weight: Peso para trocas de wavelength no fitness
            w: Inércia do PSO
            c1: Aceleração cognitiva (componente pessoal)
            c2: Aceleração social (componente global)
            n_gen: Número de gerações do PSO
        """
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.gene_size = gene_size
        self.k = k
        self.manual_pairs = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
        self.manual_selection = manual_selection
        self.hops_weight = hops_weight
        self.wavelength_weight = wavelength_weight
        
        # Parâmetros PSO
        self.population_size = population_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_gen = n_gen

        # Calcula todos os k-shortest paths uma vez
        self.k_shortest_paths = self._get_all_k_shortest_paths(k=self.k)
        self.reset_network()

    def reset_network(self) -> None:
        """Reseta os canais de comprimento de onda na rede."""
        for u, v in self.graph.edges:
            self.graph[u][v]['wavelengths'] = np.ones(self.num_wavelengths, dtype=bool)
            self.graph[u][v]['current_wavelength'] = -1

    def _get_k_shortest_paths(self, source: int, target: int, k: int) -> List[List[int]]:
        """Calcula os k menores caminhos entre dois nós."""
        if not nx.has_path(self.graph, source, target):
            return []
        try:
            return list(islice(nx.shortest_simple_paths(self.graph, source, target), k))
        except nx.NetworkXNoPath:
            return []

    def _get_all_k_shortest_paths(self, k: int) -> Dict[Tuple[int, int], List[List[int]]]:
        """Calcula os k menores caminhos para todos os pares origem-destino."""
        paths = {}
        pairs_to_process = self.manual_pairs if self.manual_selection else []

        if not self.manual_selection:
            nodes = list(self.graph.nodes)
            for _ in range(self.gene_size):
                source, target = random.sample(nodes, 2)
                pairs_to_process.append((source, target))

        for source, target in pairs_to_process:
            paths[(source, target)] = self._get_k_shortest_paths(source, target, k)

        return paths

    def _fitness_route(self, route: List[int]) -> float:
        """
        Calcula a aptidão de uma rota específica.

        Args:
            route: Lista de nós representando a rota

        Returns:
            Valor de fitness da rota
        """
        if len(route) < 2:
            return 0.0

        # Número de saltos
        hops = len(route) - 1

        # Número de trocas de comprimentos de onda necessárias
        wavelength_changes = 0
        for i in range(len(route) - 2):
            u1, v1 = route[i], route[i + 1]
            u2, v2 = route[i + 1], route[i + 2]

            if (self.graph.has_edge(u1, v1) and self.graph.has_edge(u2, v2) and
                    self.graph[u1][v1].get('current_wavelength', -1) != -1 and
                    self.graph[u2][v2].get('current_wavelength', -1) != -1 and
                    self.graph[u1][v1]['current_wavelength'] != 
                    self.graph[u2][v2]['current_wavelength']):
                wavelength_changes += 1

        # Função de fitness combinada (normalizada)
        fitness = (self.hops_weight * (1 / (hops + 1)) +
                   self.wavelength_weight * (1 / (wavelength_changes + 1)))

        return fitness

    def _fitness(self, individual: List[int], source_targets: List[Tuple[int, int]]) -> float:
        """
        Calcula a aptidão de um indivíduo.

        Args:
            individual: Lista de índices de rotas
            source_targets: Lista de pares origem-destino

        Returns:
            Aptidão total
        """
        total_fitness = 0.0

        for i, (source, target) in enumerate(source_targets):
            if i >= len(individual):
                continue

            route_idx = individual[i]
            available_routes = self.k_shortest_paths.get((source, target), [])

            if not available_routes or route_idx >= len(available_routes):
                continue  # Penaliza rotas inválidas

            route = available_routes[route_idx]
            total_fitness += self._fitness_route(route)

        return total_fitness

    def pso_algorithm(self) -> Tuple[List[int], float]:
        """
        Executa o algoritmo PSO.

        Returns:
            Tupla com (melhor indivíduo, melhor fitness)
        """
        print(f"Iniciando PSO com {self.population_size} partículas, "
              f"{self.n_gen} gerações...")
        print(f"Parâmetros PSO: w={self.w}, c1={self.c1}, c2={self.c2}")
    
        # Definição do problema para pymoo
        problem = MyProblem(self.gene_size, self._fitness, self.manual_pairs)

        from pymoo.algorithms.soo.nonconvex.pso import PSO
        from pymoo.operators.sampling.rnd import IntegerRandomSampling

        algorithm = PSO(
            pop_size=self.population_size,
            w=self.w,
            c1=self.c1,
            c2=self.c2,
            sampling=IntegerRandomSampling(),
        )

        from pymoo.termination import get_termination
        termination = get_termination("n_gen", self.n_gen)

        from pymoo.optimize import minimize
        res_pso = minimize(
            problem, 
            algorithm,
            termination,
            seed=None,
            save_history=True,
            verbose=False
        )

        X_PSO = res_pso.X.astype(int)
        F_PSO = -res_pso.F[0]  # Inverte o sinal de volta

        print("-" * 50)
        print("RESULTADOS PSO")
        print(f"Melhor Indivíduo: {X_PSO}")
        print(f"Fitness: {F_PSO:.6f}")
        print("-" * 50)

        return X_PSO.tolist(), F_PSO

    def _calculate_blocking_probability(self, route: List[int], load: int) -> float:
        """
        Calcula a probabilidade de bloqueio para uma rota e carga específicas.
        Usa a mesma lógica do AG original.

        Args:
            route: Rota a ser avaliada
            load: Carga de tráfego

        Returns:
            Probabilidade de bloqueio
        """
        # Modelo de probabilidade de bloqueio baseado na teoria de tráfego
        route_load = 0.05 * load
        utilization = route_load * len(route) / (self.num_wavelengths + 5)

        # Aplica limite para manter probabilidade entre 0 e 1
        return max(0.0, min(1.0, utilization))

    def simulate_network(
        self, 
        num_simulations: int = 1,
        output_file: str = "blocking_results_pso.txt"
    ) -> Tuple[Dict[str, List[float]], List[int], float]:
        """
        Simula a rede WDM e calcula probabilidades de bloqueio.

        Args:
            num_simulations: Número de simulações
            output_file: Arquivo de saída para resultados

        Returns:
            Tupla com (resultados por rota, melhor indivíduo, melhor fitness)
        """
        # Inicia contador de tempo
        start_time = time.time()
        
        results = {}
        best_individual = None
        best_fitness_value = 0.0
        
        # Dicionário para armazenar todas as simulações por gene
        all_simulations = {i: [] for i in range(self.gene_size)}

        with open(output_file, "w") as f:
            f.write("# Resultados de Probabilidade de Bloqueio - PSO\n")
            f.write("# Formato: Load Probabilidade\n")

            for sim in range(num_simulations):
                print(f"\n{'='*50}")
                print(f"Executando simulação {sim + 1}/{num_simulations}")
                print(f"{'='*50}")

                # Executa algoritmo PSO
                best_individual, best_fitness_value = self.pso_algorithm()

                for gene_idx, (source, target) in enumerate(self.manual_pairs):
                    if gene_idx >= len(best_individual):
                        continue

                    routes = self.k_shortest_paths.get((source, target), [])
                    if not routes or best_individual[gene_idx] >= len(routes):
                        continue

                    best_route = routes[best_individual[gene_idx]]
                    
                    print(f"\nProcessando rota [{source},{target}]...")
                    print(f"Caminho selecionado: {best_route}")
                    print(f"Número de saltos: {len(best_route) - 1}")

                    # Calcula probabilidades de bloqueio
                    blocking_probs = []
                    for load in range(1, 201):
                        blocked_calls = 0
                        total_calls = 1000

                        for _ in range(total_calls):
                            block_prob = self._calculate_blocking_probability(
                                best_route, load
                            )
                            if random.random() < block_prob:
                                blocked_calls += 1

                        prob = blocked_calls / total_calls
                        blocking_probs.append(prob)
                        f.write(f"{load} {prob}\n")

                    # Armazena resultados da simulação atual
                    all_simulations[gene_idx].append(blocking_probs)

                    # Armazena resultados (para compatibilidade)
                    label = f'[{source},{target}]'
                    if label not in results:
                        results[label] = []
                    results[label].extend(blocking_probs)

        # Calcula tempo total
        end_time = time.time()
        total_time = end_time - start_time
        
        # Salva estatísticas
        self._save_statistics(all_simulations, num_simulations)
        
        # Salva tempo de execução
        self._save_execution_time(total_time, num_simulations)

        print(f"\n{'='*50}")
        print(f"Resultados salvos em {output_file}")
        print(f"Tempo total de execução: {total_time:.2f} segundos")
        print(f"{'='*50}")
        
        return results, best_individual, best_fitness_value

    def _save_statistics(self, all_simulations: Dict[int, List[List[float]]], num_sims: int) -> None:
        """
        Calcula e salva média e desvio padrão das simulações.
        
        Args:
            all_simulations: Dicionário com todas as simulações por gene
            num_sims: Número de simulações realizadas
        """
        print(f"\n{'='*50}")
        print("SALVANDO ESTATÍSTICAS")
        print(f"{'='*50}")
        
        for gene_idx in range(self.gene_size):
            if gene_idx not in all_simulations or not all_simulations[gene_idx]:
                print(f"⚠️  Gene {gene_idx + 1}: Sem dados para processar")
                continue
            
            source, target = self.manual_pairs[gene_idx]
            
            # Converte lista de listas para array numpy
            sim_array = np.array(all_simulations[gene_idx])
            
            print(f"\n📊 Gene {gene_idx + 1} [{source}->{target}]:")
            print(f"   Simulações coletadas: {len(all_simulations[gene_idx])}")
            print(f"   Shape do array: {sim_array.shape}")
            
            # Calcula média e desvio padrão por load
            mean_probs = np.mean(sim_array, axis=0)
            std_probs = np.std(sim_array, axis=0)
            
            # Arquivo de médias
            mean_filename = f"gene_{gene_idx + 1}_pso_mean.txt"
            with open(mean_filename, "w") as f:
                f.write(f"# [{source} -> {target}] - Média de {num_sims} simulações\n")
                f.write("# Formato: Load Probabilidade_Media\n")
                for load, prob in enumerate(mean_probs, 1):
                    f.write(f"{load} {prob:.6f}\n")
            print(f"   ✓ Salvo: {mean_filename}")
            
            # Arquivo de desvios padrão
            std_filename = f"gene_{gene_idx + 1}_pso_std.txt"
            with open(std_filename, "w") as f:
                f.write(f"# [{source} -> {target}] - Desvio Padrão de {num_sims} simulações\n")
                f.write("# Formato: Load Desvio_Padrao\n")
                for load, std in enumerate(std_probs, 1):
                    f.write(f"{load} {std:.6f}\n")
            print(f"   ✓ Salvo: {std_filename}")
            
            # Arquivo combinado
            combined_filename = f"gene_{gene_idx + 1}_pso_stats.txt"
            with open(combined_filename, "w") as f:
                f.write(f"# [{source} -> {target}] - Estatísticas de {num_sims} simulações\n")
                f.write("# Formato: Load Media Desvio_Padrao\n")
                for load, (mean, std) in enumerate(zip(mean_probs, std_probs), 1):
                    f.write(f"{load} {mean:.6f} {std:.6f}\n")
            print(f"   ✓ Salvo: {combined_filename}")

    def _save_execution_time(self, total_time: float, num_sims: int) -> None:
        """
        Salva o tempo de execução em arquivo.
        
        Args:
            total_time: Tempo total em segundos
            num_sims: Número de simulações
        """
        filename = "execution_time_pso.txt"
        with open(filename, "w") as f:
            f.write("# Tempo de Execução - PSO\n")
            f.write(f"# Número de simulações: {num_sims}\n")
            f.write(f"# Número de wavelengths: {self.num_wavelengths}\n")
            f.write(f"# População: {self.population_size}\n")
            f.write(f"# Gerações: {self.n_gen}\n")
            f.write(f"#\n")
            f.write(f"Tempo total (segundos): {total_time:.2f}\n")
            f.write(f"Tempo médio por simulação (segundos): {total_time/num_sims:.2f}\n")
            f.write(f"Tempo em minutos: {total_time/60:.2f}\n")
            f.write(f"Tempo em horas: {total_time/3600:.2f}\n")
        
        print(f"\n⏱️  Tempo de execução salvo em: {filename}")

    def plot_individual_genes(
        self, 
        save_path: str = "grafico_wdm_simulation_pso.png"
    ) -> None:
        """
        Gera gráfico com resultados de todos os genes.

        Args:
            save_path: Caminho para salvar o gráfico
        """
        plt.figure(figsize=(12, 8))
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for gene_idx in range(self.gene_size):
            # Usa os arquivos de média
            filename = f"gene_{gene_idx + 1}_pso_mean.txt"

            if not os.path.exists(filename):
                print(f"Arquivo {filename} não encontrado.")
                continue

            try:
                data = np.loadtxt(filename, comments='#')
                if data.size == 0:
                    continue

                if data.ndim == 1:
                    loads = [1]
                    probs = [data[1]] if len(data) > 1 else [data]
                else:
                    loads = data[:, 0]
                    probs = data[:, 1]

                # Lê o cabeçalho
                with open(filename, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('#'):
                        label = first_line[2:].split('-')[0].strip()
                    else:
                        label = f"Gene {gene_idx + 1}"

                plt.plot(
                    loads, 
                    probs,
                    label=label,
                    color=colors[gene_idx % len(colors)],
                    linewidth=2,
                    marker='o',
                    markersize=4
                )

            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")
                continue

        plt.xlabel('Carga de Tráfego', fontsize=12)
        plt.ylabel('Probabilidade de Bloqueio', fontsize=12)
        plt.title('Probabilidade de Bloqueio por Rota (PSO - Média)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 31, 5))
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Gráfico salvo em {save_path}")

    def generate_comparison_table(
        self, 
        output_file: str = "comparacao_rotas_pso.csv"
    ) -> pd.DataFrame:
        """
        Gera tabela de comparação das probabilidades de bloqueio.

        Args:
            output_file: Arquivo de saída CSV

        Returns:
            DataFrame com comparação
        """
        data = {}
        loads = None

        for gene_idx in range(self.gene_size):
            # Usa os arquivos de média
            filename = f"gene_{gene_idx + 1}_pso_mean.txt"

            if not os.path.exists(filename):
                continue

            try:
                file_data = np.loadtxt(filename, comments='#')
                if file_data.size == 0:
                    continue

                if file_data.ndim == 1:
                    loads = [1]
                    probs = [file_data[1]] if len(file_data) > 1 else [file_data]
                else:
                    loads = file_data[:, 0].astype(int)
                    probs = file_data[:, 1]

                data[f"Rota {gene_idx + 1}"] = probs

            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")
                continue

        if not data or loads is None:
            print("Nenhum dado válido encontrado para gerar tabela.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df.insert(0, "Carga (Load)", loads)

        print("\nTabela de Comparação:")
        print(df.to_string(index=False))

        df.to_csv(output_file, index=False)
        print(f"\nTabela salva em {output_file}")

        return df


def main():
    """Função principal para executar a simulação com PSO."""
    
    # Criação do grafo NSFNet
    graph = nx.Graph()
    nsfnet_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
        (4, 6), (4, 5), (5, 8), (5, 12), (6, 7), (7, 9), (8, 9), (9, 11),
        (9, 13), (10, 11), (10, 13), (11, 12)
    ]
    graph.add_edges_from(nsfnet_edges)

    # Configurações para teste
    num_wavelengths = 40  # Pode alterar para 8
    
    print(f"\n{'='*50}")
    print(f"SIMULAÇÃO WDM COM PSO")
    print(f"Número de wavelengths: {num_wavelengths}")
    print(f"{'='*50}\n")

    # Configuração do simulador PSO
    wdm_simulator = WDMSimulator(
        graph=graph,
        num_wavelengths=num_wavelengths,
        gene_size=5,
        manual_selection=True,
        k=150,
        population_size=120,
        hops_weight=0.7,
        wavelength_weight=0.3,
        # Parâmetros PSO
        w=0.7,
        c1=1.5,
        c2=1.5,
        n_gen=40
    )

    # Executa simulação
    output_file = f"blocking_results_pso_{num_wavelengths}w.txt"
    results, best_individual, best_fitness_value = wdm_simulator.simulate_network(
        num_simulations=20,
        output_file=output_file
    )

    # Gera visualizações e relatórios
    wdm_simulator.plot_individual_genes(
        save_path=f"grafico_wdm_pso_{num_wavelengths}w.png"
    )
    wdm_simulator.generate_comparison_table(
        output_file=f"comparacao_rotas_pso_{num_wavelengths}w.csv"
    )

    print(f"\n{'='*50}")
    print("RESULTADOS FINAIS")
    print(f"{'='*50}")
    print(f"Melhor Indivíduo: {best_individual}")
    print(f"Fitness: {best_fitness_value:.6f}")
    
    # Calcula média das probabilidades de bloqueio
    all_means = [np.mean(values) for values in results.values() if values]
    if all_means:
        avg_blocking = np.mean(all_means)
        print(f"Probabilidade média de bloqueio: {avg_blocking:.6f}")
    
    print(f"{'='*50}")
    print("Simulação concluída com sucesso!")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()