import heapq
import random
import os
import time
from itertools import islice
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class WDMSimulatorPhD:
    """
    Simulador de rede WDM para pesquisa de doutorado.
    Implementa algoritmo genético avançado com codificação eficiente de genes.
    """

    def __init__(self,
                 graph: nx.Graph,
                 num_wavelengths: int = 40,
                 gene_size: int = 5,
                 manual_selection: bool = True,
                 gene_variation_mode: str = "fixed",
                 k: int = 150):
        """
        Inicializa o simulador WDM.

        Args:
            graph: Grafo da rede
            num_wavelengths: Número de comprimentos de onda disponíveis
            gene_size: Tamanho do gene (número de pares origem-destino)
            manual_selection: Se True, usa pares manuais predefinidos
            gene_variation_mode: "fixed" ou "custom"
            k: Número máximo de caminhos mais curtos a considerar
        """
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.population_size = 120
        self.num_generations = 40
        self.crossover_rate = 0.6
        self.mutation_rate = 0.02
        self.gene_size = gene_size
        self.k = k
        self.manual_pairs = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
        self.manual_selection = manual_selection
        self.gene_variation_mode = gene_variation_mode

        # Calcula todos os k-shortest paths uma vez
        self.k_shortest_paths = self._get_all_k_shortest_paths(k=self.k)
        self.reset_network()
        
        # Para armazenar tempos de execução
        self.execution_times = []

    def reset_network(self) -> None:
        """Reseta os canais de comprimento de onda na rede."""
        for u, v in self.graph.edges:
            self.graph[u][v]['wavelengths'] = np.ones(self.num_wavelengths, dtype=bool)
            self.graph[u][v]['current_wavelength'] = -1

    def release_wavelength(self, route: List[int], wavelength: int) -> None:
        """
        Libera um comprimento de onda de uma rota.

        Args:
            route: Lista de nós representando a rota
            wavelength: Índice do comprimento de onda a ser liberado
        """
        if not (0 <= wavelength < self.num_wavelengths):
            return

        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if self.graph.has_edge(u, v):
                self.graph[u][v]['wavelengths'][wavelength] = True
                # Reset apenas se este era o wavelength em uso
                if self.graph[u][v]['current_wavelength'] == wavelength:
                    self.graph[u][v]['current_wavelength'] = -1

    def allocate_wavelength(self, route: List[int], wavelength: int) -> bool:
        """
        Aloca um comprimento de onda para uma rota.

        Args:
            route: Lista de nós representando a rota
            wavelength: Índice do comprimento de onda a ser alocado

        Returns:
            True se a alocação foi bem-sucedida, False caso contrário
        """
        if not (0 <= wavelength < self.num_wavelengths):
            return False

        # Verifica se o wavelength está disponível em toda a rota
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if not self.graph.has_edge(u, v):
                return False
            if not self.graph[u][v]['wavelengths'][wavelength]:
                return False

        # Aloca o wavelength
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            self.graph[u][v]['wavelengths'][wavelength] = False
            self.graph[u][v]['current_wavelength'] = wavelength

        return True

    def find_available_wavelength(self, route: List[int]) -> Optional[int]:
        """
        Encontra o primeiro wavelength disponível para uma rota.

        Args:
            route: Lista de nós representando a rota

        Returns:
            Índice do wavelength disponível ou None se nenhum estiver disponível
        """
        for wavelength in range(self.num_wavelengths):
            available = True
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                if not self.graph.has_edge(u, v) or not self.graph[u][v]['wavelengths'][wavelength]:
                    available = False
                    break
            if available:
                return wavelength
        return None

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

        # Se não está usando seleção manual, gera pares aleatórios
        if not self.manual_selection:
            nodes = list(self.graph.nodes)
            for _ in range(self.gene_size):
                source, target = random.sample(nodes, 2)
                pairs_to_process.append((source, target))

        for source, target in pairs_to_process:
            paths[(source, target)] = self._get_k_shortest_paths(source, target, k)

        return paths

    def _initialize_population(self, source_targets: List[Tuple[int, int]]) -> List[List[int]]:
        """
        Inicializa a população do algoritmo genético.

        Args:
            source_targets: Lista de pares (origem, destino)

        Returns:
            População inicial
        """
        population = []

        for _ in range(self.population_size):
            individual = []
            for source, target in source_targets:
                if (source, target) not in self.k_shortest_paths:
                    raise ValueError(f"Nenhum caminho encontrado para ({source}, {target})")

                num_paths = len(self.k_shortest_paths[(source, target)])
                if num_paths == 0:
                    raise ValueError(f"Nenhum caminho válido para ({source}, {target})")

                # Escolhe um índice válido baseado no modo de variação do gene
                if self.gene_variation_mode == "fixed":
                    max_index = min(self.gene_size - 1, num_paths - 1)
                else:  # custom
                    max_index = num_paths - 1

                individual.append(random.randint(0, max_index))

            population.append(individual)

        return population

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
        current_wavelength = -1
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if self.graph.has_edge(u, v):
                link_wavelength = self.graph[u][v].get('current_wavelength', -1)
                if current_wavelength != -1 and link_wavelength != -1:
                    if current_wavelength != link_wavelength:
                        wavelength_changes += 1
                if link_wavelength != -1:
                    current_wavelength = link_wavelength

        # Disponibilidade média de wavelengths na rota
        availability_sum = 0
        valid_links = 0
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if self.graph.has_edge(u, v):
                available_wavelengths = np.sum(self.graph[u][v]['wavelengths'])
                availability = available_wavelengths / self.num_wavelengths
                availability_sum += availability
                valid_links += 1
        
        mean_availability = availability_sum / valid_links if valid_links > 0 else 0

        # Função de fitness combinada (normalizada)
        fitness = (0.5 * mean_availability + 
                   0.3 * (1 / (hops + 1)) + 
                   0.2 * (1 / (wavelength_changes + 1)))

        return fitness

    def _fitness(self, individual: List[int], source_targets: List[Tuple[int, int]]) -> float:
        """
        Calcula a aptidão de um indivíduo.

        Args:
            individual: Cromossomo (lista de índices de rotas)
            source_targets: Lista de pares origem-destino

        Returns:
            Aptidão total do cromossomo
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

        # Normaliza pelo número de rotas
        return total_fitness / len(source_targets)

    def _tournament_selection(self, population: List[List[int]],
                              source_targets: List[Tuple[int, int]],
                              tournament_size: int = 3) -> List[int]:
        """
        Realiza seleção por torneio.

        Args:
            population: População atual
            source_targets: Lista de pares origem-destino
            tournament_size: Tamanho do torneio

        Returns:
            Indivíduo selecionado
        """
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda ind: self._fitness(ind, source_targets))

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Realiza crossover entre dois pais.

        Args:
            parent1: Primeiro pai
            parent2: Segundo pai

        Returns:
            Tupla com dois filhos
        """
        if len(parent1) <= 1:
            return parent1[:], parent2[:]

        cut_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:cut_point] + parent2[cut_point:]
        child2 = parent2[:cut_point] + parent1[cut_point:]

        return child1, child2

    def _mutate(self, individual: List[int], source_targets: List[Tuple[int, int]]) -> None:
        """
        Aplica mutação em um indivíduo.

        Args:
            individual: Indivíduo a ser mutado
            source_targets: Lista de pares origem-destino
        """
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                source, target = source_targets[i]
                available_routes = self.k_shortest_paths.get((source, target), [])

                if available_routes:
                    if self.gene_variation_mode == "fixed":
                        max_index = min(self.gene_size - 1, len(available_routes) - 1)
                    else:  # custom
                        max_index = len(available_routes) - 1

                    individual[i] = random.randint(0, max_index)

    def genetic_algorithm(self, source_targets: List[Tuple[int, int]]) -> List[int]:
        """
        Executa o algoritmo genético.

        Args:
            source_targets: Lista de pares origem-destino

        Returns:
            Melhor indivíduo encontrado
        """
        start_time = time.time()
        
        # Inicializa população
        population = self._initialize_population(source_targets)

        best_fitness_history = []
        stagnation_count = 0
        max_stagnation = 10

        for generation in range(self.num_generations):
            # Avalia e ordena população
            population.sort(key=lambda ind: self._fitness(ind, source_targets), reverse=True)
            population = population[:self.population_size]  # Mantém apenas os melhores

            best_individual = population[0]
            best_fitness = self._fitness(best_individual, source_targets)
            best_fitness_history.append(best_fitness)

            # Verifica estagnação
            if (len(best_fitness_history) > 1 and
                    abs(best_fitness - best_fitness_history[-2]) < 1e-6):
                stagnation_count += 1
            else:
                stagnation_count = 0

            if stagnation_count >= max_stagnation:
                print(f"Estagnação detectada após {generation + 1} gerações.")
                break

            # Cria nova geração
            next_generation = []

            # Elitismo: mantém os melhores
            elite_size = max(1, self.population_size // 10)
            next_generation.extend(population[:elite_size])

            # Gera resto da população
            while len(next_generation) < self.population_size:
                parent1 = self._tournament_selection(population, source_targets)
                parent2 = self._tournament_selection(population, source_targets)

                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                    next_generation.extend([child1, child2])
                else:
                    next_generation.extend([parent1[:], parent2[:]])

            # Aplica mutação
            for individual in next_generation[elite_size:]:  # Não muta a elite
                self._mutate(individual, source_targets)

            population = next_generation[:self.population_size]

        # Tempo de execução do AG
        ga_time = time.time() - start_time
        self.execution_times.append(ga_time)
        
        best_solution = max(population, key=lambda ind: self._fitness(ind, source_targets))

        return best_solution

    def simulate_single_gene(self, gene_idx: int, source: int, target: int, 
                           best_individual: List[int], load_values: List[int],
                           num_simulations: int = 10, calls_per_load: int = 1000) -> Dict[str, List[float]]:
        """
        Simula uma única requisição (gene/par O-D) para todos os loads.
        
        Args:
            gene_idx: Índice do gene/par O-D
            source: Nó de origem
            target: Nó de destino
            best_individual: Melhor indivíduo encontrado pelo AG
            load_values: Lista de valores de carga a simular
            num_simulations: Número de simulações independentes
            calls_per_load: Número de chamadas por load
            
        Returns:
            Dicionário com resultados para este gene
        """
        gene_results = defaultdict(list)
        
        # Obtém a rota para este gene do melhor indivíduo
        route_idx = best_individual[gene_idx]
        available_routes = self.k_shortest_paths.get((source, target), [])
        
        if not available_routes or route_idx >= len(available_routes):
            # Usa caminho mais curto se rota do AG não estiver disponível
            try:
                route = nx.shortest_path(self.graph, source, target)
            except:
                print(f"  ⚠ Nenhuma rota válida encontrada para [{source},{target}]")
                # Retorna resultados vazios
                for load in load_values:
                    gene_results[load] = [1.0] * num_simulations  # 100% bloqueio
                return dict(gene_results)
        else:
            route = available_routes[route_idx]
        
        print(f"  Simulando gene {gene_idx+1} [{source},{target}]: rota {route}")
        
        # Para cada simulação
        for sim in range(num_simulations):
            # Para cada valor de load
            for load in load_values:
                # Reset da rede
                self.reset_network()
                
                # Contadores para este load
                blocked_calls = 0
                total_calls = calls_per_load
                
                # Simulação de eventos discretos
                current_time = 0
                event_queue = []  # (tempo_fim, rota, wavelength)
                
                # Processa as chamadas
                for call_id in range(total_calls):
                    # Tempo de chegada da chamada
                    current_time += np.random.exponential(1.0 / max(load, 0.1))
                    
                    # Libera chamadas que já expiraram
                    while event_queue and event_queue[0][0] <= current_time:
                        event_time, route_to_free, wavelength_to_free = heapq.heappop(event_queue)
                        self.release_wavelength(route_to_free, wavelength_to_free)
                    
                    # Tenta alocar wavelength
                    wavelength = self.find_available_wavelength(route)
                    
                    if wavelength is None:
                        blocked_calls += 1
                    else:
                        # Aloca e agenda liberação
                        if self.allocate_wavelength(route, wavelength):
                            duration = np.random.exponential(1.0)  # Duração média = 1
                            heapq.heappush(event_queue, 
                                         (current_time + duration, route, wavelength))
                        else:
                            blocked_calls += 1
                
                # Libera todas as chamadas restantes
                while event_queue:
                    event_time, route, wavelength = heapq.heappop(event_queue)
                    self.release_wavelength(route, wavelength)
                
                # Calcula probabilidade de bloqueio
                blocking_prob = blocked_calls / total_calls
                gene_results[load].append(blocking_prob)
        
        return dict(gene_results)

    def simulate_network_real(self, load_values: List[int] = None, 
                            num_simulations: int = 10,
                            calls_per_load: int = 1000,
                            output_file: str = "simulation_phd_results.txt") -> Dict[str, Dict[int, List[float]]]:
        """
        Simulação REAL da rede WDM usando algoritmo genético.
        Separa resultados por requisição (gene/par O-D).
        
        Args:
            load_values: Lista de valores de carga a simular
            num_simulations: Número de simulações independentes
            calls_per_load: Número de chamadas por load
            output_file: Arquivo de saída para resultados
            
        Returns:
            Dicionário com resultados por gene
        """
        start_total_time = time.time()
        
        if load_values is None:
            load_values = list(range(1, 201))  # 1 a 200 por padrão
        
        print(f"\n{'='*60}")
        print("SIMULAÇÃO REAL DE REDE WDM - DOUTORADO")
        print("(Resultados separados por requisição)")
        print(f"{'='*60}")
        print(f"Configuração:")
        print(f"  • Número de simulações: {num_simulations}")
        print(f"  • Loads testados: {len(load_values)} valores (de {min(load_values)} a {max(load_values)})")
        print(f"  • Número de wavelengths: {self.num_wavelengths}")
        print(f"  • Pares O-D fixos: {self.manual_pairs}")
        print(f"  • Chamadas por load: {calls_per_load}")
        print(f"{'='*60}")
        
        # Executa algoritmo genético UMA VEZ para encontrar melhores rotas
        print("\nExecutando Algoritmo Genético para encontrar melhores rotas...")
        ga_start_time = time.time()
        best_individual = self.genetic_algorithm(self.manual_pairs)
        ga_time = time.time() - ga_start_time
        print(f"AG concluído em {ga_time:.2f}s")
        print(f"Melhor indivíduo (índices de rotas): {best_individual}")
        
        # Dicionário para armazenar resultados por gene
        gene_results = {}
        
        # Arquivos individuais para cada gene
        gene_files = {}
        
        # Cria arquivos individuais para cada gene
        for gene_idx, (source, target) in enumerate(self.manual_pairs):
            filename = f"gene_{gene_idx + 1}.txt"
            gene_files[gene_idx] = filename
            
            with open(filename, 'w') as f:
                f.write(f"# [{source} -> {target}]\n")
                f.write(f"# Melhor rota: índice {best_individual[gene_idx]}\n")
                f.write(f"# Simulações: {num_simulations}, Chamadas/load: {calls_per_load}\n")
                f.write("# Formato: Load Probabilidade_Bloqueio_Média\n")
                f.write("# " + "="*50 + "\n")
        
        # Arquivo principal com todos os resultados
        with open(output_file, 'w') as f:
            # Cabeçalho
            f.write("# Resultados de Simulação WDM - Doutorado\n")
            f.write("# Resultados separados por requisição\n")
            f.write("# " + "="*60 + "\n")
            f.write(f"# Data da simulação: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Configuração:\n")
            f.write(f"#   • Número de simulações: {num_simulations}\n")
            f.write(f"#   • Loads testados: {len(load_values)} valores\n")
            f.write(f"#   • Número de wavelengths: {self.num_wavelengths}\n")
            f.write(f"#   • Pares O-D: {self.manual_pairs}\n")
            f.write(f"#   • Chamadas por load: {calls_per_load}\n")
            f.write(f"# Melhor indivíduo (índices): {best_individual}\n")
            f.write("# " + "="*60 + "\n")
            f.write("# Formato por gene: Gene Load Probabilidade1 Probabilidade2 ...\n")
            f.write("# " + "="*60 + "\n\n")

        # Simula cada gene/requisição separadamente
        print(f"\nSimulando {len(self.manual_pairs)} genes/requisições...")
        
        for gene_idx, (source, target) in enumerate(self.manual_pairs):
            print(f"\nGene {gene_idx+1}/{len(self.manual_pairs)}: [{source},{target}]")
            
            # Simula este gene para todos os loads
            gene_start_time = time.time()
            results = self.simulate_single_gene(
                gene_idx=gene_idx,
                source=source,
                target=target,
                best_individual=best_individual,
                load_values=load_values,
                num_simulations=num_simulations,
                calls_per_load=calls_per_load
            )
            gene_time = time.time() - gene_start_time
            
            gene_results[gene_idx] = results
            
            # Calcula médias para este gene
            avg_blocking_by_load = {}
            for load in load_values:
                if load in results and results[load]:
                    avg_blocking_by_load[load] = np.mean(results[load])
            
            # Salva resultados no arquivo individual do gene
            with open(gene_files[gene_idx], 'a') as f:
                for load in sorted(avg_blocking_by_load.keys()):
                    f.write(f"{load} {avg_blocking_by_load[load]:.6f}\n")
            
            # Salva resultados no arquivo principal
            with open(output_file, 'a') as f:
                f.write(f"\n# Gene {gene_idx+1} [{source},{target}]\n")
                for load in load_values:
                    if load in results:
                        # Formato: Gene Load Prob1 Prob2 ... ProbN
                        probs_str = " ".join([f"{p:.6f}" for p in results[load]])
                        f.write(f"{gene_idx+1} {load} {probs_str}\n")
            
            print(f"  Concluído em {gene_time:.2f}s")
        
        # Tempo total de execução
        total_time = time.time() - start_total_time
        
        # Calcula estatísticas finais
        summary_stats = self._calculate_summary_statistics(gene_results, load_values)
        
        # Salva resumo final no arquivo principal
        with open(output_file, 'a') as f:
            f.write("\n" + "="*60 + "\n")
            f.write("# RESUMO FINAL\n")
            f.write("="*60 + "\n")
            f.write(f"# Tempo total de execução: {total_time:.2f} segundos ({total_time/60:.2f} minutos)\n")
            f.write(f"# Tempo do AG: {ga_time:.2f}s\n")
            f.write(f"# Número total de simulações: {num_simulations}\n")
            f.write(f"# Total de chamadas simuladas: {len(self.manual_pairs)} × {len(load_values)} × {num_simulations} × {calls_per_load} = {len(self.manual_pairs)*len(load_values)*num_simulations*calls_per_load:,}\n")
            
            f.write("\n# Probabilidade de Bloqueio Média por Gene:\n")
            f.write("# Gene  Load  Probabilidade_Média\n")
            for gene_idx in sorted(gene_results.keys()):
                source, target = self.manual_pairs[gene_idx]
                for load in sorted(summary_stats['gene_means'][gene_idx].keys()):
                    mean_prob = summary_stats['gene_means'][gene_idx][load]
                    f.write(f"{gene_idx+1:4d}  {load:4d}  {mean_prob:.6f}  # [{source},{target}]\n")
        
        # Salva arquivo de tempo de execução
        with open("tempo_execucao_phd.txt", "w") as time_file:
            time_file.write(f"Tempo total de execução: {total_time:.2f} segundos\n")
            time_file.write(f"Tempo total de execução: {total_time/60:.2f} minutos\n")
            time_file.write(f"Tempo do AG: {ga_time:.2f} segundos\n")
            time_file.write(f"Início: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_total_time))}\n")
            time_file.write(f"Fim: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}\n")
            time_file.write(f"Simulações por gene: {num_simulations}\n")
            time_file.write(f"Genes/requisições: {len(self.manual_pairs)}\n")
            time_file.write(f"Loads testados: {len(load_values)} (de {min(load_values)} a {max(load_values)})\n")
            time_file.write(f"Chamadas por load: {calls_per_load}\n")
            time_file.write(f"Total de chamadas simuladas: {len(self.manual_pairs)*len(load_values)*num_simulations*calls_per_load:,}\n")
        
        print(f"\n{'='*60}")
        print("SIMULAÇÃO CONCLUÍDA")
        print(f"{'='*60}")
        print(f"Tempo total: {total_time:.2f}s ({total_time/60:.2f} minutos)")
        print(f"Tempo do AG: {ga_time:.2f}s")
        print(f"Arquivos gerados:")
        print(f"  • {output_file} - Resultados principais")
        print(f"  • tempo_execucao_phd.txt - Tempos de execução")
        for gene_idx in range(len(self.manual_pairs)):
            print(f"  • gene_{gene_idx+1}.txt - Resultados do gene {gene_idx+1}")
        print(f"{'='*60}")
        
        # Gera gráficos
        self._plot_gene_results(gene_results, load_values, num_simulations)
        
        return gene_results

    def _calculate_summary_statistics(self, gene_results: Dict[int, Dict[int, List[float]]], 
                                    load_values: List[int]) -> Dict[str, any]:
        """
        Calcula estatísticas de resumo dos resultados.
        """
        summary = {
            'gene_means': {},
            'overall_means': {},
            'gene_stds': {},
            'overall_stds': {}
        }
        
        # Para cada gene
        for gene_idx in gene_results:
            summary['gene_means'][gene_idx] = {}
            summary['gene_stds'][gene_idx] = {}
            
            for load in load_values:
                if load in gene_results[gene_idx]:
                    probs = gene_results[gene_idx][load]
                    summary['gene_means'][gene_idx][load] = np.mean(probs)
                    summary['gene_stds'][gene_idx][load] = np.std(probs) if len(probs) > 1 else 0.0
        
        # Médias gerais por load (média entre genes)
        for load in load_values:
            all_probs_for_load = []
            for gene_idx in gene_results:
                if load in gene_results[gene_idx]:
                    all_probs_for_load.extend(gene_results[gene_idx][load])
            
            if all_probs_for_load:
                summary['overall_means'][load] = np.mean(all_probs_for_load)
                summary['overall_stds'][load] = np.std(all_probs_for_load) if len(all_probs_for_load) > 1 else 0.0
        
        return summary

    def _plot_gene_results(self, gene_results: Dict[int, Dict[int, List[float]]],
                         load_values: List[int], num_simulations: int) -> None:
        """
        Gera gráficos dos resultados por gene.
        """
        plt.figure(figsize=(15, 10))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Gráfico 1: Todos os genes juntos
        plt.subplot(2, 1, 1)
        
        for gene_idx in range(len(self.manual_pairs)):
            if gene_idx in gene_results:
                source, target = self.manual_pairs[gene_idx]
                
                # Calcula médias para este gene
                means = []
                for load in sorted(load_values):
                    if load in gene_results[gene_idx] and gene_results[gene_idx][load]:
                        means.append(np.mean(gene_results[gene_idx][load]))
                    else:
                        means.append(0.0)
                
                plt.plot(load_values, means, 
                        color=colors[gene_idx % len(colors)],
                        linewidth=2,
                        marker='o',
                        markersize=4,
                        label=f'Gene {gene_idx+1} [{source},{target}]')
        
        plt.xlabel('Carga de Tráfego (Load)', fontsize=12)
        plt.ylabel('Probabilidade de Bloqueio', fontsize=12)
        plt.title(f'Probabilidade de Bloqueio por Requisição (Gene)\n{num_simulations} simulações por gene', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim(min(load_values) - 5, max(load_values) + 5)
        plt.ylim(-0.02, 1.02)
        plt.xticks(np.arange(min(load_values), max(load_values) + 1, 20))
        
        # Gráfico 2: Média geral
        plt.subplot(2, 1, 2)
        
        overall_means = []
        overall_stds = []
        
        for load in sorted(load_values):
            all_probs = []
            for gene_idx in gene_results:
                if load in gene_results[gene_idx]:
                    all_probs.extend(gene_results[gene_idx][load])
            
            if all_probs:
                overall_means.append(np.mean(all_probs))
                overall_stds.append(np.std(all_probs) if len(all_probs) > 1 else 0.0)
            else:
                overall_means.append(0.0)
                overall_stds.append(0.0)
        
        plt.errorbar(load_values, overall_means, yerr=overall_stds,
                    fmt='o-', linewidth=2, capsize=5, capthick=2,
                    color='darkblue', ecolor='lightblue',
                    label='Média Geral ± Desvio Padrão')
        
        plt.xlabel('Carga de Tráfego (Load)', fontsize=12)
        plt.ylabel('Probabilidade de Bloqueio', fontsize=12)
        plt.title('Probabilidade de Bloqueio Média Geral', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim(min(load_values) - 5, max(load_values) + 5)
        plt.ylim(-0.02, 1.02)
        plt.xticks(np.arange(min(load_values), max(load_values) + 1, 20))
        
        plt.tight_layout()
        
        # Salva o gráfico
        plot_filename = f'simulation_phd_genes_{time.strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Gráfico salvo como: {plot_filename}")

    def plot_individual_genes(self, save_path: str = "grafico_wdm_simulation_phd.png") -> None:
        """
        Gera gráfico com resultados de todos os genes.

        Args:
            save_path: Caminho para salvar o gráfico
        """
        plt.figure(figsize=(12, 8))

        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for gene_idx in range(self.gene_size):
            filename = f"gene_{gene_idx + 1}.txt"

            if not os.path.exists(filename):
                print(f"Arquivo {filename} não encontrado.")
                continue

            try:
                data = np.loadtxt(filename, comments='#')
                if data.size == 0:
                    continue

                if data.ndim == 1:
                    # Apenas uma linha de dados
                    loads = [1]
                    probs = [data]
                else:
                    loads = data[:, 0]
                    probs = data[:, 1]

                # Lê o cabeçalho
                with open(filename, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('#'):
                        label = first_line[2:]
                    else:
                        label = f"Gene {gene_idx + 1}"

                plt.plot(loads, probs,
                         label=label,
                         color=colors[gene_idx % len(colors)],
                         linewidth=2,
                         marker='o',
                         markersize=4)

            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")
                continue

        plt.xlabel('Carga de Tráfego', fontsize=12)
        plt.ylabel('Probabilidade de Bloqueio', fontsize=12)
        plt.title('Probabilidade de Bloqueio por Rota', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
        plt.xticks(range(0, 31, 5))
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Gráfico salvo em {save_path}")

    def generate_comparison_table(self, output_file: str = "comparacao_rotas_phd.csv") -> pd.DataFrame:
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
            filename = f"gene_{gene_idx + 1}.txt"

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

                source, target = self.manual_pairs[gene_idx]
                data[f"[{source},{target}]"] = probs

            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")
                continue

        if not data or loads is None:
            print("Nenhum dado válido encontrado para gerar tabela.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df.insert(0, "Carga (Load)", loads)

        print("\nTabela de Comparação (Probabilidades de Bloqueio por Requisição):")
        print(df.to_string(index=False))

        df.to_csv(output_file, index=False)
        print(f"\nTabela salva em {output_file}")

        return df


def main():
    """Função principal para executar a simulação."""
    # Início da medição do tempo total
    total_start_time = time.time()
    
    # Criação do grafo NSFNet
    print("Criando topologia NSFNet...")
    graph = nx.Graph()
    nsfnet_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
        (4, 6), (4, 5), (5, 8), (5, 12), (6, 7), (7, 9), (8, 9), (9, 11),
        (9, 13), (10, 11), (10, 13), (11, 12)
    ]
    graph.add_edges_from(nsfnet_edges)
    
    print(f"Topologia criada: {len(graph.nodes)} nós, {len(graph.edges)} links")

    # Configuração e execução da simulação
    print("\nInicializando simulador WDM para doutorado...")
    wdm_simulator = WDMSimulatorPhD(
        graph=graph,
        num_wavelengths=40,
        gene_size=5,
        manual_selection=True,
        gene_variation_mode="fixed",
        k=83
    )

    # Executa simulação REAL com resultados separados por requisição
    print("\nIniciando simulação real da rede (resultados por requisição)...")
    results = wdm_simulator.simulate_network_real(
        load_values=list(range(1, 201)),  # Loads de 1 a 200, passo 10 para ser mais rápido
        num_simulations=20,  # x simulações por gene
        calls_per_load=1000,
        output_file="simulation_results_phd_por_requisicao.txt"
    )

    # Gera visualizações
    print("\nGerando visualizações...")
    wdm_simulator.plot_individual_genes()
    wdm_simulator.generate_comparison_table()
    
    # Tempo total
    total_time = time.time() - total_start_time
    
    print("\n" + "="*60)
    print("SIMULAÇÃO DOUTORADO CONCLUÍDA COM SUCESSO!")
    print("="*60)
    print(f"Tempo total de execução: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
    print(f"Arquivos gerados:")
    print(f"  • simulation_results_phd_por_requisicao.txt - Resultados principais")
    print(f"  • tempo_execucao_phd.txt - Tempos de execução")
    print(f"  • gene_1.txt a gene_5.txt - Resultados individuais por requisição")
    print(f"  • grafico_wdm_simulation_phd.png - Gráfico das rotas")
    print(f"  • comparacao_rotas_phd.csv - Tabela de comparação")
    print("="*60)


if __name__ == "__main__":
    main()