import heapq
import random
import os
import time
from itertools import islice
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class WDMSimulatorPhD_Fixed:
    """
    Versão corrigida do AG do Doutorado.
    - Remove limitação artificial de caminhos (gene_variation_mode="custom")
    - Fitness function melhorada
    - Implementação mais eficiente
    """
    def __init__(self,
                 graph: nx.Graph,
                 num_wavelengths: int = 40,
                 gene_size: int = 5,
                 manual_selection: bool = True,
                 k: int = 30):  # k razoável, não 150 ou 83
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
        """Libera um comprimento de onda de uma rota."""
        if not (0 <= wavelength < self.num_wavelengths):
            return

        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if self.graph.has_edge(u, v):
                self.graph[u][v]['wavelengths'][wavelength] = True
                if self.graph[u][v]['current_wavelength'] == wavelength:
                    self.graph[u][v]['current_wavelength'] = -1

    def allocate_wavelength(self, route: List[int], wavelength: int) -> bool:
        """Aloca um comprimento de onda para uma rota (SEM conversão)."""
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
        """Encontra o primeiro wavelength disponível para uma rota."""
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
        CORRIGIDO: Permite todos os caminhos disponíveis.
        """
        population = []

        for _ in range(self.population_size):
            individual = []
            for source, target in source_targets:
                available_routes = self.k_shortest_paths.get((source, target), [])
                
                if not available_routes:
                    individual.append(0)  # Índice padrão se não houver caminhos
                else:
                    # ESCOLHE QUALQUER CAMINHO DISPONÍVEL (não limitado por gene_size)
                    individual.append(random.randint(0, len(available_routes) - 1))

            population.append(individual)

        return population

    def _fitness_route(self, route: List[int]) -> float:
        """
        Calcula a aptidão de uma rota específica.
        CORRIGIDO: Penalidade mais forte para caminhos longos.
        """
        if len(route) < 2:
            return 0.0

        # Número de saltos (hops)
        hops = len(route) - 1
        
        # Penalidade por hops (logarítmica para não ser muito severa)
        hops_penalty = 1.0 / (1 + np.log1p(hops))

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
        
        if valid_links == 0:
            return 0.0
        
        mean_availability = availability_sum / valid_links

        # Fitness balanceado: 70% disponibilidade, 30% penalidade por hops
        fitness = 0.7 * mean_availability + 0.3 * hops_penalty

        return fitness

    def _fitness(self, individual: List[int], source_targets: List[Tuple[int, int]]) -> float:
        """Calcula a aptidão total de um indivíduo."""
        total_fitness = 0.0

        for i, (source, target) in enumerate(source_targets):
            if i >= len(individual):
                continue

            route_idx = individual[i]
            available_routes = self.k_shortest_paths.get((source, target), [])

            if not available_routes or route_idx >= len(available_routes):
                continue

            route = available_routes[route_idx]
            total_fitness += self._fitness_route(route)

        # Normaliza pelo número de rotas
        return total_fitness / len(source_targets)

    def _tournament_selection(self, population: List[List[int]],
                              source_targets: List[Tuple[int, int]],
                              tournament_size: int = 3) -> List[int]:
        """Realiza seleção por torneio."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda ind: self._fitness(ind, source_targets))

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Realiza crossover entre dois pais."""
        if len(parent1) <= 1:
            return parent1[:], parent2[:]

        cut_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:cut_point] + parent2[cut_point:]
        child2 = parent2[:cut_point] + parent1[cut_point:]

        return child1, child2

    def _mutate(self, individual: List[int], source_targets: List[Tuple[int, int]]) -> None:
        """
        Aplica mutação em um indivíduo.
        CORRIGIDO: Permite mutação para qualquer caminho disponível.
        """
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                source, target = source_targets[i]
                available_routes = self.k_shortest_paths.get((source, target), [])

                if available_routes:
                    # Pode mutar para QUALQUER caminho disponível
                    individual[i] = random.randint(0, len(available_routes) - 1)

    def genetic_algorithm(self, source_targets: List[Tuple[int, int]]) -> List[int]:
        """
        Executa o algoritmo genético.
        CORRIGIDO: Sem limitações artificiais.
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
            population = population[:self.population_size]

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
                print(f"  Estagnação detectada após {generation + 1} gerações.")
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
            for individual in next_generation[elite_size:]:
                self._mutate(individual, source_targets)

            population = next_generation[:self.population_size]

        # Tempo de execução do AG
        ga_time = time.time() - start_time
        self.execution_times.append(ga_time)
        
        # Encontra o melhor indivíduo final
        best_solution = max(population, key=lambda ind: self._fitness(ind, source_targets))

        return best_solution

    def simulate_single_gene(self, gene_idx: int, source: int, target: int, 
                           best_individual: List[int], load_values: List[int],
                           num_simulations: int = 10, calls_per_load: int = 1000) -> Dict[str, List[float]]:
        """
        Simula uma única requisição (gene/par O-D) para todos os loads.
        """
        gene_results = defaultdict(list)
        
        # Obtém a rota para este gene do melhor indivíduo
        route_idx = best_individual[gene_idx]
        available_routes = self.k_shortest_paths.get((source, target), [])
        
        if not available_routes or route_idx >= len(available_routes):
            # Usa caminho mais curto se rota do AG não estiver disponível
            try:
                route = nx.shortest_path(self.graph, source, target)
                print(f"  ⚠ Usando caminho mais curto para [{source},{target}] (AG não encontrou)")
            except:
                print(f"  ⚠ Nenhuma rota válida encontrada para [{source},{target}]")
                for load in load_values:
                    gene_results[load] = [1.0] * num_simulations
                return dict(gene_results)
        else:
            route = available_routes[route_idx]
        
        print(f"  Simulando gene {gene_idx+1} [{source},{target}]: rota com {len(route)-1} hops")
        
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
                            duration = np.random.exponential(1.0)
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

    def simulate_all_requisitions(self, load_values: List[int] = None, 
                                num_simulations: int = 10,
                                calls_per_load: int = 1000,
                                output_prefix: str = "simulation_phd_fixed") -> Dict[str, Dict[int, List[float]]]:
        """
        Simulação REAL da rede WDM usando algoritmo genético CORRIGIDO.
        Separa resultados por requisição (gene/par O-D).
        """
        start_total_time = time.time()
        
        if load_values is None:
            load_values = list(range(1, 201))
        
        print(f"\n{'='*60}")
        print("AG DOUTORADO CORRIGIDO - RESULTADOS POR REQUISIÇÃO")
        print(f"{'='*60}")
        print(f"Configuração:")
        print(f"  • Requisições: {len(self.manual_pairs)}")
        print(f"  • Loads testados: {len(load_values)} (de {min(load_values)} a {max(load_values)})")
        print(f"  • Simulações por load: {num_simulations}")
        print(f"  • Número de wavelengths: {self.num_wavelengths}")
        print(f"  • Pares O-D fixos: {self.manual_pairs}")
        print(f"  • Chamadas por load: {calls_per_load}")
        print(f"{'='*60}")
        
        # Executa algoritmo genético UMA VEZ para encontrar melhores rotas
        print("\nExecutando Algoritmo Genético (sem limitações)...")
        ga_start_time = time.time()
        best_individual = self.genetic_algorithm(self.manual_pairs)
        ga_time = time.time() - ga_start_time
        
        print(f"AG concluído em {ga_time:.2f}s")
        print(f"Melhor indivíduo (índices de rotas): {best_individual}")
        
        # Mostra as rotas escolhidas
        print("\nRotas escolhidas pelo AG:")
        for gene_idx, (source, target) in enumerate(self.manual_pairs):
            route_idx = best_individual[gene_idx]
            available_routes = self.k_shortest_paths.get((source, target), [])
            if route_idx < len(available_routes):
                route = available_routes[route_idx]
                print(f"  [{source},{target}]: {route} ({len(route)-1} hops)")
        
        # Dicionário para armazenar resultados por gene
        gene_results = {}
        
        # Arquivos individuais para cada gene
        gene_files = {}
        
        # Cria arquivos individuais para cada gene
        for gene_idx, (source, target) in enumerate(self.manual_pairs):
            filename = f"{output_prefix}_gene_{gene_idx + 1}.txt"
            gene_files[gene_idx] = filename
            
            with open(filename, 'w') as f:
                f.write(f"# Requisição {gene_idx + 1}: [{source},{target}]\n")
                f.write(f"# Índice da rota no AG: {best_individual[gene_idx]}\n")
                f.write(f"# Simulações: {num_simulations}, Chamadas/load: {calls_per_load}\n")
                f.write("# Formato: Load Probabilidade_Bloqueio_Média\n")
                f.write("# " + "="*50 + "\n")
        
        # Arquivo principal com todos os resultados
        main_file = f"{output_prefix}_all_results.txt"
        with open(main_file, 'w') as f:
            f.write("# Resultados AG Doutorado Corrigido\n")
            f.write("# Resultados separados por requisição\n")
            f.write("# " + "="*60 + "\n")
            f.write(f"# Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Configuração:\n")
            f.write(f"#   • Requisições: {self.manual_pairs}\n")
            f.write(f"#   • Simulações: {num_simulations}\n")
            f.write(f"#   • Loads: {len(load_values)}\n")
            f.write(f"#   • Chamadas/load: {calls_per_load}\n")
            f.write(f"# Melhor indivíduo (índices): {best_individual}\n")
            f.write("# " + "="*60 + "\n")
            f.write("# Formato: Load Prob1 Prob2 ... ProbN\n")
            f.write("# " + "="*60 + "\n\n")

        # Simula cada gene/requisição separadamente
        print(f"\nSimulando {len(self.manual_pairs)} requisições...")
        
        for gene_idx, (source, target) in enumerate(self.manual_pairs):
            print(f"\nRequisição {gene_idx+1}/{len(self.manual_pairs)}: [{source},{target}]")
            
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
            with open(main_file, 'a') as f:
                f.write(f"\n# Requisição {gene_idx+1} [{source},{target}]\n")
                for load in load_values:
                    if load in results:
                        probs_str = " ".join([f"{p:.6f}" for p in results[load]])
                        f.write(f"{load} {probs_str}\n")
            
            print(f"  Concluído em {gene_time:.2f}s")
        
        # Tempo total de execução
        total_time = time.time() - start_total_time
        
        # Calcula estatísticas finais
        summary_stats = self._calculate_summary_statistics(gene_results, load_values)
        
        # Salva resumo final no arquivo principal
        with open(main_file, 'a') as f:
            f.write("\n" + "="*60 + "\n")
            f.write("# RESUMO FINAL\n")
            f.write("="*60 + "\n")
            f.write(f"# Tempo total: {total_time:.2f}s ({total_time/60:.2f} minutos)\n")
            f.write(f"# Tempo do AG: {ga_time:.2f}s\n")
            f.write(f"# Simulações: {num_simulations}\n")
            f.write(f"# Total de chamadas simuladas: {len(self.manual_pairs)} × {len(load_values)} × {num_simulations} × {calls_per_load} = {len(self.manual_pairs)*len(load_values)*num_simulations*calls_per_load:,}\n")
            
            f.write("\n# Probabilidade de Bloqueio Média por Requisição:\n")
            f.write("# Req  Load  Probabilidade_Média\n")
            for gene_idx in sorted(gene_results.keys()):
                source, target = self.manual_pairs[gene_idx]
                for load in sorted(summary_stats['gene_means'][gene_idx].keys()):
                    mean_prob = summary_stats['gene_means'][gene_idx][load]
                    f.write(f"{gene_idx+1:3d}  {load:4d}  {mean_prob:.6f}  # [{source},{target}]\n")
        
        # Salva arquivo de tempo de execução
        with open(f"{output_prefix}_tempo.txt", "w") as time_file:
            time_file.write(f"Tempo total: {total_time:.2f}s\n")
            time_file.write(f"Tempo total: {total_time/60:.2f} minutos\n")
            time_file.write(f"Tempo do AG: {ga_time:.2f}s\n")
            time_file.write(f"Início: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_total_time))}\n")
            time_file.write(f"Fim: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}\n")
            time_file.write(f"Requisições: {len(self.manual_pairs)}\n")
            time_file.write(f"Loads testados: {len(load_values)}\n")
            time_file.write(f"Simulações: {num_simulations}\n")
            time_file.write(f"Chamadas por load: {calls_per_load}\n")
        
        print(f"\n{'='*60}")
        print("SIMULAÇÃO CONCLUÍDA")
        print(f"{'='*60}")
        print(f"Tempo total: {total_time:.2f}s ({total_time/60:.2f} minutos)")
        print(f"Tempo do AG: {ga_time:.2f}s")
        print(f"Arquivos gerados:")
        print(f"  • {main_file} - Resultados principais")
        print(f"  • {output_prefix}_tempo.txt - Tempos de execução")
        for gene_idx in range(len(self.manual_pairs)):
            print(f"  • {output_prefix}_gene_{gene_idx+1}.txt - Requisição {gene_idx+1}")
        print(f"{'='*60}")
        
        # Gera gráficos
        self._plot_gene_results(gene_results, load_values, num_simulations, output_prefix)
        
        return gene_results

    def _calculate_summary_statistics(self, gene_results: Dict[int, Dict[int, List[float]]], 
                                    load_values: List[int]) -> Dict[str, Any]:
        """Calcula estatísticas de resumo dos resultados."""
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
                         load_values: List[int], num_simulations: int,
                         output_prefix: str) -> None:
        """Gera gráficos dos resultados por gene."""
        plt.figure(figsize=(15, 10))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Gráfico 1: Todas as requisições
        plt.subplot(2, 1, 1)
        
        for gene_idx in range(len(self.manual_pairs)):
            if gene_idx in gene_results:
                source, target = self.manual_pairs[gene_idx]
                
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
                        label=f'[{source},{target}]')
        
        plt.xlabel('Load', fontsize=12)
        plt.ylabel('Probabilidade de Bloqueio', fontsize=12)
        plt.title(f'AG Doutorado Corrigido - Probabilidade por Requisição\n{num_simulations} simulações',
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
        
        plt.xlabel('Load', fontsize=12)
        plt.ylabel('Probabilidade de Bloqueio', fontsize=12)
        plt.title('Média Geral de Todas as Requisições', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim(min(load_values) - 5, max(load_values) + 5)
        plt.ylim(-0.02, 1.02)
        plt.xticks(np.arange(min(load_values), max(load_values) + 1, 20))
        
        plt.tight_layout()
        
        # Salva o gráfico
        plot_filename = f'{output_prefix}_plot_{time.strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Gráfico salvo como: {plot_filename}")

class WDMSimulatorPhD_Conjunto(WDMSimulatorPhD_Fixed):
    """
    AG do Doutorado que otimiza as 5 requisições JUNTAS.
    Implementa a ideia original do doutorado.
    """
    
    def simulate_network_conjunto(self, load_values: List[int] = None,
                                 num_simulations: int = 10,
                                 calls_per_load: int = 1000,
                                 output_prefix: str = "ag_doutorado_conjunto") -> Dict[int, Dict[int, List[float]]]:
        """
        Simula as 5 requisições JUNTAS competindo por recursos.
        """
        start_total_time = time.time()
        
        if load_values is None:
            load_values = list(range(1, 201))
        
        print(f"\n{'='*60}")
        print("AG DOUTORADO - 5 REQUISIÇÕES CONJUNTAS")
        print("(Otimização e simulação conjunta)")
        print(f"{'='*60}")
        print(f"Configuração:")
        print(f"  • Requisições: {len(self.manual_pairs)}")
        print(f"  • Loads testados: {len(load_values)} (de {min(load_values)} a {max(load_values)})")
        print(f"  • Simulações por load: {num_simulations}")
        print(f"  • Número de wavelengths: {self.num_wavelengths}")
        print(f"  • Pares O-D fixos: {self.manual_pairs}")
        print(f"  • Chamadas por load: {calls_per_load}")
        print(f"{'='*60}")
        
        # Executa AG para otimizar as 5 rotas JUNTAS
        print("\nExecutando AG para otimizar 5 rotas conjuntas...")
        ga_start_time = time.time()
        best_individual = self.genetic_algorithm(self.manual_pairs)
        ga_time = time.time() - ga_start_time
        
        print(f"AG concluído em {ga_time:.2f}s")
        print(f"Melhor indivíduo (índices de rotas): {best_individual}")
        
        # Obtém as 5 rotas do melhor indivíduo
        rotas_conjuntas = []
        for gene_idx, (source, target) in enumerate(self.manual_pairs):
            route_idx = best_individual[gene_idx]
            available_routes = self.k_shortest_paths.get((source, target), [])
            if route_idx < len(available_routes):
                rotas_conjuntas.append(available_routes[route_idx])
                print(f"  [{source},{target}]: {rotas_conjuntas[-1]} ({len(rotas_conjuntas[-1])-1} hops)")
            else:
                # Fallback para shortest path
                rota_fallback = nx.shortest_path(self.graph, source, target)
                rotas_conjuntas.append(rota_fallback)
                print(f"  [{source},{target}]: {rota_fallback} ({len(rota_fallback)-1} hops) [FALLBACK]")
        
        # Dicionário para resultados
        results = {gene_idx: {load: [] for load in load_values} 
                  for gene_idx in range(len(self.manual_pairs))}
        
        # Arquivos individuais para cada gene
        gene_files = {}
        
        # Cria arquivos individuais para cada gene
        for gene_idx, (source, target) in enumerate(self.manual_pairs):
            filename = f"{output_prefix}_gene_{gene_idx + 1}.txt"
            gene_files[gene_idx] = filename
            
            with open(filename, 'w') as f:
                f.write(f"# Requisição {gene_idx + 1}: [{source},{target}]\n")
                f.write(f"# Índice da rota no AG: {best_individual[gene_idx]}\n")
                f.write(f"# Simulações CONJUNTAS: {num_simulations}, Chamadas/load: {calls_per_load}\n")
                f.write("# Formato: Load Probabilidade_Bloqueio_Média\n")
                f.write("# " + "="*50 + "\n")
        
        # Arquivo principal
        main_file = f"{output_prefix}_all_results.txt"
        with open(main_file, 'w') as f:
            f.write("# Resultados AG Doutorado Conjunto\n")
            f.write("# 5 requisições otimizadas e simuladas JUNTAS\n")
            f.write("# " + "="*60 + "\n")
            f.write(f"# Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Configuração:\n")
            f.write(f"#   • Requisições: {self.manual_pairs}\n")
            f.write(f"#   • Simulações: {num_simulations}\n")
            f.write(f"#   • Loads: {len(load_values)}\n")
            f.write(f"#   • Chamadas/load: {calls_per_load}\n")
            f.write(f"# Melhor indivíduo (índices): {best_individual}\n")
            f.write("# " + "="*60 + "\n\n")
        
        # Simulação CONJUNTA
        print(f"\nSimulando {len(load_values)} loads com competição entre requisições...")
        
        for sim in range(num_simulations):
            print(f"  Simulação {sim+1}/{num_simulations}...", end='\r')
            
            for load in load_values:
                # Reset da rede
                self.reset_network()
                
                # Contadores por requisição
                blocked_calls = [0] * len(self.manual_pairs)
                total_calls = calls_per_load
                
                # Event queue para TODAS as requisições
                current_time = 0
                event_queue = []  # (tempo_fim, rota_idx, rota, wavelength)
                
                # Gera chamadas para CADA requisição (competindo)
                for call_id in range(total_calls):
                    # Para cada chamada, seleciona aleatoriamente uma das 5 requisições
                    req_idx = random.randint(0, len(self.manual_pairs) - 1)
                    rota = rotas_conjuntas[req_idx]
                    
                    current_time += np.random.exponential(1.0 / max(load, 0.1))
                    
                    # Libera chamadas expiradas
                    while event_queue and event_queue[0][0] <= current_time:
                        event_time, released_req_idx, route_to_free, wavelength_to_free = heapq.heappop(event_queue)
                        self.release_wavelength(route_to_free, wavelength_to_free)
                    
                    # Tenta alocar
                    wavelength = self.find_available_wavelength(rota)
                    
                    if wavelength is None:
                        blocked_calls[req_idx] += 1
                    else:
                        if self.allocate_wavelength(rota, wavelength):
                            duration = np.random.exponential(1.0)
                            heapq.heappush(event_queue, 
                                         (current_time + duration, req_idx, rota, wavelength))
                        else:
                            blocked_calls[req_idx] += 1
                
                # Libera chamadas restantes
                while event_queue:
                    event_time, req_idx, route, wavelength = heapq.heappop(event_queue)
                    self.release_wavelength(route, wavelength)
                
                # Armazena resultados
                for req_idx in range(len(self.manual_pairs)):
                    blocking_prob = blocked_calls[req_idx] / total_calls
                    results[req_idx][load].append(blocking_prob)
        
        print(f"\nSimulação conjunta concluída!")
        
        # Calcula médias e salva arquivos
        for gene_idx in range(len(self.manual_pairs)):
            source, target = self.manual_pairs[gene_idx]
            
            # Calcula médias para este gene
            avg_blocking_by_load = {}
            for load in load_values:
                if load in results[gene_idx] and results[gene_idx][load]:
                    avg_blocking_by_load[load] = np.mean(results[gene_idx][load])
            
            # Salva no arquivo individual
            with open(gene_files[gene_idx], 'a') as f:
                for load in sorted(avg_blocking_by_load.keys()):
                    f.write(f"{load} {avg_blocking_by_load[load]:.6f}\n")
            
            # Salva no arquivo principal
            with open(main_file, 'a') as f:
                f.write(f"\n# Requisição {gene_idx+1} [{source},{target}]\n")
                for load in load_values:
                    if load in results[gene_idx]:
                        probs_str = " ".join([f"{p:.6f}" for p in results[gene_idx][load]])
                        f.write(f"{load} {probs_str}\n")
        
        # Tempo total
        total_time = time.time() - start_total_time
        
        # Calcula estatísticas
        summary_stats = self._calculate_summary_statistics(results, load_values)
        
        # Salva resumo
        with open(main_file, 'a') as f:
            f.write("\n" + "="*60 + "\n")
            f.write("# RESUMO FINAL (CONJUNTO)\n")
            f.write("="*60 + "\n")
            f.write(f"# Tempo total: {total_time:.2f}s ({total_time/60:.2f} minutos)\n")
            f.write(f"# Tempo do AG: {ga_time:.2f}s\n")
            
            f.write("\n# Probabilidade de Bloqueio Média por Requisição (CONJUNTO):\n")
            f.write("# Req  Load  Probabilidade_Média\n")
            for gene_idx in sorted(results.keys()):
                source, target = self.manual_pairs[gene_idx]
                for load in sorted(summary_stats['gene_means'][gene_idx].keys()):
                    mean_prob = summary_stats['gene_means'][gene_idx][load]
                    f.write(f"{gene_idx+1:3d}  {load:4d}  {mean_prob:.6f}  # [{source},{target}]\n")
        
        # Salva tempo
        with open(f"{output_prefix}_tempo.txt", "w") as time_file:
            time_file.write(f"Tempo total: {total_time:.2f}s\n")
            time_file.write(f"Tempo do AG: {ga_time:.2f}s\n")
            time_file.write(f"Requisições: {len(self.manual_pairs)} (CONJUNTAS)\n")
            time_file.write(f"Loads: {len(load_values)}\n")
            time_file.write(f"Simulações: {num_simulations}\n")
        
        print(f"\n{'='*60}")
        print("SIMULAÇÃO CONJUNTA CONCLUÍDA")
        print(f"{'='*60}")
        print(f"Tempo total: {total_time:.2f}s")
        print(f"Arquivos gerados:")
        print(f"  • {main_file} - Resultados principais")
        print(f"  • {output_prefix}_tempo.txt - Tempos")
        for gene_idx in range(len(self.manual_pairs)):
            print(f"  • {output_prefix}_gene_{gene_idx+1}.txt - Requisição {gene_idx+1}")
        print(f"{'='*60}")
        
        # Gera gráficos
        self._plot_gene_results(results, load_values, num_simulations, output_prefix)
        
        return results


def main_phd_fixed(conjunto: bool = False):
    """Função principal para executar o AG do doutorado."""
    start_time = time.time()
    
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
    
    if conjunto:
        print("\n" + "="*60)
        print("EXECUTANDO AG DOUTORADO CONJUNTO")
        print("(5 requisições otimizadas e simuladas juntas)")
        print("="*60)
        
        # Usa a classe CONJUNTO
        wdm_simulator = WDMSimulatorPhD_Conjunto(
            graph=graph,
            num_wavelengths=40,
            gene_size=5,
            manual_selection=True,
            k=30
        )
        
        # Executa simulação CONJUNTA
        results = wdm_simulator.simulate_network_conjunto(
            load_values=list(range(1, 201)),
            num_simulations=20,
            calls_per_load=1000,
            output_prefix="ag_doutorado_conjunto"
        )
        
    else:
        print("\n" + "="*60)
        print("EXECUTANDO AG DOUTORADO SEPARADO")
        print("(Cada requisição otimizada e simulada separadamente)")
        print("="*60)
        
        # Usa a classe SEPARADA (original)
        wdm_simulator = WDMSimulatorPhD_Fixed(
            graph=graph,
            num_wavelengths=40,
            gene_size=5,
            manual_selection=True,
            k=30
        )
        
        # Executa simulação SEPARADA
        results = wdm_simulator.simulate_all_requisitions(
            load_values=list(range(1, 201)),
            num_simulations=20,
            calls_per_load=1000,
            output_prefix="ag_doutorado_separado"
        )
    
    total_time = time.time() - start_time
    
    print(f"\nTempo total de execução: {total_time:.2f}s ({total_time/60:.2f} minutos)")
    if conjunto:
        print("Resultados salvos em arquivos 'ag_doutorado_conjunto_*.txt'")
    else:
        print("Resultados salvos em arquivos 'ag_doutorado_separado_*.txt'")


def compare_results():
    """
    Compara resultados do AG antigo e AG do doutorado corrigido.
    Lê os arquivos gerados e cria comparação.
    """
    print(f"\n{'='*60}")
    print("COMPARAÇÃO: AG ANTIGO vs AG DOUTORADO CORRIGIDO")
    print(f"{'='*60}")
    
    # Configuração
    requisições = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
    loads = list(range(1, 201))
    
    # Arquivos de entrada
    old_prefix = "ag_antigo"
    phd_prefix = "ag_doutorado_corrigido"
    
    # Arquivo de comparação
    comp_file = "comparacao_ag_antigo_vs_doutorado.txt"
    
    with open(comp_file, 'w') as f:
        f.write("# COMPARAÇÃO: AG ANTIGO vs AG DOUTORADO CORRIGIDO\n")
        f.write("# " + "="*60 + "\n")
        f.write(f"# Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Requisições: {requisições}\n")
        f.write(f"# Loads: 1-200\n")
        f.write(f"# Simulações: 20\n")
        f.write(f"# Chamadas/load: 1000\n")
        f.write("# " + "="*60 + "\n")
        f.write("# Formato: Req Load Prob_Antigo Prob_Doutorado Diferença\n")
        f.write("# " + "="*60 + "\n\n")
    
    # Processa cada requisição
    for idx, (source, target) in enumerate(requisições):
        print(f"\nProcessando requisição {idx+1}: [{source},{target}]")
        
        # Arquivos de entrada
        old_file = f"{old_prefix}_req_{idx+1}.txt"
        phd_file = f"{phd_prefix}_gene_{idx+1}.txt"
        
        if not os.path.exists(old_file):
            print(f"  ⚠ Arquivo não encontrado: {old_file}")
            continue
        if not os.path.exists(phd_file):
            print(f"  ⚠ Arquivo não encontrado: {phd_file}")
            continue
        
        # Lê resultados do AG antigo
        old_results = {}
        with open(old_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    load = int(parts[0])
                    prob = float(parts[1])
                    old_results[load] = prob
        
        # Lê resultados do AG doutorado
        phd_results = {}
        with open(phd_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    load = int(parts[0])
                    prob = float(parts[1])
                    phd_results[load] = prob
        
        # Compara e salva
        with open(comp_file, 'a') as f:
            f.write(f"\n# Requisição {idx+1} [{source},{target}]\n")
            
            for load in loads:
                if load in old_results and load in phd_results:
                    prob_old = old_results[load]
                    prob_phd = phd_results[load]
                    diff = prob_phd - prob_old
                    diff_pct = (diff / prob_old * 100) if prob_old > 0 else 0
                    
                    f.write(f"{idx+1} {load} {prob_old:.6f} {prob_phd:.6f} {diff:+.6f} ({diff_pc:+.1f}%)\n")
        
        print(f"  ✓ Processado {len(old_results)} loads")
    
    # Cria resumo estatístico
    create_comparison_summary(comp_file, requisições, loads)
    
    print(f"\n{'='*60}")
    print("COMPARAÇÃO CONCLUÍDA")
    print(f"{'='*60}")
    print(f"Arquivo de comparação: {comp_file}")
    print(f"Execute 'python comparacao.py' para gerar gráficos comparativos")


def create_comparison_summary(comp_file: str, requisições: List[Tuple[int, int]], loads: List[int]):
    """Cria resumo estatístico da comparação."""
    summary_file = "comparacao_resumo.txt"
    
    # Leitura dos dados
    data = defaultdict(list)
    
    with open(comp_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                req_idx = int(parts[0])
                load = int(parts[1])
                prob_old = float(parts[2])
                prob_phd = float(parts[3])
                diff = float(parts[4])
                
                data[req_idx].append((load, prob_old, prob_phd, diff))
    
    # Cria resumo
    with open(summary_file, 'w') as f:
        f.write("# RESUMO DA COMPARAÇÃO\n")
        f.write("# " + "="*60 + "\n")
        
        for req_idx in sorted(data.keys()):
            source, target = requisições[req_idx-1]
            f.write(f"\n# Requisição {req_idx} [{source},{target}]:\n")
            
            diffs = [d[3] for d in data[req_idx]]
            probs_old = [d[1] for d in data[req_idx]]
            probs_phd = [d[2] for d in data[req_idx]]
            
            # Estatísticas
            avg_diff = np.mean(diffs)
            avg_old = np.mean(probs_old)
            avg_phd = np.mean(probs_phd)
            
            melhor = "AG Doutorado" if avg_phd < avg_old else "AG Antigo"
            percentual = abs((avg_phd - avg_old) / avg_old * 100) if avg_old > 0 else 0
            
            f.write(f"  Probabilidade média (AG Antigo): {avg_old:.6f}\n")
            f.write(f"  Probabilidade média (AG Doutorado): {avg_phd:.6f}\n")
            f.write(f"  Diferença média: {avg_diff:+.6f}\n")
            f.write(f"  Melhor: {melhor} ({percentual:.1f}% de diferença)\n")
        
        # Geral
        f.write("\n" + "="*60 + "\n")
        f.write("# GERAL:\n")
        
        all_diffs = []
        all_old = []
        all_phd = []
        
        for req_idx in data:
            all_diffs.extend([d[3] for d in data[req_idx]])
            all_old.extend([d[1] for d in data[req_idx]])
            all_phd.extend([d[2] for d in data[req_idx]])
        
        if all_diffs:
            avg_diff_total = np.mean(all_diffs)
            avg_old_total = np.mean(all_old)
            avg_phd_total = np.mean(all_phd)
            
            f.write(f"  Média geral (AG Antigo): {avg_old_total:.6f}\n")
            f.write(f"  Média geral (AG Doutorado): {avg_phd_total:.6f}\n")
            f.write(f"  Diferença média geral: {avg_diff_total:+.6f}\n")
            
            if avg_phd_total < avg_old_total:
                f.write("  ✅ AG DOUTORADO É MELHOR EM MÉDIA!\n")
            else:
                f.write("  ⚠ AG ANTIGO É MELHOR EM MÉDIA\n")
    
    print(f"Resumo salvo em: {summary_file}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SISTEMA DE COMPARAÇÃO DE ALGORITMOS GENÉTICOS")
    print("="*60)
    print("\nEscolha uma opção:")
    print("1. Executar AG Antigo com resultados separados")
    print("2. Executar AG Doutorado SEPARADO (para comparação)")
    print("3. Executar AG Doutorado CONJUNTO (ideia original)")
    print("4. Comparar resultados já existentes")
    print("5. Comparar todas as versões (Antigo, Separado, Conjunto)")
    print("6. Sair")
    
    choice = input("\nDigite sua escolha (1-6): ")
    
    if choice == "1":
        print("\nExecutando AG Antigo...")
        # Você precisaria ter a função main_old() definida
        # main_old()
        print("Nota: Execute o código do AG antigo separadamente primeiro")
    elif choice == "2":
        print("\nExecutando AG Doutorado SEPARADO...")
        main_phd_fixed(conjunto=False)
    elif choice == "3":
        print("\nExecutando AG Doutorado CONJUNTO...")
        main_phd_fixed(conjunto=True)
    elif choice == "4":
        print("\nComparando resultados...")
        compare_results()
    elif choice == "5":
        print("\nComparando todas as versões...")
        compare_all_versions()
    elif choice == "6":
        print("\nSaindo...")
    else:
        print("\nOpção inválida!")