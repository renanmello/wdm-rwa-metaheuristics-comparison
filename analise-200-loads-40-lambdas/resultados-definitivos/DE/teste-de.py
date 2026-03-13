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
    """Classe do problema para o algoritmo DE usando pymoo."""
    
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
        # DE minimiza, então invertemos o sinal do fitness
        fitness = -self.calc_fitness(x_int.tolist(), self.manual_pairs)
        out["F"] = [fitness]


class WDMSimulator:
    """
    Simulador de rede WDM (Wavelength Division Multiplexing) usando
    algoritmo DE (Differential Evolution) para resolver o problema 
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
        # Parâmetros DE
        CR: float = 0.9,
        F: float = 0.8,
        n_gen: int = 40
    ):
        """
        Inicializa o simulador WDM com DE.

        Args:
            graph: Grafo da rede
            num_wavelengths: Número de comprimentos de onda disponíveis
            gene_size: Tamanho do gene (número de pares origem-destino)
            manual_selection: Se True, usa pares manuais predefinidos
            k: Número máximo de caminhos mais curtos a considerar
            population_size: Tamanho da população
            hops_weight: Peso para o número de saltos no fitness
            wavelength_weight: Peso para trocas de wavelength no fitness
            CR: Crossover Rate (Taxa de Crossover), define a chance de usar o vetor mutante
            F: Scale Factor (Fator de Escala), define o quanto o vetor mutante é escalado
            n_gen: Número de gerações do DE
        """
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.gene_size = gene_size
        self.k = k
        self.manual_pairs = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
        self.manual_selection = manual_selection
        self.hops_weight = hops_weight
        self.wavelength_weight = wavelength_weight
        
        # Parâmetros DE
        self.population_size = population_size
        self.CR = CR
        self.F = F
        self.n_gen = n_gen

        # Parâmetros da simulação (CORRIGIDO: load afeta número de tentativas)
        self.base_calls = 100  # Chamadas base por load 1
        self.calls_multiplier = 5  # Incremento por load
        
        # Parâmetros de fitness IGUAIS AO AGP
        self.penalty_per_hop = 0.2  # Igual ao AGP: 0.2 por hop
        self.penalty_per_congested_link = 0.5  # Igual ao AGP: 0.5
        
        # Estrutura para rastrear uso de links (para cálculo de congestionamento)
        self.link_usage_count = defaultdict(int)
        
        # Calcula todos os k-shortest paths uma vez
        self.k_shortest_paths = self._get_all_k_shortest_paths(k=self.k)
        
        # Dicionário para guardar rotas selecionadas pelo melhor indivíduo
        self.selected_routes = {}
        
        self.reset_network()

    def reset_network(self) -> None:
        """Reseta completamente a rede para estado inicial."""
        # Reinicia os wavelengths disponíveis em cada link
        for u, v in self.graph.edges:
            self.graph[u][v]['available_wavelengths'] = [True] * self.num_wavelengths
        # Reinicia contagem de uso de links
        self.link_usage_count.clear()

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

    def _check_wavelength_availability(self, path: List[int], wavelength: int) -> bool:
        """
        Verifica se um wavelength específico está disponível em todos os links do caminho.
        
        Args:
            path: Lista de nós representando o caminho
            wavelength: Índice do wavelength a verificar
            
        Returns:
            True se o wavelength está disponível em todos os links
        """
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not self.graph.has_edge(u, v):
                return False
            if not self.graph[u][v]['available_wavelengths'][wavelength]:
                return False
        return True

    def _allocate_wavelength(self, path: List[int], wavelength: int) -> None:
        """
        Aloca um wavelength específico em todos os links do caminho.
        
        Args:
            path: Lista de nós representando o caminho
            wavelength: Índice do wavelength a alocar
        """
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            self.graph[u][v]['available_wavelengths'][wavelength] = False
            # Atualiza uso do link
            self.link_usage_count[(u, v)] += 1

    def _release_wavelength(self, path: List[int], wavelength: int) -> None:
        """
        Libera um wavelength específico em todos os links do caminho.
        
        Args:
            path: Lista de nós representando o caminho
            wavelength: Índice do wavelength a liberar
        """
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.graph.has_edge(u, v):
                self.graph[u][v]['available_wavelengths'][wavelength] = True
                # Atualiza uso do link
                if (u, v) in self.link_usage_count:
                    self.link_usage_count[(u, v)] = max(0, self.link_usage_count[(u, v)] - 1)

    def _find_available_wavelength(self, path: List[int]) -> Optional[int]:
        """
        Encontra o primeiro wavelength disponível para um caminho.
        
        Args:
            path: Lista de nós representando o caminho
            
        Returns:
            Índice do wavelength disponível ou None se nenhum disponível
        """
        for w in range(self.num_wavelengths):
            if self._check_wavelength_availability(path, w):
                return w
        return None

    def _estimate_congestion(self, route: List[int]) -> float:
        """
        ESTIMA congestionamento da rota (SIMPLIFICADO, mas baseado no AGP).
        
        No AGP: congestion = used_wavelengths / total_wavelengths por link
        Aqui: estimamos baseado no histórico de uso dos links
        
        Args:
            route: Lista de nós representando a rota
            
        Returns:
            Valor estimado de congestionamento (0-1)
        """
        if len(route) < 2:
            return 1.0
        
        total_congestion = 0.0
        links = 0
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            link_key = (u, v) if (u, v) in self.link_usage_count else (v, u)
            
            if link_key in self.link_usage_count:
                # Estima congestionamento baseado no uso histórico
                usage = self.link_usage_count[link_key]
                # Normaliza pelo número de wavelengths (estimativa)
                congestion = min(1.0, usage / (self.num_wavelengths * 2))
            else:
                # Link não usado recentemente
                congestion = 0.0
            
            total_congestion += congestion
            links += 1
        
        return total_congestion / links if links > 0 else 1.0

    def _fitness_route(self, route: List[int]) -> float:
        """
        Calcula a aptidão de uma rota ESPECÍFICA IGUAL AO AGP.
        
        MODIFICAÇÃO: Agora usa a MESMA lógica do AGP:
        1. Penalidade por hops (saltos)
        2. Penalidade por congestionamento estimado
        
        Args:
            route: Lista de nós representando a rota

        Returns:
            Valor de fitness da rota (maior é melhor)
        """
        if len(route) < 2:
            return 0.0

        # 1. Penalidade por número de hops (IGUAL AO AGP)
        hops = len(route) - 1
        hops_penalty = hops * self.penalty_per_hop  # 0.2 por hop, igual AGP
        
        # 2. Penalidade por congestionamento (IGUAL AO AGP)
        congestion = self._estimate_congestion(route)
        congestion_penalty = congestion * self.penalty_per_congested_link  # 0.5, igual AGP
        
        # Fitness base (quanto maior, melhor)
        fitness = 1.0
        
        # Aplica penalidades (IGUAL AO AGP)
        fitness -= hops_penalty
        fitness -= congestion_penalty
        
        # 3. Bônus adicional se a rota for muito curta (IGUAL AO AGP)
        if hops <= 3:  # Rotas muito curtas recebem bônus
            fitness *= 1.5
        
        # Garante que fitness não seja negativo (IGUAL AO AGP)
        return max(0.01, fitness)

    def _fitness(self, individual: List[int], source_targets: List[Tuple[int, int]]) -> float:
        """
        Calcula a aptidão de um indivíduo (conjunto de rotas).
        
        MODIFICAÇÃO: Usa fitness igual ao AGP (soma das fitness das rotas)
        
        Args:
            individual: Lista de índices de rotas
            source_targets: Lista de pares origem-destino

        Returns:
            Aptidão total
        """
        total_fitness = 0.0
        valid_routes = 0

        for i, (source, target) in enumerate(source_targets):
            if i >= len(individual):
                continue

            route_idx = individual[i]
            available_routes = self.k_shortest_paths.get((source, target), [])

            if not available_routes or route_idx >= len(available_routes):
                continue  # Penaliza rotas inválidas

            route = available_routes[route_idx]
            if len(route) >= 2:
                route_fitness = self._fitness_route(route)
                total_fitness += route_fitness
                valid_routes += 1

        # Retorna média das fitness válidas (similar ao AGP conjunto)
        return total_fitness / valid_routes if valid_routes > 0 else 0.0

    def _simulate_calls_for_load(self, load: int, individual: List[int]) -> Dict[int, float]:
        """
        Simula chamadas onde LOAD AFETA o número de tentativas.
        
        CORREÇÃO: Load determina quantas tentativas de chamada são feitas
        Load 1 = 100 tentativas, Load 200 = 1100 tentativas
        
        Args:
            load: Intensidade de tráfego (1-200)
            individual: Indivíduo (lista de índices de rotas) a ser testado
            
        Returns:
            Dicionário com blocking probability por par OD
        """
        # Reinicia a rede para esta simulação
        self.reset_network()
        
        # Guarda as rotas selecionadas pelo indivíduo
        self.selected_routes = {}
        for i, (source, target) in enumerate(self.manual_pairs):
            if i < len(individual):
                routes = self.k_shortest_paths.get((source, target), [])
                if routes and individual[i] < len(routes):
                    self.selected_routes[i] = routes[individual[i]]
        
        # Variáveis de estatística
        blocked_by_pair = defaultdict(int)
        arrivals_by_pair = defaultdict(int)
        
        # CORREÇÃO CRÍTICA: Load afeta número de tentativas
        # Load 1 = 100 tentativas (pouca carga)
        # Load 200 = 1100 tentativas (muita carga)
        total_attempts = self.base_calls + (load * self.calls_multiplier)
        
        print(f"    Load {load}: {total_attempts} tentativas de chamada")
        
        # Lista para controlar chamadas ativas (para liberação)
        active_calls = []  # (pair_idx, route, wavelength)
        
        # Para cada tentativa de chamada
        for attempt in range(total_attempts):
            # Escolhe um par OD aleatoriamente
            pair_idx = random.randint(0, len(self.manual_pairs) - 1)
            arrivals_by_pair[pair_idx] += 1
            
            # Verifica se temos rota para este par
            if pair_idx in self.selected_routes:
                route = self.selected_routes[pair_idx]
                
                # Tenta alocar um wavelength
                wavelength = self._find_available_wavelength(route)
                
                if wavelength is not None:
                    # Aloca o wavelength
                    self._allocate_wavelength(route, wavelength)
                    
                    # Registra chamada ativa
                    active_calls.append((pair_idx, route, wavelength))
                    
                    # CORREÇÃO: Libera algumas chamadas aleatoriamente para simular término
                    # Probabilidade de liberação aumenta com o número de chamadas ativas
                    if len(active_calls) > self.num_wavelengths * 10:  # Se muitas chamadas ativas
                        if random.random() < 0.1:  # 10% chance de liberar uma
                            if active_calls:
                                release_idx = random.randint(0, len(active_calls) - 1)
                                released_pair, released_route, released_wavelength = active_calls.pop(release_idx)
                                self._release_wavelength(released_route, released_wavelength)
                else:
                    # Chamada bloqueada
                    blocked_by_pair[pair_idx] += 1
                    
                    # Tenta liberar uma chamada aleatoriamente para fazer espaço
                    if active_calls and random.random() < 0.3:  # 30% chance quando bloqueia
                        release_idx = random.randint(0, len(active_calls) - 1)
                        released_pair, released_route, released_wavelength = active_calls.pop(release_idx)
                        self._release_wavelength(released_route, released_wavelength)
            else:
                # Sem rota definida - considera bloqueada
                blocked_by_pair[pair_idx] += 1
        
        # Calcula probabilidades de bloqueio
        blocking_probs_by_pair = {}
        for pair_idx in range(len(self.manual_pairs)):
            if arrivals_by_pair[pair_idx] > 0:
                blocking_probs_by_pair[pair_idx] = blocked_by_pair[pair_idx] / arrivals_by_pair[pair_idx]
            else:
                blocking_probs_by_pair[pair_idx] = 0.0
            
        return blocking_probs_by_pair

    def de_algorithm(self) -> Tuple[List[int], float]:
        """
        Executa o algoritmo DE (Differential Evolution).

        Returns:
            Tupla com (melhor indivíduo, melhor fitness)
        """
        print(f"Iniciando DE com {self.population_size} indivíduos, "
              f"{self.n_gen} gerações...")
        print(f"Parâmetros DE: CR={self.CR}, F={self.F}")
        print(f"Fitness igual ao AGP: considera hops e congestionamento")
    
        # Definição do problema para pymoo
        problem = MyProblem(self.gene_size, self._fitness, self.manual_pairs)

        from pymoo.algorithms.soo.nonconvex.de import DE
        from pymoo.operators.sampling.rnd import IntegerRandomSampling

        algorithm = DE(
            pop_size=self.population_size,
            CR=self.CR,
            F=self.F,
            variant="DE/rand/1/bin",
            sampling=IntegerRandomSampling(),
        )

        from pymoo.termination import get_termination
        termination = get_termination("n_gen", self.n_gen)

        from pymoo.optimize import minimize
        res_de = minimize(
            problem, 
            algorithm,
            termination,
            seed=None,
            save_history=True,
            verbose=False
        )

        X_DE = res_de.X.astype(int)
        F_DE = -res_de.F[0]  # Inverte o sinal de volta

        print("-" * 50)
        print("RESULTADOS DE (COM FITNESS IGUAL AO AGP)")
        print(f"Melhor Indivíduo: {X_DE}")
        print(f"Fitness: {F_DE:.6f}")
        
        # Mostra as rotas selecionadas
        print("Rotas selecionadas:")
        for i, (source, target) in enumerate(self.manual_pairs):
            if i < len(X_DE):
                route_idx = X_DE[i]
                routes = self.k_shortest_paths.get((source, target), [])
                if routes and route_idx < len(routes):
                    route = routes[route_idx]
                    hops = len(route) - 1
                    congestion = self._estimate_congestion(route)
                    print(f"  [{source},{target}]: Índice {route_idx}, {hops} hops, "
                          f"congestionamento estimado: {congestion:.3f}")
        
        print("-" * 50)

        return X_DE.tolist(), F_DE

    def simulate_network(
        self, 
        num_simulations: int = 1,
        output_file: str = "blocking_results_de.txt"
    ) -> Tuple[Dict[str, List[float]], List[int], float]:
        """
        Simula a rede WDM com load afetando número de tentativas.

        Args:
            num_simulations: Número de simulações
            output_file: Arquivo de saída para resultados

        Returns:
            Tupla com (resultados por rota, melhor indivíduo, melhor fitness)
        """
        # Inicia contador de tempo
        start_time = time.time()
        
        # Executa DE para encontrar o melhor indivíduo
        print(f"\n{'='*50}")
        print(f"EXECUTANDO DE COM FITNESS IGUAL AO AGP")
        print(f"{'='*50}")
        best_individual, best_fitness_value = self.de_algorithm()
        
        # Dicionário para armazenar todas as simulações por gene
        all_simulations = {i: [] for i in range(self.gene_size)}
        
        # Dicionário para resultados finais por rota
        results_by_route = {}
        
        # Abre arquivos individuais para cada rota
        route_files = {}
        for gene_idx in range(self.gene_size):
            source, target = self.manual_pairs[gene_idx]
            filename = f"route_{source}_{target}_de_agp_fitness.txt"  # Nome diferente
            route_files[gene_idx] = open(filename, 'w')
            route_files[gene_idx].write(f"# Resultados para rota [{source},{target}] (DE com fitness AGP)\n")
            route_files[gene_idx].write("# Formato: Load Probabilidade_Media Desvio_Padrao\n")
            results_by_route[gene_idx] = []

        print(f"\n{'='*50}")
        print(f"EXECUTANDO SIMULAÇÕES (DE com fitness igual ao AGP)")
        print(f"Número de simulações: {num_simulations}")
        print(f"Loads: 1-200")
        print(f"Tentativas: {self.base_calls} (load 1) até {self.base_calls + 200*self.calls_multiplier} (load 200)")
        print(f"{'='*50}")

        # Para cada load de 1 a 200
        for load in range(1, 201):
            print(f"\nProcessando Load {load}/200...")
            
            # Listas para armazenar resultados deste load
            load_results_by_route = {i: [] for i in range(self.gene_size)}
            
            # Executa múltiplas simulações para este load
            for sim in range(num_simulations):
                if (sim + 1) % 5 == 0:
                    print(f"  Simulação {sim + 1}/{num_simulations}")
                
                # Simula com número de tentativas baseado no load
                blocking_by_pair = self._simulate_calls_for_load(load, best_individual)
                
                # Armazena resultados por rota
                for pair_idx, blocking_prob in blocking_by_pair.items():
                    load_results_by_route[pair_idx].append(blocking_prob)
            
            # Calcula estatísticas para cada rota
            for gene_idx in range(self.gene_size):
                if load_results_by_route[gene_idx]:
                    mean_prob = np.mean(load_results_by_route[gene_idx])
                    std_prob = np.std(load_results_by_route[gene_idx])
                    
                    # Armazena para esta rota
                    results_by_route[gene_idx].append((load, mean_prob, std_prob))
                    
                    # Escreve no arquivo da rota
                    route_files[gene_idx].write(f"{load} {mean_prob:.6f} {std_prob:.6f}\n")
                    
                    # Armazena para estatísticas gerais
                    all_simulations[gene_idx].append(load_results_by_route[gene_idx])
            
            # Mostra blocking probability médio
            if load % 20 == 0:
                all_probs = []
                for gene_idx in range(self.gene_size):
                    if load_results_by_route[gene_idx]:
                        all_probs.extend(load_results_by_route[gene_idx])
                if all_probs:
                    avg_blocking = np.mean(all_probs)
                    print(f"  Load {load}: Pblock médio = {avg_blocking:.4f}")

        # Fecha arquivos das rotas
        for gene_idx, file in route_files.items():
            file.close()
            source, target = self.manual_pairs[gene_idx]
            print(f"✓ Resultados da rota [{source},{target}] salvos em route_{source}_{target}_de_agp_fitness.txt")

        # Calcula tempo total
        end_time = time.time()
        total_time = end_time - start_time
        
        # Salva estatísticas consolidadas
        self._save_statistics(all_simulations, num_simulations, "agp_fitness")
        
        # Salva tempo de execução
        self._save_execution_time(total_time, num_simulations, "agp_fitness")

        print(f"\n{'='*50}")
        print(f"Resultados salvos em arquivos individuais por rota")
        print(f"Tempo total de execução: {total_time:.2f} segundos")
        print(f"{'='*50}")
        
        # Converte resultados para formato de retorno
        final_results = {}
        for gene_idx in range(self.gene_size):
            source, target = self.manual_pairs[gene_idx]
            label = f'[{source},{target}]'
            if results_by_route[gene_idx]:
                # Extrai apenas as probabilidades médias
                probs = [item[1] for item in results_by_route[gene_idx]]
                final_results[label] = probs
        
        return final_results, best_individual, best_fitness_value

    def _save_statistics(self, all_simulations: Dict[int, List[List[float]]], 
                        num_sims: int, suffix: str = "") -> None:
        """
        Calcula e salva média e desvio padrão das simulações.
        
        Args:
            all_simulations: Dicionário com todas as simulações por gene
            num_sims: Número de simulações realizadas
            suffix: Sufixo para nome dos arquivos
        """
        print(f"\n{'='*50}")
        print("SALVANDO ESTATÍSTICAS CONSOLIDADAS")
        print(f"{'='*50}")
        
        for gene_idx in range(self.gene_size):
            if gene_idx not in all_simulations or not all_simulations[gene_idx]:
                print(f"⚠️  Gene {gene_idx + 1}: Sem dados para processar")
                continue
            
            source, target = self.manual_pairs[gene_idx]
            
            # Converte lista de listas para array numpy
            transposed_data = []
            num_loads = 200
            
            for load_idx in range(num_loads):
                load_data = []
                for sim_list in all_simulations[gene_idx]:
                    if load_idx < len(sim_list):
                        load_data.append(sim_list[load_idx])
                transposed_data.append(load_data)
            
            # Calcula média e desvio padrão por load
            mean_probs = [np.mean(data) if data else 0.0 for data in transposed_data]
            std_probs = [np.std(data) if data else 0.0 for data in transposed_data]
            
            # Sufixo para identificar que é com fitness igual ao AGP
            fitness_suffix = f"_{suffix}" if suffix else ""
            
            # Arquivo de médias
            mean_filename = f"gene_{gene_idx + 1}_de_mean{fitness_suffix}.txt"
            with open(mean_filename, "w") as f:
                f.write(f"# [{source} -> {target}] - Média de {num_sims} simulações (DE com fitness AGP)\n")
                f.write("# Formato: Load Probabilidade_Media\n")
                for load, prob in enumerate(mean_probs, 1):
                    f.write(f"{load} {prob:.6f}\n")
            print(f"   ✓ Salvo: {mean_filename}")
            
            # Arquivo de desvios padrão
            std_filename = f"gene_{gene_idx + 1}_de_std{fitness_suffix}.txt"
            with open(std_filename, "w") as f:
                f.write(f"# [{source} -> {target}] - Desvio Padrão de {num_sims} simulações (DE com fitness AGP)\n")
                f.write("# Formato: Load Desvio_Padrao\n")
                for load, std in enumerate(std_probs, 1):
                    f.write(f"{load} {std:.6f}\n")
            print(f"   ✓ Salvo: {std_filename}")
            
            # Arquivo combinado
            combined_filename = f"gene_{gene_idx + 1}_de_stats{fitness_suffix}.txt"
            with open(combined_filename, "w") as f:
                f.write(f"# [{source} -> {target}] - Estatísticas de {num_sims} simulações (DE com fitness AGP)\n")
                f.write("# Formato: Load Media Desvio_Padrao\n")
                for load, (mean, std) in enumerate(zip(mean_probs, std_probs), 1):
                    f.write(f"{load} {mean:.6f} {std:.6f}\n")
            print(f"   ✓ Salvo: {combined_filename}")

    def _save_execution_time(self, total_time: float, num_sims: int, 
                            suffix: str = "") -> None:
        """
        Salva o tempo de execução em arquivo.
        
        Args:
            total_time: Tempo total em segundos
            num_sims: Número de simulações
            suffix: Sufixo para nome do arquivo
        """
        fitness_suffix = f"_{suffix}" if suffix else ""
        filename = f"execution_time_de{fitness_suffix}.txt"
        
        with open(filename, "w") as f:
            f.write("# Tempo de Execução - DE COM FITNESS IGUAL AO AGP\n")
            f.write(f"# Número de simulações: {num_sims}\n")
            f.write(f"# Número de wavelengths: {self.num_wavelengths}\n")
            f.write(f"# População: {self.population_size}\n")
            f.write(f"# Gerações: {self.n_gen}\n")
            f.write(f"# Chamadas base: {self.base_calls}\n")
            f.write(f"# Multiplicador: {self.calls_multiplier}\n")
            f.write(f"# Loads simulados: 1-200\n")
            f.write(f"# Penalidade por hop: {self.penalty_per_hop}\n")
            f.write(f"# Penalidade por congestionamento: {self.penalty_per_congested_link}\n")
            f.write(f"#\n")
            f.write(f"Tempo total (segundos): {total_time:.2f}\n")
            f.write(f"Tempo médio por simulação (segundos): {total_time/num_sims:.2f}\n")
            f.write(f"Tempo em minutos: {total_time/60:.2f}\n")
            f.write(f"Tempo em horas: {total_time/3600:.2f}\n")
        
        print(f"\n⏱️  Tempo de execução salvo em: {filename}")

    def plot_individual_genes(
        self, 
        save_path: str = "grafico_wdm_simulation_de_agp_fitness.png"
    ) -> None:
        """
        Gera gráfico com resultados de todos os genes.

        Args:
            save_path: Caminho para salvar o gráfico
        """
        plt.figure(figsize=(12, 8))
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for gene_idx in range(self.gene_size):
            # Usa os arquivos de média com sufixo agp_fitness
            filename = f"gene_{gene_idx + 1}_de_mean_agp_fitness.txt"

            if not os.path.exists(filename):
                # Tenta sem sufixo
                filename = f"gene_{gene_idx + 1}_de_mean.txt"
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
                    markersize=4,
                    markevery=10
                )

            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")
                continue

        plt.xlabel('Carga de Tráfego (Load)', fontsize=12)
        plt.ylabel('Probabilidade de Bloqueio', fontsize=12)
        plt.title('Probabilidade de Bloqueio por Rota (DE com fitness igual ao AGP)', fontsize=14)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlim(1, 200)
        plt.xticks(range(0, 201, 20))
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Gráfico salvo em {save_path}")

    def generate_comparison_table(
        self, 
        output_file: str = "comparacao_rotas_de_agp_fitness.csv"
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
            # Tenta primeiro com sufixo agp_fitness
            filename = f"gene_{gene_idx + 1}_de_mean_agp_fitness.txt"
            
            if not os.path.exists(filename):
                # Tenta sem sufixo
                filename = f"gene_{gene_idx + 1}_de_mean.txt"
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

        # Adiciona estatísticas
        df["Média Geral"] = df.iloc[:, 1:].mean(axis=1)
        df["Desvio Padrão"] = df.iloc[:, 1:-1].std(axis=1)

        print("\nTabela de Comparação (DE com fitness AGP - primeiras 10 cargas):")
        print(df.head(10).to_string(index=False))

        df.to_csv(output_file, index=False)
        print(f"\nTabela salva em {output_file}")

        return df


def main():
    """Função principal para executar a simulação com DE."""
    
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
    print(f"SIMULAÇÃO WDM COM DE (FITNESS IGUAL AO AGP)")
    print(f"Número de wavelengths: {num_wavelengths}")
    print(f"Tentativas: 100 (load 1) até 1100 (load 200)")
    print(f"Loads: 1-200")
    print(f"Fitness igual ao AGP: considera hops e congestionamento")
    print(f"{'='*50}\n")

    # Configuração do simulador DE
    wdm_simulator = WDMSimulator(
        graph=graph,
        num_wavelengths=num_wavelengths,
        gene_size=5,
        manual_selection=True,
        k=150,
        population_size=120,
        hops_weight=0.7,  # Ainda mantido, mas fitness agora usa lógica do AGP
        wavelength_weight=0.3,
        # Parâmetros DE
        CR=0.9,
        F=0.8,
        n_gen=40
    )

    # Executa simulação
    output_file = f"blocking_results_de_agp_fitness_{num_wavelengths}w.txt"
    results, best_individual, best_fitness_value = wdm_simulator.simulate_network(
        num_simulations=20,
        output_file=output_file
    )

    # Gera visualizações e relatórios
    wdm_simulator.plot_individual_genes(
        save_path=f"grafico_wdm_de_agp_fitness_{num_wavelengths}w.png"
    )
    wdm_simulator.generate_comparison_table(
        output_file=f"comparacao_rotas_de_agp_fitness_{num_wavelengths}w.csv"
    )

    print(f"\n{'='*50}")
    print("RESULTADOS FINAIS (DE COM FITNESS IGUAL AO AGP)")
    print(f"{'='*50}")
    print(f"Melhor Indivíduo: {best_individual}")
    print(f"Fitness: {best_fitness_value:.6f}")
    
    # Calcula média das probabilidades de bloqueio
    all_means = [np.mean(values) for values in results.values() if values]
    if all_means:
        avg_blocking = np.mean(all_means)
        print(f"Probabilidade média de bloqueio: {avg_blocking:.6f}")
        print(f"Esperado: 50-70% (similar ao AGP, não 90% como DE anterior)")
    
    print(f"{'='*50}")
    print("Simulação concluída com sucesso!")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()