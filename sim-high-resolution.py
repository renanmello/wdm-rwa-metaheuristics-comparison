"""
COMPREHENSIVE STATISTICAL ANALYSIS FOR PSO, DE, AND GA
Problem: RWA in WDM networks with dynamic traffic
High Resolution: Loads from 1 to 200 (or 1 to 400)

Author: PhD Thesis

CORRECTIONS APPLIED (v8):
- Fitness keeps EXACTLY the paper's Equation 1 (per OD pair):
        fit_r = 1 / (alpha * nc_r + beta * rl_r)          total = sum_r fit_r
  but nc_r is now the number of wavelength conversions REALLY observed, and
  rl_r is the real number of hops of the route (no more nc = rl - 1 proxy).
- nc_r is measured with a short dynamic-traffic evaluation of the candidate
  solution (all 5 routes together, so shared links and congestion matter):
  warm-up requests are discarded, then the mean number of conversions per
  established connection of each OD pair is computed with the same
  hop-by-hop wavelength assignment used in the final simulation.
- The evaluation traffic uses its own RNG (never touches the global random /
  numpy state used by GA/PSO/DE) and seeds that never coincide with the seeds
  of the final evaluation (no leakage), so the fitness is deterministic for a
  given solution and can be cached.
- The evaluation load is calibrated once per (network, lambdas): the load where
  fixed shortest-path routing reaches `fitness_target_bp` blocking (default 2%).
- fitness_mode='structural' restores the old behaviour (nc = rl - 1, empty network).
- fitness_offset (default 0.0) is a constant added to the denominator
  (use 1.0 to get the "+1" variant).
- The best-solutions CSV also stores the fitness mode/load and the observed nc per pair.

CORRECTIONS APPLIED (v7):
- Wavelength conversion is implemented in the simulator (any node can convert),
  hop by hop from the source: on the first link take the first free wavelength;
  on each next link keep the wavelength used so far if it is free, otherwise
  take the first free one on that link and count one conversion. A call is
  blocked only if some link of the route has NO free wavelength at all.
  Conversions are counted per connection and saved (*_conversions_*.csv).
  (continuity_first=True is an optional variant: try one wavelength free on
  the whole route before going hop by hop; default is False.)

CORRECTIONS APPLIED (v6):
- Fitness now matches Eq. 1 exactly: fit = 1 / (alpha*nc + beta*rl)   (no "+1").
- Janet6 has 9 links (BIR-LON edge removed), as in Fig. 2 / Fig. 5b.
- PSO/DE discretization: clip(rint(x), 0, K_i - 1) applied consistently
  in the fitness evaluation AND in the final solution used in the simulation.
- PSO: adaptive=False (fixed w, c1, c2) when supported by the installed pymoo.
- Best solution (route indices, hops, fitness) and algorithm time are saved
  per execution; execution_times.csv separates algorithm time from total time.
- NST/inflection: first load with mean blocking >= 1% (as in the paper).

CORRECTIONS APPLIED (v5):
- FIXED: Per-gene upper bounds based on actual number of routes per OD pair
- Common Random Numbers (CRN): separation of algo_seed (different per
  algorithm) and traffic_seed (SAME for GA, PSO, and DE in the same
  run + SAME load).
- traffic_seed depends on (execution, load) — NOT only on execution.
- num_requests fixed at 5000 for all load levels (consistent with the paper).
- OPTIMIZED_PARAMS with two scenarios (40λ and 80λ).
- run_single_experiment selects parameters by the number of λ.
"""

import random
import os
from itertools import islice
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t as t_dist
from datetime import datetime
import json
import inspect
import heapq

# ============================================
# PYMOO IMPORTS (for PSO and DE)
# ============================================
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.pso import PSO as PSOAlgorithm
from pymoo.algorithms.soo.nonconvex.de import DE as DEAlgorithm
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize


# ============================================
# GLOBAL CONFIGURATION
# ============================================
NUM_REQUESTS_FIXED = 5000  # Fixed according to paper methodology


# ============================================
# DISCRETIZATION (PSO / DE -> LUT indices)
# ============================================
# Seeds for the fitness evaluation traffic. The final evaluation uses seeds
# exec*1_000_000 + load*100 + 42 (< 10**8), so these never coincide with them.
FITNESS_SEED_BASE = 10**12


def discretize_to_lut(x, xu_int) -> np.ndarray:
    """Convert continuous values to valid LUT indices.

    index_i = clip(round(x_i), 0, K_i - 1), with xu_int[i] = K_i - 1.
    Used both inside the fitness evaluation and for the final solution,
    so the evaluated solution is exactly the one simulated afterwards.
    """
    x = np.asarray(x, dtype=float)
    return np.clip(np.rint(x), 0, np.asarray(xu_int)).astype(int)


# ============================================
# PYMOO PROBLEM CLASS
# ============================================
class RWAAProblem(ElementwiseProblem):
    """RWA problem for use with pymoo (PSO and DE).
    
    Each gene corresponds to an OD pair. The upper bound of each gene
    is the actual number of available routes for that OD pair (minus 1),
    because Yen's algorithm may return fewer than K paths.
    """
    
    def __init__(self, gene_size, fitness_func, manual_pairs, k_shortest_paths, k=150):
        self.fitness_func = fitness_func
        self.manual_pairs = manual_pairs
        self.k_shortest_paths = k_shortest_paths
        self.k = k
        
        # Per-gene upper bounds based on actual number of routes
        xu_per_gene = []
        for pair in manual_pairs:
            routes = k_shortest_paths.get(pair, [])
            xu_per_gene.append(max(0, len(routes) - 1))
        
        print(f"  [RWAAProblem] Per-gene upper bounds: {xu_per_gene}")
        self.xu_int = np.array(xu_per_gene, dtype=int)
        
        super().__init__(n_var=gene_size,
                         n_obj=1,
                         xl=np.array([0]*gene_size),
                         xu=np.array(xu_per_gene),
                         vtype=int)
    
    def _evaluate(self, x, out, *args, **kwargs):
        x_int = discretize_to_lut(x, self.xu_int)
        fitness = -self.fitness_func(x_int.tolist(), self.manual_pairs)
        out["F"] = [fitness]


# ============================================
# WDM SIMULATOR WITH STATISTICAL ANALYSIS
# ============================================
class WDMSimulatorStatistical:
    """
    WDM network simulator supporting PSO, DE, and GA,
    including comprehensive statistical analysis and high-resolution runs.
    
    Uses Common Random Numbers (CRN) to ensure pairing between
    algorithms in statistical comparisons:
        - Same (execution, load) → same traffic realization for the 3 algorithms
        - Different (execution, load) → different traffic realization
    """

    def __init__(self,
                 graph: nx.Graph,
                 num_wavelengths: int = 40,
                 gene_size: int = 5,
                 manual_pairs: List[Tuple[int, int]] = None,
                 k: int = 150,
                 # Common parameters
                 population_size: int = 120,
                 n_gen: int = 40,
                 hops_weight: float = 0.55,
                 wavelength_weight: float = 0.45,
                 # PSO parameters
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 # DE parameters
                 CR: float = 0.9,
                 F: float = 0.8,
                 # GA parameters
                 crossover_rate: float = 0.6,
                 mutation_rate: float = 0.02,
                 tournament_size: int = 3,
                 num_generations_ag: int = 40,
                 # Wavelength assignment / conversion
                 conversion_enabled: bool = True,
                 continuity_first: bool = False,
                 # Fitness (Equation 1) evaluation
                 fitness_mode: str = 'dynamic',      # 'dynamic' (real nc) | 'structural' (nc = rl - 1)
                 fitness_requests: int = 400,        # requests per evaluation replica (incl. warm-up)
                 fitness_warmup: int = 100,          # initial requests discarded
                 fitness_replicas: int = 2,          # independent traffic replicas averaged
                 fitness_target_bp: float = 0.02,    # blocking used to calibrate the evaluation load
                 fitness_load: float = None,         # fixed evaluation load (None -> calibrate)
                 fitness_offset: float = 0.0         # added to the denominator (0 = exact Eq. 1)
                 ):
        
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.gene_size = gene_size
        self.manual_pairs = manual_pairs if manual_pairs else [(0, 6), (2, 5), (0, 3), (1, 4), (2, 6)]
        self.k = k
        
        # Common parameters
        self.population_size = population_size
        self.n_gen = n_gen
        self.hops_weight = hops_weight
        self.wavelength_weight = wavelength_weight
        
        # PSO parameters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # DE parameters
        self.CR = CR
        self.F = F
        
        # GA parameters
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.num_generations_ag = num_generations_ag
        
        # Wavelength assignment policy (hop by hop, from the source):
        #  - first link      : first free wavelength (first-fit)
        #  - each next link  : keep the wavelength used on the previous link if it
        #                      is free; otherwise take the first free wavelength on
        #                      that link -> one wavelength conversion is counted.
        #  - blocking        : only if some link has NO free wavelength at all.
        #  conversion_enabled=False -> pure wavelength continuity (no conversion).
        #  continuity_first=True    -> optional variant: try one wavelength free on
        #                              the whole route (0 conversions) before the
        #                              hop-by-hop rule. Default: False.
        self.conversion_enabled = conversion_enabled
        self.continuity_first = continuity_first
        
        # Fitness evaluation settings
        assert fitness_mode in ('dynamic', 'structural')
        self.fitness_mode = fitness_mode
        self.fitness_requests = fitness_requests
        self.fitness_warmup = fitness_warmup
        self.fitness_replicas = fitness_replicas
        self.fitness_target_bp = fitness_target_bp
        self.fitness_load = fitness_load
        self.fitness_offset = fitness_offset
        self._fit_cache = {}
        
        # Pre-compute K-shortest paths
        self.k_shortest_paths = self._get_all_k_shortest_paths()
        self.reset_network()
        
        # Assign a name to the graph for identification
        if not hasattr(self.graph, 'name'):
            self.graph.name = "Custom"

    def reset_network(self) -> None:
        """Reset wavelength channels in the network."""
        for u, v in self.graph.edges:
            self.graph[u][v]['wavelengths'] = np.ones(self.num_wavelengths, dtype=bool)
            self.graph[u][v]['current_wavelength'] = -1

    def release_wavelength(self, route: List[int], wavelength: int) -> None:
        """Release a wavelength from a route."""
        if not (0 <= wavelength < self.num_wavelengths):
            return
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if self.graph.has_edge(u, v):
                self.graph[u][v]['wavelengths'][wavelength] = True
                if self.graph[u][v]['current_wavelength'] == wavelength:
                    self.graph[u][v]['current_wavelength'] = -1

    def allocate_wavelength(self, route: List[int], wavelength: int) -> bool:
        """Allocate a wavelength along a route."""
        if not (0 <= wavelength < self.num_wavelengths):
            return False
        
        # Check availability along the entire route
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if not self.graph.has_edge(u, v) or not self.graph[u][v]['wavelengths'][wavelength]:
                return False
        
        # Allocate along the entire route
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            self.graph[u][v]['wavelengths'][wavelength] = False
            self.graph[u][v]['current_wavelength'] = wavelength
        
        return True

    def find_available_wavelength(self, route: List[int]) -> Optional[int]:
        """Find the first available wavelength along a route."""
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

    def assign_wavelengths(self, route: List[int]) -> Optional[List[int]]:
        """Return one wavelength per link of the route (or None -> blocked).

        Consecutive different values mean a wavelength conversion at the
        intermediate node between those two links.
        """
        n_links = len(route) - 1
        if n_links < 1:
            return None
        
        # Optional variant / no-conversion mode: single wavelength on the whole route
        if (not self.conversion_enabled) or self.continuity_first:
            w = self.find_available_wavelength(route)
            if w is not None:
                return [w] * n_links
            if not self.conversion_enabled:
                return None
        
        # Hop-by-hop: keep the previous wavelength when free, else first free (conversion)
        wls: List[int] = []
        prev = None
        for i in range(n_links):
            free = self.graph[route[i]][route[i + 1]]['wavelengths']
            if prev is not None and free[prev]:
                w = prev
            else:
                idx = np.flatnonzero(free)
                if idx.size == 0:
                    return None  # no free wavelength on this link -> blocked
                w = int(idx[0])
            wls.append(w)
            prev = w
        return wls

    def allocate_lightpath(self, route: List[int], wls: List[int]) -> None:
        """Occupy wls[i] on link i of the route."""
        for i, w in enumerate(wls):
            self.graph[route[i]][route[i + 1]]['wavelengths'][w] = False

    def release_lightpath(self, route: List[int], wls: List[int]) -> None:
        """Free wls[i] on link i of the route."""
        for i, w in enumerate(wls):
            self.graph[route[i]][route[i + 1]]['wavelengths'][w] = True

    @staticmethod
    def count_conversions(wls: List[int]) -> int:
        return sum(1 for i in range(1, len(wls)) if wls[i] != wls[i - 1])

    def _get_k_shortest_paths(self, source: int, target: int) -> List[List[int]]:
        """Compute K-shortest paths between two nodes (Yen's algorithm)."""
        if not nx.has_path(self.graph, source, target):
            return []
        try:
            return list(islice(nx.shortest_simple_paths(self.graph, source, target), self.k))
        except nx.NetworkXNoPath:
            return []

    def _get_all_k_shortest_paths(self) -> Dict[Tuple[int, int], List[List[int]]]:
        """Compute K-shortest paths for all source-destination pairs."""
        paths = {}
        for source, target in self.manual_pairs:
            paths[(source, target)] = self._get_k_shortest_paths(source, target)
        return paths

    def _fitness_route(self, route: List[int]) -> float:
        """Compute fitness for a single route.
        
        Unified fitness function (Equation 1 in the paper):
            fit = 1 / (alpha * nc + beta * rl)
        
        where:
            nc    = number of wavelength conversions along the route
            rl    = number of links (hops) in the route
            alpha = wavelength_weight (conversion weight)
            beta  = hops_weight (link weight)
        
        NOTE: LEGACY structural version (empty network). Kept for reference; the
        optimization now uses _solution_metrics(), which measures nc with a short
        dynamic evaluation (fitness_mode='dynamic') or reproduces this proxy
        (fitness_mode='structural').
        """
        if len(route) < 2:
            return 0.0
        
        hops = len(route) - 1
        
        # Structural estimate of conversions:
        # Each intermediate node is a potential conversion point.
        # A route with N nodes has (N - 2) intermediate nodes.
        # If the route is a direct link (2 nodes, 1 hop), there are
        # no intermediate nodes, so no conversions.
        wavelength_changes = max(0, len(route) - 2)
        
        # Unified fitness function (Equation 1 in the paper)
        # fit = 1 / (alpha * nc + beta * rl)
        fitness = 1.0 / (self.wavelength_weight * wavelength_changes +
                         self.hops_weight * hops)
        
        return fitness

    # ============================================
    # FITNESS (Equation 1) - real conversions via short dynamic evaluation
    # ============================================
    def _run_short_sim(self, routes: List[List[int]], load: float, total_requests: int,
                       warmup: int, seed: int):
        """Short dynamic-traffic simulation used ONLY to evaluate a candidate solution.

        Uses a local RNG (global random/numpy state is untouched) and the same
        hop-by-hop wavelength assignment as the final simulation.
        Returns per-pair lists (conversions, established, blocked), counted after warm-up.
        """
        n_pairs = len(routes)
        rng = np.random.default_rng(seed)
        arrivals = np.cumsum(rng.exponential(1.0 / load, total_requests))
        holds = rng.exponential(1.0, total_requests)
        pair_idx = rng.integers(0, n_pairs, total_requests)
        
        self.reset_network()
        heap: list = []
        active: dict = {}
        conv = [0] * n_pairs
        est = [0] * n_pairs
        blk = [0] * n_pairs
        
        for k in range(total_requests):
            t = arrivals[k]
            while heap and heap[0][0] <= t:
                _, cid = heapq.heappop(heap)
                r, w = active.pop(cid)
                self.release_lightpath(r, w)
            p = int(pair_idx[k])
            route = routes[p]
            wls = self.assign_wavelengths(route)
            counted = k >= warmup
            if wls is None:
                if counted:
                    blk[p] += 1
            else:
                self.allocate_lightpath(route, wls)
                active[k] = (route, wls)
                heapq.heappush(heap, (t + holds[k], k))
                if counted:
                    est[p] += 1
                    conv[p] += self.count_conversions(wls)
        
        self.reset_network()
        return conv, est, blk

    def calibrate_fitness_load(self) -> float:
        """Evaluation load = load where fixed shortest-path routing reaches the target blocking."""
        if self.fitness_load is not None:
            return self.fitness_load
        routes = [self.k_shortest_paths[p][0] for p in self.manual_pairs]
        lo, hi = 1, 1000
        for _ in range(10):
            mid = (lo + hi) // 2
            conv, est, blk = self._run_short_sim(routes, mid, 3000, 500, FITNESS_SEED_BASE + 777)
            total = sum(est) + sum(blk)
            bp = sum(blk) / total if total else 0.0
            if bp < self.fitness_target_bp:
                lo = mid + 1
            else:
                hi = mid
        self.fitness_load = float(lo)
        return self.fitness_load

    def _solution_metrics(self, individual: List[int]):
        """Return (fitness, nc_per_pair, rl_per_pair) for a solution (cached)."""
        key = tuple(int(v) for v in individual)
        cached = self._fit_cache.get(key)
        if cached is not None:
            return cached
        
        routes = [self.k_shortest_paths[pair][idx] for pair, idx in zip(self.manual_pairs, key)]
        rls = [len(r) - 1 for r in routes]
        
        if self.fitness_mode == 'structural':
            ncs = [max(0, len(r) - 2) for r in routes]          # old proxy: nc = rl - 1
        else:
            load = self.calibrate_fitness_load()
            n = len(routes)
            conv = [0] * n
            est = [0] * n
            for rep_idx in range(self.fitness_replicas):
                c, e, _ = self._run_short_sim(routes, load, self.fitness_requests,
                                              self.fitness_warmup, FITNESS_SEED_BASE + rep_idx)
                for i in range(n):
                    conv[i] += c[i]
                    est[i] += e[i]
            # mean conversions per established connection; if a pair had no established
            # connection, assume the worst case (conversion at every intermediate node)
            ncs = [conv[i] / est[i] if est[i] > 0 else float(max(0, rls[i] - 1)) for i in range(n)]
        
        total = 0.0
        for nc, rl in zip(ncs, rls):
            # Equation 1 (per OD pair): fit = 1 / (alpha * nc + beta * rl)
            total += 1.0 / (self.wavelength_weight * nc + self.hops_weight * rl + self.fitness_offset)
        
        result = (total, ncs, rls)
        self._fit_cache[key] = result
        return result

    def _fitness(self, individual: List[int], source_targets: List[Tuple[int, int]]) -> float:
        """Total fitness of a solution = sum over OD pairs of Equation 1."""
        return self._solution_metrics(individual)[0]

    def run_pso(self, seed: int = None) -> Tuple[np.ndarray, float]:
        """Run PSO (Particle Swarm Optimization)."""
        problem = RWAAProblem(self.gene_size, self._fitness, self.manual_pairs,
                              self.k_shortest_paths, k=self.k)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        pso_kwargs = dict(
            pop_size=self.population_size,
            w=self.w,
            c1=self.c1,
            c2=self.c2,
            sampling=IntegerRandomSampling(),
        )
        # Paper: classical PSO with fixed w, c1, c2. Disable pymoo's
        # adaptive coefficients when the installed version supports it.
        if 'adaptive' in inspect.signature(PSOAlgorithm.__init__).parameters:
            pso_kwargs['adaptive'] = False
        else:
            print("  [WARN] this pymoo version has no 'adaptive' argument for PSO")
        algorithm = PSOAlgorithm(**pso_kwargs)
        
        termination = get_termination("n_gen", self.n_gen)
        
        res = minimize(problem, algorithm, termination,
                       seed=seed if seed else 1,
                       save_history=False,
                       verbose=False)
        
        X = discretize_to_lut(res.X, problem.xu_int)
        # PSO: the problem minimizes, but fitness was defined to maximize.
        # Therefore, we invert the sign.
        F = -res.F if hasattr(res.F, '__len__') else -res.F
        
        return X, F

    # ============================================
    # DE ALGORITHM (via pymoo)  ← CORREÇÃO
    # ============================================
    def run_de(self, seed: int = None) -> Tuple[np.ndarray, float]:
        """Run DE (Differential Evolution) with DE/rand/1/bin variant."""
        problem = RWAAProblem(self.gene_size, self._fitness, self.manual_pairs,
                              self.k_shortest_paths, k=self.k)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        algorithm = DEAlgorithm(
            pop_size=self.population_size,
            CR=self.CR,
            F=self.F,
            variant="DE/rand/1/bin",  # Classical DE variant
            sampling=IntegerRandomSampling(),
        )
        
        termination = get_termination("n_gen", self.n_gen)
        
        res = minimize(problem, algorithm, termination,
                       seed=seed if seed else 1,
                       save_history=False,
                       verbose=False)
        
        X = discretize_to_lut(res.X, problem.xu_int)
        # F is the negated fitness (pymoo minimizes); only X is used afterwards.
        F = -res.F
        
        return X, F

    # ============================================
    # GA ALGORITHM (manual implementation)
    # ============================================
    def _initialize_population_ag(self) -> List[List[int]]:
        """Initialize GA population."""
        population = []
        for _ in range(self.population_size):
            individual = []
            for source, target in self.manual_pairs:
                routes = self.k_shortest_paths.get((source, target), [])
                if routes:
                    max_idx = len(routes) - 1
                    individual.append(random.randint(0, max_idx))
                else:
                    individual.append(0)
            population.append(individual)
        return population

    def _tournament_selection_ag(self, population: List[List[int]], 
                                   fitness_scores: List[float]) -> List[int]:
        """GA tournament selection."""
        tournament = random.sample(list(zip(population, fitness_scores)), 
                                   min(self.tournament_size, len(population)))
        return max(tournament, key=lambda x: x[1])[0]

    def _crossover_ag(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """GA single-point crossover."""
        if len(parent1) <= 1:
            return parent1[:], parent2[:]
        
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutate_ag(self, individual: List[int]) -> None:
        """GA uniform mutation (gene-by-gene)."""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                source, target = self.manual_pairs[i]
                routes = self.k_shortest_paths.get((source, target), [])
                if routes:
                    max_idx = len(routes) - 1
                    individual[i] = random.randint(0, max_idx)

    def run_ag(self, seed: int = None) -> Tuple[List[int], float]:
        """Run GA (Genetic Algorithm) with elitism, tournament, and single-point crossover."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialization
        population = self._initialize_population_ag()
        
        best_individual_overall = None
        best_fitness_overall = -float('inf')
        
        for generation in range(self.num_generations_ag):
            # Evaluation
            fitness_scores = [self._fitness(ind, self.manual_pairs) for ind in population]
            
            # Best of generation
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            
            if best_fitness > best_fitness_overall:
                best_fitness_overall = best_fitness
                best_individual_overall = population[best_idx].copy()
            
            # Elitism: preserve the best 10% intact
            elite_size = max(1, self.population_size // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            new_population = [population[i] for i in elite_indices]
            
            # Generate the new population
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection_ag(population, fitness_scores)
                parent2 = self._tournament_selection_ag(population, fitness_scores)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover_ag(parent1, parent2)
                    new_population.extend([child1, child2])
                else:
                    new_population.extend([parent1.copy(), parent2.copy()])
            
            # Mutation (except elite)
            for i in range(elite_size, len(new_population)):
                self._mutate_ag(new_population[i])
            
            population = new_population[:self.population_size]
        
        return best_individual_overall, best_fitness_overall

    # ============================================
    # DYNAMIC TRAFFIC SIMULATION (WITH CRN)
    # ============================================
    def simulate_dynamic_traffic_optimized(self, 
                                            best_individual: List[int],
                                            load: float,
                                            traffic_seed: int = None,
                                            num_requests: int = None) -> Dict[Tuple[int, int], float]:
        """
        Simulate dynamic traffic using Common Random Numbers (CRN).
        
        Args:
            best_individual : routes selected by the algorithm
            load            : traffic load in Erlangs
            traffic_seed    : traffic seed (SAME for the three algorithms
                              in the same (execution, load) combination)
            num_requests    : number of calls (default: 5000 fixed)
        
        Returns:
            Dictionary with blocking probability per OD pair and global average.
        """
        # =========================================
        # COMMON RANDOM NUMBERS (CRN)
        # =========================================
        if traffic_seed is not None:
            np.random.seed(traffic_seed)
            random.seed(traffic_seed)
        
        # =========================================
        # FIXED NUMBER OF REQUESTS
        # =========================================
        if num_requests is None:
            actual_requests = NUM_REQUESTS_FIXED
        else:
            actual_requests = num_requests
        
        hold_time_mean = 1.0
        arrival_rate = load / hold_time_mean
        mean_interarrival = 1.0 / arrival_rate if arrival_rate > 0 else float('inf')
        
        # Extremely low load: zero blocking
        if mean_interarrival > 1e6:
            bp_by_pair = {pair: 0.0 for pair in self.manual_pairs}
            bp_by_pair['global'] = 0.0
            return bp_by_pair
        
        # Per-OD-pair counters
        blocked_by_pair = {pair: 0 for pair in self.manual_pairs}
        total_by_pair = {pair: 0 for pair in self.manual_pairs}
        
        active_connections = {}
        next_id = 0
        current_time = 0.0
        
        # Conversion statistics of this simulation
        established = 0
        conversions_total = 0
        conns_with_conversion = 0
        
        # Batch-generate arrival and hold times (vectorized)
        interarrival_times = np.random.exponential(mean_interarrival, actual_requests)
        arrival_times = np.cumsum(interarrival_times)
        hold_times = np.random.exponential(hold_time_mean, actual_requests)
        
        for req_idx in range(actual_requests):
            current_time = arrival_times[req_idx]
            release_time = current_time + hold_times[req_idx]
            
            # Release expired connections
            to_remove = [cid for cid, (_, _, rtime) in active_connections.items() 
                         if rtime <= current_time]
            
            for conn_id in to_remove:
                conn_route, conn_wls, _ = active_connections[conn_id]
                self.release_lightpath(conn_route, conn_wls)
                del active_connections[conn_id]
            
            # Randomly choose an origin-destination pair
            source, target = random.choice(self.manual_pairs)
            pair = (source, target)
            total_by_pair[pair] += 1
            
            # Retrieve route
            pair_idx = self.manual_pairs.index((source, target))
            if pair_idx < len(best_individual):
                route_idx = best_individual[pair_idx]
                routes = self.k_shortest_paths.get((source, target), [])
                if route_idx < len(routes):
                    route = routes[route_idx]
                else:
                    blocked_by_pair[pair] += 1
                    continue
            else:
                blocked_by_pair[pair] += 1
                continue
            
            # Try to allocate a wavelength
            wls = self.assign_wavelengths(route)
            
            if wls is not None:
                self.allocate_lightpath(route, wls)
                active_connections[next_id] = (route, wls, release_time)
                next_id += 1
                established += 1
                n_conv = self.count_conversions(wls)
                conversions_total += n_conv
                if n_conv > 0:
                    conns_with_conversion += 1
            else:
                blocked_by_pair[pair] += 1
        
        # Release remaining connections
        for conn_route, conn_wls, _ in active_connections.values():
            self.release_lightpath(conn_route, conn_wls)
        
        # Compute BP per pair
        bp_by_pair = {}
        total_blocked = 0
        total_requests = 0
        
        for pair in self.manual_pairs:
            if total_by_pair[pair] > 0:
                bp = blocked_by_pair[pair] / total_by_pair[pair]
                bp_by_pair[pair] = bp
                total_blocked += blocked_by_pair[pair]
                total_requests += total_by_pair[pair]
            else:
                bp_by_pair[pair] = 0.0
        
        # Compute global average
        bp_by_pair['global'] = total_blocked / total_requests if total_requests > 0 else 0.0
        
        # Conversion metrics (extra keys; per-pair loops only read OD-pair keys)
        bp_by_pair['conv_per_conn'] = conversions_total / established if established > 0 else 0.0
        bp_by_pair['frac_conv_conns'] = conns_with_conversion / established if established > 0 else 0.0
        
        return bp_by_pair

    # ============================================
    # HIGH-RESOLUTION EXPERIMENT (WITH CRN)
    # ============================================
    def run_high_resolution_experiment(self,
                                        algorithm: str,
                                        max_load: int,
                                        num_executions: int = 10,
                                        save_results: bool = True,
                                        results_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Run high-resolution experiment (from 1 to max_load).
        Saves GLOBAL and per-OD-pair data.
        """
        if results_dir is None:
            results_dir = f"results_{algorithm}_highres"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create subfolders for per-pair data
        pairs_dir = os.path.join(results_dir, "por_par")
        os.makedirs(pairs_dir, exist_ok=True)
        
        loads = list(range(1, max_load + 1))
        
        # Global results
        results_global = {load: [] for load in loads}
        
        # Per-OD-pair results
        results_by_pair = {pair: {load: [] for load in loads} for pair in self.manual_pairs}
        
        # Conversion metrics per load (one value per execution)
        conv_per_conn = {load: [] for load in loads}
        frac_conv_conns = {load: [] for load in loads}
        
        # Per-execution records (best solution and timings)
        exec_records = []
        
        # Total time measurement
        start_total_time = time.time()
        
        print("="*80)
        print(f"HIGH-RESOLUTION EXPERIMENT - {algorithm}")
        print(f"Network: {self.graph.name}")
        print(f"Wavelengths: {self.num_wavelengths}")
        print(f"Loads: 1 to {max_load} ({len(loads)} points)")
        print(f"Executions: {num_executions}")
        print(f"OD pairs: {self.manual_pairs}")
        print(f"Requests per simulation: {NUM_REQUESTS_FIXED} (fixed)")
        print(f"Mode: Common Random Numbers (CRN) per (execution, load)")
        print(f"Wavelength conversion: {'ON' if self.conversion_enabled else 'OFF'} | continuity first: {self.continuity_first}")
        if self.fitness_mode == 'dynamic':
            t_cal = time.time()
            L_fit = self.calibrate_fitness_load()
            print(f"Fitness: Eq.1 with REAL conversions | evaluation load = {L_fit:.0f} Erlangs "
                  f"(shortest-path blocking ~ {self.fitness_target_bp:.0%}) | {self.fitness_requests} req x "
                  f"{self.fitness_replicas} replicas, warm-up {self.fitness_warmup} | calibration {time.time() - t_cal:.1f}s")
        else:
            print("Fitness: Eq.1 with structural proxy nc = rl - 1 (legacy)")
        print("="*80)
        
        for exec_idx in range(num_executions):
            print(f"\nExecution {exec_idx + 1}/{num_executions}")
            
            # Measure time for this execution (fitness cache cleared: fair algorithm timing)
            self._fit_cache.clear()
            start_exec_time = time.time()
            
            # =========================================
            # ALGORITHM seed: different per algorithm
            # =========================================
            base_seed = exec_idx * 100 + 42
            
            if algorithm == 'PSO':
                algo_seed = base_seed
            elif algorithm == 'DE':
                algo_seed = base_seed + 1000
            else:  # GA
                algo_seed = base_seed + 2000
            
            # Run the algorithm
            if algorithm == 'PSO':
                best_ind, _ = self.run_pso(seed=algo_seed)
            elif algorithm == 'DE':
                best_ind, _ = self.run_de(seed=algo_seed)
            else:  # GA
                best_ind, _ = self.run_ag(seed=algo_seed)
            
            algo_time = time.time() - start_exec_time
            best_ind = [int(v) for v in best_ind]
            best_routes = [self.k_shortest_paths[pair][idx]
                           for pair, idx in zip(self.manual_pairs, best_ind)]
            best_hops = [len(r) - 1 for r in best_routes]
            min_hops = [len(self.k_shortest_paths[pair][0]) - 1 for pair in self.manual_pairs]
            
            # =========================================
            # CRN: traffic_seed depends on (exec, load)
            # =========================================
            for i, load in enumerate(loads):
                # Deterministic seed based on (execution, load)
                traffic_seed = exec_idx * 1_000_000 + load * 100 + 42
                
                self.reset_network()
                bp_dict = self.simulate_dynamic_traffic_optimized(
                    best_ind,
                    load,
                    traffic_seed=traffic_seed,
                )
                
                # Store global average
                results_global[load].append(bp_dict['global'])
                conv_per_conn[load].append(bp_dict.get('conv_per_conn', 0.0))
                frac_conv_conns[load].append(bp_dict.get('frac_conv_conns', 0.0))
                
                # Store per-pair
                for pair in self.manual_pairs:
                    results_by_pair[pair][load].append(bp_dict[pair])
                
                if exec_idx == 0 and (i + 1) % 50 == 0:
                    print(f"  Progress: {i+1}/{len(loads)} loads")
            
            # Statistics for this execution
            bp_values = [results_global[load][-1] for load in loads]
            bp_array = np.array(bp_values)
            exec_time = time.time() - start_exec_time
            exec_records.append({
                'execution': exec_idx,
                'algo_seed': algo_seed,
                'best_ind': json.dumps(best_ind),
                'best_hops': json.dumps(best_hops),
                'min_possible_hops': json.dumps(min_hops),
                'is_min_hops_everywhere': best_hops == min_hops,
                'fitness_total': self._fitness(best_ind, self.manual_pairs),
                'fitness_mode': self.fitness_mode,
                'fitness_load': self.fitness_load if self.fitness_mode == 'dynamic' else None,
                'best_nc_per_pair': json.dumps([round(v, 4) for v in self._solution_metrics(best_ind)[1]]),
                'best_routes': json.dumps(best_routes),
                'algo_time_s': round(algo_time, 4),
                'simulation_time_s': round(exec_time - algo_time, 4),
                'exec_time_s': round(exec_time, 4),
            })
            print(f"  Best solution (route idx per OD pair): {best_ind} | hops: {best_hops} | min hops: {min_hops}")
            print(f"  Global BP: min={bp_array.min():.6f}, max={bp_array.max():.6f}")
            print(f"  Execution time: {exec_time:.2f}s (algorithm: {algo_time:.2f}s)")
        
        total_time = time.time() - start_total_time
        print(f"\nTOTAL time for {algorithm}: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        # Global results DataFrame
        df_results_global = pd.DataFrame(results_global)
        stats_summary_global = self._compute_statistics_highres(df_results_global, loads)
        
        # Per-pair DataFrames
        stats_by_pair = {}
        df_by_pair = {}
        
        for pair in self.manual_pairs:
            pair_name = f"{pair[0]}_{pair[1]}"
            df_by_pair[pair_name] = pd.DataFrame(results_by_pair[pair])
            stats_by_pair[pair_name] = self._compute_statistics_highres(df_by_pair[pair_name], loads)
        
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save GLOBAL data
            raw_filename = f"{algorithm}_raw_{self.graph.name}_{self.num_wavelengths}l_{max_load}loads_{timestamp}.csv"
            df_results_global.to_csv(f"{results_dir}/{raw_filename}", index=False)
            
            stats_filename = f"{algorithm}_stats_{self.graph.name}_{self.num_wavelengths}l_{max_load}loads_{timestamp}.csv"
            stats_summary_global.to_csv(f"{results_dir}/{stats_filename}", index=False)
            
            # Global plot
            self._plot_highres_curve(df_results_global, stats_summary_global, loads, algorithm, max_load, results_dir, timestamp)
            
            # Save per-OD-pair data
            for pair in self.manual_pairs:
                pair_name = f"{pair[0]}_{pair[1]}"
                pair_dir = os.path.join(pairs_dir, pair_name)
                os.makedirs(pair_dir, exist_ok=True)
                
                # Per-pair raw data
                raw_pair_filename = f"{algorithm}_raw_{self.graph.name}_{self.num_wavelengths}l_{max_load}loads_par_{pair_name}_{timestamp}.csv"
                df_by_pair[pair_name].to_csv(f"{pair_dir}/{raw_pair_filename}", index=False)
                
                # Per-pair statistics
                stats_pair_filename = f"{algorithm}_stats_{self.graph.name}_{self.num_wavelengths}l_{max_load}loads_par_{pair_name}_{timestamp}.csv"
                stats_by_pair[pair_name].to_csv(f"{pair_dir}/{stats_pair_filename}", index=False)
                
                # Per-pair plot
                self._plot_highres_curve_pair(df_by_pair[pair_name], stats_by_pair[pair_name], 
                                              loads, algorithm, max_load, pair_dir, timestamp, pair_name)
            
            # Save conversion metrics (mean/std over executions, per load)
            df_conv = pd.DataFrame({
                'load': loads,
                'conv_per_conn_mean': [np.mean(conv_per_conn[l]) for l in loads],
                'conv_per_conn_std': [np.std(conv_per_conn[l], ddof=1) if len(conv_per_conn[l]) > 1 else 0.0 for l in loads],
                'frac_conv_conns_mean': [np.mean(frac_conv_conns[l]) for l in loads],
                'frac_conv_conns_std': [np.std(frac_conv_conns[l], ddof=1) if len(frac_conv_conns[l]) > 1 else 0.0 for l in loads],
            })
            conv_filename = f"{algorithm}_conversions_{self.graph.name}_{self.num_wavelengths}l_{max_load}loads_{timestamp}.csv"
            df_conv.to_csv(f"{results_dir}/{conv_filename}", index=False)
            
            # Save best solutions per execution
            df_best = pd.DataFrame(exec_records)
            best_filename = f"{algorithm}_best_solutions_{self.graph.name}_{self.num_wavelengths}l_{max_load}loads_{timestamp}.csv"
            df_best.to_csv(f"{results_dir}/{best_filename}", index=False)
            
            # Save execution time
            self._save_execution_time(algorithm, max_load, num_executions, total_time, 
                                       self.graph.name, self.num_wavelengths, results_dir, timestamp,
                                       algo_times=[r['algo_time_s'] for r in exec_records])
            
            # Save pair summary
            self._save_pairs_summary(stats_by_pair, algorithm, self.graph.name, self.num_wavelengths, 
                                     max_load, results_dir, timestamp)
        
        return df_results_global, stats_summary_global, results_by_pair

    def _compute_statistics_highres(self, df_results: pd.DataFrame, loads: List[float]) -> pd.DataFrame:
        """Compute high-resolution statistics."""
        from scipy import stats as scipy_stats
        
        stats = []
        
        for load in loads:
            data = df_results[load].dropna()
            
            if len(data) > 0:
                mean_val = np.mean(data)
                std_val = np.std(data, ddof=1)
                median_val = np.median(data)
                min_val = np.min(data)
                max_val = np.max(data)
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                
                # 95% Confidence Interval
                conf_int = scipy_stats.t.interval(0.95, df=len(data)-1, 
                                                   loc=mean_val, 
                                                   scale=std_val/np.sqrt(len(data)))
                
                stats.append({
                    'load': load,
                    'mean': mean_val,
                    'std': std_val,
                    'median': median_val,
                    'min': min_val,
                    'max': max_val,
                    'q1': q1,
                    'q3': q3,
                    'iqr': q3 - q1,
                    'ci_lower': conf_int[0],
                    'ci_upper': conf_int[1],
                    'n_executions': len(data)
                })
        
        return pd.DataFrame(stats)

    def _plot_highres_curve(self, df_results, stats_summary, loads, algorithm, max_load, results_dir, timestamp):
        """High-resolution plot (global data)."""
        colors = {'PSO': 'blue', 'DE': 'green', 'GA': 'red'}
        color = colors.get(algorithm, 'black')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        means = stats_summary['mean'].values
        ci_lower = stats_summary['ci_lower'].values
        ci_upper = stats_summary['ci_upper'].values
        
        ax.plot(loads, means, color=color, linewidth=1.5, alpha=0.8, label=f'{algorithm} (mean)')
        ax.fill_between(loads, ci_lower, ci_upper, alpha=0.2, color=color, label='95% CI')
        
        # Inflection point (1% blocking)
        inflexion_load = None
        for i, mean in enumerate(means):
            if mean >= 0.01:
                inflexion_load = loads[i]
                ax.axvline(x=inflexion_load, color='red', linestyle='--', linewidth=2,
                          label=f'1% blocking: {inflexion_load:.0f} Erlangs')
                break
        
        # 1% reference line
        ax.axhline(y=0.01, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='1% blocking')
        
        ax.set_xlabel('Traffic Load (Erlangs)', fontsize=12)
        ax.set_ylabel('Blocking Probability', fontsize=12)
        ax.set_title(f'{algorithm} - {self.graph.name} ({self.num_wavelengths} wavelengths)\n'
                     f'Blocking Curve - Loads 1 to {max_load}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_xlim(0, max_load)
        
        plt.tight_layout()
        filename = f"{results_dir}/{algorithm}_curve_{self.graph.name}_{self.num_wavelengths}l_{max_load}loads_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_highres_curve_pair(self, df_results, stats_summary, loads, algorithm, max_load, results_dir, timestamp, pair_name):
        """High-resolution plot for a specific OD pair."""
        colors = {'PSO': 'blue', 'DE': 'green', 'GA': 'red'}
        color = colors.get(algorithm, 'black')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        means = stats_summary['mean'].values
        ci_lower = stats_summary['ci_lower'].values
        ci_upper = stats_summary['ci_upper'].values
        
        ax.plot(loads, means, color=color, linewidth=1.5, alpha=0.8, label=f'{algorithm} (mean)')
        ax.fill_between(loads, ci_lower, ci_upper, alpha=0.2, color=color, label='95% CI')
        
        # Inflection point (1% blocking)
        for i, mean in enumerate(means):
            if mean >= 0.01:
                ax.axvline(x=loads[i], color='red', linestyle='--', linewidth=1, alpha=0.5)
                break
        
        ax.axhline(y=0.01, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        
        ax.set_xlabel('Traffic Load (Erlangs)', fontsize=12)
        ax.set_ylabel('Blocking Probability', fontsize=12)
        ax.set_title(f'{algorithm} - Pair ({pair_name.replace("_",",")}) - {self.graph.name} ({self.num_wavelengths} wavelengths)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_xlim(0, max_load)
        
        plt.tight_layout()
        filename = f"{results_dir}/{algorithm}_curve_{self.graph.name}_{self.num_wavelengths}l_{max_load}loads_par_{pair_name}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_execution_time(self, algorithm: str, max_load: int, num_executions: int,
                              total_time: float, network: str, lambdas: int, 
                              results_dir: str, timestamp: str, algo_times: List[float] = None):
        """Save execution time to a consolidated file.

        total_time_seconds = whole experiment (optimization + traffic simulation
        of every load). algo_* columns isolate the optimization step only.
        """
        import csv
        
        # Consolidated time file
        time_file = "execution_times.csv"
        time_file_path = os.path.join(results_dir, time_file)
        
        # Data to save
        data = {
            'timestamp': timestamp,
            'algorithm': algorithm,
            'network': network,
            'num_wavelengths': lambdas,
            'max_load': max_load,
            'num_executions': num_executions,
            'num_requests': NUM_REQUESTS_FIXED,
            'total_time_seconds': round(total_time, 2),
            'total_time_minutes': round(total_time / 60, 2),
            'mean_exec_time_seconds': round(total_time / num_executions, 4),
            'total_algo_time_seconds': round(float(np.sum(algo_times)), 4) if algo_times else None,
            'mean_algo_time_seconds': round(float(np.mean(algo_times)), 4) if algo_times else None,
        }
        
        # Append safely even if an older file (with fewer columns) already exists:
        # merge with pandas so the header always matches the rows.
        new_row = pd.DataFrame([data])
        if os.path.exists(time_file_path):
            try:
                old_df = pd.read_csv(time_file_path, engine='python', on_bad_lines='skip')
                new_row = pd.concat([old_df, new_row], ignore_index=True)
            except Exception as e:
                print(f"  [WARN] could not read existing {time_file_path}: {e}. Rewriting it.")
        new_row.to_csv(time_file_path, index=False)
        
        print(f"  Execution time saved to: {time_file_path}")
        
        # Also save a separate file for this run
        time_detail_file = f"{results_dir}/{algorithm}_time_{network}_{lambdas}l_{max_load}loads_{timestamp}.json"
        with open(time_detail_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_pairs_summary(self, stats_by_pair, algorithm, network, lambdas, max_load, results_dir, timestamp):
        """Save a comparative summary of OD pairs."""
        summary_data = []
        
        for pair_name, df_stats in stats_by_pair.items():
            # Find inflection point for this pair
            inflexion = None
            for _, row in df_stats.iterrows():
                if row['mean'] >= 0.01:
                    inflexion = row['load']
                    break
            
            summary_data.append({
                'od_pair': pair_name.replace('_', '->'),
                'bp_mean': df_stats['mean'].mean(),
                'bp_max': df_stats['mean'].max(),
                'bp_min': df_stats['mean'].min(),
                'inflection_1pct': inflexion if inflexion else f">{max_load}",
                'load_max_bp': df_stats.loc[df_stats['mean'].idxmax(), 'load'],
                'max_bp_value': df_stats['mean'].max()
            })
        
        df_summary = pd.DataFrame(summary_data)
        summary_file = f"{results_dir}/{algorithm}_pairs_summary_{network}_{lambdas}l_{max_load}loads_{timestamp}.csv"
        df_summary.to_csv(summary_file, index=False)
        print(f"  Pair summary saved to: {summary_file}")


# ============================================
# NETWORK CONFIGURATION
# ============================================
def get_janet6_graph():
    """Janet6 network - 7 nodes, 9 links"""
    G = nx.Graph()
    edges = [
            (0, 1), (0, 2),
            (1, 2), (1, 3),
            (2, 4),
            (3, 4), (3, 5),  #(3,6),
            (4, 6),
            (5, 6)
        ]

    G.add_edges_from(edges)
    G.name = "Janet6"
    
    # Node label mapping for UK cities
    G.node_labels = {
        0:  "GLA",   # Glasgow
        1:  "MAN",   # Manchester
        2:  "LEE",   # Leeds
        3:  "BIR",   # Birmingham
        4:  "NOT",   # Nottingham
        5:  "BRI",   # Bristol
        6:  "LON",   # London
        
    }
    
    print(f"Janet6: {G.number_of_nodes()} nodes, {G.number_of_edges()} links")
    return G



def get_redclara_graph():
    """RedCLARA network - 13 nodes, 17 links"""
    G = nx.Graph()
    edges = [
        (0, 1), (0, 5), (0, 8), (0, 11),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6), (5, 7), (5, 11),
        (7, 8),
        (8, 9), (8, 11),
        (9, 10), (9, 11),
        (11, 12)
    ]
    G.add_edges_from(edges)
    G.name = "RedCLARA"
    
    # Node label mapping for countries
    G.node_labels = {
        0: "US", 1: "MX", 2: "GT", 3: "SV",
        4: "CR", 5: "PN", 6: "VE", 7: "CO",
        8: "CL", 9: "AR", 10: "UY", 11: "BR", 12: "UK"
    }
    
    print(f"RedCLARA: {G.number_of_nodes()} nodes, {G.number_of_edges()} links")
    return G


def get_ipe_graph():
    """Ipê network - 28 nodes, 39 links"""
    G = nx.Graph()
    edges =  [
            (0, 1),
            (1, 3), (1, 4),
            (2, 4),
            (3, 4), (3, 7), (3, 17), (3, 19), (3, 25),
            (4, 6), (4, 12),
            (5, 25),
            (6, 7),
            (7, 8), (7, 11), (7, 18), (7, 19),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12), (11, 13), (11, 15),
            (13, 14),
            (14, 15),
            (15, 16), (15, 19),
            (16, 17),
            (17, 18),
            (18, 19), (18, 20), (18, 22),
            (20, 21),
            (21, 22),
            (22, 23),
            (23, 24),
            (24, 25), (24, 26),
            (26, 27)
        ]
    G.add_edges_from(edges)
    G.name = "IPE"
    
    # Node label mapping for Brazilian states/capitals
    G.node_labels = {
        0:  "RR",   # Roraima
        1:  "AM",   # Amazonas
        2:  "Ap",   # Amapá
        3:  "DF",   # Distrito federal
        4:  "PA",   # Pará
        5:  "TO",   # Tocantins
        6:  "MA",   # Maranhão
        7:  "CE",   # Ceará
        8:  "RN",   # Rio Grande do Norte
        9:  "PB1",  # Paraíba (1)
        10: "PB2",  # Paraíba (2)
        11: "PE",   # Pernambuco
        12: "PI",   # Piauí
        13: "AL",   # Alagoas
        14: "SE",   # Sergipe
        15: "BA",   # Bahia
        16: "ES",   # Espirito Santo
        17: "RJ",   # Rio de Janeiro
        18: "SAO",  # São Paulo
        19: "GM",   # Minhas Gerais
        20: "SC",   # Santa Catarina
        21: "RS",   # Rio Grande do Sul
        22: "PR",   # Paraná
        23: "MS",   # Mato Grosso do Sul
        24: "MT",   # Mato Grosso
        25: "GOI",  # Goiânia
        26: "RO",   # Rondônia
        27: "AC",   # Acre
        
    }
    
    print(f"IPE: {G.number_of_nodes()} nodes, {G.number_of_edges()} links")
    return G


# ============================================
# OPTIMIZED PARAMETERS (BY SCENARIO)
# ============================================
OPTIMIZED_PARAMS = {
    40: {
        'PSO': {
            'population_size': 120,
            'n_gen': 35,
            'w': 0.6719,
            'c1': 1.9636,
            'c2': 1.4178,
            'hops_weight': 0.55,
            'wavelength_weight': 0.45,
            'k': 150
        },
        'DE': {
            'population_size': 120,
            'n_gen': 35,
            'CR': 0.5355,
            'F': 0.4713,
            'hops_weight': 0.55,
            'wavelength_weight': 0.45,
            'k': 150
        },
        'GA': {
            'population_size': 120,
            'num_generations_ag': 35,
            'crossover_rate': 0.2710,
            'mutation_rate': 0.2952,
            'tournament_size': 4,
            'hops_weight': 0.55,
            'wavelength_weight': 0.45,
            'k': 150
        }
    },
    80: {
        'PSO': {
            'population_size': 120,
            'n_gen': 35,
            'w': 0.7333,
            'c1': 1.6527,
            'c2': 2.0948,
            'hops_weight': 0.55,
            'wavelength_weight': 0.45,
            'k': 150
        },
        'DE': {
            'population_size': 120,
            'n_gen': 35,
            'CR': 0.3402,
            'F': 0.5436,
            'hops_weight': 0.55,
            'wavelength_weight': 0.45,
            'k': 150
        },
        'GA': {
            'population_size': 120,
            'num_generations_ag': 35,
            'crossover_rate': 0.6451,
            'mutation_rate': 0.1997,
            'tournament_size': 3,
            'hops_weight': 0.55,
            'wavelength_weight': 0.45,
            'k': 150
        }
    }
}


# ============================================
# OD PAIRS PER NETWORK
# ============================================

JANET_PAIRS = [(0, 6),
               (2, 5),
               (0, 3),
               (1, 4),
               (2, 6)]

REDCLARA_PAIRS = [(0, 12),
                  (2, 6),
                  (5, 10),
                  (4, 11),
                  (3, 8)]

IPE_PAIRS = [(0, 12),
             (2, 6),
             (5, 10),
             (4, 11),
             (3, 8)]


# ============================================
# MAIN FUNCTION
# ============================================
def run_single_experiment(network_name: str, graph_func, pairs, lambdas: int, max_load: int, num_executions: int = 10):
    """
    Run experiment for a specific network.
    Runs PSO, DE, and GA in sequence.
    """
    graph = graph_func()
    
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {network_name} | {lambdas} wavelengths | Loads 1 to {max_load}")
    print(f"{'='*80}")
    
    for algo in ['PSO', 'DE', 'GA']:
        print(f"\n{'#'*80}")
        print(f"# STARTING {algo} IN {network_name} WITH {lambdas} WAVELENGTHS")
        print(f"{'#'*80}")
        
        # Select parameters by number of wavelengths
        params = OPTIMIZED_PARAMS[lambdas][algo].copy()
        
        simulator = WDMSimulatorStatistical(
            graph=graph,
            num_wavelengths=lambdas,
            gene_size=5,
            manual_pairs=pairs,
            k=params.pop('k', 150),
            population_size=params.pop('population_size', 120),
            n_gen=params.pop('n_gen', 40),
            hops_weight=params.pop('hops_weight', 0.55),
            wavelength_weight=params.pop('wavelength_weight', 0.45),
            **params
        )
        
        simulator.run_high_resolution_experiment(
            algorithm=algo,
            max_load=max_load,
            num_executions=num_executions,
            save_results=True
        )
        
        print(f"\n✓ {algo} completed for {network_name} ({lambdas} wavelengths)")


# ============================================
# FUNCTION TO CONSOLIDATE EXECUTION TIMES
# ============================================

def consolidate_execution_times(base_dir="."):
    """
    Consolidate all execution time files found.
    """
    import glob
    
    all_times = []
    
    # Search in all results_*_highres folders
    pattern = "results_*_highres/execution_times.csv"
    time_files = glob.glob(pattern)
    
    for file in time_files:
        try:
            df = pd.read_csv(file)
            all_times.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_times:
        print("No execution time file found!")
        return None
    
    # Concatenate all
    df_times = pd.concat(all_times, ignore_index=True)
    
    # Remove duplicates (keep the most recent per algorithm/network/wavelengths)
    df_times = df_times.sort_values('timestamp').drop_duplicates(
        subset=['algorithm', 'network', 'num_wavelengths', 'max_load'], 
        keep='last'
    )
    
    # Sort
    df_times = df_times.sort_values(['network', 'num_wavelengths', 'algorithm'])
    
    # Save consolidated
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"consolidated_execution_times_{timestamp}.csv"
    df_times.to_csv(output_file, index=False)
    
    print(f"\nExecution times consolidated into: {output_file}")
    print("\nTime summary (minutes):")
    print(df_times[['algorithm', 'network', 'num_wavelengths', 'total_time_minutes']].to_string(index=False))
    
    return df_times


def main():
    """Main function."""
    
    # General settings
    NUM_EXECUTIONS = 20  # Number of executions per algorithm
    
    # Experiments: (name, graph_function, pairs, wavelengths, max_load)
    experiments = [
        ('JANET6', get_janet6_graph, JANET_PAIRS, 40, 200),
        ('JANET6', get_janet6_graph, JANET_PAIRS, 80, 400),
        ('RedCLARA', get_redclara_graph, REDCLARA_PAIRS, 40, 200),
        ('RedCLARA', get_redclara_graph, REDCLARA_PAIRS, 80, 400),
        ('IPE', get_ipe_graph, IPE_PAIRS, 40, 200),
        ('IPE', get_ipe_graph, IPE_PAIRS, 80, 400),
    ]
    
    for net_name, graph_func, pairs, lambdas, max_load in experiments:
        run_single_experiment(
            network_name=net_name,
            graph_func=graph_func,
            pairs=pairs,
            lambdas=lambdas,
            max_load=max_load,
            num_executions=NUM_EXECUTIONS
        )
    
    print("\n" + "="*80)
    print("✅ ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    
    # Consolidate execution times
    print("\n" + "="*80)
    print("CONSOLIDATING EXECUTION TIMES...")
    print("="*80)
    consolidate_execution_times()
    
    print("\n" + "="*80)
    print("📁 GENERATED FILES:")
    print("="*80)
    print("  - *raw_*.csv : Raw data (global)")
    print("  - *stats_*.csv : Statistics (global)")
    print("  - *curve_*.png : Plots (global)")
    print("  - por_par/ : Data separated by OD pair")
    print("    - *raw_*.csv : Raw data per pair")
    print("    - *stats_*.csv : Statistics per pair")
    print("    - *curve_*.png : Plots per pair")
    print("  - *pairs_summary_*.csv : Comparative pair summary")
    print("  - execution_times.csv : Execution times")
    print("="*80)


if __name__ == "__main__":
    main()