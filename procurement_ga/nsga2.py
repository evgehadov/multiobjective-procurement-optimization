
"""A self-contained implementation of the NSGA-II algorithm."""

from __future__ import annotations

import random
import time
from math import inf
from typing import Callable

from config import DEFAULT_CONFIG, NSGA2Config
from procurement_ga.models import (
    EvaluatedSolution,
    ProcurementProblem,
    ProcurementSolution,
    RunResult,
)
from procurement_ga.objective import evaluate_solution
from procurement_ga.operators import crossover, mutate, random_solution
from procurement_ga.pareto import crowding_distance, first_front, nondominated_sort


HypervolumeFunction = Callable[[tuple[EvaluatedSolution, ...]], float]


def run_nsga2(
    problem: ProcurementProblem,
    config: NSGA2Config | None = None,
    seed: int | None = None,
    hypervolume_function: HypervolumeFunction | None = None,
) -> RunResult:
    """Run NSGA-II and return its nondominated archive and run statistics."""
    settings = config if config is not None else DEFAULT_CONFIG.nsga2
    actual_seed = settings.seed if seed is None else seed

    _validate_config(settings, actual_seed)
    rng = random.Random(actual_seed)
    started_at = time.perf_counter()

    population = _initialize_population(
        problem=problem,
        population_size=settings.population_size,
        rng=rng,
    )

    hypervolume_history: list[float] = []
    if hypervolume_function is not None:
        initial_front = tuple(first_front(population))
        hypervolume_history.append(
            _validated_hypervolume(hypervolume_function(initial_front))
        )

    for _ in range(settings.generations):
        fronts = nondominated_sort(population)
        ranks, distances = _rank_and_crowding(fronts, population)

        offspring: list[EvaluatedSolution] = []

        while len(offspring) < settings.population_size:
            parent_a = _tournament_select(population, ranks, distances, rng)
            parent_b = _tournament_select(population, ranks, distances, rng)

            child_a, child_b = crossover(
                parent_a.solution,
                parent_b.solution,
                problem,
                rng,
                probability=settings.crossover_probability,
            )

            for child in (child_a, child_b):
                mutated = mutate(
                    child,
                    problem,
                    rng,
                    probability=settings.mutation_probability,
                    gene_probability=settings.mutation_gene_probability,
                )
                offspring.append(
                    EvaluatedSolution(
                        solution=mutated,
                        objectives=evaluate_solution(mutated, problem),
                    )
                )

                if len(offspring) >= settings.population_size:
                    break

        combined = population + offspring
        combined_fronts = nondominated_sort(combined)
        population = _environmental_selection(
            combined_fronts,
            settings.population_size,
        )

        if hypervolume_function is not None:
            current_front = tuple(first_front(population))
            hypervolume_history.append(
                _validated_hypervolume(
                    hypervolume_function(current_front)
                )
            )

    archive = tuple(first_front(population))
    elapsed_seconds = time.perf_counter() - started_at

    hypervolume = (
        hypervolume_history[-1] if hypervolume_history else 0.0
    )

    return RunResult(
        seed=actual_seed,
        generations=settings.generations,
        population_size=settings.population_size,
        elapsed_seconds=elapsed_seconds,
        archive=archive,
        hypervolume=hypervolume,
        hypervolume_history=tuple(hypervolume_history),
    )


def _initialize_population(
    problem: ProcurementProblem,
    population_size: int,
    rng: random.Random,
) -> list[EvaluatedSolution]:
    """Create the initial population of evaluated feasible solutions."""
    population: list[EvaluatedSolution] = []

    for _ in range(population_size):
        solution = random_solution(problem, rng)
        population.append(
            EvaluatedSolution(
                solution=solution,
                objectives=evaluate_solution(solution, problem),
            )
        )

    return population


def _rank_and_crowding(
    fronts: list[list[EvaluatedSolution]],
    population: list[EvaluatedSolution],
) -> tuple[dict[int, int], dict[int, float]]:
    """Map population object identities to rank and crowding distance."""
    ranks: dict[int, int] = {}
    distances: dict[int, float] = {}

    for rank, front in enumerate(fronts):
        front_distances = crowding_distance(front)

        for local_index, individual in enumerate(front):
            identity = id(individual)
            ranks[identity] = rank
            distances[identity] = front_distances[local_index]

    # Every population member must be present in the computed fronts.
    if any(id(individual) not in ranks for individual in population):
        raise RuntimeError("Nondominated sorting omitted a population member.")

    return ranks, distances


def _tournament_select(
    population: list[EvaluatedSolution],
    ranks: dict[int, int],
    distances: dict[int, float],
    rng: random.Random,
) -> EvaluatedSolution:
    """Select one individual by rank first and crowding distance second."""
    first, second = rng.sample(population, 2) if len(population) > 1 else (
        population[0],
        population[0],
    )

    first_rank = ranks[id(first)]
    second_rank = ranks[id(second)]

    if first_rank < second_rank:
        return first
    if second_rank < first_rank:
        return second

    first_distance = distances[id(first)]
    second_distance = distances[id(second)]

    if first_distance > second_distance:
        return first
    if second_distance > first_distance:
        return second

    return rng.choice((first, second))


def _environmental_selection(
    fronts: list[list[EvaluatedSolution]],
    population_size: int,
) -> list[EvaluatedSolution]:
    """Select the next generation by fronts and crowding distance."""
    selected: list[EvaluatedSolution] = []

    for front in fronts:
        remaining = population_size - len(selected)

        if remaining <= 0:
            break

        if len(front) <= remaining:
            selected.extend(front)
            continue

        distances = crowding_distance(front)
        ordered_indices = sorted(
            range(len(front)),
            key=lambda index: distances[index],
            reverse=True,
        )
        selected.extend(front[index] for index in ordered_indices[:remaining])
        break

    if len(selected) != population_size:
        raise RuntimeError(
            "Environmental selection returned an unexpected population size."
        )

    return selected


def _validated_hypervolume(value: float) -> float:
    """Ensure a hypervolume callback returns a valid non-negative number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Hypervolume function must return a numeric value.")
    if value < 0 or value == inf or value != value:
        raise ValueError("Hypervolume must be finite and non-negative.")
    return float(value)


def _validate_config(config: NSGA2Config, seed: int) -> None:
    """Validate the algorithm parameters before starting a run."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    if config.population_size < 2:
        raise ValueError("population_size must be at least 2.")
    if config.generations < 0:
        raise ValueError("generations must be non-negative.")

    for name, value in (
        ("crossover_probability", config.crossover_probability),
        ("mutation_probability", config.mutation_probability),
        ("mutation_gene_probability", config.mutation_gene_probability),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in the interval [0, 1].")
