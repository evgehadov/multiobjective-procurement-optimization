
"""Genetic operators for integer-coded procurement plans."""

from __future__ import annotations

import random

from procurement_ga.constraints import repair_budget
from procurement_ga.models import (
    ProcurementProblem,
    ProcurementSolution,
    validate_solution_for_problem,
)


def random_solution(
    problem: ProcurementProblem,
    rng: random.Random,
) -> ProcurementSolution:
    """Generate a random integer solution and repair its budget if needed."""
    quantities = tuple(
        rng.randint(0, material.max_purchase)
        for material in problem.materials
    )
    candidate = ProcurementSolution(quantities)
    return repair_budget(candidate, problem, rng)


def crossover(
    parent_a: ProcurementSolution,
    parent_b: ProcurementSolution,
    problem: ProcurementProblem,
    rng: random.Random,
    probability: float = 0.9,
) -> tuple[ProcurementSolution, ProcurementSolution]:
    """Apply uniform crossover and repair both offspring."""
    validate_solution_for_problem(parent_a, problem)
    validate_solution_for_problem(parent_b, problem)
    _validate_probability(probability, "probability")

    if rng.random() >= probability:
        return (
            repair_budget(parent_a, problem, rng),
            repair_budget(parent_b, problem, rng),
        )

    child_a: list[int] = []
    child_b: list[int] = []

    for gene_a, gene_b in zip(
        parent_a.quantities,
        parent_b.quantities,
    ):
        if rng.random() < 0.5:
            child_a.append(gene_a)
            child_b.append(gene_b)
        else:
            child_a.append(gene_b)
            child_b.append(gene_a)

    offspring_a = ProcurementSolution(tuple(child_a))
    offspring_b = ProcurementSolution(tuple(child_b))

    return (
        repair_budget(offspring_a, problem, rng),
        repair_budget(offspring_b, problem, rng),
    )


def mutate(
    solution: ProcurementSolution,
    problem: ProcurementProblem,
    rng: random.Random,
    probability: float = 0.1,
    gene_probability: float = 0.1,
) -> ProcurementSolution:
    """Mutate selected genes by reassigning them within valid integer bounds.

    The per-gene probability is applied independently. A selected gene is
    assigned a different value when its bounds permit one.
    """
    validate_solution_for_problem(solution, problem)
    _validate_probability(probability, "probability")
    _validate_probability(gene_probability, "gene_probability")

    if rng.random() >= probability:
        return repair_budget(solution, problem, rng)

    quantities = list(solution.quantities)

    for index, material in enumerate(problem.materials):
        if material.max_purchase == 0:
            continue

        if rng.random() < gene_probability:
            old_value = quantities[index]
            new_value = rng.randint(0, material.max_purchase)

            if new_value == old_value:
                if old_value < material.max_purchase:
                    new_value = old_value + 1
                else:
                    new_value = old_value - 1

            quantities[index] = new_value

    mutated = ProcurementSolution(tuple(quantities))
    return repair_budget(mutated, problem, rng)


def _validate_probability(value: float, name: str) -> None:
    """Ensure a probability belongs to the closed interval [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in the interval [0, 1].")
