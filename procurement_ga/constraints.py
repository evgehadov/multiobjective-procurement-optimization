
"""Constraint validation and budget repair for procurement solutions."""

from __future__ import annotations

import random
from numbers import Integral

from procurement_ga.models import (
    ProcurementProblem,
    ProcurementSolution,
    validate_solution_for_problem,
)


def is_within_budget(
    solution: ProcurementSolution,
    problem: ProcurementProblem,
) -> bool:
    """Return whether the purchase cost does not exceed the budget."""
    validate_solution_for_problem(solution, problem)

    total_cost = sum(
        material.unit_price * quantity
        for material, quantity in zip(
            problem.materials, solution.quantities
        )
    )
    return total_cost <= problem.budget + 1e-9


def is_feasible(
    solution: ProcurementSolution,
    problem: ProcurementProblem,
) -> bool:
    """Check genotype validity, purchase bounds, and budget feasibility."""
    try:
        validate_solution_for_problem(solution, problem)
    except (TypeError, ValueError):
        return False

    return is_within_budget(solution, problem)


def validate_genotype(
    quantities: tuple[int, ...] | list[int],
    problem: ProcurementProblem,
) -> ProcurementSolution:
    """Validate a raw genotype and return it as a solution object."""
    solution = ProcurementSolution(tuple(quantities))
    validate_solution_for_problem(solution, problem)
    return solution


def repair_budget(
    solution: ProcurementSolution,
    problem: ProcurementProblem,
    rng: random.Random,
) -> ProcurementSolution:
    """Reduce positive genes randomly until the budget constraint is met.

    The input solution is never mutated. If it already satisfies the budget,
    the same solution object is returned unchanged.
    """
    validate_solution_for_problem(solution, problem)

    if is_within_budget(solution, problem):
        return solution

    quantities = list(solution.quantities)
    total_cost = sum(
        material.unit_price * quantity
        for material, quantity in zip(problem.materials, quantities)
    )

    positive_indices = [
        index for index, quantity in enumerate(quantities) if quantity > 0
    ]

    while total_cost > problem.budget + 1e-9:
        if not positive_indices:
            raise RuntimeError(
                "Budget repair failed although the zero plan must be feasible."
            )

        index = rng.choice(positive_indices)
        material = problem.materials[index]

        # Reduce by one unit to preserve the requested random decrement
        # behavior and ensure the repair process always makes progress.
        quantities[index] -= 1
        total_cost -= material.unit_price

        if quantities[index] == 0:
            positive_indices.remove(index)

    repaired = ProcurementSolution(tuple(quantities))
    validate_solution_for_problem(repaired, problem)

    if not is_within_budget(repaired, problem):
        raise RuntimeError("Budget repair produced an infeasible solution.")

    return repaired


def validate_budget_value(budget: float) -> None:
    """Validate a budget value when it is supplied independently."""
    from math import isfinite

    if isinstance(budget, bool) or not isinstance(budget, (int, float, Integral)):
        raise TypeError("budget must be numeric.")
    if not isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and non-negative.")
