
"""Objective functions for multi-objective procurement optimization."""

from __future__ import annotations

from procurement_ga.models import (
    ObjectiveValues,
    ProcurementProblem,
    ProcurementSolution,
    validate_solution_for_problem,
)


def required_purchase(problem: ProcurementProblem) -> tuple[int, ...]:
    """Return purchases required to cover demand and target stock."""
    return problem.required_purchase


def calculate_deficits(
    solution: ProcurementSolution,
    problem: ProcurementProblem,
) -> tuple[int, ...]:
    """Calculate shortages relative to required purchase quantities."""
    validate_solution_for_problem(solution, problem)

    required = problem.required_purchase
    return tuple(
        max(0, required_quantity - purchased_quantity)
        for required_quantity, purchased_quantity in zip(
            required, solution.quantities
        )
    )


def purchase_cost(
    solution: ProcurementSolution,
    problem: ProcurementProblem,
) -> float:
    """Calculate the total cost of purchased materials."""
    validate_solution_for_problem(solution, problem)

    return sum(
        material.unit_price * quantity
        for material, quantity in zip(
            problem.materials, solution.quantities
        )
    )


def weighted_deficit(
    solution: ProcurementSolution,
    problem: ProcurementProblem,
) -> float:
    """Calculate the weighted shortage relative to target purchases."""
    deficits = calculate_deficits(solution, problem)

    return sum(
        material.deficit_weight * deficit
        for material, deficit in zip(problem.materials, deficits)
    )


def holding_cost(
    solution: ProcurementSolution,
    problem: ProcurementProblem,
) -> float:
    """Calculate holding costs for stock remaining after monthly demand."""
    validate_solution_for_problem(solution, problem)

    return sum(
        material.holding_cost
        * max(
            0,
            material.initial_stock
            + quantity
            - material.monthly_demand,
        )
        for material, quantity in zip(
            problem.materials, solution.quantities
        )
    )


def evaluate_solution(
    solution: ProcurementSolution,
    problem: ProcurementProblem,
) -> ObjectiveValues:
    """Evaluate a plan using all three minimization objectives."""
    return ObjectiveValues(
        purchase_cost=purchase_cost(solution, problem),
        weighted_deficit=weighted_deficit(solution, problem),
        holding_cost=holding_cost(solution, problem),
    )


def objective_vector(
    solution: ProcurementSolution,
    problem: ProcurementProblem,
) -> tuple[float, float, float]:
    """Return objectives in the canonical order: cost, deficit, storage."""
    return evaluate_solution(solution, problem).as_tuple
