
"""Weighted-sum baseline for multi-objective procurement optimization."""

from __future__ import annotations

import random
from dataclasses import dataclass

from procurement_ga.constraints import is_feasible
from procurement_ga.models import (
    EvaluatedSolution,
    ProcurementProblem,
    ProcurementSolution,
)
from procurement_ga.objective import evaluate_solution
from procurement_ga.operators import random_solution


@dataclass(frozen=True, slots=True)
class WeightedSumResult:
    """Result of one weighted-sum optimization run."""

    weights: tuple[float, float, float]
    best: EvaluatedSolution
    evaluations: int
    seed: int
    normalized_score: float


def normalize_objectives(
    objective_vectors: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Min-max normalize each objective across a collection of vectors."""
    if not objective_vectors:
        return []

    if any(len(vector) != 3 for vector in objective_vectors):
        raise ValueError("Each objective vector must contain three values.")

    columns = list(zip(*objective_vectors))
    minima = [min(column) for column in columns]
    maxima = [max(column) for column in columns]

    normalized: list[tuple[float, float, float]] = []

    for vector in objective_vectors:
        values = tuple(
            0.0 if maxima[index] == minima[index]
            else (vector[index] - minima[index])
            / (maxima[index] - minima[index])
            for index in range(3)
        )
        normalized.append(values)

    return normalized


def weighted_score(
    normalized_objectives: tuple[float, float, float],
    weights: tuple[float, float, float],
) -> float:
    """Calculate the scalar objective for normalized minimization criteria."""
    _validate_weights(weights)

    return sum(
        value * weight
        for value, weight in zip(normalized_objectives, weights)
    )


def optimize_weighted_sum(
    problem: ProcurementProblem,
    weights: tuple[float, float, float],
    seed: int = 20261018,
    iterations: int = 2000,
    candidates_per_iteration: int = 2,
) -> WeightedSumResult:
    """Search feasible purchase plans using randomized local improvement.

    Each iteration proposes one or more integer-neighbor plans. Candidate
    plans are evaluated together with the current incumbent, and the best
    normalized weighted score is retained.
    """
    _validate_weights(weights)

    if iterations < 1:
        raise ValueError("iterations must be at least 1.")
    if candidates_per_iteration < 1:
        raise ValueError("candidates_per_iteration must be at least 1.")

    rng = random.Random(seed)
    current = random_solution(problem, rng)
    evaluations = 1

    # Establish fixed normalization scales from the all-zero and maximum
    # purchase plans, then use them throughout this run.
    zero_solution = ProcurementSolution(
        tuple(0 for _ in problem.materials)
    )
    maximum_solution = ProcurementSolution(
        tuple(material.max_purchase for material in problem.materials)
    )

    reference_vectors = [
        evaluate_solution(zero_solution, problem).as_tuple,
        evaluate_solution(maximum_solution, problem).as_tuple,
        evaluate_solution(current, problem).as_tuple,
    ]
    normalized_references = normalize_objectives(reference_vectors)

    minima = tuple(
        min(vector[index] for vector in reference_vectors)
        for index in range(3)
    )
    maxima = tuple(
        max(vector[index] for vector in reference_vectors)
        for index in range(3)
    )

    def score(solution: ProcurementSolution) -> tuple[float, EvaluatedSolution]:
        evaluated = EvaluatedSolution(
            solution=solution,
            objectives=evaluate_solution(solution, problem),
        )
        vector = evaluated.objectives.as_tuple
        normalized = tuple(
            0.0 if maxima[index] == minima[index]
            else (vector[index] - minima[index])
            / (maxima[index] - minima[index])
            for index in range(3)
        )
        return weighted_score(normalized, weights), evaluated

    current_score, current_evaluated = score(current)

    for _ in range(iterations):
        candidates: list[ProcurementSolution] = []

        for _ in range(candidates_per_iteration):
            quantities = list(current.quantities)
            index = rng.randrange(problem.n_materials)
            material = problem.materials[index]

            if material.max_purchase > 0:
                step = rng.choice((-1, 1))
                quantities[index] = max(
                    0,
                    min(material.max_purchase, quantities[index] + step),
                )

            candidates.append(ProcurementSolution(tuple(quantities)))

        for candidate in candidates:
            # Skip candidates that violate the budget rather than comparing
            # infeasible plans against feasible plans.
            if not is_feasible(candidate, problem):
                continue

            candidate_score, candidate_evaluated = score(candidate)
            evaluations += 1

            if candidate_score < current_score:
                current = candidate
                current_score = candidate_score
                current_evaluated = candidate_evaluated

    return WeightedSumResult(
        weights=weights,
        best=current_evaluated,
        evaluations=evaluations,
        seed=seed,
        normalized_score=current_score,
    )


def _validate_weights(weights: tuple[float, float, float]) -> None:
    """Validate three non-negative weights with positive total weight."""
    if len(weights) != 3:
        raise ValueError("Exactly three objective weights are required.")

    if any(weight < 0 for weight in weights):
        raise ValueError("Objective weights must be non-negative.")

    if sum(weights) <= 0:
        raise ValueError("At least one objective weight must be positive.")
