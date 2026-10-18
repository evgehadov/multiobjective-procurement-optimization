
"""Pareto dominance, nondominated sorting, and crowding distance."""

from __future__ import annotations

from math import inf, isfinite
from typing import Sequence, TypeVar

from procurement_ga.models import EvaluatedSolution, ObjectiveValues


T = TypeVar("T", bound=EvaluatedSolution)


def dominates(
    objectives_a: ObjectiveValues | tuple[float, ...],
    objectives_b: ObjectiveValues | tuple[float, ...],
) -> bool:
    """Return whether A Pareto-dominates B for minimization objectives."""
    values_a = _as_objective_tuple(objectives_a)
    values_b = _as_objective_tuple(objectives_b)

    if len(values_a) != len(values_b):
        raise ValueError("Objective vectors must have the same dimension.")

    no_worse = all(a <= b for a, b in zip(values_a, values_b))
    strictly_better = any(a < b for a, b in zip(values_a, values_b))
    return no_worse and strictly_better


def nondominated_sort(
    population: Sequence[T],
) -> list[list[T]]:
    """Split a population into Pareto fronts using pairwise comparisons."""
    size = len(population)
    if size == 0:
        return []

    objective_vectors = [
        _as_objective_tuple(item.objectives) for item in population
    ]

    domination_counts = [0] * size
    dominated_indices: list[list[int]] = [[] for _ in range(size)]
    first_front_indices: list[int] = []

    for i in range(size):
        for j in range(i + 1, size):
            if dominates(objective_vectors[i], objective_vectors[j]):
                dominated_indices[i].append(j)
                domination_counts[j] += 1
            elif dominates(objective_vectors[j], objective_vectors[i]):
                dominated_indices[j].append(i)
                domination_counts[i] += 1

    for index, count in enumerate(domination_counts):
        if count == 0:
            first_front_indices.append(index)

    fronts: list[list[int]] = []
    current_front = first_front_indices

    while current_front:
        fronts.append(current_front)
        next_front: list[int] = []

        for index in current_front:
            for dominated_index in dominated_indices[index]:
                domination_counts[dominated_index] -= 1
                if domination_counts[dominated_index] == 0:
                    next_front.append(dominated_index)

        current_front = next_front

    return [[population[index] for index in front] for front in fronts]


def first_front(population: Sequence[T]) -> list[T]:
    """Return the first nondominated front of a population."""
    fronts = nondominated_sort(population)
    return fronts[0] if fronts else []


def crowding_distance(
    front: Sequence[T],
) -> dict[int, float]:
    """Compute crowding distances keyed by each individual's input index.

    Boundary solutions receive infinite distance. An objective with zero
    range contributes nothing to the normalized distance.
    """
    size = len(front)
    distances = {index: 0.0 for index in range(size)}

    if size <= 2:
        return {index: inf for index in range(size)}

    vectors = [
        _as_objective_tuple(item.objectives) for item in front
    ]
    objective_count = len(vectors[0])

    if any(len(vector) != objective_count for vector in vectors):
        raise ValueError("All objective vectors must have the same dimension.")

    for objective_index in range(objective_count):
        sorted_indices = sorted(
            range(size),
            key=lambda index: vectors[index][objective_index],
        )

        distances[sorted_indices[0]] = inf
        distances[sorted_indices[-1]] = inf

        minimum = vectors[sorted_indices[0]][objective_index]
        maximum = vectors[sorted_indices[-1]][objective_index]
        objective_range = maximum - minimum

        if objective_range == 0:
            continue

        for position in range(1, size - 1):
            index = sorted_indices[position]

            if distances[index] == inf:
                continue

            previous_value = vectors[sorted_indices[position - 1]][
                objective_index
            ]
            next_value = vectors[sorted_indices[position + 1]][
                objective_index
            ]

            distances[index] += (
                next_value - previous_value
            ) / objective_range

    return distances


def _as_objective_tuple(
    objectives: ObjectiveValues | tuple[float, ...],
) -> tuple[float, ...]:
    """Normalize supported objective representations to a numeric tuple."""
    if isinstance(objectives, ObjectiveValues):
        values = objectives.as_tuple
    else:
        values = tuple(objectives)

    if not values:
        raise ValueError("An objective vector must not be empty.")

    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        for value in values
    ):
        raise ValueError("Objective values must be finite numbers.")

    return tuple(float(value) for value in values)
