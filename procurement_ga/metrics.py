
"""Quality metrics for multi-objective optimization results."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import mean, median, pstdev
from typing import Sequence

from config import DEFAULT_CONFIG, HypervolumeConfig
from procurement_ga.models import EvaluatedSolution
from procurement_ga.pareto import first_front, nondominated_sort


@dataclass(frozen=True, slots=True)
class NormalizationBounds:
    """Fixed lower and upper bounds for the three objective dimensions."""

    minima: tuple[float, float, float]
    maxima: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.minima) != 3 or len(self.maxima) != 3:
            raise ValueError("Exactly three objective bounds are required.")

        for minimum, maximum in zip(self.minima, self.maxima):
            if not isfinite(minimum) or not isfinite(maximum):
                raise ValueError("Normalization bounds must be finite.")
            if maximum < minimum:
                raise ValueError("An upper bound cannot be below its lower bound.")


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Quality metrics for one Pareto archive."""

    nondominated_count: int
    hypervolume: float
    mean_objectives: tuple[float, float, float]
    best_objectives: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Aggregate statistics for a series of independent runs."""

    runs: int
    mean_hypervolume: float
    median_hypervolume: float
    std_hypervolume: float
    mean_nondominated_count: float
    mean_objectives: tuple[float, float, float]


def derive_normalization_bounds(
    solutions: Sequence[EvaluatedSolution],
) -> NormalizationBounds:
    """Derive objective bounds from a reference collection of solutions."""
    if not solutions:
        raise ValueError("Cannot derive bounds from an empty collection.")

    vectors = [item.objectives.as_tuple for item in solutions]

    minima = tuple(min(vector[i] for vector in vectors) for i in range(3))
    maxima = tuple(max(vector[i] for vector in vectors) for i in range(3))

    return NormalizationBounds(minima=minima, maxima=maxima)


def normalize_objectives(
    objectives: tuple[float, float, float],
    bounds: NormalizationBounds,
) -> tuple[float, float, float]:
    """Normalize objectives to [0, 1] using fixed shared bounds."""
    if len(objectives) != 3:
        raise ValueError("Exactly three objective values are required.")

    normalized: list[float] = []

    for index, value in enumerate(objectives):
        if not isfinite(value):
            raise ValueError("Objective values must be finite.")

        minimum = bounds.minima[index]
        maximum = bounds.maxima[index]

        if maximum == minimum:
            normalized.append(0.0)
        else:
            normalized.append((value - minimum) / (maximum - minimum))

    return tuple(normalized)


def hypervolume_3d(
    solutions: Sequence[EvaluatedSolution],
    bounds: NormalizationBounds,
    config: HypervolumeConfig | None = None,
) -> float:
    """Compute exact 3D dominated hypervolume for minimization objectives.

    The input is normalized using shared bounds. The configured reference
    point must be no better than any included point in every dimension.
    """
    settings = config if config is not None else DEFAULT_CONFIG.hypervolume
    reference = settings.reference_point

    if len(reference) != 3:
        raise ValueError("The hypervolume reference point must be 3D.")

    if any(not isfinite(value) for value in reference):
        raise ValueError("Reference point coordinates must be finite.")

    if not solutions:
        return 0.0

    points = [
        normalize_objectives(item.objectives.as_tuple, bounds)
        for item in solutions
    ]

    # Remove duplicate points and points outside the reference box.
    unique_points = sorted(set(points))
    valid_points = [
        point
        for point in unique_points
        if all(point[i] < reference[i] for i in range(3))
    ]

    if not valid_points:
        return 0.0

    # The union of axis-aligned boxes is computed by slicing along the
    # first objective dimension. Within each slice, a 2D union area is used.
    x_coordinates = sorted(
        {point[0] for point in valid_points} | {reference[0]}
    )
    volume = 0.0

    for index in range(len(x_coordinates) - 1):
        x_left = x_coordinates[index]
        x_right = x_coordinates[index + 1]

        if x_right <= x_left:
            continue

        active_points = [
            (point[1], point[2])
            for point in valid_points
            if point[0] <= x_left
        ]

        area = _hypervolume_2d(active_points, reference[1], reference[2])
        volume += (x_right - x_left) * area

    return max(0.0, volume)


def _hypervolume_2d(
    points: Sequence[tuple[float, float]],
    reference_y: float,
    reference_z: float,
) -> float:
    """Compute the exact union area of 2D minimization rectangles."""
    valid_points = sorted(
        {
            (y, z)
            for y, z in points
            if y < reference_y and z < reference_z
        }
    )

    if not valid_points:
        return 0.0

    y_coordinates = sorted(
        {point[0] for point in valid_points} | {reference_y}
    )
    area = 0.0

    for index in range(len(y_coordinates) - 1):
        y_left = y_coordinates[index]
        y_right = y_coordinates[index + 1]

        if y_right <= y_left:
            continue

        active_z = [
            z
            for y, z in valid_points
            if y <= y_left
        ]

        if active_z:
            area += (y_right - y_left) * (
                reference_z - min(active_z)
            )

    return max(0.0, area)


def calculate_run_metrics(
    solutions: Sequence[EvaluatedSolution],
    bounds: NormalizationBounds,
    config: HypervolumeConfig | None = None,
) -> RunMetrics:
    """Calculate quality metrics for the nondominated front of one run."""
    front = first_front(solutions)

    if not front:
        return RunMetrics(
            nondominated_count=0,
            hypervolume=0.0,
            mean_objectives=(0.0, 0.0, 0.0),
            best_objectives=(0.0, 0.0, 0.0),
        )

    vectors = [item.objectives.as_tuple for item in front]

    mean_objectives = tuple(
        mean(vector[index] for vector in vectors)
        for index in range(3)
    )

    best_objectives = tuple(
        min(vector[index] for vector in vectors)
        for index in range(3)
    )

    return RunMetrics(
        nondominated_count=len(front),
        hypervolume=hypervolume_3d(front, bounds, config),
        mean_objectives=mean_objectives,
        best_objectives=best_objectives,
    )


def aggregate_run_metrics(
    metrics: Sequence[RunMetrics],
) -> AggregateMetrics:
    """Aggregate metrics across independent runs."""
    if not metrics:
        raise ValueError("At least one run metric is required.")

    hypervolumes = [item.hypervolume for item in metrics]
    counts = [item.nondominated_count for item in metrics]

    mean_objectives = tuple(
        mean(item.mean_objectives[index] for item in metrics)
        for index in range(3)
    )

    return AggregateMetrics(
        runs=len(metrics),
        mean_hypervolume=mean(hypervolumes),
        median_hypervolume=median(hypervolumes),
        std_hypervolume=pstdev(hypervolumes),
        mean_nondominated_count=mean(counts),
        mean_objectives=mean_objectives,
    )
