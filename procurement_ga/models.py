
"""Validated data models for multi-objective procurement optimization."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Sequence


@dataclass(frozen=True, slots=True)
class Material:
    """One type of consumable material."""

    id: str
    name: str
    unit_price: float
    initial_stock: int
    monthly_demand: int
    minimum_stock: int
    deficit_weight: float
    holding_cost: float
    max_purchase: int

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Material id must be a non-empty string.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Material name must be a non-empty string.")

        if not isfinite(self.unit_price) or self.unit_price <= 0:
            raise ValueError("unit_price must be finite and strictly positive.")

        for field_name in (
            "initial_stock",
            "monthly_demand",
            "minimum_stock",
            "max_purchase",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")

        for field_name in ("deficit_weight", "holding_cost"):
            value = getattr(self, field_name)
            if not isfinite(value) or value < 0:
                raise ValueError(
                    f"{field_name} must be finite and non-negative."
                )


@dataclass(frozen=True, slots=True)
class ProcurementProblem:
    """A procurement instance with materials and a total budget."""

    materials: tuple[Material, ...]
    budget: float

    def __post_init__(self) -> None:
        normalized_materials = tuple(self.materials)
        object.__setattr__(self, "materials", normalized_materials)

        if not normalized_materials:
            raise ValueError("A procurement problem must contain materials.")

        if not isfinite(self.budget) or self.budget < 0:
            raise ValueError("budget must be finite and non-negative.")

        identifiers = [material.id for material in normalized_materials]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Material identifiers must be unique.")

        if not all(isinstance(material, Material) for material in normalized_materials):
            raise TypeError("All entries in materials must be Material objects.")

    @property
    def n_materials(self) -> int:
        """Return the number of material types."""
        return len(self.materials)

    @property
    def required_purchase(self) -> tuple[int, ...]:
        """Minimum purchases needed to cover demand and target stock."""
        return tuple(
            max(
                0,
                material.monthly_demand
                + material.minimum_stock
                - material.initial_stock,
            )
            for material in self.materials
        )


@dataclass(frozen=True, slots=True)
class ProcurementSolution:
    """An integer purchase plan."""

    quantities: tuple[int, ...]

    def __post_init__(self) -> None:
        normalized_quantities = tuple(self.quantities)
        object.__setattr__(self, "quantities", normalized_quantities)

        for index, quantity in enumerate(normalized_quantities):
            if isinstance(quantity, bool) or not isinstance(quantity, Integral):
                raise TypeError(f"Gene {index} must be an integer.")
            if quantity < 0:
                raise ValueError(f"Gene {index} must be non-negative.")

    @property
    def total_units(self) -> int:
        """Return the total number of units purchased."""
        return sum(self.quantities)


@dataclass(frozen=True, slots=True)
class ObjectiveValues:
    """Values of the three minimization objectives."""

    purchase_cost: float
    weighted_deficit: float
    holding_cost: float

    def __post_init__(self) -> None:
        for field_name in (
            "purchase_cost",
            "weighted_deficit",
            "holding_cost",
        ):
            value = getattr(self, field_name)
            if not isfinite(value) or value < 0:
                raise ValueError(
                    f"{field_name} must be finite and non-negative."
                )

    @property
    def as_tuple(self) -> tuple[float, float, float]:
        """Return objectives in the project's canonical order."""
        return (
            self.purchase_cost,
            self.weighted_deficit,
            self.holding_cost,
        )


@dataclass(frozen=True, slots=True)
class EvaluatedSolution:
    """A solution together with its objective values."""

    solution: ProcurementSolution
    objectives: ObjectiveValues


@dataclass(frozen=True, slots=True)
class RunResult:
    """Summary of one evolutionary algorithm run."""

    seed: int
    generations: int
    population_size: int
    elapsed_seconds: float
    archive: tuple[EvaluatedSolution, ...]
    hypervolume: float
    hypervolume_history: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("seed", "generations", "population_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")

        if self.population_size == 0:
            raise ValueError("population_size must be positive.")

        for field_name in ("elapsed_seconds", "hypervolume"):
            value = getattr(self, field_name)
            if not isfinite(value) or value < 0:
                raise ValueError(
                    f"{field_name} must be finite and non-negative."
                )

        if not all(
            isinstance(item, EvaluatedSolution) for item in self.archive
        ):
            raise TypeError("archive must contain EvaluatedSolution objects.")

        if any(
            not isfinite(value) or value < 0
            for value in self.hypervolume_history
        ):
            raise ValueError(
                "Hypervolume history must contain finite non-negative values."
            )


def validate_solution_for_problem(
    solution: ProcurementSolution,
    problem: ProcurementProblem,
) -> None:
    """Validate solution dimension and per-material purchase limits.

    The budget is deliberately checked separately by the constraints module.
    """
    if len(solution.quantities) != problem.n_materials:
        raise ValueError(
            "Solution dimension does not match the number of materials."
        )

    for index, (quantity, material) in enumerate(
        zip(solution.quantities, problem.materials)
    ):
        if quantity > material.max_purchase:
            raise ValueError(
                f"Gene {index} exceeds max_purchase "
                f"({quantity} > {material.max_purchase})."
            )
