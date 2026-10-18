
"""Reproducible synthetic data generation for procurement optimization."""

from __future__ import annotations

import random

from config import DEFAULT_CONFIG, DataConfig
from procurement_ga.models import Material, ProcurementProblem


def generate_problem(
    seed: int | None = None,
    config: DataConfig | None = None,
) -> ProcurementProblem:
    """Generate a synthetic procurement problem deterministically.

    Args:
        seed: Random seed. If omitted, the seed from the data configuration
            is used.
        config: Data generation parameters. Defaults to DEFAULT_CONFIG.data.

    Returns:
        A procurement problem containing the generated materials and budget.
    """
    settings = config if config is not None else DEFAULT_CONFIG.data
    actual_seed = settings.seed if seed is None else seed

    if isinstance(actual_seed, bool) or not isinstance(actual_seed, int):
        raise TypeError("seed must be an integer.")
    if settings.n_materials < 30:
        raise ValueError("n_materials must be at least 30.")
    if not 0.0 < settings.budget_fraction <= 1.0:
        raise ValueError("budget_fraction must be in the interval (0, 1].")

    if settings.min_price <= 0 or settings.max_price < settings.min_price:
        raise ValueError("Invalid price range.")
    if settings.min_initial_stock < 0:
        raise ValueError("min_initial_stock must be non-negative.")
    if settings.max_initial_stock < settings.min_initial_stock:
        raise ValueError("Invalid initial stock range.")
    if settings.min_monthly_demand < 0:
        raise ValueError("min_monthly_demand must be non-negative.")
    if settings.max_monthly_demand < settings.min_monthly_demand:
        raise ValueError("Invalid monthly demand range.")
    if settings.min_required_stock < 0:
        raise ValueError("min_required_stock must be non-negative.")
    if settings.max_required_stock < settings.min_required_stock:
        raise ValueError("Invalid minimum stock range.")
    if settings.min_deficit_weight <= 0:
        raise ValueError("min_deficit_weight must be strictly positive.")
    if settings.max_deficit_weight < settings.min_deficit_weight:
        raise ValueError("Invalid deficit weight range.")
    if settings.min_holding_cost <= 0:
        raise ValueError("min_holding_cost must be strictly positive.")
    if settings.max_holding_cost < settings.min_holding_cost:
        raise ValueError("Invalid holding cost range.")
    if settings.max_purchase_quantity < 0:
        raise ValueError("max_purchase_quantity must be non-negative.")

    rng = random.Random(actual_seed)
    materials: list[Material] = []

    for index in range(settings.n_materials):
        material = Material(
            id=f"M{index + 1:03d}",
            name=f"Consumable {index + 1:03d}",
            unit_price=float(
                rng.randint(settings.min_price, settings.max_price)
            ),
            initial_stock=rng.randint(
                settings.min_initial_stock,
                settings.max_initial_stock,
            ),
            monthly_demand=rng.randint(
                settings.min_monthly_demand,
                settings.max_monthly_demand,
            ),
            minimum_stock=rng.randint(
                settings.min_required_stock,
                settings.max_required_stock,
            ),
            deficit_weight=round(
                rng.uniform(
                    settings.min_deficit_weight,
                    settings.max_deficit_weight,
                ),
                3,
            ),
            holding_cost=round(
                rng.uniform(
                    settings.min_holding_cost,
                    settings.max_holding_cost,
                ),
                3,
            ),
            max_purchase=settings.max_purchase_quantity,
        )
        materials.append(material)

    required_purchases = [
        max(
            0,
            material.monthly_demand
            + material.minimum_stock
            - material.initial_stock,
        )
        for material in materials
    ]

    full_required_cost = sum(
        material.unit_price * quantity
        for material, quantity in zip(materials, required_purchases)
    )

    budget = round(full_required_cost * settings.budget_fraction, 2)

    return ProcurementProblem(
        materials=tuple(materials),
        budget=budget,
    )


def generate_materials(
    seed: int | None = None,
    config: DataConfig | None = None,
) -> tuple[Material, ...]:
    """Generate materials using the same reproducible configuration."""
    return generate_problem(seed=seed, config=config).materials
