
"""Reproducible experiment suites for NSGA-II and the weighted-sum baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from config import (
    DEFAULT_CONFIG,
    ExperimentConfig,
    NSGA2Config,
)
from procurement_ga.baseline import WeightedSumResult, optimize_weighted_sum
from procurement_ga.data_generator import generate_problem
from procurement_ga.metrics import (
    AggregateMetrics,
    NormalizationBounds,
    RunMetrics,
    aggregate_run_metrics,
    calculate_run_metrics,
    derive_normalization_bounds,
)
from procurement_ga.models import EvaluatedSolution, ProcurementProblem, RunResult
from procurement_ga.nsga2 import run_nsga2
from procurement_ga.pareto import first_front


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """One independently seeded NSGA-II run and its metrics."""

    experiment: str
    parameter_name: str
    parameter_value: float
    run_index: int
    result: RunResult
    metrics: RunMetrics


@dataclass(frozen=True, slots=True)
class ExperimentSuite:
    """Collection of experiment outputs and shared comparison metadata."""

    problem: ProcurementProblem
    normalization_bounds: NormalizationBounds
    runs: tuple[ExperimentRun, ...]
    aggregates: dict[str, AggregateMetrics]
    baseline_results: tuple[WeightedSumResult, ...] = ()


def run_experiment_a(
    problem: ProcurementProblem | None = None,
    experiment_config: ExperimentConfig | None = None,
    nsga2_config: NSGA2Config | None = None,
) -> ExperimentSuite:
    """Run at least 20 independent NSGA-II trials with distinct seeds."""
    exp_settings = (
        experiment_config
        if experiment_config is not None
        else DEFAULT_CONFIG.experiments
    )
    algorithm_settings = (
        nsga2_config if nsga2_config is not None else DEFAULT_CONFIG.nsga2
    )
    actual_problem = problem or generate_problem()

    if exp_settings.independent_runs < 20:
        raise ValueError("Experiment A requires at least 20 independent runs.")

    seed_values = [
        exp_settings.base_seed + index
        for index in range(exp_settings.independent_runs)
    ]

    results = [
        run_nsga2(
            actual_problem,
            algorithm_settings,
            seed=seed,
        )
        for seed in seed_values
    ]

    bounds = _shared_bounds_from_results(results)
    runs = tuple(
        ExperimentRun(
            experiment="A",
            parameter_name="run",
            parameter_value=float(index + 1),
            run_index=index + 1,
            result=result,
            metrics=calculate_run_metrics(
                result.archive,
                bounds,
            ),
        )
        for index, result in enumerate(results)
    )

    return ExperimentSuite(
        problem=actual_problem,
        normalization_bounds=bounds,
        runs=runs,
        aggregates={"A": aggregate_run_metrics([run.metrics for run in runs])},
    )


def run_experiment_b(
    problem: ProcurementProblem | None = None,
    experiment_config: ExperimentConfig | None = None,
    base_config: NSGA2Config | None = None,
    runs_per_size: int = 20,
) -> ExperimentSuite:
    """Compare at least three population sizes over independent runs."""
    exp_settings = (
        experiment_config
        if experiment_config is not None
        else DEFAULT_CONFIG.experiments
    )
    algorithm_settings = base_config or DEFAULT_CONFIG.nsga2
    actual_problem = problem or generate_problem()

    sizes = tuple(sorted(set(exp_settings.population_sizes)))
    if len(sizes) < 3:
        raise ValueError("Experiment B requires at least three population sizes.")
    if runs_per_size < 1:
        raise ValueError("runs_per_size must be at least 1.")

    raw: list[tuple[int, int, RunResult]] = []

    for size_index, population_size in enumerate(sizes):
        for run_index in range(runs_per_size):
            seed = (
                exp_settings.base_seed
                + size_index * 100_000
                + run_index
            )
            config = _replace_nsga2_config(
                algorithm_settings,
                population_size=population_size,
            )
            result = run_nsga2(actual_problem, config, seed=seed)
            raw.append((population_size, run_index + 1, result))

    bounds = _shared_bounds_from_results([item[2] for item in raw])
    runs = tuple(
        ExperimentRun(
            experiment="B",
            parameter_name="population_size",
            parameter_value=float(size),
            run_index=run_index,
            result=result,
            metrics=calculate_run_metrics(result.archive, bounds),
        )
        for size, run_index, result in raw
    )

    aggregates = _aggregate_by_parameter(runs)

    return ExperimentSuite(
        problem=actual_problem,
        normalization_bounds=bounds,
        runs=runs,
        aggregates=aggregates,
    )


def run_experiment_c(
    problem: ProcurementProblem | None = None,
    experiment_config: ExperimentConfig | None = None,
    base_config: NSGA2Config | None = None,
    runs_per_probability: int = 20,
) -> ExperimentSuite:
    """Compare at least three mutation probabilities over independent runs."""
    exp_settings = (
        experiment_config
        if experiment_config is not None
        else DEFAULT_CONFIG.experiments
    )
    algorithm_settings = base_config or DEFAULT_CONFIG.nsga2
    actual_problem = problem or generate_problem()

    probabilities = tuple(sorted(set(exp_settings.mutation_probabilities)))
    if len(probabilities) < 3:
        raise ValueError(
            "Experiment C requires at least three mutation probabilities."
        )
    if runs_per_probability < 1:
        raise ValueError("runs_per_probability must be at least 1.")

    raw: list[tuple[float, int, RunResult]] = []

    for probability_index, probability in enumerate(probabilities):
        for run_index in range(runs_per_probability):
            seed = (
                exp_settings.base_seed
                + probability_index * 100_000
                + run_index
            )
            config = _replace_nsga2_config(
                algorithm_settings,
                mutation_probability=probability,
            )
            result = run_nsga2(actual_problem, config, seed=seed)
            raw.append((probability, run_index + 1, result))

    bounds = _shared_bounds_from_results([item[2] for item in raw])
    runs = tuple(
        ExperimentRun(
            experiment="C",
            parameter_name="mutation_probability",
            parameter_value=probability,
            run_index=run_index,
            result=result,
            metrics=calculate_run_metrics(result.archive, bounds),
        )
        for probability, run_index, result in raw
    )

    aggregates = _aggregate_by_parameter(runs)

    return ExperimentSuite(
        problem=actual_problem,
        normalization_bounds=bounds,
        runs=runs,
        aggregates=aggregates,
    )


def run_experiment_d(
    problem: ProcurementProblem | None = None,
    experiment_config: ExperimentConfig | None = None,
    nsga2_config: NSGA2Config | None = None,
    baseline_iterations: int = 2000,
) -> ExperimentSuite:
    """Compare NSGA-II against weighted-sum optimization with shared bounds."""
    exp_settings = (
        experiment_config
        if experiment_config is not None
        else DEFAULT_CONFIG.experiments
    )
    algorithm_settings = nsga2_config or DEFAULT_CONFIG.nsga2
    actual_problem = problem or generate_problem()

    nsga_runs = [
        run_nsga2(
            actual_problem,
            algorithm_settings,
            seed=exp_settings.base_seed + index,
        )
        for index in range(exp_settings.independent_runs)
    ]

    baseline_results = tuple(
        optimize_weighted_sum(
            actual_problem,
            weights=weights,
            seed=exp_settings.base_seed + 500_000 + index,
            iterations=baseline_iterations,
        )
        for index, weights in enumerate(exp_settings.baseline_weights)
    )

    all_solutions: list[EvaluatedSolution] = []
    for result in nsga_runs:
        all_solutions.extend(result.archive)
    all_solutions.extend(result.best for result in baseline_results)

    bounds = derive_normalization_bounds(all_solutions)

    nsga_fronts = [tuple(first_front(result.archive)) for result in nsga_runs]
    nsga_run_metrics = [
        calculate_run_metrics(front, bounds) for front in nsga_fronts
    ]
    baseline_metrics = [
        calculate_run_metrics((result.best,), bounds)
        for result in baseline_results
    ]

    runs: list[ExperimentRun] = []

    for index, (result, metrics) in enumerate(
        zip(nsga_runs, nsga_run_metrics),
        start=1,
    ):
        runs.append(
            ExperimentRun(
                experiment="D",
                parameter_name="algorithm",
                parameter_value=0.0,
                run_index=index,
                result=result,
                metrics=metrics,
            )
        )

    for index, (result, metrics) in enumerate(
        zip(baseline_results, baseline_metrics),
        start=1,
    ):
        baseline_run = RunResult(
            seed=result.seed,
            generations=0,
            population_size=1,
            elapsed_seconds=0.0,
            archive=(result.best,),
            hypervolume=metrics.hypervolume,
        )
        runs.append(
            ExperimentRun(
                experiment="D",
                parameter_name="baseline_weight_configuration",
                parameter_value=float(index),
                run_index=index,
                result=baseline_run,
                metrics=metrics,
            )
        )

    aggregates = {
        "NSGA-II": aggregate_run_metrics(nsga_run_metrics),
        "Weighted sum": aggregate_run_metrics(baseline_metrics),
    }

    return ExperimentSuite(
        problem=actual_problem,
        normalization_bounds=bounds,
        runs=tuple(runs),
        aggregates=aggregates,
        baseline_results=baseline_results,
    )


def _shared_bounds_from_results(
    results: list[RunResult],
) -> NormalizationBounds:
    """Derive shared normalization bounds from all archives in a suite."""
    solutions = [
        item
        for result in results
        for item in result.archive
    ]

    if not solutions:
        raise ValueError("Cannot derive bounds from empty experiment archives.")

    return derive_normalization_bounds(solutions)


def _replace_nsga2_config(
    config: NSGA2Config,
    **changes: Any,
) -> NSGA2Config:
    """Create a new immutable NSGA-II configuration with selected changes."""
    values = asdict(config)
    values.update(changes)
    return NSGA2Config(**values)


def _aggregate_by_parameter(
    runs: tuple[ExperimentRun, ...],
) -> dict[str, AggregateMetrics]:
    """Aggregate runs by their varied parameter value."""
    grouped: dict[str, list[RunMetrics]] = {}

    for run in runs:
        key = f"{run.parameter_name}={run.parameter_value:g}"
        grouped.setdefault(key, []).append(run.metrics)

    return {
        key: aggregate_run_metrics(metrics)
        for key, metrics in grouped.items()
    }
