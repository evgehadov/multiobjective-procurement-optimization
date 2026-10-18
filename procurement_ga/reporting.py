
"""Export experiment results to CSV, JSON, and PNG plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config import RESULTS_DIR
from procurement_ga.experiments import ExperimentRun, ExperimentSuite
from procurement_ga.metrics import normalize_objectives
from procurement_ga.models import EvaluatedSolution


OBJECTIVE_NAMES = (
    "purchase_cost",
    "weighted_deficit",
    "holding_cost",
)


def save_experiment_suite(
    suite: ExperimentSuite,
    output_dir: str | Path | None = None,
    prefix: str = "experiment",
) -> dict[str, Path]:
    """Save a suite's run metrics, solutions, and plots."""
    directory = Path(output_dir) if output_dir is not None else RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    metrics_path = directory / f"{prefix}_metrics.csv"
    _write_metrics_csv(suite.runs, metrics_path)
    paths["metrics_csv"] = metrics_path

    solutions_path = directory / f"{prefix}_solutions.csv"
    _write_solutions_csv(
        suite.runs,
        solutions_path,
        suite.problem.budget,
    )
    paths["solutions_csv"] = solutions_path

    summary_path = directory / f"{prefix}_summary.json"
    _write_summary_json(suite, summary_path)
    paths["summary_json"] = summary_path

    plots = save_pareto_plots(suite, directory, prefix)
    paths.update(plots)

    return paths


def _write_metrics_csv(
    runs: Sequence[ExperimentRun],
    path: Path,
) -> None:
    """Write per-run quality metrics."""
    fieldnames = [
        "experiment",
        "parameter_name",
        "parameter_value",
        "run_index",
        "seed",
        "generations",
        "population_size",
        "elapsed_seconds",
        "nondominated_count",
        "hypervolume",
        "mean_purchase_cost",
        "mean_weighted_deficit",
        "mean_holding_cost",
        "best_purchase_cost",
        "best_weighted_deficit",
        "best_holding_cost",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for run in runs:
            mean_values = run.metrics.mean_objectives
            best_values = run.metrics.best_objectives

            writer.writerow(
                {
                    "experiment": run.experiment,
                    "parameter_name": run.parameter_name,
                    "parameter_value": run.parameter_value,
                    "run_index": run.run_index,
                    "seed": run.result.seed,
                    "generations": run.result.generations,
                    "population_size": run.result.population_size,
                    "elapsed_seconds": run.result.elapsed_seconds,
                    "nondominated_count": run.metrics.nondominated_count,
                    "hypervolume": run.metrics.hypervolume,
                    "mean_purchase_cost": mean_values[0],
                    "mean_weighted_deficit": mean_values[1],
                    "mean_holding_cost": mean_values[2],
                    "best_purchase_cost": best_values[0],
                    "best_weighted_deficit": best_values[1],
                    "best_holding_cost": best_values[2],
                }
            )


def _write_solutions_csv(
    runs: Sequence[ExperimentRun],
    path: Path,
    budget: float,
) -> None:
    """Write the objective values and integer genotype of each archived plan."""
    fieldnames = [
        "experiment",
        "parameter_name",
        "parameter_value",
        "run_index",
        "solution_index",
        "genotype",
        "purchase_cost",
        "weighted_deficit",
        "holding_cost",
        "within_budget",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for run in runs:
            for index, solution in enumerate(run.result.archive, start=1):
                writer.writerow(
                    {
                        "experiment": run.experiment,
                        "parameter_name": run.parameter_name,
                        "parameter_value": run.parameter_value,
                        "run_index": run.run_index,
                        "solution_index": index,
                        "genotype": json.dumps(
                            solution.solution.quantities,
                            ensure_ascii=False,
                        ),
                        "purchase_cost": solution.objectives.purchase_cost,
                        "weighted_deficit": solution.objectives.weighted_deficit,
                        "holding_cost": solution.objectives.holding_cost,
                        "within_budget": solution.objectives.purchase_cost <= budget,
                    }
                )


def _write_summary_json(
    suite: ExperimentSuite,
    path: Path,
) -> None:
    """Write configuration, aggregate metrics, and comparison metadata."""
    summary: dict[str, Any] = {
        "problem": {
            "material_count": len(suite.problem.materials),
            "budget": suite.problem.budget,
        },
        "normalization_bounds": {
            "minima": suite.normalization_bounds.minima,
            "maxima": suite.normalization_bounds.maxima,
        },
        "aggregates": {
            name: {
                "runs": metrics.runs,
                "mean_hypervolume": metrics.mean_hypervolume,
                "median_hypervolume": metrics.median_hypervolume,
                "std_hypervolume": metrics.std_hypervolume,
                "mean_nondominated_count": metrics.mean_nondominated_count,
                "mean_objectives": metrics.mean_objectives,
            }
            for name, metrics in suite.aggregates.items()
        },
        "runs": [
            {
                "experiment": run.experiment,
                "parameter_name": run.parameter_name,
                "parameter_value": run.parameter_value,
                "run_index": run.run_index,
                "seed": run.result.seed,
                "generations": run.result.generations,
                "population_size": run.result.population_size,
                "hypervolume": run.metrics.hypervolume,
                "nondominated_count": run.metrics.nondominated_count,
            }
            for run in suite.runs
        ],
    }

    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_pareto_plots(
    suite: ExperimentSuite,
    output_dir: str | Path | None = None,
    prefix: str = "experiment",
) -> dict[str, Path]:
    """Save all three pairwise projections of the objective space."""
    directory = Path(output_dir) if output_dir is not None else RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)

    solutions = _unique_solutions(suite.runs)
    normalized = [
        normalize_objectives(item.objectives.as_tuple, suite.normalization_bounds)
        for item in solutions
    ]

    projections = (
        (0, 1, "purchase_cost", "weighted_deficit"),
        (0, 2, "purchase_cost", "holding_cost"),
        (1, 2, "weighted_deficit", "holding_cost"),
    )

    paths: dict[str, Path] = {}

    for x_index, y_index, x_name, y_name in projections:
        path = directory / f"{prefix}_pareto_{x_name}_vs_{y_name}.png"

        figure, axis = plt.subplots(figsize=(8, 6))
        if normalized:
            axis.scatter(
                [point[x_index] for point in normalized],
                [point[y_index] for point in normalized],
                alpha=0.65,
                s=24,
            )

        axis.set_xlabel(f"{x_name} (normalized)")
        axis.set_ylabel(f"{y_name} (normalized)")
        axis.set_title(f"Pareto projection: {x_name} vs {y_name}")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)

        paths[f"pareto_{x_name}_vs_{y_name}"] = path

    return paths


def save_hypervolume_history(
    history: Sequence[float],
    output_path: str | Path,
) -> Path:
    """Save a hypervolume-by-generation plot when history is available."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(range(1, len(history) + 1), history, marker="o", markersize=3)
    axis.set_xlabel("Generation")
    axis.set_ylabel("Hypervolume")
    axis.set_title("Hypervolume history")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)

    return path


def _unique_solutions(
    runs: Sequence[ExperimentRun],
) -> list[EvaluatedSolution]:
    """Collect unique objective/genotype pairs across experiment archives."""
    unique: dict[tuple[tuple[int, ...], tuple[float, float, float]], EvaluatedSolution] = {}

    for run in runs:
        for solution in run.result.archive:
            key = (
                solution.solution.quantities,
                solution.objectives.as_tuple,
            )
            unique[key] = solution

    return list(unique.values())
