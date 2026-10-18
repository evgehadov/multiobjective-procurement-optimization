
"""Command-line entry point for the multi-objective procurement project."""

from __future__ import annotations

import argparse
import sys

from config import (
    DEFAULT_CONFIG,
    ExperimentConfig,
    NSGA2Config,
)
from procurement_ga.data_generator import generate_problem
from procurement_ga.experiments import (
    ExperimentSuite,
    run_experiment_a,
    run_experiment_b,
    run_experiment_c,
    run_experiment_d,
)
from procurement_ga.reporting import save_experiment_suite


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Multi-objective procurement optimization using NSGA-II "
            "and a weighted-sum baseline."
        )
    )

    parser.add_argument(
        "--experiment",
        choices=("A", "B", "C", "D", "all"),
        default="all",
        help="Experiment to run (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG.data.seed,
        help="Seed for reproducible problem generation.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_CONFIG.experiments.independent_runs,
        help="Number of independent runs for each applicable configuration.",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=DEFAULT_CONFIG.nsga2.population_size,
        help="NSGA-II population size.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_CONFIG.nsga2.generations,
        help="Number of NSGA-II generations.",
    )
    parser.add_argument(
        "--mutation",
        type=float,
        default=DEFAULT_CONFIG.nsga2.mutation_probability,
        help="NSGA-II mutation probability.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_CONFIG.results_dir),
        help="Directory for CSV, JSON, and PNG results.",
    )
    parser.add_argument(
        "--baseline-iterations",
        type=int,
        default=2000,
        help="Local-search iterations for each weighted-sum baseline run.",
    )

    return parser


def _make_experiment_config(
    args: argparse.Namespace,
) -> ExperimentConfig:
    """Build experiment settings from command-line arguments."""
    defaults = DEFAULT_CONFIG.experiments

    return ExperimentConfig(
        independent_runs=args.runs,
        base_seed=defaults.base_seed,
        population_sizes=defaults.population_sizes,
        mutation_probabilities=defaults.mutation_probabilities,
        baseline_weights=defaults.baseline_weights,
        save_csv=True,
        save_json=True,
        save_plots=True,
    )


def _make_nsga2_config(args: argparse.Namespace) -> NSGA2Config:
    """Build NSGA-II settings from command-line arguments."""
    defaults = DEFAULT_CONFIG.nsga2

    return NSGA2Config(
        population_size=args.population,
        generations=args.generations,
        crossover_probability=defaults.crossover_probability,
        mutation_probability=args.mutation,
        mutation_gene_probability=defaults.mutation_gene_probability,
        seed=args.seed,
    )


def _run_selected_experiment(
    name: str,
    problem,
    experiment_config: ExperimentConfig,
    algorithm_config: NSGA2Config,
    baseline_iterations: int,
) -> ExperimentSuite:
    """Run one selected experiment."""
    if name == "A":
        return run_experiment_a(
            problem,
            experiment_config,
            algorithm_config,
        )

    if name == "B":
        return run_experiment_b(
            problem,
            experiment_config,
            algorithm_config,
            runs_per_size=experiment_config.independent_runs,
        )

    if name == "C":
        return run_experiment_c(
            problem,
            experiment_config,
            algorithm_config,
            runs_per_probability=experiment_config.independent_runs,
        )

    if name == "D":
        return run_experiment_d(
            problem,
            experiment_config,
            algorithm_config,
            baseline_iterations=baseline_iterations,
        )

    raise ValueError(f"Unknown experiment: {name}")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, execute selected experiments, and save results."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.runs < 1:
        parser.error("--runs must be at least 1.")
    if args.population < 2:
        parser.error("--population must be at least 2.")
    if args.generations < 1:
        parser.error("--generations must be at least 1.")
    if not 0.0 <= args.mutation <= 1.0:
        parser.error("--mutation must be between 0 and 1.")
    if args.baseline_iterations < 1:
        parser.error("--baseline-iterations must be at least 1.")

    if args.experiment in ("A", "all") and args.runs < 20:
        parser.error("Experiment A requires --runs to be at least 20.")

    experiment_config = _make_experiment_config(args)
    algorithm_config = _make_nsga2_config(args)
    problem = generate_problem(seed=args.seed)

    selected = ("A", "B", "C", "D") if args.experiment == "all" else (args.experiment,)

    for name in selected:
        print(f"\n=== Эксперимент {name} ===")
        suite = _run_selected_experiment(
            name,
            problem,
            experiment_config,
            algorithm_config,
            args.baseline_iterations,
        )

        paths = save_experiment_suite(
            suite,
            output_dir=args.output,
            prefix=f"experiment_{name.lower()}",
        )

        print(f"Материалов: {len(problem.materials)}")
        print(f"Бюджет: {problem.budget:.2f}")
        for label, metrics in suite.aggregates.items():
            print(
                f"{label}: запусков={metrics.runs}, "
                f"средний HV={metrics.mean_hypervolume:.6f}, "
                f"среднее число решений={metrics.mean_nondominated_count:.2f}"
            )

        print("Сохранённые файлы:")
        for label, path in paths.items():
            print(f"  {label}: {path}")

    print("\nВсе выбранные эксперименты завершены.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nВыполнение прервано пользователем.", file=sys.stderr)
        raise SystemExit(130)
