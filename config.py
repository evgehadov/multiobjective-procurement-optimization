
"""Central configuration for the multi-objective procurement optimization lab."""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Parameters used to generate the synthetic procurement instance."""

    n_materials: int = 30
    seed: int = 20261018

    min_price: int = 10
    max_price: int = 500

    min_initial_stock: int = 0
    max_initial_stock: int = 80

    min_monthly_demand: int = 10
    max_monthly_demand: int = 100

    min_required_stock: int = 0
    max_required_stock: int = 30

    min_deficit_weight: float = 1.0
    max_deficit_weight: float = 10.0

    min_holding_cost: float = 0.1
    max_holding_cost: float = 5.0

    max_purchase_quantity: int = 100
    budget_fraction: float = 0.65


@dataclass(frozen=True, slots=True)
class NSGA2Config:
    """Default parameters of the self-implemented NSGA-II algorithm."""

    population_size: int = 100
    generations: int = 200
    crossover_probability: float = 0.9
    mutation_probability: float = 0.1
    mutation_gene_probability: float = 0.1
    seed: int = 20261018


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Parameters controlling reproducible experiment series."""

    independent_runs: int = 20
    base_seed: int = 20261018

    population_sizes: tuple[int, ...] = (50, 100, 150)
    mutation_probabilities: tuple[float, ...] = (0.05, 0.1, 0.2)

    # The weights sum to one. Each configuration emphasizes a different
    # objective, with one balanced configuration.
    baseline_weights: tuple[tuple[float, float, float], ...] = (
        (0.8, 0.1, 0.1),
        (0.1, 0.8, 0.1),
        (0.1, 0.1, 0.8),
        (1 / 3, 1 / 3, 1 / 3),
    )

    save_csv: bool = True
    save_json: bool = True
    save_plots: bool = True


@dataclass(frozen=True, slots=True)
class HypervolumeConfig:
    """Shared normalization and reference-point settings for hypervolume."""

    # A reference point in normalized objective space. All compared fronts
    # must use the same point.
    reference_point: tuple[float, float, float] = (1.1, 1.1, 1.1)


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level project configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    nsga2: NSGA2Config = field(default_factory=NSGA2Config)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)
    hypervolume: HypervolumeConfig = field(default_factory=HypervolumeConfig)

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR


DEFAULT_CONFIG = AppConfig()
