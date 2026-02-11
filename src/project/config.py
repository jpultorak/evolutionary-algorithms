from dataclasses import dataclass


@dataclass
class Config:
    # Environment
    env_name: str = "BipedalWalker-v3"
    hardcore: bool = False
    seed: int = 42

    # Network
    input_size: int = 24
    hidden_size: int = 48
    output_size: int = 4

    # CMA-ES
    pop_size: int = 96
    sigma_init: float = 0.5
    max_generations: int = 1000

    # Evaluation & optimization
    total_rollouts: int = 3
    action_smoothing: float = 0.0  # 0.0 to turn off
    normalization: float = 5.0  # 1.0 to turn off
    early_termination: bool = True

    # Resources
    n_workers: int = 8


cfg = Config()
