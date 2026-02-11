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
    max_generations: int = 3000

    # Evaluation
    total_eval: int = 3
    action_smoothing: float = 0.8

    # Resources
    n_workers: int = 8


cfg = Config()
