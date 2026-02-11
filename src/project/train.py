import multiprocessing
import os
import pickle
import time

import cma
import numpy as np
from config import cfg
from evaluate import evaluate
from models.model import MLPPolicy


def save_checkpoint(es, generation, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    weights_path = os.path.join(checkpoint_dir, f"weights_gen_{generation}.pkl")
    with open(weights_path, "wb") as f:
        pickle.dump(es.result.xbest, f)

    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_gen_{generation}.pkl")
    with open(optimizer_path, "wb") as f:
        pickle.dump(es, f)

    print(f"Checkpoint saved: Generation {generation}")


def main():
    print(f"Starting trainging for {cfg.env_name}")
    print(50 * "-")

    log_file = "training_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("generation,min_fitness,avg_fitness,max_fitness\n")

    dummy_policy = MLPPolicy(cfg.input_size, cfg.hidden_size, cfg.output_size)
    n_params = dummy_policy.param_count

    print(f"Genome Size: {n_params}")
    print(f"Population: {cfg.pop_size} | Workers: {cfg.n_workers}")

    es = cma.CMAEvolutionStrategy(
        np.zeros(n_params),
        cfg.sigma_init,
        {
            "popsize": cfg.pop_size,
            "seed": cfg.seed,
            "verb_disp": 1,
        },
    )

    start_time = time.time()

    with multiprocessing.Pool(cfg.n_workers) as pool:
        while not es.stop():
            gen = es.countiter

            solutions = es.ask()
            fitness_values = pool.map(evaluate, solutions)
            es.tell(solutions, fitness_values)
            es.disp()

            #  Logs
            rewards = [-f for f in fitness_values]
            max_reward = np.max(rewards)
            avg_reward = np.mean(rewards)
            min_reward = np.min(rewards)

            with open(log_file, "a") as f:
                f.write(f"{gen},{min_reward:.2f},{avg_reward:.2f},{max_reward:.2f}\n")

            # Checkpoint
            if gen % 50 == 0:
                save_checkpoint(es, gen)

            # Termination condition
            if max_reward > 300:
                print(f"\n Solved in generation {gen}")
                save_checkpoint(es, gen)
                break

            if gen >= cfg.max_generations:
                save_checkpoint(es, gen)
                print("\nMax generations reached.")
                break

    total_time = (time.time() - start_time) / 60
    print(f"Total time: {total_time:.2f} minutes")


if __name__ == "__main__":
    # multiprocessing.freeze_support()
    main()
