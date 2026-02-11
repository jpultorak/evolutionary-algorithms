import os
import pickle

import gymnasium as gym
import numpy as np
from config import cfg
from models.model import MLPPolicy


def load_weights(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Weights file not found: {filename}")

    with open(filename, "rb") as f:
        return pickle.load(f)


def visualization(weights_path):
    print(f"Loading weights from: {weights_path}")
    weights = load_weights(weights_path)

    policy = MLPPolicy(cfg.input_size, cfg.hidden_size, cfg.output_size)
    env = gym.make(cfg.env_name, render_mode="human", hardcore=cfg.hardcore)

    print(f"Visualizing {cfg.env_name}")
    print(50 * "-")

    while True:
        obs, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False

        prev_action = np.zeros(cfg.output_size)

        while not (terminated or truncated):
            env.render()

            # 1. Normalize
            norm_obs = obs / cfg.normalization

            # 2. MLP output
            raw_action = policy.get_action(norm_obs, weights)

            # 3. Smoothing
            action = (cfg.action_smoothing * prev_action) + (
                (1.0 - cfg.action_smoothing) * raw_action
            )
            prev_action = action

            # 4. Step
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        print(f"Episode Finished. Score: {total_reward:.2f}")


if __name__ == "__main__":
    filename = "checkpoints_no_early_termination/weights_gen_250.pkl"
    # filename = "winner-normal.pkl"
    visualization(filename)
