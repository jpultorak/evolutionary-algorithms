import os
import pickle

import gymnasium as gym
import imageio
import numpy as np

# Import your existing project modules
from config import cfg
from models.model import MLPPolicy


def generate_gif(weights_path, output_filename="walker_demo.gif"):
    """
    Loads weights, runs one episode, and saves the result as a GIF.
    """
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        return

    print(f"Loading weights from: {weights_path}")
    with open(weights_path, "rb") as f:
        weights = pickle.load(f)

    # crucial: render_mode="rgb_array" allows us to capture frames without a window
    env = gym.make(cfg.env_name, render_mode="rgb_array", hardcore=cfg.hardcore)
    policy = MLPPolicy(cfg.input_size, cfg.hidden_size, cfg.output_size)

    frames = []
    obs, _ = env.reset()

    prev_action = np.zeros(cfg.output_size)
    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        # 1. Capture Frame
        frames.append(env.render())

        # 2. Normalize Observation
        norm_obs = obs / cfg.normalization

        # 3. Get Action
        raw_action = policy.get_action(norm_obs, weights)

        # 4. Apply Smoothing
        action = (cfg.action_smoothing * prev_action) + (
            (1.0 - cfg.action_smoothing) * raw_action
        )
        prev_action = action

        # 5. Step
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    env.close()
    print(f"Episode finished. Total Score: {total_reward:.2f}")

    print(f"Saving GIF to {output_filename}...")
    imageio.mimsave(output_filename, frames, fps=30, loop=0)
    print("Done!")


if __name__ == "__main__":
    w_path = "checkpoints_no_smoothing/weights_gen_300.pkl"
    o_path = "results/no_smoothing/result.gif"

    generate_gif(w_path, o_path)
