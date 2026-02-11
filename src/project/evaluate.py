import gymnasium as gym
import numpy as np
from config import cfg
from models.model import MLPPolicy

policy = MLPPolicy(cfg.input_size, cfg.hidden_size, cfg.output_size)


def run_single_episode(env, weights):
    obs, _ = env.reset()
    total_reward = 0
    step_count = 0

    prev_action = np.zeros(cfg.output_size)

    terminated = False
    truncated = False

    while not (terminated or truncated):
        # 1. Normalize
        norm_obs = obs / cfg.normalization

        # 2. MLP Output
        raw_action = policy.get_action(norm_obs, weights)

        # 3. Smoothing
        action = (cfg.action_smoothing * prev_action) + (
            (1.0 - cfg.action_smoothing) * raw_action
        )
        prev_action = action

        # 4. Step
        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        step_count += 1

        # 5. Early Termination
        if cfg.early_termination and total_reward < -90 and step_count < 50:
            return total_reward - 50

    return total_reward


def evaluate(weights):
    env = gym.make(cfg.env_name, hardcore=cfg.hardcore)

    rewards = []
    for _ in range(cfg.total_rollouts):
        r = run_single_episode(env, weights)
        rewards.append(r)

    env.close()

    avg_reward = np.mean(rewards)
    return -avg_reward
