import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch

from ml_fun.games.reacher.utils import SAC, Config, ReplayBuffer, evaluate, save_model


def train(cfg: Config):
    timestamp = time.strftime("%Y%m%d%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    env = gym.make(cfg.env_id)
    assert env.observation_space.shape is not None
    obs_dim = env.observation_space.shape[0]
    assert env.action_space.shape is not None
    action_dim = env.action_space.shape[0]
    print(f"Env: {cfg.env_id} | obs_dim={obs_dim} | action_dim={action_dim}")

    agent = SAC(obs_dim, action_dim, env.action_space, cfg, device)
    buffer = ReplayBuffer(cfg.buffer_size, obs_dim, action_dim, device)

    obs, _ = env.reset(seed=cfg.seed)
    ep_return, ep_len, ep_num = 0.0, 0, 0
    recent_returns = deque(maxlen=20)
    start_time = time.time()
    metrics = {}

    print(
        f"\n{'Step':>10}  {'Ep':>6}  {'Return':>10}  {'Avg20':>10}  {'Alpha':>8}  {'Steps/s':>8}"
    )
    print("─" * 70)

    for step in range(1, cfg.total_steps + 1):
        if step < cfg.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        assert isinstance(reward, (int, float)), "Reward must be a scalar"
        ep_return += reward
        ep_len += 1

        # Store transition (mask done if episode ended due to time limit, not terminal state)
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs

        # Episode end
        if done:
            obs, _ = env.reset()
            recent_returns.append(ep_return)
            ep_num += 1
            ep_return = 0.0
            ep_len = 0

        # Training
        if step >= cfg.learning_starts and step % cfg.train_freq == 0:
            for _ in range(cfg.gradient_steps):
                metrics = agent.update(buffer)

        # Logging
        if step % cfg.log_interval == 0 and recent_returns:
            elapsed = time.time() - start_time
            steps_sec = step / elapsed
            avg20 = np.mean(recent_returns)
            print(
                f"{step:>10,}  {ep_num:>6}  {recent_returns[-1]:>10.2f}  {avg20:>10.2f}  {metrics['alpha']:>8.4f}  {steps_sec:>8.0f}"
            )

        # Evaluation
        if step % cfg.eval_interval == 0:
            eval_return = evaluate(
                agent, cfg.env_id, cfg.eval_episodes, cfg.seed, cfg.visual
            )
            print(f"\n{'─'*70}")
            print(f"  EVAL @ step {step:,}  |  mean return = {eval_return:.2f}")
            print(f"{'─'*70}\n")
            save_model(
                agent, cfg.data_dir / f"sac_{cfg.env_id}_{timestamp}_step_{step}.pth"
            )

    env.close()
    print("\nTraining complete.")

    # Final evaluation
    final_return = evaluate(agent, cfg.env_id, 20, cfg.seed)
    print(f"Final evaluation (20 episodes): {final_return:.2f}")
    save_model(agent, cfg.data_dir / f"sac_{cfg.env_id}_{timestamp}_final.pth")

    return agent


if __name__ == "__main__":
    cfg = Config()
    agent = train(cfg)
