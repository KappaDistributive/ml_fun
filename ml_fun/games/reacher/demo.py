from pathlib import Path

import gymnasium as gym
import torch

from ml_fun.games.reacher.utils import SAC, Config, evaluate

if __name__ == "__main__":
    cfg = Config()
    env = gym.make(cfg.env_id)
    assert env.observation_space.shape is not None
    obs_dim = env.observation_space.shape[0]
    assert env.action_space.shape is not None
    action_dim = env.action_space.shape[0]
    env.close()
    model_path = (
        Path(__file__).parent / "data" / "sac_Reacher-v5_20260417121238_step_100000.pth"
    )
    agent = SAC(obs_dim, action_dim, env.action_space, cfg, torch.device("cpu"))
    agent_state = torch.load(model_path, map_location="cpu")
    agent.load_state_dict(agent_state)

    reward = evaluate(agent, cfg.env_id, n_episodes=10, visual=True)
