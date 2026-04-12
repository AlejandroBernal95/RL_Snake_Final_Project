import numpy as np
import torch
import random
import os
from stable_baselines3 import DQN

from environments_fully_observable import OriginalSnakeEnvironment as SnakeFullEnv
from wrapper import Snake_Fully_Observable

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def run_evaluation():
    seed = 0
    set_seed(seed)
    env = Snake_Fully_Observable(board_size=7)
    model = DQN.load("Model7x7")
    
    n_episodes = 100
    REAL_FRUIT = 0.5
    REAL_HIT_WALL = -0.1 
    REAL_ATE_HIMSELF = -0.2
    REAL_WIN = 1.0   
    REAL_STEP = 0.0

    shaping_rewards = []
    real_rewards = []
    fruits_captured = []
    wall_deaths = 0
    self_deaths = 0
    wins = 0
    timeouts = 0

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = truncated = False
        ep_shaping_rew = 0
        ep_real_rew = 0
        ep_fruits = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            ep_shaping_rew += reward
            raw = info.get("raw_reward", 0)
            
            if np.isclose(raw, env.env.FRUIT_REWARD):
                ep_real_rew += REAL_FRUIT
                ep_fruits += 1
            elif np.isclose(raw, env.env.HIT_WALL_REWARD):
                ep_real_rew += REAL_HIT_WALL
                wall_deaths += 1
            elif np.isclose(raw, env.env.ATE_HIMSELF_REWARD):
                ep_real_rew += REAL_ATE_HIMSELF
                self_deaths += 1
            elif np.isclose(raw, env.env.WIN_REWARD):
                ep_real_rew += REAL_WIN
                ep_fruits += 1
                wins += 1
            else:
                if reward == env.REWARD_TIMEOUT:
                    timeouts += 1
                else:
                    ep_real_rew += REAL_STEP
        
        shaping_rewards.append(ep_shaping_rew)
        real_rewards.append(ep_real_rew)
        fruits_captured.append(ep_fruits)

    print("="*50)
    print(f"{'DQN AGENT FINAL EVALUATION':^50}")
    print("="*50)
    print(f"Episodes:               {n_episodes}")
    print(f"Mean Reward (Shaping):  {np.mean(shaping_rewards):.2f}")
    print(f"Mean Reward (Real):     {np.mean(real_rewards):.2f}")
    print(f"Mean Fruits:            {np.mean(fruits_captured):.2f}")
    print("-" *50)
    print(f"Win Rate:               {(wins/n_episodes)*100:>6.1f}%")
    print(f"Wall Death Rate:        {(wall_deaths/n_episodes)*100:>6.1f}%")
    print(f"Self-Eat Rate:          {(self_deaths/n_episodes)*100:>6.1f}%")
    print(f"Timeout Rate:           {(timeouts/n_episodes)*100:>6.1f}%")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()