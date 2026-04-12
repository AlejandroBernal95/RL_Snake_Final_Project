import numpy as np
import random

from environments_fully_observable import OriginalSnakeEnvironment as SnakeFullEnv
from wrapper import Snake_Fully_Observable


class HeuristicSnakeModel:

    def __init__(self, env):
        self.env = env

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        inner_env = self.env.env 
        
        heads = np.argwhere(inner_env.boards == inner_env.HEAD)
        fruits = np.argwhere(inner_env.boards == inner_env.FRUIT)
        
        if len(heads) == 0 or len(fruits) == 0:
            return 0, None

        h_pos = heads[0][1:]
        f_pos = fruits[0][1:]
        
        h_row, h_col = h_pos[0], h_pos[1]
        f_row, f_col = f_pos[0], f_pos[1]
        
        candidates = []
        if f_row > h_row: candidates.append(inner_env.UP)
        elif f_row < h_row: candidates.append(inner_env.DOWN)
        
        if f_col > h_col: candidates.append(inner_env.RIGHT)
        elif f_col < h_col: candidates.append(inner_env.LEFT)

        safe_actions = []
        for a in [inner_env.UP, inner_env.RIGHT, inner_env.DOWN, inner_env.LEFT]:
            nr, nc = h_row, h_col
            if a == inner_env.UP: nr += 1
            elif a == inner_env.DOWN: nr -= 1
            elif a == inner_env.RIGHT: nc += 1
            elif a == inner_env.LEFT: nc -= 1
            
            if 0 <= nr < inner_env.board_size and 0 <= nc < inner_env.board_size:
                cell_value = inner_env.boards[0, nr, nc]
                if cell_value in [inner_env.EMPTY, inner_env.FRUIT]:
                    safe_actions.append(a)
        
        best_safe = [a for a in candidates if a in safe_actions]
        
        if best_safe:
            action = np.random.choice(best_safe)
        elif safe_actions:
            action = np.random.choice(safe_actions)
        else:
            action = inner_env.UP
            
        return np.int64(action), None


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

def run_evaluation_baseline():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    env = Snake_Fully_Observable(board_size=7)
    model = HeuristicSnakeModel(env) 
    
    n_episodes = 100
    
    REAL_FRUIT, REAL_HIT_WALL, REAL_ATE_HIMSELF, REAL_WIN, REAL_STEP = 0.5, -0.1, -0.2, 1.0, 0.0

    shaping_rewards, real_rewards, fruits_captured, steps_list = [], [], [], []
    wall_deaths, self_deaths, wins, timeouts = 0, 0, 0, 0

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = truncated = False
        ep_shaping_rew, ep_real_rew, ep_fruits, ep_steps = 0, 0, 0, 0
        
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
            ep_steps += 1
        
        shaping_rewards.append(ep_shaping_rew)
        real_rewards.append(ep_real_rew)
        fruits_captured.append(ep_fruits)
        steps_list.append(ep_steps)

    print("="*50)
    print(f"{'HEURISTIC BASELINE RESULTS':^50}")
    print("="*50)
    print(f"Episodes:               {n_episodes}")
    print(f"Mean Reward (Shaping):  {np.mean(shaping_rewards):.2f}")
    print(f"Mean Reward (Real):     {np.mean(real_rewards):.2f}")
    print(f"Mean Fruits:            {np.mean(fruits_captured):.2f}")
    print(f"Mean Steps/Episode:     {np.mean(steps_list):.1f}")
    print("-" *50)
    print(f"Win Rate:               {(wins/n_episodes)*100:>6.1f}%")
    print(f"Wall Death Rate:        {(wall_deaths/n_episodes)*100:>6.1f}%")
    print(f"Self-Eat Rate:          {(self_deaths/n_episodes)*100:>6.1f}%")
    print(f"Timeout Rate:           {(timeouts/n_episodes)*100:>6.1f}%")
    print("="*50)

if __name__ == "__main__":
    run_evaluation_baseline()