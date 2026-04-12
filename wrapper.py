import numpy as np
import tensorflow as tf
import gymnasium as gym
from gymnasium import spaces

from environments_fully_observable import OriginalSnakeEnvironment as SnakeFullEnv

class Snake_Fully_Observable(gym.Env):
    def __init__(self, board_size=7, num_obstacles=0, reward_config=None):
        super().__init__()

        self.board_size = board_size
        self.num_obstacles = num_obstacles
        self.env = SnakeFullEnv(n_boards=1, board_size=board_size)

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(board_size * board_size * 4,),
            dtype=np.float32
        )

        if reward_config is None:
            reward_config = {}

        self.REWARD_FRUIT = reward_config.get('fruit', 1.2)
        self.REWARD_HIT_WALL = reward_config.get('wall', -1.2)
        self.REWARD_ATE_HIMSELF = reward_config.get('ate himself', -1.2)
        self.REWARD_STEP  = reward_config.get('step', -0.03)
        self.REWARD_WIN   = reward_config.get('win', 3.0)
        self.REWARD_TIMEOUT = reward_config.get('timeout', -0.5)

        self.max_steps_no_fruit = board_size * board_size * 2
        self.steps_since_fruit = 0
        self.extra_walls = []

        self._obs_buffer = np.zeros((board_size, board_size, 4), dtype=np.float32)

    def _get_obs(self):
        board = self.env.boards[0]
        obs = self._obs_buffer
        obs.fill(0)

        obs[..., 0] = (board == self.env.WALL)
        if self.num_obstacles > 0:
            for (r, c) in self.extra_walls:
                obs[r, c, 0] = 1.0

        obs[..., 1] = (board == self.env.FRUIT)
        obs[..., 2] = (board == self.env.BODY)
        obs[..., 3] = (board == self.env.HEAD)

        return obs.copy().ravel()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed) # Requires import tensorflow

        self.env = SnakeFullEnv(n_boards=1, board_size=self.board_size)
        self.steps_since_fruit = 0
        self.extra_walls = []
        
        if self.num_obstacles > 0:
            current_board = self.env.boards[0]
            possible_locs = [tuple(p) for p in np.argwhere(current_board == self.env.EMPTY)]
            if possible_locs:
                n = min(self.num_obstacles, len(possible_locs))
                idx = np.random.choice(len(possible_locs), n, replace=False)
                self.extra_walls = [possible_locs[i] for i in idx]
                for (r, c) in self.extra_walls:
                    self.env.boards[0, r, c] = self.env.WALL 

        return self._get_obs(), {}

    def step(self, action):
        self.steps_since_fruit += 1

        if self.num_obstacles > 0:
            for (r, c) in self.extra_walls:
                self.env.boards[0, r, c] = self.env.WALL

        action_input = np.array([[action]])
        rewards_tf = self.env.move(action_input)
        raw_reward = float(rewards_tf.numpy()[0][0])

        terminated = False
        reward = self.REWARD_STEP

        if np.isclose(raw_reward, self.env.HIT_WALL_REWARD):
            reward = self.REWARD_HIT_WALL
            terminated = True
        elif np.isclose(raw_reward, self.env.ATE_HIMSELF_REWARD):
            reward = self.REWARD_ATE_HIMSELF
            terminated = True
        elif np.isclose(raw_reward, self.env.FRUIT_REWARD):
            reward = self.REWARD_FRUIT
            self.steps_since_fruit = 0
        elif np.isclose(raw_reward, self.env.WIN_REWARD):
            reward = self.REWARD_WIN
            terminated = True

        if not terminated and self.num_obstacles > 0:
            head_pos = np.argwhere(self.env.boards[0] == self.env.HEAD)
            if head_pos.size > 0:
                if tuple(head_pos[0]) in self.extra_walls:
                    reward = self.REWARD_HIT_WALL
                    terminated = True
            else:
                terminated = True

        if not terminated and self.steps_since_fruit >= self.max_steps_no_fruit:
            reward = self.REWARD_TIMEOUT
            terminated = True

        return self._get_obs(), float(reward), terminated, False, {"raw_reward": raw_reward}