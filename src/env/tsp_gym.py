import os.path
import time

import gymnasium as gym
import numpy as np
import pygame as pygame
from gym.spaces import Discrete, Dict, Box, MultiBinary
from gymnasium.utils import seeding

from src.common.data_manipulator import make_cord, make_demands
from src.common.utils import cal_distance


class TSPEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, num_nodes,
                 step_reward=False, render_mode=None, training=True, seed=None, data_path='./data', **kwargs):
        super(TSPEnv, self).__init__()
        self.action_size = num_nodes
        self.num_nodes = num_nodes
        self.step_reward = step_reward
        self.training = training
        self.seed = seed
        self.data_path = data_path

        self.observation_space = Dict(
            {
                "xy": Box(0.0, 1.0, (self.action_size, 2), dtype=np.float32),
                "pos": Discrete(self.action_size),
                "available": MultiBinary(self.action_size)
             }
        )

        self.action_space = Discrete(self.action_size, seed=seed)

        # rendering fields
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen = None

        # observation fields
        self._xy, self._pos, self._visited = None, None, None
        self._visiting_seq = None
        self._available = None

        self._t = 0

    def seed(self, seed):
        self._np_random, self._seed = seeding.np_random(seed)

    def _get_obs(self):
        return {"xy": self._xy, "pos": self._pos, "available": self._available, "_t": self._t}

    def _make_problems(self, num_rollouts, num_nodes):
        xy = make_cord(num_rollouts, 0, num_nodes)

        if num_rollouts == 1:
            xy = xy.squeeze(0)

        return xy

    def _load_problem(self):
        file_path = f"{self.data_path}/tsp/N_{self.num_nodes}.npz"

        if os.path.isfile(file_path):
            loaded_data = np.load(file_path)
            xy = loaded_data['xy']

        else:
            xy = make_cord(1, 0, self.num_nodes)

            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path, exist_ok=True)

            np.savez_compressed(file_path, xy=xy, demands=None)

        if xy.ndim == 3:
            xy = xy.reshape(-1, 2)

        return xy

    def _init_rendering(self):
        if self.render_mode is not None:
            # Set screen dimensions
            self.screen_width = 900
            self.screen_height = 600

            # Set node size and edge width
            self.node_size = 30
            self.edge_width = 2

            scaler = 450
            left_margin = scaler * 0.28
            top_margin = scaler * 0.2
            bottom_margin = 100
            right_margin = 100

            # Define colors
            self.WHITE = (255, 255, 255)
            self.BLACK = (0, 0, 0)
            self.RED = (255, 0, 0)
            self.BLUE = (0, 0, 255)

            self.scaled_xy = [(float(x * scaler) + left_margin, float(y * scaler) + top_margin) for x, y in self._xy]

            pygame.init()
            # Define font
            self.display_font = pygame.font.Font(None, 30)
            self.node_font = pygame.font.Font(None, 20)

            # Create screen
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), flags=pygame.HIDDEN)

    def get_reward(self):
        if self._is_done() or self.step_reward:
            visitng_idx = np.array(self._visiting_seq, dtype=int)[None, :]
            dist = cal_distance(self._xy[None, :], visitng_idx)
            return -float(dist)

        else:
            return 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.training:
            self._xy = self._make_problems(1, self.num_nodes)

        else:
            self._xy = self._load_problem()

        init_depot = 0
        self._pos = init_depot
        self._visited = np.zeros(self.action_size, dtype=bool)
        self._visited[self._pos] = True
        self._visiting_seq = []

        self._visiting_seq.append(init_depot)
        self._available = np.ones(self.action_size, dtype=bool)
        self._available[init_depot] = False  # for the initial depot

        self._init_rendering()
        self._t = 0

        obs = self._get_obs()

        return obs

    def step(self, action):
        # action: (1, )

        assert action not in self._visiting_seq, f"visited nodes: {self._visiting_seq}, selected node: {action}"

        # update the current pos
        self._pos = action

        # append the visited node idx
        self._visiting_seq.append(action)

        # update visited nodes
        self._visited[action] = True

        # assign avail to field
        self._available, done = self.get_avail_mask()

        reward = self.get_reward()

        info = {}

        self._t += 1

        obs = self._get_obs()

        if self.training:
            return obs, reward, done, info

        else:
            return obs, reward, done, False, info

    def _is_done(self):
        done_flag = (self._visited[:] == True).all()
        return bool(done_flag)

    def get_avail_mask(self):
        # get a copy of avail
        avail = ~self._visited.copy()

        done = self._is_done()

        return avail, done

    def render(self):
        if self.render_mode is None:
            return

        assert self.screen is not None, "render mode setting is wrong"

        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill(self.WHITE)

        # Draw edge
        if len(self._visiting_seq) > 1:
            current_node = self._visiting_seq[0]
            prev_xy = self.scaled_xy[current_node]

            for next_node in self._visiting_seq[1:]:
                next_xy = self.scaled_xy[next_node]

                # if current_node != next_node:
                pygame.draw.line(canvas, self.BLACK, prev_xy, next_xy, self.edge_width)

                prev_xy = next_xy

        # Draw nodes and edges
        for i, (x, y) in enumerate(self.scaled_xy):
            if i == self._pos:
                node_color = self.BLUE
            elif i == 0:
                node_color = self.RED
            else:
                node_color = self.BLACK

            # Draw node
            pygame.draw.circle(canvas, node_color, (x, y), self.node_size)

            text_surface = self.node_font.render(f"{i}", True, self.WHITE)
            text_rect = text_surface.get_rect(center=(x, y))
            canvas.blit(text_surface, text_rect)

        # Current distance cal
        self.step_reward = True
        reward = -self.get_reward()
        self.step_reward = False

        # current dist show
        dist_text = self.display_font.render("Distance: {:.3f}".format(float(reward)), True, self.BLACK)
        dist_rect = dist_text.get_rect(topright=(self.screen_width -  self.screen_width*0.1, self.screen_height*0.1))
        canvas.blit(dist_text, dist_rect)

        if self.render_mode == 'human':
            # Update screen
            self.screen.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2)
            )

    def close(self):
        pygame.quit()

    def set_test_mode(self):
        self.training = False


if __name__ == '__main__':
    done = False
    env = TSPEnv(10)
    from src.env.VecEnv import RoutingVecEnv
    from stable_baselines3.common.env_util import make_vec_env

    env = make_vec_env(TSPEnv, n_envs=1, env_kwargs={'num_nodes': 10}, vec_env_cls=RoutingVecEnv)
    obs = env.reset()
    # mask = env.get_avail_mask()[0].astype(np.int8)

    while not done:
        action = np.array([env.action_space.sample()], dtype=int)
        obs, reward, done, info = env.step(action)
        # mask = env.get_avail_mask()[0].astype(np.int8)

    print(reward)