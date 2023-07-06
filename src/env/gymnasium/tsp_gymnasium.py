import os.path
import pickle

import gymnasium as gym
import numpy as np
import pygame as pygame
from gymnasium.spaces import Discrete, Dict, Box, MultiBinary
from gymnasium.utils import seeding

from src.common.data_manipulator import make_cord
from src.common.utils import cal_distance


class TSPEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 3}

    def __init__(self, num_nodes,
                 step_reward=False, render_mode=None, training=True, seed=None, data_path='./data', **kwargs):
        super(TSPEnv, self).__init__()
        self.num_nodes = num_nodes
        self.step_reward = step_reward
        self.training = training
        self.seed = seed
        self.data_path = data_path
        self.env_type = 'tsp'
        self.test_num = kwargs.get('test_num')
        self.action_size = self.num_nodes if self.test_num is None else self.test_num

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
        self.xy, self.pos, self.visited = None, None, None
        self.visiting_seq = None
        self.available = None
        self.t = 0

        self.test_data_type = kwargs.get('test_data_type')
        self._load_data_idx = kwargs.get('test_data_idx')

    def seed(self, seed):
        self._np_random, self._seed = seeding.np_random(seed)

    def _get_obs(self):
        return {"xy": self.xy, "pos": self.pos, "available": self.available, "t": self.t}

    def _make_problems(self, num_rollouts, num_nodes):
        xy = make_cord(num_rollouts, 0, num_nodes)

        if num_rollouts == 1:
            xy = xy.squeeze(0)

        return xy

    def _load_data(self, filepath):
        # data format: (xy)
        ext = filepath.split('.')[-1]

        if ext == 'npz':
            loaded_data = np.load(filepath)
            xy = loaded_data['xy']

        elif ext == 'pkl':
            with open(filepath, 'rb') as f:
                xy = pickle.load(f)

            xy = np.array(xy, dtype=np.float32)[self._load_data_idx, :]
            # self._load_data_idx += 1

        else:
            raise ValueError(f"Invalid file extension for loading data: {ext}")

        # xy must be in shape of [batch, num_nodes, 2]
        return xy

    def _load_problem(self):
        if self.test_data_type == 'npz':
            file_path = f"{self.data_path}/tsp/N_{self.test_num}.npz"

        else:
            file_path = f"{self.data_path}/tsp/tsp{self.test_num}_test_seed1234.pkl"

        if os.path.isfile(file_path):
            xy = self._load_data(file_path)

        else:
            xy = make_cord(1, 0, self.test_num)

            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path, exist_ok=True)

            np.savez_compressed(file_path, xy=xy, demands=None)

        if xy.ndim == 2:
            xy = xy.reshape(1, -1, 2)

        return xy

    def _init_rendering(self):
        if self.render_mode is not None:
            # Set screen dimensions
            self.screen_width = 900
            self.screen_height = 600

            # Set node size and edge width
            self.node_size = 20
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

            self.scaled_xy = [(float(x * scaler) + left_margin, float(y * scaler) + top_margin) for x, y in self.xy.reshape(-1,2)]

            pygame.init()
            # Define font

    def get_reward(self):
        if (self._is_done() or self.step_reward) and self.t > 0:
            # batch_size, pomo_size = self.pos.shape
            visitng_idx = np.concatenate(self.visiting_seq, axis=2)
            # (num_env, pomo_size, num_nodes):
            dist = cal_distance(self.xy, visitng_idx, axis=2)
            return float(dist.reshape(-1))

        else:
            return 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.training:
            self.xy = self._make_problems(1, self.num_nodes)

        else:
            self.xy = self._load_problem()

        self.pos = None
        self.visited = np.zeros((1, 1, self.action_size), dtype=bool)

        self.visiting_seq = []

        self.available = np.ones((1, 1, self.action_size), dtype=bool)  # all nodes are available at the beginning

        self.t = 0

        self._init_rendering()

        obs = self._get_obs()

        return obs, {}

    def step(self, action):
        action = np.array([[[action]]])
        # action: (1, 1, 1)
        self.t += 1

        # update the current pos
        self.pos = action.reshape(1, 1)

        # append the visited node idx
        self.visiting_seq.append(action)

        # update visited nodes
        np.put_along_axis(self.visited, action, True, axis=2)

        # assign avail to field
        self.available, done = self.get_avail_mask()

        reward = self.get_reward()

        info = {}

        obs = self._get_obs()

        return obs, reward, done, False, info

    def _is_done(self):
        done_flag = (self.visited == True).all()
        return done_flag

    def get_avail_mask(self):
        # get a copy of avail
        avail = ~self.visited.copy()

        done = self._is_done()

        return avail, done

    def render(self):
        if self.render_mode is None:
            return

        pygame.font.init()
        display_font = pygame.font.Font(None, 30)
        node_font = pygame.font.Font(None, 20)

        # Create screen
        screen = pygame.display.set_mode((self.screen_width, self.screen_height), flags=pygame.HIDDEN)

        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill(self.WHITE)

        # Draw edge
        if len(self.visiting_seq) > 1:
            current_node = self.visiting_seq[0]
            prev_xy = self.scaled_xy[current_node]

            for next_node in self.visiting_seq[1:]:
                next_xy = self.scaled_xy[next_node]

                # if current_node != next_node:
                pygame.draw.line(canvas, self.BLACK, prev_xy, next_xy, self.edge_width)

                prev_xy = next_xy

        # Draw nodes and edges
        for i, (x, y) in enumerate(self.scaled_xy):
            if i == self.pos:
                node_color = self.BLUE
            elif i == 0:
                node_color = self.RED
            else:
                node_color = self.BLACK

            # Draw node
            pygame.draw.circle(canvas, node_color, (x, y), self.node_size)

            text_surface = node_font.render(f"{i}", True, self.WHITE)
            text_rect = text_surface.get_rect(center=(x, y))
            canvas.blit(text_surface, text_rect)

        # Current distance cal
        self.step_reward = True
        reward = -self.get_reward()
        self.step_reward = False

        # current dist show
        dist_text = display_font.render("Distance: {:.3f}".format(float(reward)), True, self.BLACK)
        dist_rect = dist_text.get_rect(topright=(self.screen_width - self.screen_width * 0.1, self.screen_height * 0.1))
        canvas.blit(dist_text, dist_rect)

        if self.render_mode == 'human':
            # Update screen
            screen.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        pygame.quit()

    def set_test_mode(self):
        self.training = False
