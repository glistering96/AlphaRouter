import os.path
import pickle

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Dict, Box, MultiBinary
from gymnasium.utils import seeding

from src.common.data_manipulator import make_cord, make_demands
from src.common.utils import cal_distance


class CVRPEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 3}

    def __init__(self, num_depots, num_nodes, step_reward=False, render_mode=None, training=True, seed=None,
                 data_path='./data', **kwargs):
        super(CVRPEnv, self).__init__()
        self.num_depots = num_depots
        self.num_nodes = num_nodes
        self.step_reward = step_reward
        self.training = training
        self.seed = seed
        self.data_path = data_path
        self.env_type = 'cvrp'
        self.test_num = kwargs.get('test_num')
        self.action_size = self.num_nodes + num_depots if self.test_num is None else self.test_num + num_depots

        self.observation_space = Dict(
            {
                "xy": Box(0.0, 1.0, (self.action_size, 2), dtype=np.float32),
                "demands": Box(
                    low=np.array([0.0 for _ in range(self.action_size)], dtype=np.float32),
                    high=np.array([0.0] + [1.0 for _ in range(num_nodes)], dtype=np.float32),
                    dtype=np.float32),
                "pos": Discrete(self.action_size),
                "load": Box(0, 1, (1,), dtype=np.float32),
                "available": MultiBinary(self.action_size)
            }
        )

        self.action_space = Discrete(self.action_size, seed=seed)

        # rendering fields
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen = None

        # observation fields
        self.xy, self.demand, self.pos, self.visited = None, None, None, None
        self.visiting_seq = None
        self.load = None
        self.available = None

        self.t = 0

        self.test_data_type = kwargs.get('test_data_type')
        self._load_data_idx = 0

        self.num_env = 1
        self.pomo_size = 1

    def seed(self, seed):
        self._np_random, self._seed = seeding.np_random(seed)

    def _get_obs(self):
        return {"xy": self.xy, "demands": self.demand, "pos": self.pos, "load": self.load,
                "available": self.available, "t": self.t}

    def _make_problems(self, num_rollouts, num_depots, num_nodes):
        xy = make_cord(num_rollouts, num_depots, num_nodes)
        demands = make_demands(num_rollouts, num_depots, num_nodes)

        if num_rollouts == 1:
            xy = xy.squeeze(0)
            demands = demands.squeeze(0)

        return xy, demands

    def _load_data(self, filepath):
        # data format: ([depot xy, node_xy, node_demand, capacity])
        ext = filepath.split('.')[-1]

        if ext == 'npz':
            loaded_data = np.load(filepath)
            xy = loaded_data['xy']
            demands = loaded_data['demands']

        elif ext == 'pkl':
            with open(filepath, 'rb') as f:
                depot_xy, node_xy, node_demand, capacity = pickle.load(f)
                xy = np.array([depot_xy, node_xy], dtype=np.float32)[self._load_data_idx, :]
                demands = np.array([[0, 1], node_demand], dtype=np.float32) / capacity
                demands = demands[self._load_data_idx, :]

            self._load_data_idx += 1

        else:
            raise ValueError(f"Invalid file extension for loading data: {ext}")

        if xy.ndim == 2:
            xy = xy.reshape(1, -1, 2)

        if demands.ndim == 2:
            demands = demands.reshape(1, -1)

        return xy, demands

    def _load_problem(self):
        if self.test_data_type == 'npz':
            file_path = f"{self.data_path}/cvrp/N_{self.test_num}.npz"

        else:
            file_path = f"{self.data_path}/cvrp/cvrp{self.test_num}_test_seed1234.pkl"

        if os.path.isfile(file_path):
            xy, demands = self._load_data(file_path)

        else:
            xy = make_cord(1, self.num_depots, self.test_num)
            demands = make_demands(1, self.num_depots, self.test_num)

            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path, exist_ok=True)

            np.savez_compressed(file_path, xy=xy, demands=demands)

        return xy, demands

    def _init_rendering(self):
        if self.render_mode is not None:
            import pygame as pygame

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

            self.scaled_xy = [(float(x * scaler) + left_margin, float(y * scaler) + top_margin) for x, y in self.xy]

            pygame.init()
            # Define font
            self.display_font = pygame.font.Font(None, 30)
            self.node_font = pygame.font.Font(None, 15)

            # Create screen
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), flags=pygame.HIDDEN)

    def get_reward(self):
        if self._is_done().all() or self.step_reward:
            visitng_idx = np.concatenate(self.visiting_seq, axis=2)  # (num_env, num_nodes)
            dist = cal_distance(self.xy, visitng_idx, axis=2)
            return dist

        else:
            return 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.training:
            self.xy, self.demand = self._make_problems(1, self.num_depots, self.num_nodes)

        else:
            self.xy, self.demand = self._load_problem()

        self.pos = None
        self.visited = np.zeros((self.num_env, self.pomo_size, self.action_size), dtype=bool)

        self.visiting_seq = []

        self.available = np.ones((self.num_env, self.pomo_size, self.action_size),
                                 dtype=bool)  # all nodes are available at the beginning

        self.load = np.ones((self.num_env, self.pomo_size, 1), dtype=np.float16)  # all vehicles start with full load
        self.t = 0

        obs = self._get_obs()  # this must come after resetting all the fields

        return obs, {}

    def _is_on_depot(self):
        return (self.pos == 0).squeeze(-1)

    def step(self, action):
        # action: (num_env, pomo_size)
        action = np.array([[[action]]]).astype(np.int64)

        # update the current pos
        self.pos = action

        # append the visited node idx
        self.visiting_seq.append(action)

        # check on depot
        on_depot = self._is_on_depot()
        # on_depot: (num_env, pomo_size, 1)

        # get the demands of the current node
        demand = np.take_along_axis(self.demand[:, None, :], self.pos, axis=2)
        # demand: (num_env, pomo_size, 1)

        # update load
        self.load -= demand

        # reload the vehicles that are o
        # depot
        # self.load = np.where(on_depot[:, :, None], self.load, 1)
        self.load[on_depot] = 1

        # update visited nodes
        # self.visited[action] = True
        np.put_along_axis(self.visited, action, True, axis=2)

        # depot is always set as not visited if the vehicle is not on the depot
        # here 0 is the depot idx
        self.visited[~on_depot, 0] = False
        # self.visited = np.where(~on_depot[:, :, None], self.visited, False)

        # assign avail to field
        self.available, done = self.get_avail_mask()

        reward = self.get_reward()

        info = {}

        self.t += 1

        obs = self._get_obs()

        return obs, reward, done, False, info

    def _is_done(self):
        # here 1 is depot
        done_flag = (self.visited[:, :, 1:] == True).all(axis=-1)
        return done_flag

    def get_avail_mask(self):
        # get a copy of avail
        avail = ~self.visited.copy()

        # mark unavail for nodes where the demands are larger than the current load
        unreachable = self.load + 1e-6 < self.demand[:, None, :]

        # mark unavail for nodes in which the demands cannot be fulfilled
        avail = avail & ~unreachable

        done = self._is_done()

        # for done episodes, set the depot as available
        avail[done, 0] = True

        return avail, done

    def render(self):
        if self.render_mode is None:
            return

        import pygame
        pygame.font.init()

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

            # Round up demands to 3 decimal points
            demand = np.round(self.demand[i], 3)

            # Draw node
            pygame.draw.circle(canvas, node_color, (x, y), self.node_size)

            # Draw demands text
            # demands = i  # for debugging

            if i == 0:  # DEPOT
                i = "D"
                demand = 1

            text_surface = self.node_font.render(f"{i}:{demand:.3f}", True, self.WHITE)
            text_rect = text_surface.get_rect(center=(x, y))
            canvas.blit(text_surface, text_rect)

        # Show current load
        load_text = self.display_font.render("Load: {:.3f}".format(float(self.load)), True, self.BLACK)
        load_rect = load_text.get_rect(
            topright=(self.screen_width - self.screen_width * 0.05, self.screen_height * 0.05))
        canvas.blit(load_text, load_rect)

        # Current distance cal
        self.step_reward = True
        reward = -self.get_reward()
        self.step_reward = False

        # current dist show
        dist_text = self.display_font.render("Distance: {:.3f}".format(float(reward)), True, self.BLACK)
        dist_rect = load_text.get_rect(topright=(self.screen_width - self.screen_width * 0.1, self.screen_height * 0.1))
        canvas.blit(dist_text, dist_rect)

        if self.render_mode == 'human':
            # Update screen
            self.screen.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.quit()

    def set_test_mode(self):
        self.training = False
