import os.path
import pickle

import gymnasium as gym
import numpy as np
import pygame as pygame
from gymnasium.spaces import Discrete, Dict, Box, MultiBinary
from gymnasium.utils import seeding
from matplotlib import pyplot as plt

from src.common.data_manipulator import make_cord
from src.common.utils import cal_distance


class TSPEnv:
    def __init__(self, num_nodes,
                 step_reward=False, training=True, seed=None, data_path='./data', **kwargs):
        super(TSPEnv, self).__init__()
        self.num_nodes = num_nodes
        self.step_reward = step_reward
        self.training = training
        self.seed = seed
        self.data_path = data_path
        self.env_type = 'tsp'
        self.test_num = kwargs.get('test_num')
        self.action_size = self.num_nodes if self.test_num is None else self.test_num

        # observation fields
        self.xy, self.pos, self.visited = None, None, None
        self.visiting_seq = None
        self.available = None
        self.t = 0

        self.test_data_type = kwargs.get('test_data_type')
        self._load_data_idx = kwargs.get('test_data_idx')

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

    def get_reward(self, visited, visiting_seq):
        if self.is_done(visited) or self.step_reward:
            # batch_size, pomo_size = self.pos.shape
            visitng_idx = np.concatenate(visiting_seq, axis=2)
            # (num_env, pomo_size, num_nodes):
            dist = cal_distance(self.xy, visitng_idx, axis=2)
            return float(dist.reshape(-1))

        else:
            return 0

    def _get_obs(self, pos, visited, visiting_seq, available, t):
        return {"xy": self.xy, "pos": pos, "visited": visited, "visiting_seq": visiting_seq, "available": available, "t": t}

    def reset(self):
        if self.training:
            self.xy = self._make_problems(1, self.num_nodes)

        else:
            self.xy = self._load_problem()

        pos = None
        visited = np.zeros((1, 1, self.action_size), dtype=bool)

        visiting_seq = []

        available = np.ones((1, 1, self.action_size), dtype=bool)  # all nodes are available at the beginning

        t = 0

        obs = self._get_obs(pos, visited, visiting_seq, available, t)

        return obs, {}

    @staticmethod
    def extract_obs(obs):
        return obs["pos"], obs["visited"], obs["visiting_seq"], obs["available"], obs["t"]

    def step(self, prev_obs, action):
        pos, visited, visiting_seq, available, t = self.extract_obs(prev_obs)

        action = np.array([[[action]]])
        # action: (1, 1, 1)

        t += 1

        # update the current pos
        pos = action.reshape(1, 1)

        # append the visited node idx
        visiting_seq.append(action)

        # update visited nodes
        np.put_along_axis(visited, action, True, axis=2)

        # assign avail to field
        available, done = self.get_avail_mask(visited)

        reward = self.get_reward(visited, visiting_seq)

        info = {}

        obs = self._get_obs(pos, visited, visiting_seq, available, t)

        return obs, reward, done, False, info

    def is_done(self, visited):
        done_flag = (visited == True).all()
        return done_flag

    def get_avail_mask(self, visited):
        # get a copy of avail
        avail = ~visited.copy()

        done = self.is_done(visited)

        return avail, done

    def set_test_mode(self):
        self.training = False

    def plot(self, obs, node_visit_count=None, priors=None, iteration=None, agent_type=None, save_path=None):
        # set the figure size
        plt.figure(figsize=(10, 10))

        # plot problem with black point and plot the visiting sequence with red line
        visiting_seq = obs['visiting_seq']
        visiting_seq = np.concatenate(visiting_seq, axis=2)
        visiting_seq = visiting_seq.reshape(-1)

        plt.scatter(self.xy[0, :, 0], self.xy[0, :, 1], c='black')

        for visted_node in visiting_seq:
            plt.scatter(self.xy[0, visted_node, 0], self.xy[0, visted_node, 1], c='red')

        # draw a line between two nodes
        if len(visiting_seq) > 1:
            for i in range(len(visiting_seq) - 1):
                plt.plot([self.xy[0, visiting_seq[i], 0], self.xy[0, visiting_seq[i+1], 0]],
                         [self.xy[0, visiting_seq[i], 1], self.xy[0, visiting_seq[i+1], 1]], c='red')

        # denote the node visit count with text on the node
        if node_visit_count is not None:
            for node_id, visit_count in node_visit_count.items():
                plt.text(self.xy[0, node_id, 0], self.xy[0, node_id, 1], f"vc: {visit_count}, p: {priors[node_id]:.2f}")

        if node_visit_count is None and priors is not None:
            for node_id, visit_count in priors.items():
                plt.text(self.xy[0, node_id, 0], self.xy[0, node_id, 1], f"p: {priors[node_id]:.2f}")

        plt.savefig(f"{save_path}/{self.test_num}-{iteration}-{agent_type}.png")
        plt.show()