import os.path
import pickle
import random

import gymnasium as gym
import numpy as np
import pygame as pygame
from gymnasium.spaces import Discrete, Dict, Box, MultiBinary
from gymnasium.utils import seeding

from src.common.data_manipulator import make_cord
from src.common.utils import cal_distance


class TSPNpVec:
    def __init__(self, num_nodes,
                 step_reward=False, num_env=128, seed=None, data_path='./data', **kwargs):
        self.action_size = num_nodes
        self.num_nodes = num_nodes
        self.step_reward = step_reward
        self.training = True
        self.seed = seed
        self.data_path = data_path
        self.env_type = 'tsp'
        self.num_env = num_env

        # observation fields
        self.xy, self.pos, self.visited = None, None, None
        self.visiting_seq = None
        self.available = None
        self.t = 0

    def _get_obs(self):
        return {"xy": self.xy, "pos": self.pos, "available": self.available}

    def _make_problems(self, num_rollouts, num_nodes):
        xy = make_cord(num_rollouts, 0, num_nodes)

        if num_rollouts == 1:
            xy = xy.squeeze(0)

        return xy

    def get_reward(self):
        if self._is_done().all() or self.step_reward:
            visitng_idx = np.hstack(self.visiting_seq, dtype=int)   # (num_env, num_nodes)
            dist = cal_distance(self.xy, visitng_idx)
            return -dist

        else:
            return 0

    def reset(self):
        self.xy = self._make_problems(self.num_env, self.num_nodes)

        self.pos = np.zeros((self.num_env, 1), dtype=int)
        self.visited = np.zeros((self.num_env, self.action_size), dtype=bool)
        np.put_along_axis(self.visited, self.pos, True, axis=1) # set the current pos as visited

        self.visiting_seq = []
        self.load = np.ones((self.num_env, 1), dtype=np.float32) # all vehicles start with full load

        self.visiting_seq.append(self.pos) # append the depot position
        self.available = np.ones((self.num_env, self.action_size), dtype=bool) # all nodes are available at the beginning
        np.put_along_axis(self.available, self.pos, False, axis=1)  # set the current pos to False

        obs = self._get_obs()

        return obs, {}

    def step(self, action):
        # action: (num_env, 1)

        # check if the action is already visited
        for env_id, a in enumerate(action.squeeze(1)):
            # if the action is already in visit_seq from the env_id, raise an error
            if int(a) != 0 and self.visited[env_id, a] == True:
                raise ValueError(f'The selected action {a} is already visited in env {env_id}.'
                                 f' Visit seq for env {env_id}: {self.visited[env_id]}')

        # update the current pos
        self.pos = action

        # append the visited node idx
        self.visiting_seq.append(action)

        # update visited nodes
        np.put_along_axis(self.visited, action, True, axis=1)

        # assign avail to field
        self.available, done = self.get_avail_mask()

        reward = self.get_reward()

        info = {}

        self.t += 1

        obs = self._get_obs()

        return obs, reward, done, False, info

    def _is_done(self):
        done_flag = (self.visited == True).all(-1)
        return done_flag

    def get_avail_mask(self):
        # get a copy of avail
        avail = ~self.visited.copy()

        done = self._is_done()

        return avail, done


