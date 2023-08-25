import os.path
import pickle

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Dict, Box, MultiBinary
from gymnasium.utils import seeding
from matplotlib import pyplot as plt

from src.common.data_manipulator import make_cord, make_demands
from src.common.utils import cal_distance


class CVRPEnv:
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

        # observation fields
        self.xy, self.demand, self.pos, self.visited = None, None, None, None
        self.visiting_seq = None
        self.load = None
        self.available = None

        self.t = 0

        self.test_data_type = kwargs.get('test_data_type')
        self._load_data_idx = kwargs.get('test_data_idx')

        self.num_env = 1
        self.pomo_size = 1

    def seed(self, seed):
        self._np_random, self._seed = seeding.np_random(seed)

    def _get_obs(self, load, pos, visited, visiting_seq, available, t):
        return {"load": load, "pos": pos, "visited": visited, "visiting_seq": visiting_seq,
                "available": available, "t": t}

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
                loaded_data = pickle.load(f)
                # loaded_data: [(depot_xy, node_xy, node_demand, capacity), (...), ...]
                depot_xy, node_xy, node_demand, capacity = loaded_data[self._load_data_idx]

                depot_xy = np.array(depot_xy, dtype=np.float32).reshape(-1, 2)
                node_xy = np.array(node_xy, dtype=np.float32)

                xy = np.concatenate([depot_xy, node_xy])

                node_demands = np.array(node_demand, dtype=np.float32) / capacity
                depot_demands = np.array([0.0 for _ in range(self.num_depots)], dtype=np.float32)

                demands = np.concatenate([depot_demands, node_demands])

            self._load_data_idx += 1

        else:
            raise ValueError(f"Invalid file extension for loading data: {ext}")

        if xy.ndim == 2:
            xy = xy.reshape(1, -1, 2)

        if demands.ndim < 2:
            demands = demands.reshape(1, -1)

        return xy, demands

    def _load_problem(self):
        if self.test_data_type == 'npz':
            file_path = f"{self.data_path}/cvrp/N_{self.test_num}.npz"

        else:
            file_path = f"{self.data_path}/cvrp/vrp{self.test_num}_test_seed1234.pkl"

        if os.path.isfile(file_path):
            xy, demands = self._load_data(file_path)

        else:
            xy = make_cord(1, self.num_depots, self.test_num)
            demands = make_demands(1, self.num_depots, self.test_num)

            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path, exist_ok=True)

            np.savez_compressed(file_path, xy=xy, demands=demands)

        return xy, demands

    def get_reward(self, visited, visiting_seq):
        if self.is_done(visited).all() or self.step_reward:
            visitng_idx = np.concatenate(visiting_seq, axis=2)  # (num_env, num_nodes)
            dist = cal_distance(self.xy, visitng_idx, axis=2)
            return float(dist.reshape(-1))

        else:
            return 0

    def reset(self):
        if self.training:
            self.xy, self.demand = self._make_problems(1, self.num_depots, self.num_nodes)

        else:
            self.xy, self.demand = self._load_problem()

        load = np.ones((self.num_env, self.pomo_size, 1), dtype=np.float16)  # all vehicles start with full load
        pos = None
        visited = np.zeros((self.num_env, self.pomo_size, self.action_size), dtype=bool)
        # visiting_seq = [pos[None, :, :]]
        visiting_seq = []
        available = np.ones((self.num_env, self.pomo_size, self.action_size),
                                 dtype=bool)  # all nodes are available at the beginning
        t = 0

        # np.put_along_axis(available, pos[None, :, :], False, axis=2)  # set the current pos to unavailable
        # np.put_along_axis(visited, pos[None, :, :], True, axis=2)  # set the current pos to visited

        obs = self._get_obs(load, pos, visited, visiting_seq, available, t)  # this must come after resetting all the fields

        return obs, {}

    def _is_on_depot(self, pos):
        return (pos == 0).squeeze(-1)

    @staticmethod
    def extract_obs(obs):
        return obs["load"], obs["pos"], obs["visited"], obs["visiting_seq"], obs["available"], obs["t"]

    def step(self, prev_obs, action):
        load, pos, visited, visiting_seq, available, t = self.extract_obs(prev_obs)

        # action: int
        action = np.array([[[action]]]).astype(np.int64)

        # update the current pos
        pos = action

        # append the visited node idx
        visiting_seq.append(action)

        # check on depot
        on_depot = self._is_on_depot(pos)
        # on_depot: (num_env, pomo_size, 1)

        # get the demands of the current node
        demand = np.take_along_axis(self.demand[:, None, :], pos, axis=2)
        # demand: (num_env, pomo_size, 1)

        # update load
        load -= demand

        # reload the vehicles that are o
        # depot
        # self.load = np.where(on_depot[:, :, None], self.load, 1)
        load[on_depot] = 1

        # update visited nodes
        # self.visited[action] = True
        np.put_along_axis(visited, action, True, axis=2)

        # depot is always set as not visited if the vehicle is not on the depot
        # here 0 is the depot idx
        visited[~on_depot, 0] = False
        # self.visited = np.where(~on_depot[:, :, None], self.visited, False)

        # assign avail to field
        available, done = self.get_avail_mask(visited, load)

        reward = self.get_reward(visited, visiting_seq)

        info = {}

        t += 1

        obs = self._get_obs(load, pos, visited, visiting_seq, available, t)

        return obs, reward, done, False, info

    def is_done(self, visited):
        # here 1 is depot
        done_flag = (visited[:, :, 1:] == True).all(axis=-1)
        return done_flag

    def get_avail_mask(self, visited, load):
        # get a copy of avail
        avail = ~visited.copy()

        # mark unavail for nodes where the demands are larger than the current load
        unreachable = load + 1e-6 < self.demand[:, None, :]

        # mark unavail for nodes in which the demands cannot be fulfilled
        avail = avail & ~unreachable

        done = self.is_done(visited)

        # for done episodes, set the depot as available
        avail[done, 0] = True

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

        plt.scatter(self.xy[0, 0, 0], self.xy[0, 0, 1], marker="*", c='green')  # depot node is green
        plt.scatter(self.xy[0, 1:, 0], self.xy[0, 1:, 1], c='black')

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
                demand = self.demand[0, node_id]
                plt.text(self.xy[0, node_id, 0], self.xy[0, node_id, 1], f"vc: {visit_count}, p: {priors[node_id]:.2f}, d: {demand:.2f}")

        if node_visit_count is None and priors is not None:
            for node_id, visit_count in priors.items():
                demand = self.demand[0, node_id]
                plt.text(self.xy[0, node_id, 0], self.xy[0, node_id, 1], f"p: {priors[node_id]:.2f}, d: {demand:.2f}")

        plt.savefig(f"{save_path}/{self.test_num}-{iteration}-{agent_type}.png")
        plt.show()