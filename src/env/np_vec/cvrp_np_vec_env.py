import numpy as np

from src.common.data_manipulator import make_cord, make_demands
from src.common.utils import cal_distance


class CVRPNpVec:
    def __init__(self,
                 num_depots,
                 num_nodes,
                 num_env=128,
                 step_reward=False,
                 seed=None,
                 data_path='./data',
                 **kwargs):
        self.action_size = num_nodes + num_depots
        self.num_depots = num_depots
        self.num_nodes = num_nodes
        self.step_reward = step_reward
        self.training = True
        self.seed = seed
        self.data_path = data_path
        self.env_type = 'cvrp'
        self.num_env = num_env

        # observation fields
        self.xy, self.demand, self.pos, self.visited = None, None, None, None
        self.visiting_seq = None
        self.load = None
        self.available = None

        self.t = 0

        self.test_data_type = kwargs.get('test_data_type')
        self._load_data_idx = 0

    def _get_obs(self):
        return {"xy": self.xy, "demands": self.demand, "pos": self.pos, "load": self.load, "available": self.available}

    def _make_problems(self, num_rollouts, num_depots, num_nodes):
        xy = make_cord(num_rollouts, num_depots, num_nodes)
        demands = make_demands(num_rollouts, num_depots, num_nodes)

        if num_rollouts == 1:
            xy = xy.squeeze(0)
            demands = demands.squeeze(0)

        return xy, demands

    def get_reward(self):
        if self._is_done().all() or self.step_reward:
            visitng_idx = np.hstack(self.visiting_seq, dtype=int)   # (num_env, num_nodes)
            dist = cal_distance(self.xy, visitng_idx)
            return -dist

        else:
            return 0

    def reset(self):
        self.xy, self.demand = self._make_problems(self.num_env, self.num_depots, self.num_nodes)

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

    def _is_on_depot(self):
        return (self.pos == 0).squeeze(-1)

    def step(self, action):
        # action: (num_env, 1)
        if action.shape != (self.num_env, 1):
            action = action.reshape(self.num_env, 1)

        # update the current pos
        self.pos = action

        # append the visited node idx
        self.visiting_seq.append(action)

        # check on depot
        on_depot = self._is_on_depot()
        # on_depot: (num_env, 1)

        # get the demands of the current node
        demand = np.take_along_axis(self.demand, self.pos, axis=1)

        # update load
        self.load -= demand

        # reload the vehicles that are on depot
        self.load[on_depot, :] = 1

        # update visited nodes
        np.put_along_axis(self.visited, action, True, axis=1)

        # depot is always set as not visited if the vehicle is not on the depot
        self.visited[~on_depot, 0] = False

        # assign avail to field
        self.available, done = self.get_avail_mask()

        reward = self.get_reward()

        info = {}

        self.t += 1

        obs = self._get_obs()

        return obs, reward, done, False, info

    def _is_done(self):
        done_flag = (self.visited[:, :] == True).all(axis=1)
        return done_flag

    def get_avail_mask(self):
        # get a copy of avail
        avail = ~self.visited.copy()

        # mark unavail for nodes where the demands are larger than the current load
        unreachable = self.load + 1e-6 < self.demand

        # mark unavail for nodes in which the demands cannot be fulfilled
        avail = avail & ~unreachable

        done = self._is_done()

        # for done episodes, set the depot as available
        avail[done, 0] = True

        # if (avail == False).all(-1).all():
        #     print(avail)

        # for available nodes, if the node is already visited, raise an error
        for env_id, a in enumerate(avail):
            for idx, availability in enumerate(a):
                if idx != 0 and self.visited[env_id, idx] is True and availability is True:
                    raise ValueError(f'The available node {idx} is already visited in env {env_id}.'
                                     f' Visit seq for env {env_id}: {self.visiting_seq[env_id]}')

        return avail, done