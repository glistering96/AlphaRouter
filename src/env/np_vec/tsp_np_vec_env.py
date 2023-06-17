import numpy as np

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
        
        self.pomo_size = self.num_nodes

    def _get_obs(self):
        return {"xy": self.xy, "pos": self.pos, "available": self.available, "t": self.t}

    def _make_problems(self, num_rollouts, num_nodes):
        xy = make_cord(num_rollouts, 0, num_nodes)

        if num_rollouts == 1:
            xy = xy.squeeze(0)

        return xy

    def get_reward(self):
        if self._is_done().all() or self.step_reward:
            batch_size, pomo_size = self.pos.shape
            visitng_idx = np.concatenate(self.visiting_seq, axis=2)
            # (num_env, pomo_size, num_nodes): 
            dist = cal_distance(self.xy, visitng_idx)
            return dist

        else:
            return 0

    def reset(self):
        self.xy = self._make_problems(self.num_env, self.num_nodes)

        self.pos = None
        self.visited = np.zeros((self.num_env, self.pomo_size, self.action_size), dtype=bool)

        self.visiting_seq = []

        self.available = np.ones((self.num_env, self.pomo_size, self.action_size),
                                 dtype=bool)  # all nodes are available at the beginning
        
        self.t = 0
        obs = self._get_obs()

        return obs, {}

    def step(self, action):
        action = action[:, :, None]
        # action: (num_env, pomo_size, 1)
        self.t += 1

        # update the current pos
        self.pos = action.reshape(self.num_env, self.pomo_size)

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


if __name__ == '__main__':
    tsp = TSPNpVec(20)
    