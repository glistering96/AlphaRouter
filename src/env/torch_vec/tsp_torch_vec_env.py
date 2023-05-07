import torch

from src.env.torch_vec.torch_vec_base_ import TorchVecEnvBase


class TSPTorchVec(TorchVecEnvBase):
    def __init__(self, num_nodes,
                 step_reward=False, num_env=128, seed=None, **kwargs):
        super(TSPTorchVec, self).__init__(num_env, 0, num_nodes, step_reward, seed)
        # There are no depots for TSP
        self.action_size = num_nodes
        self.num_nodes = num_nodes
        self.num_depots = 0
        self.training = True
        self.env_type = 'tsp'

        # observation fields
        self.xy, self.pos, self.visited = None, None, None
        self.visiting_seq = None
        self.available = None
        self.t = 0

    def _get_obs(self):
        return {"xy": self.xy, "pos": self.pos, "available": self.available}

    def _reset_individual(self):
        # no need to reset anything in TSP
        pass

    def step(self, action: torch.Tensor):
        # action: (num_env, 1)
        if action.shape != (self.num_env, 1):
            action = action.reshape(self.num_env, 1)

        # update the current pos
        self.pos = action

        # append the visited node idx
        self.visiting_seq.append(action)

        # update visited nodes tensor
        self.visited[self.BATCH_IDX, action.squeeze(-1)] = True

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
        avail = ~self.visited.clone()

        done = self._is_done()

        return avail, done

