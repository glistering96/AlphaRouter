import torch

from src.env.torch_vec.torch_vec_base_ import TorchVecEnvBase


class CVRPTorchVec(TorchVecEnvBase):
    def __init__(self,
                 num_depots,
                 num_nodes,
                 num_env=128,
                 step_reward=False,
                 seed=None,
                 **kwargs):
        super(CVRPTorchVec, self).__init__(num_env, num_depots, num_nodes, step_reward, seed)
        self.num_depots = num_depots
        self.num_nodes = num_nodes
        self.step_reward = step_reward
        self.seed = seed
        self.env_type = 'cvrp'

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

    def _reset_individual(self):
        self.load = torch.ones((self.num_env, 1), dtype=torch.float32) # all vehicles start with full load

    def _is_on_depot(self):
        return (self.pos == 0).squeeze(-1)

    def step(self, action: torch.Tensor):
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

        # gather the demands of the current node from self.demand tensor
        demand = torch.gather(self.demand, 1, action)

        # update load
        self.load -= demand

        # reload the vehicles that are on depot
        self.load[on_depot, :] = 1

        # update visited nodes tensor
        self.visited[self.BATCH_IDX, action.squeeze(-1)] = True

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
        avail = ~self.visited.clone()

        # mark unavail for nodes where the demands are larger than the current load
        unreachable = self.load + 1e-6 < self.demand

        # mark unavail for nodes in which the demands cannot be fulfilled
        avail = avail & ~unreachable

        done = self._is_done()

        # for done episodes, set the depot as available
        avail[done, 0] = True

        return avail, done