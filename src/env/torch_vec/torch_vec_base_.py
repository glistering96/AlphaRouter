import torch


class TorchVecEnvBase:
    def __init__(self, num_env, num_nodes, num_depots, step_reward, seed):
        self.num_env = num_env
        self.num_nodes = num_nodes
        self.num_depots = num_depots
        self.step_reward = step_reward

        self.seed = seed
        self.training = True
        self.action_size = num_nodes + num_depots

        self.xy = None
        self.visiting_seq = []
        self.BATCH_IDX = range(self.num_env)

    @staticmethod
    def _make_problems(num_rollouts, num_depots, num_nodes):
        # make xy torch tensor
        xy = torch.rand((num_rollouts, num_depots + num_nodes, 2))

        # make demand torch tensor
        depot_demands = torch.zeros((num_rollouts, num_depots))

        if num_nodes == 20 or num_nodes == 10:
            demand_scaler = 30
        elif num_nodes == 50:
            demand_scaler = 40
        elif num_nodes == 100:
            demand_scaler = 50
        else:
            raise NotImplementedError

        node_demands = torch.randint(1, 10, size=(num_rollouts, num_nodes)) / demand_scaler
        demands = torch.concatenate([depot_demands, node_demands], dim=1)

        return xy, demands

    @staticmethod
    def cal_distance(xy, visiting_seq):
        """
        :param xy: coordinates of nodes
        :param visiting_seq: sequence of visiting node idx
        :return:

        1. Gather coordinates on a given sequence of nodes
        2. roll by -1
        3. calculate the distance
        4. return distance

        """
        desired_shape = tuple(list(visiting_seq.shape) + [2])
        gather_idx = torch.broadcast_to(visiting_seq[:, :, None], desired_shape)

        original_seq = torch.gather(xy, 1, gather_idx)
        rolled_seq = torch.roll(original_seq, -1, 1)

        segments = torch.sqrt(((original_seq - rolled_seq) ** 2).sum(-1))
        # segments: (num_env, num_nodes)
        distance = segments.sum(1).to(torch.float16)
        return distance

    def _reset_common(self):
        self.xy, self.demand = self._make_problems(self.num_env, self.num_depots, self.num_nodes)

        self.pos = torch.zeros((self.num_env, 1), dtype=int)
        self.visited = torch.zeros((self.num_env, self.action_size), dtype=bool)

        # set the current pos as visited
        self.visited[self.BATCH_IDX, self.pos.squeeze(-1)] = True # (num_env, num_nodes)

        self.visiting_seq = []  # list of visiting sequence

        self.visiting_seq.append(self.pos) # append the depot position
        self.available = torch.ones((self.num_env, self.action_size), dtype=bool) # all nodes are available at the beginning
        # set the current pos to False
        self.available[self.BATCH_IDX, self.pos.squeeze(-1)] = False

    def _get_obs(self):
        raise NotImplementedError

    def _reset_individual(self):
        raise NotImplementedError("Implement this method in the child class. If no special reset is needed, just pass.")

    def reset(self):
        self._reset_common()
        self._reset_individual()
        return self._get_obs(), {}

    def step(self, actions):
        raise NotImplementedError

    def _is_done(self):
        raise NotImplementedError

    def get_reward(self):
        if self._is_done().all() or self.step_reward:
            visitng_idx = torch.hstack(self.visiting_seq)   # (num_env, num_nodes)
            dist = self.cal_distance(self.xy, visitng_idx)
            return -dist

        else:
            return 0