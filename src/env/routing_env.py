from copy import deepcopy

from src.env.gymnasium.cvrp_gymnasium import CVRPEnv
from src.env.gymnasium.tsp_gymnasium import TSPEnv
from src.env.np_vec.cvrp_np_vec_env import CVRPNpVec
from src.env.np_vec.tsp_np_vec_env import TSPNpVec


class RoutingEnv:
    def __init__(self, env_params):
        self.env_params = env_params

    def create_env(self, test=True, **kwargs):
        num_episode = self.env_params['num_parallel_env']
        env_type = self.env_params['env_type']
        env_params = deepcopy(self.env_params)

        for k in kwargs:
            env_params[k] = kwargs[k]

        if test:
            if env_type == 'tsp':
                env = TSPEnv(**env_params)
            elif env_type == 'cvrp':
                env = CVRPEnv(**env_params)
            else:
                raise NotImplementedError

        else:
            if env_type == 'tsp':
                env = TSPNpVec(num_env=num_episode, **env_params)

            elif env_type == 'cvrp':
                env = CVRPNpVec(num_env=num_episode, **env_params)
            else:
                raise NotImplementedError

        return env
