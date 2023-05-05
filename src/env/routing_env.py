from gymnasium.vector import SyncVectorEnv
from src.env.np_vec.cvrp_np_vec_env import CVRPNpVec
from src.env.np_vec.tsp_np_vec_env import TSPNpVec
from src.env.gymnasium.cvrp_gymnasium import CVRPEnv
from src.env.gymnasium.tsp_gymnasium import TSPEnv


class RoutingEnv:
    def __init__(self, env_params, run_params):
        self.env_params = env_params
        self.run_params = run_params

    def create_env(self, test=True):
        num_episode = self.run_params['num_episode']
        env_type = self.env_params['env_type']

        if test:
            if env_type == 'tsp':
                env = TSPEnv(**self.env_params)
            elif env_type == 'cvrp':
                env = CVRPEnv(**self.env_params)
            else:
                raise NotImplementedError

        else:
            if env_type == 'tsp':
                env = TSPNpVec(**self.env_params)

            elif env_type == 'cvrp':
                env = CVRPNpVec(**self.env_params)
            else:
                raise NotImplementedError

        return env