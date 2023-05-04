from gymnasium.vector import SyncVectorEnv

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
                env = SyncVectorEnv([lambda: TSPEnv(**self.env_params) for _ in range(num_episode)])

            elif env_type == 'cvrp':
                env = SyncVectorEnv([lambda: TSPEnv(**self.env_params) for _ in range(num_episode)])
            else:
                raise NotImplementedError

        return env