from src.env.cvrp_gym import CVRPEnv
from src.env.tsp_gym import TSPEnv
from src.env.VecEnv import RoutingVecEnv
from stable_baselines3.common.env_util import make_vec_env


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
                env = make_vec_env(TSPEnv, n_envs=num_episode, env_kwargs=self.env_params,
                                   vec_env_cls=RoutingVecEnv)
            elif env_type == 'cvrp':
                env = make_vec_env(CVRPEnv, n_envs=num_episode, env_kwargs=self.env_params,
                                   vec_env_cls=RoutingVecEnv)
            else:
                raise NotImplementedError

        return env