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
                env = TSPNpVec(num_env=num_episode, **self.env_params)

            elif env_type == 'cvrp':
                env = CVRPNpVec(num_env=num_episode, **self.env_params)
            else:
                raise NotImplementedError

        return env




if __name__ == '__main__':
    env_params = {
        'env_type': 'cvrp',
        'num_nodes': 5,
        'num_depots': 1,

    }
    num_episode= 4

    env = SyncVectorEnv([lambda: CVRPEnv(**env_params) for _ in range(num_episode)])

    obs, _ = env.reset()

    all_done = False

    while not all_done:
        mask = tuple(obs['available'][i] for i in range(num_episode))
        action = env.action_space.sample(mask=mask)
        obs, reward, done, _, info = env.step(action)
        all_done = done.all()
        # print(obs, reward, done, info)
