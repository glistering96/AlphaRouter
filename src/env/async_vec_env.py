import os
from typing import Callable

import numpy as np

from src.env.tsp_gymnasium import TSPEnv
import gymnasium as gym
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

import logging
import time

logger = logging.getLogger("async_debug")
logger.setLevel(logging.INFO)
simple_formatter = logging.Formatter("[%(name)s] %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(simple_formatter)

logger.addHandler(stream_handler)


class AsyncVecEnv:
    def __init__(self,
                 env_fns: Callable[[], gym.Env],
                 num_envs: int = 256,
                 env_kwargs=None,
                 num_processes:int = None,
                 ):
        self.num_processes = num_processes if not None else os.cpu_count()-2
        self.env_fns = env_fns
        self.num_envs = num_envs
        self.env_kwargs = env_kwargs

        self.envs = [env_fns(**env_kwargs) for _ in range(num_envs)]

        self.mp_type = 'p'

    def _reset(self, env_idx, seed=None):
        obs, _ = self.envs[env_idx].reset(seed=seed)
        return obs

    def _step(self, env_idx, action):
        try:
            obs, reward, done, _, info = self.envs[env_idx].step(action)
            return obs, reward, done, info

        except:
            print(self.envs[env_idx].available)
            print(f"env_idx: {env_idx}, action: {action}")

            raise Exception

    def reset(self, seed=None):
        if seed is not None:
            to_do = [(i, seed+i) for i in range(self.num_envs)]

        else:
            to_do = [(i, None) for i in range(self.num_envs)]

        if self.mp_type == 't':
            logger.debug(f"threading reset start")
            st = time.time()

            pool = ThreadPool(processes=self.num_processes)

            results = pool.starmap(self._reset, to_do)

            pool.close()
            pool.join()

            logger.debug(f"threading reset end, time: {time.time()-st}")

        else:
            logger.debug(f"Sequential reset start")
            st = time.time()

            results = [self._reset(*args) for args in to_do]

            logger.debug(f"Sequential reset end, time: {time.time()-st}")

        # stack the results into a single numpy array as obs variable
        obs = {}
        for key in results[0].keys():
            obs[key] = np.stack([result[key] for result in results])

        return obs

    def step(self, action: np.ndarray):
        # action: (num_envs, )
        if self.mp_type == 't':
            logger.debug(f"threading step start")
            st = time.time()

            pool = ThreadPool(processes=self.num_processes)

            results = pool.starmap(self._step, [(i, action[i]) for i in range(self.num_envs)])

            pool.close()
            pool.join()

            logger.debug(f"threading step end, time: {time.time()-st}")

        else:
            logger.debug(f"Sequential step start")
            st = time.time()

            results = [self._step(i, action[i]) for i in range(self.num_envs)]

            logger.debug(f"Sequential step end, time: {time.time()-st}")

        # results: [(obs, reward, done, debug), ...]
        obs = {}

        for key in results[0][0].keys():
            obs[key] = np.stack([result[0][key] for result in results])

        rewards = np.stack([result[1] for result in results])
        dones = np.stack([result[2] for result in results])

        return obs, rewards, dones, {}


def run_exp(mp_type, env_params, n_env):
    env = AsyncVecEnv(TSPEnv, env_kwargs=env_params, num_envs=n_env, num_processes=4)
    env.mp_type = mp_type

    logger.info(f"Exp using {mp_type}")

    st = time.time()
    env.reset(seed=None)

    done = False

    debug_action = 1

    while not done:
        action = np.array([debug_action for _ in range(n_env)])
        obs, rewards, dones, debugs = env.step(action)

        done = (dones == True).all()

        debug_action += 1

    logger.info(f"{mp_type} time: {time.time() - st}")



if __name__ == '__main__':
    env_params = {
        'num_nodes': 100,
        'num_depots': 1,
        'seed': 1,
        'step_reward': False,
        'env_type': 'tsp',
        'render_mode': None,

    }
    n_env = 512

    for mp_type in ['t', 's']:
        run_exp(mp_type, env_params, n_env)

    from gymnasium.vector import SyncVectorEnv

    env = SyncVectorEnv([lambda: TSPEnv(**env_params) for _ in range(n_env)])

    logger.info(f"Exp using SyncVectorEnv")

    st = time.time()
    env.reset()

    done = False

    debug_action = 1

    while not done:
        action = np.array([debug_action for _ in range(n_env)])
        obs, rewards, dones, _, debugs = env.step(action)

        done = (dones == True).all()

        debug_action += 1

    logger.info(f"SyncVectorEnv time: {time.time() - st}")

    from src.env.VecEnv import RoutingVecEnv
    from src.env.tsp_gym import TSPEnv as TSPGymEnv

    env = RoutingVecEnv([lambda: TSPGymEnv(**env_params) for _ in range(n_env)])

    logger.info(f"Exp using VecEnv")

    st = time.time()

    env.reset()

    done = False

    debug_action = 1

    while not done:
        action = np.array([debug_action for _ in range(n_env)])
        obs, rewards, dones, debugs = env.step(action)

        done = (dones == True).all()

        debug_action += 1

    logger.info(f"VecEnv time: {time.time() - st}")



