from copy import deepcopy
from typing import List, Callable
from collections import OrderedDict

import gym
import gymnasium
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs
from stable_baselines3.common.vec_env.util import obs_space_info


class CVRPVecEnv(DummyVecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        super(CVRPVecEnv, self).__init__(env_fns)

    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            if not self.buf_dones[env_idx]:
                obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                    self.actions[env_idx]
                )

            else:
                obs = self.buf_infos[env_idx]["terminal_observation"]

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                # obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def reset(self) -> VecEnvObs:
        obs_space = self.envs[0].observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict(
            [(k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = self.envs[0].metadata

        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)

        return self._obs_from_buf()

    def set_test_mode(self):
        for env_idx in range(self.num_envs):
            self.envs[env_idx].set_test_mode()

