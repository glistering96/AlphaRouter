import time

import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

from src.mcts import MCTS
from src.module_base import RolloutBase


class MCTSTesterModule(RolloutBase):
    def __init__(self, env_params, model_params, mcts_params, logger_params, run_params, dir_parser):
        # save arguments
        super().__init__(env_params, model_params, mcts_params, logger_params, run_params, dir_parser)
        global hparam_writer

        self.env = self.env_setup.create_env(test=True)
        load_epoch = run_params['model_load']['epoch']

        self._load_model(load_epoch)

        video_dir = self.result_folder + f'/videos/'
        self.test_env_with_vide = RecordVideo(self.env_setup.create_env(test=True, render_mode='rgb_array'), video_dir,
                                              name_prefix=f'test_on_{env_params["test_data_idx"]}_with_{load_epoch}')

    def run(self):
        self.time_estimator.reset(self.epochs)
        global hparam_writer

        test_score, runtime = test_one_episode(self.env, self.model, self.mcts_params, 1)

        self.logger.info(f"Test score: {test_score: .5f}")
        self.logger.info(" *** Testing Done *** ")

        self.record_video()

        return test_score, runtime

    def record_video(self):
        test_one_episode(self.test_env_with_vide, self.model, self.mcts_params, 1)


def test_one_episode(env, agent, mcts_params, temp):
    env.set_test_mode()
    obs, _ = env.reset()
    done = False
    agent.eval()
    debug = 0

    agent.encoding = None

    start = time.time()

    with torch.no_grad():
        while not done:
            mcts = MCTS(env, agent, mcts_params, training=False)
            action_probs = mcts.get_action_prob(obs, temp=temp)
            action = int(np.argmax(action_probs, -1))  # type must be python native

            next_state, reward, done, _, _ = env.step(action)

            obs = next_state
            debug += 1

            if done:
                return -reward, time.time() - start
