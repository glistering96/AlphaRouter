import time
from copy import  deepcopy
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo
from src.env.cvrp_gym import CVRPEnv
from src.mcts import MCTS
from src.module_base import RolloutBase, rollout_episode


class TesterModule(RolloutBase):
    def __init__(self, env_params, model_params, mcts_params, logger_params, run_params):
        # save arguments
        super().__init__(env_params, model_params, mcts_params, logger_params, run_params)
        global hparam_writer

        self.env = self.env_setup.create_env(test=True)

        self.start_epoch = 1
        self.best_score = float('inf')
        self.best_loss = float('inf')

        self.debug_epoch = 0

        self.min_reward = float('inf')
        self.max_reward = float('-inf')

        self._load_model(run_params['model_load']['epoch'])

        video_dir = self.run_params['logging']['result_folder_name'] + f'/videos/'
        self.test_env_with_vide = RecordVideo(self.env_setup.create_env(test=True), video_dir, name_prefix='test')

    def run(self):
        self.time_estimator.reset(self.epochs)
        global hparam_writer
        start = time.time()
        test_score = test_one_episode(self.env, self.model, self.mcts_params, 1)
        runtime = time.time() - start
        self.logger.info(f"Test score: {test_score: .5f}")
        self.logger.info(" *** Testing Done *** ")
        return test_score, runtime

    def record_video(self):
        test_one_episode(self.test_env_with_vide, self.model, self.mcts_params, 1)


def test_one_episode(env, agent, mcts_params, temp):
    env.set_test_mode()
    obs = env.reset()
    done = False
    agent.eval()
    debug = 0

    agent.encoding = None

    with torch.no_grad():
        while not done:
            mcts = MCTS(env, agent, mcts_params, training=False)
            action_probs = mcts.get_action_prob(obs, temp=temp)
            action = int(np.argmax(action_probs, -1))   # type must be python native

            next_state, reward, done, _, _ = env.step(action)

            obs = next_state
            debug += 1

            if done:
                return -reward