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

        self.start_epoch = 1
        self.best_score = float('inf')
        self.best_loss = float('inf')

        self.debug_epoch = 0

        self.min_reward = float('inf')
        self.max_reward = float('-inf')

        self._load_model(run_params['model_load']['path'])

    def _load_model(self, path):
        loaded = torch.load(path, map_location=self.device)

        self.model.load_state_dict(loaded['model_state_dict'])
        self.best_score = loaded['best_score']
        self.logger.info(f"Successfully loaded pre-trained policy_net from {path}")

    def _record_video(self, epoch):
        mode = "rgb_array"
        video_dir = self.run_params['logging']['result_folder_name'] + f'/videos/'
        data_path = self.run_params['data_path']

        env_params = deepcopy(self.env_params)
        env_params['render_mode'] = mode
        env_params['training'] = False
        env_params['seed'] = 5
        env_params['data_path'] = data_path

        env = CVRPEnv(**env_params)
        env = RecordVideo(env, video_dir, name_prefix=str(epoch))

        # render and interact with the environment as usual
        obs = env.reset()
        done = False

        with torch.no_grad():
            while not done:
                # env.render()
                mcts = MCTS(env, self.model, self.mcts_params)
                action_probs = mcts.get_action_prob(obs, temp=1)
                action = action_probs.argmax(-1)
                obs, reward, done, truncated, info = env.step(int(action))

        # close the environment and the video recorder
        env.close()
        return -reward

    def run(self):
        self.time_estimator.reset(self.epochs)
        global hparam_writer
        start = time.time()
        test_score = test_one_episode(self.env, self.model, self.mcts_params, 1)
        runtime = time.time() - start
        self.logger.info(f"Test score: {test_score: .5f}")
        self.logger.info(" *** Testing Done *** ")
        return test_score, runtime


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