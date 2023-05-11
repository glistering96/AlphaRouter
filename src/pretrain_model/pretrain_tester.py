import time

import torch
from gymnasium.wrappers import RecordVideo

from src.module_base import RolloutBase


class AMTesterModule(RolloutBase):
    def __init__(self, env_params, model_params, logger_params, run_params, dir_parser):
        # save arguments
        super().__init__(env_params, model_params, None, logger_params, run_params, dir_parser)

        self.start_epoch = 1
        self.debug_epoch = 0

        self._load_model(run_params['model_load']['path'])
        self.env = self.env_setup.create_env(test=False)

    def _record_video(self, epoch):
        video_dir = self.result_folder + f'/videos/'

        env = RecordVideo(self.video_env, video_dir, name_prefix=str(epoch))

        # render and interact with the environment as usual
        obs = env.reset()
        done = False
        self.model.encoding = None

        with torch.no_grad():
            while not done:
                # env.render()
                action, _ = self.model.predict(obs)
                obs, reward, done, truncated, info = env.step(int(action))

        # close the environment and the video recorder
        env.close()
        return -reward

    def run(self):
        self.time_estimator.reset(self.epochs)

        test_score, runtime = test_one_episode(self.env, self.model)

        self.logger.info(f"Test score: {test_score: .5f}")

        self.logger.info(" *** Testing Done *** ")
        return test_score, runtime


def test_one_episode(env, agent):
    env.set_test_mode()
    obs = env.reset()
    done = False
    agent.eval()
    debug = 0
    agent.encoding = None

    start = time.time()

    with torch.no_grad():
        while not done:
            action_probs, _ = agent(obs)
            action = action_probs.argmax(-1).detach().item()  # type must be python native

            next_state, reward, done, _, _ = env.step(action)

            obs = next_state
            debug += 1

            if done:
                runtime = time.time() - start
                return -reward, runtime
