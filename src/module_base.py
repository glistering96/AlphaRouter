from copy import deepcopy
from logging import getLogger
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecNormalize

from src.common.dataclass import rollout_result
from src.common.utils import TimeEstimator, deepcopy_state, get_result_folder
from src.env.cvrp_gym import CVRPEnv as Env, CVRPEnv
from src.env.routing_env import RoutingEnv
from src.env.tsp_gym import TSPEnv
from src.models.cvrp_model.models import CVRPModel
from src.mcts import MCTS
from src.models.routing_model import RoutingModel


class RolloutBase:
    def __init__(self,
                 env_params,
                 model_params,
                 mcts_params,
                 logger_params,
                 run_params):
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.run_params = run_params
        self.logger_params = logger_params
        self.mcts_params = mcts_params

        # cuda
        USE_CUDA = self.run_params['use_cuda']

        self.logger = getLogger(name='trainer')
        self.result_folder = self.run_params['logging']['result_folder_name']

        Path(self.result_folder).mkdir(parents=True, exist_ok=True)

        if USE_CUDA:
            cuda_device_num = self.run_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        self.device = device

        # Env
        self.env_setup = RoutingEnv(self.env_params, self.run_params)
        self.video_env = self.init_test_env()

        # Model
        self.model_params['device'] = device

        self.model = RoutingModel(self.model_params, self.env_params).create_model(self.env_params['env_type'])

        # etc.
        self.epochs = 1
        self.best_score = float('inf')
        self.time_estimator = TimeEstimator()

    def _save_checkpoints(self, epoch, is_best=False):
        file_name = 'best' if is_best else epoch

        checkpoint_dict = {
            'epoch': epoch,
            'model_params': self.model_params,
            'best_score': self.best_score,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        path = f"{self.result_folder}/saved_models"

        if not Path(path).exists():
            Path(path).mkdir(parents=True, exist_ok=False)

        torch.save(checkpoint_dict, '{}/saved_models/checkpoint-{}.pt'.format(self.result_folder, file_name))

    def _load_model(self, model_load):
        checkpoint = torch.load(model_load, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1

        self.best_score = checkpoint['best_score']

        loaded_state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(loaded_state_dict)

        self.logger.info(f"Successfully loaded pre-trained policy_net from {model_load}")

    def _log_info(self, epoch, train_score, total_loss, p_loss, val_loss, elapsed_time_str,
                  remain_time_str):

        self.logger.info(
            f'Epoch {epoch:3d}: Score: {train_score:.4f}, total_loss: {total_loss:.4f}, p_loss: {p_loss:.4f}, '
            f'val_loss: {val_loss:.4f}, Best: {self.best_score:.4f}')

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            epoch, self.run_params['epochs'], elapsed_time_str, remain_time_str))
        self.logger.info('=================================================================')

    def init_test_env(self):
        env_params = deepcopy(self.env_params)
        env_params['training'] = False
        env_params['seed'] = 5
        env_params['data_path'] = self.run_params['data_path']

        if env_params['env_type'] == 'cvrp':
            env = CVRPEnv(**env_params)

        elif env_params['env_type'] == 'tsp':
            env = TSPEnv(**env_params)

        else:
            raise NotImplementedError

        return env

    def run(self):
        # abstract method
        raise NotImplementedError


def get_model(model_params):
    nn = model_params['nn']

    if nn == 'shared_mha':
        return CVRPModel(**model_params)

    else:
        raise ValueError(f"Unsupported model: {nn}")


def load_model(model_load, agent_params, device):
    model = get_model(agent_params).to(device)

    if model_load is not None:
        checkpoint = torch.load(model_load, map_location=device)
        loaded_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(loaded_state_dict)

    return model


def rollout_episode(env, agent_load_dir, agent_params, device, mcts_params):
    obs = env.reset()

    agent = load_model(agent_load_dir, agent_params, device)
    buffer = []
    done = False

    # episode rollout
    # gather probability of the action and value estimates for the state
    debug = 0
    agent.eval()

    with torch.no_grad():
        while not done:
            mcts = MCTS(env, agent, mcts_params)
            temp = 1 if debug < mcts_params['temp_threshold'] else 0
            action_probs = mcts.get_action_prob(obs, temp=temp)
            action = np.random.choice(len(action_probs), p=action_probs)

            buffer.append((obs, action_probs))

            next_state, reward, done, _ = env.step(action)

            obs = next_state

            debug += 1

            if done:
                result = [(x[0], x[1], float(reward), debug) for x in buffer]
                return result