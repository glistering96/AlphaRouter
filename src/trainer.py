import random
from collections import deque
from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.multiprocessing import Pool
import torch.nn.functional as F
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.env_util import make_vec_env
from torch.optim import Adam as Optimizer
from torch.utils.tensorboard import SummaryWriter

from src.common.utils import add_hparams, check_debug, explained_variance, parse_saved_model_dir
from src.env.VecEnv import RoutingVecEnv
from src.env.cvrp_gym import CVRPEnv
from src.models.common_modules import get_batch_tensor
from src.module_base import RolloutBase, rollout_episode
from src.tester import test_one_episode

tb = None
hparam_writer = None


class TrainerModule(RolloutBase):
    def __init__(self, env_params, model_params, mcts_params, logger_params, optimizer_params, run_params,
                 h_params, args):
        # save arguments
        super().__init__(env_params, model_params, mcts_params, logger_params, run_params)
        global tb, hparam_writer

        self.optimizer_params = optimizer_params
        self.args = args
        logging_params = logger_params["log_file"]
        filename = logging_params['result_dir']
        tb_log_dir = logger_params['tb_log_dir']

        tb_log_path = f'{tb_log_dir}/{filename}/'
        tb_hparam_path = f'./hparams/{tb_log_dir}/{filename}/'

        tb = SummaryWriter(tb_log_path)
        hparam_writer = SummaryWriter(tb_hparam_path)

        self.model_load_elem = run_params['logging']['result_folder_name'].split('/')[:-1]

        self.hparam = h_params

        # policy_optimizer
        self.optimizer = Optimizer(self.model.parameters(), **optimizer_params)

        self.start_epoch = 1
        self.best_score = float('inf')
        self.best_loss = float('inf')
        self.current_lr = optimizer_params['lr']
        self.trainExamplesHistory = deque([], maxlen=500000)
        self.model_load_dir = None
        self.eval_score = float('inf')

        if Path('../data/mcts_train_data.pt').exists():
            self.trainExamplesHistory = torch.load('../data/mcts_train_data.pt')

        if run_params['model_load']['enable'] is True:
            self.model_load_dir = run_params['model_load']['path']
            self._load_model(run_params['model_load']['path'])

        self.debug_epoch = 0

        self.min_reward = float('inf')
        self.max_reward = float('-inf')

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
        self.model.encoding = None

        with torch.no_grad():
            while not done:
                # env.render()
                action, _ = self.model.predict(obs)
                obs, reward, done, truncated, info = env.step(int(action))

        # close the environment and the video recorder
        env.close()
        self.model.encoding = None
        return -reward

    def _evaluate(self):
        env_params = deepcopy(self.env_params)
        env_params['render_mode'] = None
        env_params['training'] = False
        env_params['seed'] = None

        train_score_lst = []

        for _ in range(10):
            env = CVRPEnv(**env_params)
            train_score = test_one_episode(env, self.model, self.mcts_params, 1)
            train_score_lst.append(train_score)

        return float(np.mean(train_score_lst))

    def run(self):
        self.time_estimator.reset(self.epochs)
        model_save_interval = self.run_params['logging']['model_save_interval']
        log_interval = self.run_params['logging']['log_interval']

        global tb, hparam_writer
        total_epochs = self.run_params['epochs']

        for epoch in range(self.start_epoch, total_epochs + 1):
            # Train
            # print("epochs ", epochs) # debugging

            score, total_loss, p_loss, val_loss, explained_var, epi_len = self._train_one_epoch(epoch)

            score = self._evaluate()

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, total_epochs)

            # for k, v in self.model.state_dict().items():
            #     print(f"{k}: {v.requires_grad}")

            ############################
            # Logs & Checkpoint
            ############################
            all_done = (epoch == total_epochs)
            to_compare_score = score

            updated = False
            logged = False

            eval_score = self._record_video(f"{epoch}")

            if self.eval_score > eval_score:
                prev_score = self.eval_score
                self.eval_score = eval_score
                self._save_checkpoints("rollout")
                self.model_load_dir = parse_saved_model_dir(self.args, self.args.result_dir, self.args.name_prefix,
                                                            mcts_param=True, load_epoch='rollout',
                                                            return_checkpoint=True)
                self.logger.info(f"Rollout policy changed. Prev: {prev_score:.4f}, Now: {self.eval_score:.4f}")

            if to_compare_score < self.best_score:
                # normal logging interval
                self.logger.info("Saving the best policy_net")
                self.best_score = to_compare_score
                self._save_checkpoints(epoch, is_best=True)
                self._log_info(epoch, score, total_loss, p_loss,
                               val_loss, elapsed_time_str, remain_time_str)
                logged = True

            if all_done or (epoch % model_save_interval) == 0:
                # when the best score is collected
                self.logger.info(f"Saving the trained policy_net. Current lr: {self.current_lr}")
                self._save_checkpoints(epoch, is_best=False)

                if not logged:
                    self._log_info(epoch, score, total_loss, p_loss,
                               val_loss, elapsed_time_str, remain_time_str)

            elif epoch % log_interval == 0:
                # logging interval
                if not logged:
                    self._log_info(epoch, score, total_loss, p_loss, val_loss, elapsed_time_str, remain_time_str)

            # self._save_checkpoints("last", is_best=False)
            tb.add_scalar('score/train_score', score, epoch)
            tb.add_scalar('score/episode_length', epi_len, epoch)
            tb.add_scalar('loss/total_loss', total_loss, epoch)
            tb.add_scalar('loss/p_loss', p_loss, epoch)
            tb.add_scalar('loss/val_loss', val_loss, epoch)
            tb.add_scalar('loss/explained_var', explained_var, epoch)

            add_hparams(hparam_writer, self.hparam, {'train_score': score, 'best_score': self.best_score},
                        epoch)

            self.debug_epoch += 1

            # All-done announcement
            if all_done:
                tb.flush()
                tb.close()
                self.logger.info(" *** Training Done *** ")

    def _set_lr(self):
        self.current_lr = self.current_lr * 0.9999

        self.optimizer.param_groups[0]["lr"] = self.current_lr

    def _train_one_epoch(self, epoch):
        # train for one epoch.
        # In one epoch, the policy_net trains over given number of scenarios from tester parameters
        # The scenarios are trained in batched.
        num_episodes = self.run_params['num_episode']

        self._set_lr()

        iterationTrainExamples = []

        num_cpus = self.run_params['num_proc']

        if check_debug():
            num_cpus = 2

        pool = Pool(processes=num_cpus)
        params = [(CVRPEnv(**self.env_params), self.model_load_dir, self.model_params,
                   self.device, self.mcts_params) for _ in range(num_episodes)]

        result = pool.starmap_async(rollout_episode, params)

        pool.close()
        pool.join()

        for r in result.get():
            data_chunk = r
            reward = data_chunk[0][-1] # 0 th reward

            if reward < self.min_reward:
                self.min_reward = reward

            if reward > self.max_reward:
                self.max_reward = reward

            iterationTrainExamples += data_chunk

        self.trainExamplesHistory.extend(iterationTrainExamples)

        self.logger.info(
            f"Simulating episodes done on {num_episodes}. Number of data is {len(self.trainExamplesHistory)}")

        reward, total_loss, pi_loss, v_loss, explained_var, epi_len_total = self._train_model(self.trainExamplesHistory, epoch)
        # reward, total_loss, pi_loss, v_loss, explained_var = 0,0,0,0,0

        return reward, total_loss, pi_loss, v_loss, explained_var, epi_len_total

    def _train_model(self, examples, epoch):
        # trainExamples: [(obs, action_prob_dist, reward)]
        self.model.train()
        batch_size = min(len(examples), self.run_params['mini_batch_size'])
        train_epochs = self.run_params['train_epochs']

        t_losses = 0
        pi_losses = 0
        v_losses = 0
        rewards = 0
        epi_len_total = 0
        num_observations = 0

        train_epochs = min([max(10, epoch), train_epochs])
        exp_var_rewards = []
        exp_var_values = []
        # train_epochs = 1

        for epoch in range(train_epochs):
            batch_from = 0
            remaining = len(examples)
            batch_idx = list(range(len(examples)))
            random.shuffle(batch_idx)

            while remaining > 0:
                B = min(batch_size, remaining)
                selected_batch_idx = batch_idx[batch_from:batch_from + B]
                obs, policy, reward, epi_len = list(zip(*[examples[i] for i in selected_batch_idx]))

                obs_batch_tensor = get_batch_tensor(obs)

                target_probs = torch.tensor(policy, dtype=torch.float32, device=self.device).squeeze(1).detach()
                # (B, num_vehicles)

                target_reward = -torch.tensor(reward, dtype=torch.float32, device=self.device).detach()
                # (B, )

                # target_reward = (target_reward - self.min_reward) / (self.max_reward - self.min_reward + 1e-8)

                # compute output
                self.model.encoding = self.model.encoder(obs_batch_tensor['xy'],
                                                         obs_batch_tensor['demands'].unsqueeze(-1))
                out_pi, out_v = self.model(obs_batch_tensor)

                l_pi = F.cross_entropy(out_pi, target_probs)
                l_v = 0.5*F.mse_loss(out_v, target_reward)
                loss = l_pi + l_v + 0.001 * self.l2()

                # record loss
                t_losses += loss.item() * B
                rewards += sum(reward)
                pi_losses += l_pi.item() * B
                v_losses += l_v.item() * B
                epi_len_total += sum(epi_len)

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                remaining -= B
                batch_from += B
                num_observations += B

                exp_var_rewards += reward
                exp_var_values += out_v.detach().cpu().view(-1,).tolist()

        rewards /= num_observations
        t_losses /= num_observations
        pi_losses /= num_observations
        v_losses /= num_observations
        epi_len_total /= num_observations

        explained_var = explained_variance(exp_var_values, exp_var_rewards)

        del loss, out_pi, out_v, l_v, l_pi

        return -rewards, t_losses, pi_losses, v_losses, explained_var, epi_len_total

    def l2(self):
        l2_reg = torch.tensor(0., device=self.device)

        for param in self.model.parameters():
            l2_reg += torch.norm(param)

        return l2_reg
