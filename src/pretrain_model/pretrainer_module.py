import json

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.wrappers import RecordVideo
from torch.optim import Adam as Optimizer
from torch.utils.tensorboard import SummaryWriter

from src.common.lr_scheduler import CosineAnnealingWarmupRestarts
from src.module_base import RolloutBase

tb = None

import warnings


class PreTrainerModule(RolloutBase):
    def __init__(self, env_params, model_params, logger_params, optimizer_params, run_params, dir_parser):
        super(PreTrainerModule, self).__init__(env_params, model_params, None, logger_params, run_params, dir_parser)
        # save arguments
        global tb

        self.optimizer_params = optimizer_params
        tb_log_path = self.dir_parser.get_tensorboard_logging_dir()

        tb = SummaryWriter(tb_log_path)

        # env
        self.env = self.env_setup.create_env(test=False)

        # policy_optimizer
        self.optimizer = Optimizer(self.model.parameters(), **optimizer_params)

        self.scaler = torch.cuda.amp.GradScaler()

        self.start_epoch = 1
        self.best_score = float('inf')
        self.best_loss = float('inf')
        self.ent_coef = run_params['ent_coef']
        self.current_lr = optimizer_params['lr']

        if run_params['model_load']['enable'] is True:
            self._load_model(run_params['model_load']['epoch'])

        self.debug_epoch = 0

        self.min_reward = float('inf')
        self.max_reward = float('-inf')

        warmup_epochs = 2000
        first_cycle_steps = (run_params['nn_train_epochs'] + 1 - self.start_epoch)
        self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer,
                                                       first_cycle_steps=first_cycle_steps,
                                                       warmup_steps=warmup_epochs,
                                                       min_lr=optimizer_params['lr'] / 10,
                                                       max_lr=optimizer_params['lr']
                                                       )

    def _record_video(self, epoch):
        video_dir = self.result_folder + f'/videos/'

        env = RecordVideo(self.video_env, video_dir, name_prefix=str(epoch))

        # render and interact with the environment as usual
        obs, _ = env.reset()
        done = False
        self.model.encoding = None

        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16):
            while not done:
                # env.render()
                action, _ = self.model.predict(obs)
                obs, reward, done, truncated, info = env.step(int(action))

        # close the environment and the video recorder
        env.close()
        return -reward

    def get_valid_score(self):
        try:
            with open(f"{self.result_folder}/run_result.json", "r") as f:
                valid_result = json.load(f)
                self.logger.info("Previous result json record loaded.")
                return valid_result

        except json.JSONDecodeError:
            self.logger.info("Previous result json could not be loaded.")
            # raise FileNotFoundError
            raise json.JSONDecodeError

        except FileNotFoundError:
            self.logger.info("Previous result json file does not exist. Working on the new file")
            return {}

    def run(self):
        self.time_estimator.reset(self.epochs)
        model_save_interval = self.run_params['logging']['model_save_interval']
        log_interval = self.run_params['logging']['log_interval']

        global tb
        total_epochs = self.run_params['nn_train_epochs']
        valid_scores = self.get_valid_score()

        # self._record_video(0)

        for epoch in range(self.start_epoch, total_epochs + 1):
            # Train
            # print("epochs ", epochs) # debugging

            train_score, loss, p_loss, val_loss, epi_len, entropy = self._train_one_epoch()

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, total_epochs)

            ############################
            # Logs & Checkpoint
            ############################
            all_done = (epoch == total_epochs)
            to_compare_score = train_score
            logged = False

            if val_loss < self.best_loss:
                self.logger.info(f"Saving the best loss network. Prev: {self.best_loss: .4f}, Current: {val_loss: .4f}")
                self.best_loss = val_loss
                self._save_checkpoints("best_val_loss")

            if to_compare_score < self.best_score:
                # normal logging interval
                self.logger.info("Saving the best policy_net")
                self.best_score = to_compare_score
                self._save_checkpoints(epoch, is_best=True)

                # with warnings.catch_warnings():
                #     warnings.simplefilter("ignore")
                #     score = self._record_video(f"best")

                # valid_scores['best'] = round(score, 4)

                if not (all_done or (epoch % model_save_interval) == 0):
                    self._log_info(epoch, train_score, loss, p_loss, val_loss, elapsed_time_str, remain_time_str)
                    logged = True

            if all_done or (epoch % model_save_interval) == 0:
                # when the best score is collected
                self.logger.info(f"Saving the trained policy_net. Current lr: {self.current_lr: .5f}")
                self._save_checkpoints(epoch, is_best=False)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    score = self._record_video(f"{epoch}")

                valid_scores[epoch] = round(score, 4)
                self._log_info(epoch, train_score, loss, p_loss, val_loss, elapsed_time_str, remain_time_str)

            elif epoch % log_interval == 0:
                # logging interval
                if not logged:
                    self._log_info(epoch, train_score, loss, p_loss, val_loss, elapsed_time_str, remain_time_str)

            try:
                with open(f"{self.result_folder}/run_result.json", "w") as f:
                    json.dump(valid_scores, f, indent=4)

            except:
                pass

            # self._save_checkpoints("last", is_best=False)
            tb.add_scalar('score/train_score', train_score, epoch)
            tb.add_scalar('score/episode_length', epi_len, epoch)
            tb.add_scalar('loss/total_loss', loss, epoch)
            tb.add_scalar('loss/p_loss', p_loss, epoch)
            tb.add_scalar('loss/val_loss', val_loss, epoch)
            tb.add_scalar('loss/entropy', entropy, epoch)
            tb.add_scalar('debug/lr', self.current_lr, epoch)

            self.debug_epoch += 1

            # All-done announcement
            if all_done:
                tb.flush()
                tb.close()
                self.logger.info(" *** Training Done *** ")

    def _train_one_epoch(self):
        # train for one epoch.
        # In one epoch, the policy_net trains over given number of scenarios from tester parameters
        # The scenarios are trained in batched.
        done = False
        self.model.encoding = None

        obs, _ = self.env.reset()
        prob_lst = []
        entropy_lst = []
        val_lst = []
        reward = 0

        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            while not done:
                action_probs, val = self.model(obs)

                probs = torch.distributions.Categorical(probs=action_probs)
                action = probs.sample()

                obs, reward, dones, _, _ = self.env.step(action.detach().cpu().numpy())

                done = bool(np.all(dones == True))

                prob_lst.append(probs.log_prob(action)[:, None])
                entropy_lst.append(probs.entropy()[:, None])
                val_lst.append(val[:, None])

            reward = -torch.as_tensor(reward, device=self.device, dtype=torch.float16)
            val_tensor = torch.cat(val_lst, dim=-1)
            # val_tensor: (batch, time)
            baseline = val_tensor
            adv = torch.broadcast_to(reward[:, None], val_tensor.shape) - baseline

            log_prob = torch.cat(prob_lst, dim=-1)
            p_loss = (adv.detach() * log_prob).sum(dim=-1).mean()
            entropy = -torch.cat(entropy_lst, dim=-1).mean()

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore")  # broad casting below line is intended. It is much faster than manual calculation
                val_loss = 0.5 * F.mse_loss(val_tensor, reward.unsqueeze(-1).detach())

            loss = p_loss + val_loss + self.ent_coef * entropy

        self.scaler.scale(loss).backward()  # scaled loss backward call first to create scaled gradients
        self.scaler.step(self.optimizer)  # scaler step call
        self.scaler.update()  # scaler update call

        self.optimizer.zero_grad()
        self.scheduler.step()

        self.current_lr = self.optimizer.param_groups[0]['lr']
        # self.model.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # self.scheduler.step()

        return reward.mean().item(), loss, p_loss, val_loss, len(prob_lst), entropy
