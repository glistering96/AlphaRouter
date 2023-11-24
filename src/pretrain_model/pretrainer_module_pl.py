import lightning.pytorch as pl
import numpy as np
import torch
from torch.optim import Adam as Optimizer

from src.common.lr_scheduler import CosineAnnealingWarmupRestarts
from src.env.routing_env import RoutingEnv
from src.models.routing_model import RoutingModel


class AMTrainer(pl.LightningModule):
    def __init__(self, env_params, model_params, optimizer_params, run_params, config=None):
        super(AMTrainer, self).__init__()

        if config is not None:
            for key, value in config.items():
                for params in [env_params, model_params, optimizer_params, run_params]:
                    if key in params:
                        setattr(params, key, value)

        # save arguments
        self.optimizer_params = optimizer_params

        # model
        self.model = RoutingModel(model_params, env_params).create_model(env_params['env_type'])

        # env
        self.env = RoutingEnv(env_params).create_env(test=False)

        # etc
        self.ent_coef = run_params['ent_coef']
        self.nn_train_epochs = run_params['nn_train_epochs']

        self.baseline = run_params['baseline']

    def training_step(self, batch, _):
        # train for one epoch.
        # In one epoch, the policy_net trains over given number of scenarios from tester parameters
        # The scenarios are trained in batched.
        done = False
        self.model.encoding = None
        self.model.device = self.device

        obs, _ = self.env.reset()
        prob_lst = []
        entropy_lst = []
        val_lst = []
        reward = 0
                
        while not done:
            action_probs, val = self.model(obs)
            # action_probs: (batch, pomo, N)
            # val: (batch, pomo, 1)

            dist = torch.distributions.Categorical(probs=action_probs)
            action = dist.sample()
            entropy_lst.append(dist.entropy()[:, :, None])

            logit = dist.log_prob(action)[:, :, None]
            # (batch, pomo, 1)

            obs, reward, dones, _, _ = self.env.step(action.detach().cpu().numpy())

            done = bool(np.all(dones == True))

            prob_lst.append(logit)
            val_lst.append(val)

        reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
        # (batch, pomo)

        val_tensor = torch.cat(val_lst, dim=-1)
        # val_tensor: (batch, pomo, T)

        log_prob = torch.cat(prob_lst, dim=-1).sum(dim=-1, keepdim=True)
        # (batch, pomo)

        reward_broadcasted = torch.broadcast_to(reward[:, :, None], val_tensor.shape)
        val_loss = torch.nn.functional.mse_loss(val_tensor, reward_broadcasted)

        if self.baseline == 'val':
            baseline = val_tensor
            advantage = reward_broadcasted - baseline.detach()
            # advantage: (batch, pomo, T)
            p_loss = advantage * log_prob.expand_as(advantage)

        else:
            baseline = reward.mean(dim=1, keepdim=True)
            # shape: (batch, 1)
            advantage = reward - baseline.detach()
            # shape: (batch, pomo)
            p_loss = advantage * log_prob.squeeze(-1)
            # shape: (batch, pomo)

        loss = p_loss + 0.5 * val_loss 
        loss = loss.mean()
        entropy = -torch.cat(entropy_lst, dim=-1).mean()
        min_pomo_reward, _ = reward.min(dim=1)  # get best results from pomo
        score_mean = min_pomo_reward.mean()
        
        train_score, loss, p_loss, val_loss, epi_len, entropy = reward.mean().item(), loss, p_loss, val_loss, len(prob_lst), -entropy

        self.log('score/train_score', score_mean, on_step=True, on_epoch=True)
        self.log('train_score', score_mean, prog_bar=True, logger=False)
        self.log('score/episode_length', float(epi_len), prog_bar=True)
        self.log('loss/total_loss', loss)
        self.log('loss/p_loss', p_loss.mean().item())
        self.log('loss/val_loss', val_loss, prog_bar=True)
        self.log('loss/entropy', entropy, prog_bar=True)
        self.log('hp_metric', train_score)

        # self.add_histogram()
        return loss
    
    def add_histogram(self):
        for name, param in self.model.named_parameters():
            self.logger.experiment.add_histogram(f"{name}/weight", param, self.current_epoch)

            if param.grad is not None and not torch.isinf(param.grad).any() and not torch.isnan(param.grad).any():
                self.logger.experiment.add_histogram(f"{name}/grad", param.grad, self.current_epoch)

    def configure_optimizers(self):
        optimizer = Optimizer(self.parameters(), **self.optimizer_params)
        return optimizer

    # def lr_scheduler_step(self, scheduler, metric):
    #     scheduler.step(epoch=self.current_epoch)