import json

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.wrappers import RecordVideo
from torch.optim import Adam as Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings
import lightning.pytorch as pl

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
        self.warm_up_epochs = 10

    def _save_output(self, module, grad_input, grad_output):
        print(module, grad_output)
  
    def training_step(self, batch, _):
        # TODO: need to add a batch input for training step. It means that environment rollout must be isolated
        # from the training step.

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

            logit = dist.log_prob(action)[:, :, None] + 1e-10
            # (batch, pomo, 1)

            obs, reward, dones, _, _ = self.env.step(action.detach().cpu().numpy())

            done = bool(np.all(dones == True))

            prob_lst.append(logit)
            val_lst.append(val)
    
        reward = -torch.as_tensor(reward, device=self.device, dtype=torch.float16)
        # (batch, pomo)
        val_tensor = torch.cat(val_lst, dim=-1)
        # val_tensor: (batch, pomo, T)
        
        baseline = val_tensor
        reward_broadcasted = torch.broadcast_to(reward[:, :, None], baseline.shape)
        # (batch, pomo, T)
        
        val_loss = torch.nn.functional.mse_loss(val_tensor, reward_broadcasted)
        
        adv = reward_broadcasted - baseline.detach()
        # (batch, pomo, T)

        log_prob = torch.cat(prob_lst, dim=-1).sum(dim=-1)
        # (batch, pomo)
        
        p_loss = (adv * log_prob[:, :, None]).mean()       

        entropy = -torch.cat(entropy_lst, dim=-1).mean()
        loss = p_loss + val_loss   
        
        # for module in self.model.modules():
        #     module.register_full_backward_hook(self._save_output)

        # self.automatic_optimization = False
        # self.optimizers().zero_grad()
        # self.manual_backward(loss)
        # self.optimizers().step()
        
        train_score, loss, p_loss, val_loss, epi_len, entropy = reward.mean().item(), loss, p_loss, val_loss, len(prob_lst), -entropy

        self.log('score/train_score', train_score)
        self.log('train_score', train_score, prog_bar=True, logger=False)
        self.log('score/episode_length', float(epi_len), prog_bar=True)
        self.log('loss/total_loss', loss)
        self.log('loss/p_loss', p_loss)
        self.log('loss/val_loss', val_loss, prog_bar=True)
        self.log('loss/entropy', entropy, prog_bar=True)
        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_lr()[0]
        self.log('debug/lr', lr, prog_bar=True)
        self.log('hp_metric', train_score)

        # self.add_histogram()
        return loss
    
    def add_histogram(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.logger.experiment.add_histogram(name, param.grad, self.current_epoch)
            

    def configure_optimizers(self):
        optimizer = Optimizer(self.parameters(), **self.optimizer_params)

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=1000,
            warmup_steps=self.warm_up_epochs,
            max_lr=self.optimizer_params['lr'],
            min_lr=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # def lr_scheduler_step(self, scheduler, metric):
    #     scheduler.step(epoch=self.current_epoch)