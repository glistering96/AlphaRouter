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

from src.env.routing_env import RoutingEnv
from src.models.routing_model import RoutingModel


class AMTrainer(pl.LightningModule):
    def __init__(self, env_params, model_params, optimizer_params, run_params):
        super(AMTrainer, self).__init__()
        # save arguments
        self.optimizer_params = optimizer_params

        # model
        self.model = RoutingModel(model_params, env_params).create_model(env_params['env_type'])

        # env
        self.env = RoutingEnv(env_params).create_env(test=False)

        # etc
        self.ent_coef = run_params['ent_coef']
        self.warm_up_epochs = 2000

    def training_step(self):
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

        train_score, loss, p_loss, val_loss, epi_len, entropy = reward.mean().item(), loss, p_loss, val_loss, len(prob_lst), entropy

        self.log('score/train_score', train_score, prog_bar=True)
        self.log('score/episode_length', epi_len, prog_bar=True)
        self.log('loss/total_loss', loss)
        self.log('loss/p_loss', p_loss)
        self.log('loss/val_loss', val_loss, prog_bar=True)
        self.log('loss/entropy', entropy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = Optimizer(self.parameters(), **self.optimizer_params)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.warm_up_epochs,
            eta_min=1e-7)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
