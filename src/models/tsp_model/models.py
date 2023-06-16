import math

import torch
from torch.distributions import Categorical

from src.models.model_common import get_encoding, _to_tensor, Encoder, Decoder, Value, Policy
import torch.nn as nn


class TSPModel(nn.Module):
    def __init__(self, **model_params):
        super(TSPModel, self).__init__()

        self.model_params = model_params

        self.policy_net = Policy(**model_params)
        self.value_net = Value(**model_params)
        self.encoder = Encoder(2, **model_params)
        self.decoder = Decoder(model_params['embedding_dim'], **model_params)

        self.encoding = None

        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(0))
            param.data.uniform_(-stdv, stdv)

    def _get_obs(self, observations, device):
        observations = _to_tensor(observations, device)
        
        xy = observations['xy']
        # (N, 2), (N, 1)

        cur_node = observations['pos']
        # (1, )

        available = observations['available']
        # (1, )
        
        batch_size, pomo_size, _ = available.size()

        xy = xy.reshape(batch_size, -1, 2)
        
        if cur_node is not None:
            cur_node = cur_node.reshape(batch_size, pomo_size, 1)

        available = available.reshape(batch_size, pomo_size, -1)

        return xy, cur_node, available

    def forward(self, obs):
        xy, cur_node, available = self._get_obs(obs, self.device)
        # xy: (B, N, 2)
        # cur_node: (B, pomo, )
        # available: (B, pomo, N)
        
        batch_size, pomo_size, N = available.size()
        
        mask = torch.zeros_like(available).type(torch.float32)
        mask[available == False] = float('-inf')

        if self.encoding is None:
            self.encoding = self.encoder(xy)
            
        self.decoder.set_kv(self.encoding)
        
        if obs['t'] == 0:
            selected = torch.arange(pomo_size).repeat(batch_size, 1).to(self.device)
            probs = torch.zeros(size=(batch_size, pomo_size, N)).to(self.device)
            probs[:, torch.arange(pomo_size), torch.arange(N)] = 1.0

            cur_node_encoding = get_encoding(self.encoding, selected[:, :, None])
            mh_attn_out = self.decoder(cur_node_encoding, load=None, mask=mask)
            
            val = self.value_net(mh_attn_out)
            val = val.reshape(batch_size, pomo_size, 1)

        else:
            cur_node_encoding = get_encoding(self.encoding, cur_node.long())
            mh_attn_out = self.decoder(cur_node_encoding, load=None, mask=mask)
            
            probs = self.policy_net(mh_attn_out, self.decoder.single_head_key, mask)
            probs = probs.reshape(batch_size, pomo_size, N)        
            
            val = self.value_net(mh_attn_out)
            val = val.reshape(batch_size, pomo_size, 1)
            
        return probs, val

    def predict(self, obs, deterministic=False):
        probs, _, = self.forward(obs)

        if deterministic:
            action = probs.argmax(-1).item()

        else:
            action = Categorical(probs=probs).sample().item()

        return action, None


