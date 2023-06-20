import math

from torch.distributions import Categorical


from src.models.model_common import get_encoding, _to_tensor, Encoder, Decoder, Value, Policy
import torch.nn.functional as F
import torch.nn as nn
import torch


class CVRPModel(nn.Module):
    def __init__(self, **model_params):
        super(CVRPModel, self).__init__()

        self.model_params = model_params

        self.policy_net = Policy(**model_params)
        self.value_net = Value(**model_params)
        self.encoder = Encoder(3, **model_params)
        self.decoder = Decoder(model_params['embedding_dim'] + 1, **model_params)

        self.encoding = None

    def _get_obs(self, observations, device):
        observations = _to_tensor(observations, device)

        xy, demands = observations['xy'], observations['demands']
        # (N, 2), (N, 1)

        cur_node = observations['pos']
        # (1, )

        load = observations['load']
        # (1, )

        available = observations['available']
        # (1, )
        
        batch_size, pomo_size, _ = available.size()

        xy = xy.reshape(batch_size, -1, 2)

        demands = demands.reshape(batch_size, -1, 1)

        load = load.reshape(batch_size, pomo_size, 1)

        available = available.reshape(batch_size, pomo_size, -1)

        return load, cur_node, available, xy, demands

    def forward(self, obs):
        load, cur_node, available, xy, demands = self._get_obs(obs, self.device)
        # load: (B, pomo, 1)
        # cur_node: (B, pomo, 1)
        # available: (B, pomo, N)
        # xy: (B, N, 2)
        # demands: (B, N, 1)

        batch_size, pomo_size, N = available.size()

        mask = torch.zeros_like(available).type(torch.float32)
        mask[available == False] = float('-inf')

        if self.encoding is None:
            self.encoding = self.encoder(xy, demands)
            self.decoder.set_kv(self.encoding)
        
        if obs['t'] == 0:
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            probs = torch.zeros((batch_size, pomo_size, N))   # assign prob 1 to the depots
            probs[:, :, 0] = 1.0

            cur_node_encoding = get_encoding(self.encoding, selected[:, :, None])
            self.decoder.set_q1(cur_node_encoding)
            mh_attn_out = self.decoder(cur_node_encoding, load=load, mask=mask)

        elif obs['t'] == 1:
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            probs = torch.zeros(batch_size, pomo_size, N)
            probs[:, torch.arange(pomo_size), torch.arange(N)] = torch.arange(N) # assign prob 1 to the pomo tensors

            cur_node_encoding = get_encoding(self.encoding, selected[:, :, None])
            self.decoder.set_q1(cur_node_encoding)
            mh_attn_out = self.decoder(cur_node_encoding, load=load, mask=mask)

        else:
            cur_node_encoding = get_encoding(self.encoding, cur_node)
            mh_attn_out = self.decoder(cur_node_encoding, load=load, mask=mask)

            probs = self.policy_net(mh_attn_out, self.decoder.single_head_key, mask)
            probs = probs.reshape(batch_size, pomo_size, N)

        val = self.value_net(mh_attn_out)
        val = val.reshape(batch_size, pomo_size, 1)

        return probs, val

    def predict(self, obs, deterministic=False):
        probs, _ = self.forward(obs)

        if deterministic:
            action = probs.argmax(-1).item()

        else:
            action = Categorical(probs=probs).sample().item()

        return action, None