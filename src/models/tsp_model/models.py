import math

import torch
from torch.distributions import Categorical

from src.models.model_common import get_encoding, _to_tensor, EncoderLayer
from src.models.tsp_model.modules import *
import torch.nn.functional as F

class TSPModel(nn.Module):
    def __init__(self, **model_params):
        super(TSPModel, self).__init__()

        self.model_params = model_params
        self.device = model_params['device']

        self.policy_net = Policy(**model_params)
        self.value_net = Value(**model_params)
        self.encoder = Encoder(**model_params)
        self.decoder = Decoder(**model_params)

        self.encoding = None

    def _get_obs(self, observations):
        observations = _to_tensor(observations, self.device)

        xy = observations['xy']
        # (N, 2), (N, 1)

        cur_node = observations['pos']
        # (1, )

        available = observations['available']
        # (1, )

        B = xy.size(0) if xy.dim() == 3 else 1

        xy = xy.reshape(B, -1, 2)

        cur_node = cur_node.reshape(B, 1)

        available = available.reshape(B,-1)

        return xy, cur_node, available

    def forward(self, obs):
        xy, cur_node, available = self._get_obs(obs)
        # xy: (B, N, 2)
        # cur_node: (B, )
        # available: (B, N)

        B, T = xy.size(0), 1

        mask = torch.zeros_like(available).type(torch.float32)
        mask[available == False] = float('-inf')

        if self.encoding is None:
            self.encoding = self.encoder(xy)

        self.decoder.set_kv(self.encoding)

        last_node = get_encoding(self.encoding, cur_node.long(), T)

        mh_atten_out = self.decoder(last_node, mask)

        probs = self.policy_net(mh_atten_out, self.decoder.single_head_key, mask)
        probs = probs.reshape(-1, probs.size(-1))

        val = self.value_net(mh_atten_out)
        val = val.reshape(-1, )

        return probs, val

    def predict(self, obs, deterministic=False):
        probs, _ = self.forward(obs)

        if deterministic:
            action = probs.argmax(-1).item()

        else:
            action = Categorical(probs=probs).sample().item()

        return action, None


class Encoder(nn.Module):
    def __init__(self, **model_params):
        super(Encoder, self).__init__()

        self.model_params = model_params
        self.embedding_dim = model_params['embedding_dim']

        self.input_embedder = nn.Linear(2, self.embedding_dim)
        self.embedder = nn.ModuleList([EncoderLayer(**model_params) for _ in range(model_params['encoder_layer_num'])])

    def forward(self, xy):
        out = self.input_embedder(xy)

        for layer in self.embedder:
            out = layer(out)

        return out


class Policy(nn.Module):
    def __init__(self, **model_params):
        super(Policy, self).__init__()
        self.C = model_params['C']
        self.embedding_dim = model_params['embedding_dim']

    def forward(self, mh_attn_out, single_head_key, mask):
        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_attn_out, single_head_key)
        # shape: (batch, problem)

        sqrt_embedding_dim = math.sqrt(self.embedding_dim)

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, problem)

        score_clipped = self.C * torch.tanh(score_scaled)

        if score_clipped.dim() != mask.dim():
            mask = mask.reshape(score_clipped.shape)

        score_masked = score_clipped + mask

        probs = F.softmax(score_masked, dim=-1)

        return probs


class Value(nn.Module):
    def __init__(self, **model_params):
        super(Value, self).__init__()
        self.embedding_dim = model_params['embedding_dim']
        self.val = nn.Linear(self.embedding_dim, 1)

    def forward(self, mh_attn_out):
        val = self.val(mh_attn_out)
        return val


