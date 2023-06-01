import math

import torch.nn as nn
import torch.nn.functional as F

from src.common.scaler import *

INNER_MULT = 2

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_encoding(encoded_nodes, node_index_to_pick, T=1):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, 1)

    batch_size = node_index_to_pick.size(0)
    embedding_dim = encoded_nodes.size(-1)

    _to_pick = node_index_to_pick.view(batch_size, T, 1)
    desired_shape = (batch_size, T, embedding_dim)
    gathering_index = torch.broadcast_to(_to_pick, desired_shape).reshape(batch_size, -1, embedding_dim)
    picked_node_embedding = encoded_nodes.gather(dim=1, index=gathering_index)

    return picked_node_embedding


def _to_tensor(obs, device):
    if isinstance(list(obs.values())[0], torch.Tensor):
        return obs

    tensor_obs = {k: None for k in obs.keys()}

    for k, v in obs.items():
        if k != 't':
            if isinstance(v, np.ndarray):
                tensor = torch.from_numpy(v).to(device)
                tensor_obs[k] = tensor

            elif isinstance(v, int):
                tensor_obs[k] = torch.tensor([v], dtype=torch.long).to(device)

    return tensor_obs


def get_batch_tensor(obs: list):
    if not obs:
        return None

    tensor_obs = {k: [] for k in obs[0].keys()}

    for x in obs:
        for k, v in x.items():
            tensor_obs[k].append(v)

    for k, v in tensor_obs.items():
        cat = np.stack(v)
        tensor_obs[k] = torch.tensor(cat)

    return tensor_obs


class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        return self.module(input) + input


def multi_head_attention(q, k, v, mask=None):
    # q shape: (batch, head_num, N, key_dim)
    # k,v shape: (batch, head_num, N, key_dim)
    # mask.shape: (batch, N)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if mask is not None:
        if mask.dim() == 2:
            score_scaled = score_scaled + mask[:, None, None, :].expand(batch_s, head_num, n, input_s)

        elif mask.dim() == 3:
            score_scaled = score_scaled + mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=-1)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


# This class if for compatibility with different torch versions
class ScaledDotProductAttention(nn.Module):
    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v, attn_mask=None):
        if int(torch.__version__[0]) == 2:
            # native scaled dot product attention is only available in torch >= 2.0
            return F.scaled_dot_product_attention(q, k, v, attn_mask)

        else:
            return multi_head_attention(q, k, v, attn_mask)


class MHABlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.head_num = self.model_params['head_num']
        self.qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, embedding_dim)

        self.scaled_dot_product_attention = ScaledDotProductAttention(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem, embedding)
        B, N, E = input1.size()
        q = self.Wq(input1).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        k = self.Wk(input1).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        v = self.Wv(input1).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = self.scaled_dot_product_attention(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat.reshape(B, N, -1))
        # shape: (batch, problem, embedding)

        return multi_head_out


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class Activation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.act = nn.ReLU()
        self.act = SwiGLU()
        # self.act = nn.GELU()
        # self.act = nn.SiLU()
        
    def forward(self, x):
        return self.act(x)
    

class FFBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_size = embedding_dim * INNER_MULT
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_size*2),
            Activation(),
            nn.Linear(ff_size, embedding_dim)
        )

    def forward(self, input1):
        # input1.shape: (batch, problem, embedding)
        return self.feed_forward(input1)


class Normalization(nn.Module):
    def __init__(self, embed_dim):
        super(Normalization, self).__init__()

        self.normalizer = nn.InstanceNorm1d(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)


class EncoderLayer(nn.Sequential):
    def __init__(self, **model_params):
        super().__init__(
            SkipConnection(
                MHABlock(**model_params)
            ),
            Normalization(model_params['embedding_dim']),
            SkipConnection(
                FFBlock(**model_params)
            ),
            Normalization(model_params['embedding_dim'])
        )


class Encoder(nn.Module):
    def __init__(self, input_dim, **model_params):
        super(Encoder, self).__init__()

        self.model_params = model_params
        self.embedding_dim = model_params['embedding_dim']

        self.input_embedder = nn.Linear(input_dim, self.embedding_dim)
        self.embedder = nn.ModuleList([EncoderLayer(**model_params) for _ in range(model_params['encoder_layer_num'])])
        
        self.init_parameters()

    def init_parameters(self):
        for name, param in self.input_embedder.named_parameters():
            stdv = 1. / math.sqrt(param.size(0))
            param.data.uniform_(-stdv, stdv)

    def forward(self, xy):
        out = self.input_embedder(xy)

        for layer in self.embedder:
            out = layer(out) + out
        
        return out


class Decoder(nn.Module):
    def __init__(self, query_dim, **model_params):
        super(Decoder, self).__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        self.head_num = self.model_params['head_num']
        self.qkv_dim = self.model_params['qkv_dim']

        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, embedding_dim)

        self.Wq_last = nn.Linear(query_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, self.head_num * self.qkv_dim, bias=False)

        self.scaled_dot_product_attention = ScaledDotProductAttention(**model_params)

        self.k, self.v = None, None

    def set_kv(self, encoding):
        B, N, _ = encoding.shape

        self.k = self.Wk(encoding).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        self.v = self.Wv(encoding).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        # shape: (batch, head_num, problem+1, qkv_dim)

        self.single_head_key = encoding.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def forward(self, cur_node_encoding, load=None, mask=None):
        """
        :param cur_node_encoding: (B, 1 or T, d)
        :param load:   (B, 1 or T, 1)
        :param encoding: (B, N, d)
        :return:
        """
        B, N = cur_node_encoding.shape[:2]

        if load is not None:
            load_embedding = load
            query_in = torch.cat([cur_node_encoding, load_embedding[..., None]], -1)

        else:
            query_in = cur_node_encoding

        q = self.Wq_last(query_in).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        # (batch, N, embedding)

        if mask is not None and mask.dim() == 2:
            mask = mask[:, None, None, :]

        out_concat = self.scaled_dot_product_attention(q, self.k, self.v, mask)
        # (batch, 1 or T, qkv*head_num)

        attn_out = self.multi_head_combine(out_concat.reshape(B, N, -1))
        # shape: (batch, 1 or T, embedding)

        # TODO: Is it working for CVRP as well?
        mh_attn_out = attn_out + cur_node_encoding

        return mh_attn_out


class Policy(nn.Module):
    def __init__(self, **model_params):
        super(Policy, self).__init__()
        self.C = model_params['C']
        self.embedding_dim = model_params['embedding_dim']

    def forward(self, mh_attn_out, single_head_key, mask):
        # mh_attn_out: (batch, 1, embedding_dim)
        # single_head_key: (batch, embedding_dim, problem)
        # mask: (batch, problem)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_attn_out, single_head_key)
        # shape: (batch, 1, problem)

        sqrt_embedding_dim = math.sqrt(self.embedding_dim)

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, 1, problem)

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
        inner_size = self.embedding_dim * INNER_MULT
        self.val = nn.Sequential(
            nn.Linear(self.embedding_dim, inner_size*2),
            Activation(),
            nn.Linear(inner_size, 1)
        )

    def forward(self, mh_attn_out):
        val = self.val(mh_attn_out)
        return val
