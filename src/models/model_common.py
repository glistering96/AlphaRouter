import math

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

INNER_MULT = 2


def get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo, 1)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(-1)

    _to_pick = torch.broadcast_to(node_index_to_pick, (batch_size, pomo_size, embedding_dim))
    picked_node_embedding = encoded_nodes.gather(dim=2, index=_to_pick)

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


# This class if for compatibility with different torch versions
class ScaledDotProductAttention(nn.Module):
    def __init__(self, **model_params):
        super(ScaledDotProductAttention, self).__init__()
        self.embedding_dim = model_params['embedding_dim']

    def forward(self, q, k, v, attn_mask=None):
        B = q.size(0)
        head_num = q.size(1)
        n = q.size(2)
        key_dim = q.size(3)
        input_s = k.size(2)

        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask[:, None, None, :].expand(B, head_num, n, input_s)

        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask[:, None, :, :].expand(B, head_num, n, input_s)

        if int(torch.__version__[0]) == 2:
            # native scaled dot product attention is only available in torch >= 2.0
            return F.scaled_dot_product_attention(q, k, v, attn_mask)

        else:
            return self.multi_head_attention(q, k, v, attn_mask)

    def multi_head_attention(self, q, k, v, mask=None):
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
            score_scaled = score_scaled + mask

        weights = nn.Softmax(dim=-1)(score_scaled)
        # shape: (batch, head_num, n, problem)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, n, key_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, n, head_num, key_dim)

        out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
        # shape: (batch, n, head_num*key_dim)

        return out_concat


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
        self.act = nn.ReLU()
        # self.act = SwiGLU()
        # self.act = nn.GELU()
        # self.act = nn.SiLU()
        
    def forward(self, x):
        return self.act(x)
    

class FFBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_size = embedding_dim * INNER_MULT
        mult_factor = 2 if Activation().act.__class__.__name__ == 'SwiGLU' else 1
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_size*mult_factor),
            Activation(),
            nn.Linear(ff_size, embedding_dim)
        )

    def forward(self, input1):
        # input1.shape: (batch, problem, embedding)
        return self.feed_forward(input1)


class Normalization(nn.Module):
    def __init__(self, embed_dim):
        super(Normalization, self).__init__()

        self.normalizer = nn.InstanceNorm1d(embed_dim, affine=True, track_running_stats=False)

    def forward(self, input):
        transposed = input.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.normalizer(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()

        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.head_num = self.model_params['head_num']
        self.qkv_dim = self.model_params['qkv_dim']

        self.mha_block = MHABlock(**model_params)
        self.layer_norm1 = Normalization(embedding_dim)
        self.ff_block = FFBlock(**model_params)
        self.layer_norm2 = Normalization(embedding_dim)

    def forward(self, input1):
        attn_out = self.mha_block(input1)
        # (batch, problem, embedding)

        attn_normalized = self.layer_norm1(attn_out + input1)
        # (batch, problem, embedding)

        ff_out = self.ff_block(attn_normalized)
        # (batch, problem, embedding)

        ff_normalized = self.layer_norm2(attn_normalized + ff_out)
        # (batch, problem, embedding)

        return ff_normalized


class Encoder(nn.Module):
    def __init__(self, input_dim, **model_params):
        super(Encoder, self).__init__()

        self.model_params = model_params
        self.embedding_dim = model_params['embedding_dim']

        self.input_embedder = nn.Linear(input_dim, self.embedding_dim)
        self.embedder = nn.ModuleList([EncoderLayer(**model_params) for _ in range(model_params['encoder_layer_num'])])

    def forward(self, xy):
        input_emb = self.input_embedder(xy)

        out = input_emb

        for layer in self.embedder:
            out = layer(out)

        return out


class Decoder(nn.Module):
    def __init__(self, query_dim, **model_params):
        super(Decoder, self).__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        self.head_num = self.model_params['head_num']
        self.qkv_dim = self.model_params['qkv_dim']

        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, embedding_dim)

        self.Wq_first = nn.Linear(embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wq_last = nn.Linear(query_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, self.head_num * self.qkv_dim, bias=False)

        self.scaled_dot_product_attention = ScaledDotProductAttention(**model_params)

        self.k, self.v = None, None
        self.q_first, self.single_head_key = None, None

    def set_q1(self, encoding):
        B, N, embedding_dim = encoding.shape

        self.q_first = self.Wq_first(encoding).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        # shape: (batch, head_num, n, qkv_dim)

    def set_kv(self, encoding):
        B, N, _ = encoding.shape

        self.k = self.Wk(encoding).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        self.v = self.Wv(encoding).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        # shape: (batch, head_num, n, qkv_dim)

        self.single_head_key = encoding.transpose(1, 2)
        # shape: (batch, embedding, problem)

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
            q_current = torch.cat([cur_node_encoding, load_embedding[..., None]], -1)

        else:
            q_current = cur_node_encoding

        q_current_head = self.Wq_last(q_current).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        # (batch, N, embedding)

        q = q_current_head + self.q_first

        out_concat = self.scaled_dot_product_attention(q, self.k, self.v, mask)
        # (batch, pomo, qkv, head_num)

        mh_attn_out = self.multi_head_combine(out_concat.reshape(B, N, -1))
        # shape: (batch, pomo, embedding)

        return mh_attn_out


class Policy(nn.Module):
    def __init__(self, **model_params):
        super(Policy, self).__init__()
        self.C = model_params['C']
        self.embedding_dim = model_params['embedding_dim']

    def forward(self, mh_attn_out, single_head_key, mask):
        # mh_attn_out: (batch, 1, embedding_dim)
        # single_head_key: (batch, embedding_dim, N)
        # mask: (batch, pomo, N)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_attn_out, single_head_key)
        # shape: (batch, pomo, N)

        sqrt_embedding_dim = math.sqrt(self.embedding_dim)

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, N)

        score_clipped = self.C * torch.tanh(score_scaled)
        # shape: (batch, pomo, N)

        score_masked = score_clipped + mask
        # shape: (batch, pomo, N)

        probs = F.softmax(score_masked, dim=-1)

        return probs


class Value(nn.Module):
    def __init__(self, **model_params):
        super(Value, self).__init__()
        self.embedding_dim = model_params['embedding_dim']
        inner_size = self.embedding_dim
        mult_factor = 2 if Activation().act.__class__.__name__ == 'SwiGLU' else 1
        self.val = nn.Sequential(
            nn.Linear(self.embedding_dim, inner_size*mult_factor),
            Activation(),
            nn.Linear(inner_size*2, 1)
        )

    def forward(self, mh_attn_out):
        val = self.val(mh_attn_out)
        return val
