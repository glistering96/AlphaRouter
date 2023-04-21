import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoder model related


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3


class DecoderCommon(nn.Module):
    def __init__(self, **model_params):
        super(DecoderCommon, self).__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        self.head_num = self.model_params['head_num']
        self.qkv_dim = self.model_params['qkv_dim']

        # self.load_embedder = nn.Linear(1, embedding_dim)
        self.embedding_mixer = nn.Linear(embedding_dim + embedding_dim//2, embedding_dim)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, embedding_dim)

        self.Wq_last = nn.Linear(embedding_dim + 1, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, self.head_num * self.qkv_dim, bias=False)

        self.layer = AttentionLayer(**model_params)

        self.k, self.v = None, None

    def set_kv(self, encoding):
        self.k = reshape_by_heads(self.Wk(encoding), head_num=self.head_num)
        self.v = reshape_by_heads(self.Wv(encoding), head_num=self.head_num)

        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoding.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def forward(self, cur_node_encoding, load, mask):
        """
        :param cur_node_encoding: (B, 1 or T, d)
        :param load:   (B, 1 or T, 1)
        :param encoding: (B, N, d)
        :return:
        """

        load_embedding = load
        _in = torch.cat([cur_node_encoding, load_embedding[..., None]], -1)
        _in_tf = self.Wq_last(_in)

        q = reshape_by_heads(_in_tf, head_num=self.head_num)
        # (batch, N, embedding)

        out_concat = multi_head_attention(q, self.k, self.v, mask)
        # (batch, 1 or T, qkv*head_num)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, 1 or T, embedding)

        return mh_atten_out


class AttentionLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem, embedding)


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = embedding_dim*2

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


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