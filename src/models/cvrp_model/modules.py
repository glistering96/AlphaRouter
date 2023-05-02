import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model_common import reshape_by_heads


class Decoder(nn.Module):
    def __init__(self, **model_params):
        super(Decoder, self).__init__()
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

        out_concat = F.scaled_dot_product_attention(q, self.k, self.v, mask)
        # (batch, 1 or T, qkv*head_num)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, 1 or T, embedding)

        return mh_atten_out