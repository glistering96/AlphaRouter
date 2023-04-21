import numpy as np
import torch

from src.common.scaler import *


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_encoding(encoded_nodes, node_index_to_pick, T=1):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch,) or (batch, 1)

    batch_size = node_index_to_pick.size(0)
    embedding_dim = encoded_nodes.size(-1)

    _to_pick = node_index_to_pick.view(-1, T, 1)
    desired_shape = (batch_size, T, embedding_dim)
    gathering_index = torch.broadcast_to(_to_pick, desired_shape).reshape(batch_size, -1, embedding_dim)
    picked_node_embedding = encoded_nodes.gather(dim=1, index=gathering_index)

    return picked_node_embedding


def _to_tensor(obs, device):
    if isinstance(list(obs.values())[0], torch.Tensor):
        return obs

    tensor_obs = {k: None for k in obs.keys()}

    for k, v in obs.items():
        if k != '_t':
            if isinstance(v, np.ndarray):
                tensor = torch.from_numpy(v).to(device)
                tensor_obs[k] = tensor

            elif isinstance(v, int):
                tensor_obs[k] = torch.tensor([v], dtype=torch.long, device=device)

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



