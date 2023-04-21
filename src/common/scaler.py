import numpy as np
import torch


def min_max_norm(x, dim=-1):
    if isinstance(x, list):
        x = np.concatenate(x, axis=-1)

    if isinstance(x, np.ndarray):
        max_dist = x.max(axis=dim, keepdims=True)
        min_dist = x.min(axis=dim, keepdims=True)

    elif isinstance(x, torch.Tensor):
        max_dist, _ = torch.max(x, dim=dim, keepdim=True)
        min_dist, _ = torch.min(x, dim=dim, keepdim=True)

    else:
        raise TypeError(f"type of dist({type(x)}) is neither np.ndarray or torch.Tensor or list")

    diff = (max_dist - min_dist)
    out = (x - min_dist) / (diff + 1e-8)

    return out, (diff, min_dist)


def max_norm(x, dim=-1):
    if isinstance(x, np.ndarray):
        max_dist = x.max(axis=dim, keepdims=True)

    elif isinstance(x, torch.Tensor):
        max_dist, _ = torch.max(x, dim=dim, keepdim=True)

    else:
        raise TypeError(f"type of dist({type(x)}) is neither np.ndarray or torch.Tensor")

    out = (x / (max_dist+1e-6))

    return out, (max_dist, 0)


def std_norm(x, dim=-1):
    if (x == 0).all():
        return x, (1, 0)

    if isinstance(x, list):
        x = np.concatenate(x, axis=-1)

    if isinstance(x, np.ndarray):
        mean = x.mean(axis=dim)
        std = x.std(axis=dim)

    elif isinstance(x, torch.Tensor):
        mean = x.mean()
        std = x.std()

    else:
        raise TypeError(f"type of dist({type(x)}) is neither np.ndarray or torch.Tensor or list")

    out = (x - mean) / (std + 1e-8)
    return out, (std, mean)
