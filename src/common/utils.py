import itertools
import json
import logging
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from glob import glob

import numpy as np
import pytz
import torch
from torch.utils.tensorboard.summary import hparams



def cal_distance(xy, visiting_seq, axis=1):
    """
    :param xy: coordinates of nodes, (batch, N, 2)
    :param visiting_seq: sequence of visiting node idx, (batch, pomo, seq_len)
    :return:

    1. Gather coordinates on a given sequence of nodes
    2. roll by -1
    3. calculate the distance
    4. return distance

    """
    batch_size, pomo, seq_len = visiting_seq.shape
    N = xy.shape[1]

    gather_idx_shape = [batch_size, pomo, seq_len, 2]
    xy_broad_shape = [batch_size, pomo, N, 2]

    gather_idx = np.broadcast_to(visiting_seq[:, :, :, None], gather_idx_shape)
    xy_broaded = np.broadcast_to(xy[:, None, :, :], xy_broad_shape)
    original_seq = np.take_along_axis(xy_broaded, gather_idx, 2)
    rolled_seq = np.roll(original_seq, -1, axis=axis)

    segments = np.sqrt(((original_seq - rolled_seq) ** 2).sum(-1))
    distance = segments.sum(2).astype(np.float32)
    return distance


def check_debug():
    import sys

    eq = sys.gettrace() is None

    if eq is False:
        return True
    else:
        return False


def save_json(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)

    except:
        data = {}

    return data


def cal_average_std(result_dict):
    result_dict = deepcopy(result_dict)

    # sort the result_dict by test_data_idx
    result_dict = dict(sorted(result_dict.items(), key=lambda x: x[0]))

    num_problems = len(result_dict)

    avg_score = sum([result_dict[i]['score'] for i in range(num_problems)]) / num_problems
    avg_runtime = sum([result_dict[i]['runtime'] for i in range(num_problems)]) / num_problems

    std_score = sum([(result_dict[i]['score'] - avg_score) ** 2 for i in range(num_problems)]) / num_problems
    std_runtime = sum([(result_dict[i]['runtime'] - avg_runtime) ** 2 for i in range(num_problems)]) / num_problems

    result_dict['average'] = {'score': avg_score, 'runtime': avg_runtime}
    result_dict['std'] = {'score': std_score, 'runtime': std_runtime}

    return result_dict


def collect_all_checkpoints(params):
    from src.common.dir_parser import DirParser
    from src.run import parse_args

    args = parse_args()

    for k, v in params.items():
        setattr(args, k, v)

    ckpt_root = DirParser(args).get_model_checkpoint()

    all_files = glob(ckpt_root + '/*.ckpt')

    # get the checkpoint with minimum train_score from the all_files
    all_files = list(sorted(all_files, key=lambda x: float(x.split('train_score=')[1].split('.ckpt')[0])))

    return all_files, ckpt_root


def get_result_dir(params, mcts=False):
    from src.common.dir_parser import DirParser
    from src.run import parse_args

    args = parse_args()

    for k, v in params.items():
        setattr(args, k, v)

    dir = DirParser(args).get_result_dir(mcts=mcts)

    return dir


def get_param_dict(args, return_logger=False):
    # env_params
    num_demand_nodes = args.num_nodes
    num_depots = args.num_depots

    if args.test_num is None:
        args.test_num = num_demand_nodes

    action_space = num_demand_nodes + num_depots if args.test_num is None else args.test_num + num_depots

    load_epoch = args.load_epoch
    load_model = True if load_epoch is not None else False

    if check_debug():
        seed = 2
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True  # type: ignore

    # allocating hyper-parameters
    env_params = {
        'num_nodes': args.num_nodes,
        'num_depots': args.num_depots,
        'seed': args.seed,
        'step_reward': args.step_reward,
        'env_type': args.env_type,
        'render_mode': args.render_mode,
        'test_data_type': args.test_data_type,
        'test_data_idx': args.test_data_idx,
        'test_num': args.test_num,
        'num_parallel_env': args.num_parallel_env,
        'data_path': args.data_path,
        'test_data_seed': args.test_data_seed,

    }

    mcts_params = {
        'num_simulations': args.num_simulations,
        'temp_threshold': args.temp_threshold,  #
        'noise_eta': args.noise_eta,  # 0.25
        'cpuct': args.cpuct,
        'action_space': action_space,
        'normalize_value': args.normalize_value,
        'rollout_game': args.rollout_game,
        'selection_coef': args.selection_coef,
    }

    model_params = {
        'nn': args.nn,
        'embedding_dim': args.embedding_dim,
        'encoder_layer_num': args.encoder_layer_num,
        'qkv_dim': args.qkv_dim,
        'head_num': args.head_num,
        'activation': args.activation,
        'C': args.C,
    }

    run_params = {
        'use_cuda': True,
        'cuda_device_num': args.gpu_id,
        'train_epochs': args.train_epochs,
        'nn_train_epochs': args.nn_train_epochs,
        'mini_batch_size': args.mini_batch_size,
        'ent_coef': args.ent_coef,

        'model_save_interval': args.model_save_interval,
        'log_interval': args.log_interval,
        'warm_up': args.warm_up,

        'model_load': {
            'enable': load_model,
            'epoch': args.load_epoch
        },
        'baseline': args.baseline
    }

    try:
        run_params['data_path'] = args.data_path

    except:
        pass

    optimizer_params = {
        'lr': args.lr,
        'eps': 1e-7,
        'betas': (0.9, 0.95)
    }

    h_params = {
        'num_parallel_env': args.num_parallel_env,
        'embedding_dim': args.embedding_dim,
        'encoder_layer_num': args.encoder_layer_num,
        'qkv_dim': args.qkv_dim,
        'head_num': args.head_num,
        'betas': (0.9, 0.95),
        'lr': args.lr,
        'nn_train_epochs': args.nn_train_epochs,
        'ent_coef': args.ent_coef,
        'grad_acc': args.grad_acc,
        'warm_up': args.warm_up,
    }

    logger_params = {
        'log_file': {
            'filename': 'log.txt',
            'date_prefix': False
        },
    }

    # TODO: result folder src copy needs to be managed
    # if copy_src:
    #     copy_all_src(result_folder_name)

    if return_logger:
        return env_params, mcts_params, model_params, h_params, run_params, optimizer_params, logger_params

    else:
        return env_params, mcts_params, model_params, h_params, run_params, optimizer_params


def dict_product(dicts):
    return list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))
