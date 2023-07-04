import itertools
import json
import logging
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import fields
from datetime import datetime

import numpy as np
import pytz
import torch
from torch.utils.tensorboard.summary import hparams


class TimeEstimator:
    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count - 1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total - count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time * 60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time * 60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))



def get_result_folder(desc, result_dir, date_prefix=True):
    process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))

    if date_prefix is True:
        _date_prefix = process_start_time.strftime("%Y%m%d_%H%M%S")
        result_folder = f'{result_dir}/{_date_prefix}-{desc}'

    else:
        result_folder = f'{result_dir}/{desc}'

    return result_folder


def set_result_folder(folder):
    global result_folder
    result_folder = folder


def create_logger(filepath):
    filename = filepath + '/log.txt'

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    file_mode = 'a' if os.path.isfile(filename) else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    if isinstance(y_true, list):
        y_true = np.array(y_true)

    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


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


def concat_key_val(*args):
    result = deepcopy(args[0])

    for param_group in args[1:]:

        for k, v in param_group.items():
            result[k] = v

    if 'device' in result:
        del result['device']

    return result


def add_hparams(writer, param_dict, metrics_dict, step=None):
    exp, ssi, sei = hparams(param_dict, metrics_dict)

    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)

    if step is not None:
        for k, v in metrics_dict.items():
            writer.add_scalar(k, v, step)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


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

    }

    mcts_params = {
        'num_simulations': args.num_simulations,
        'temp_threshold': args.temp_threshold,  #
        'noise_eta': args.noise_eta,  # 0.25
        'cpuct': args.cpuct,
        'action_space': action_space,
        'normalize_value': args.normalize_value,
        'rollout_game': args.rollout_game
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
