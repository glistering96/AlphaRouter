import itertools
import json
import os
import time
from pathlib import Path

import torch

from src.common.utils import dict_product
from src.pretrain_model.pretrainer_module_pl import AMTrainer
from src.run import parse_args, run_pretrain

import torch.multiprocessing as mp
from src.common.dir_parser import DirParser


def _work(**kwargs):
    if 'load_from_the_latest' in kwargs:
        load_from_the_latest = kwargs.pop('load_from_the_latest')

    else:
        load_from_the_latest = False

    args = parse_args()

    for k, v in kwargs.items():
        setattr(args, k, v)

    if load_from_the_latest:
        saved_model_path = DirParser(args).get_model_checkpoint()

        latest_epoch = list(
            map(lambda x: int(x), list(
                filter(lambda x: x.isdigit(), list(
                    map(lambda x: x.split('-')[1].split('.')[0], os.listdir(saved_model_path)))
                       )
            )
                )
        )

        if latest_epoch:
            args.load_epoch = max(latest_epoch)

    run_pretrain(args)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    params = {
        'num_nodes' : 100,
        'result_dir' : 'POMO',
        'name_prefix' : "",
        'render_mode' : None,
        'qkv_dim' : 32,
        'num_heads': 4,
        'load_from_the_latest' : False,
        'env_type' : 'cvrp',
        'embedding_dim': 128,
        'encoder_layer_num': 6,
        'nn_train_epochs': 200,
        'model_save_interval': 2,
        'num_parallel_env': 64,
        'lr': 1e-4,
        'grad_acc': 1,
        'num_steps_in_epoch': 100*1000 // 64,
        'baseline': 'val',
        'activation': 'relu',
    }

    for _num_nodes in [20, 50, 100]:
        for _activation in ['relu', 'swiglu']:
            for _baseline in ['val', 'mean']:
                params['baseline'] = _baseline
                params['num_nodes'] = _num_nodes
                params['activation'] = _activation
                _work(**params)



