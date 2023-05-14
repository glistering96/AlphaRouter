import itertools
import json
import os
import time
from pathlib import Path

from src.common.utils import dict_product
from src.run import parse_args, run_pretrain

import torch.multiprocessing as mp
from src.common.dir_parser import DirParser

def _work(**kwargs):
    load_from_the_latest = kwargs.pop('load_from_the_latest')

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


def search_params(num_proc):
    hyper_param_dict = {
        'env_type':  ['cvrp'],
        'num_nodes': [20, 50, 100],
        'result_dir' : ['pretrained_result'],
        'render_mode' : [None]
    }
    save_path = None

    async_result = mp.Queue()

    def __callback(val):
        async_result.put(val)

    pool = mp.Pool(num_proc)

    for params in dict_product(hyper_param_dict):
        pool.apply_async(_work, kwds=params, callback=__callback)

    pool.close()
    pool.join()

    result = {}

    while not async_result.empty():
        k, score, save_path = async_result.get()
        result[k] = score

    assert save_path is not None, "Wrong in fetching save path"

    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True, exist_ok=False)

    filename = "param_result"
    file_dir = f'{save_path}/{filename}.json'

    num_iter = 0

    while Path(file_dir).exists():
        filename = f'{filename}_{num_iter}'
        file_dir = f'{save_path}/{filename}.json'
        num_iter += 1

    with open(file_dir, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    params = {
    'num_nodes' : 50,
    'result_dir' : 'pretrained_result',
    'name_prefix' : '4_step-gamma_0.5_lr_3e-4',
    'render_mode' : None,
    'num_episode' : 1024,
    'qkv_dim' : 32,
    'load_from_the_latest' : False,
    'env_type' : 'cvrp',
    'embedding_dim': 128,
    'nn_train_epochs': 50000,
    'lr': 3e-4,
    }

    _work(**params)

    # for qkv_dim in [32, 64]:
    #     for embedding_dim in [128, 256]:
    #         params['qkv_dim'] = qkv_dim
    #         params['embedding_dim'] = embedding_dim
    #         _work(**params)


