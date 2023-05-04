import itertools
import json
import os
import time
from pathlib import Path

from src.common.utils import dict_product
from src.run import parse_args, run_pretrain

import torch.multiprocessing as mp


def _work(**kwargs):
    args = parse_args()

    for k, v in kwargs.items():
        setattr(args, k, v)

    save_path = f"./{args.result_dir}/{args.env_type}/{args.name_prefix}/N_{args.num_nodes}/{args.nn}-{args.embedding_dim}-" \
                f"{args.encoder_layer_num}-{args.qkv_dim}-{args.head_num}-{args.C}"

    saved_model_path = save_path + "/saved_models/"

    # latest_epoch = max(
    #     map(lambda x: int(x), list(
    #         filter(lambda x: x.isdigit(), list(
    #             map(lambda x: x.split('-')[1].split('.')[0], os.listdir(saved_model_path)))
    #                )
    #     )
    #         )
    # )

    # args.model_load = latest_epoch

    score = run_pretrain(args)
    str_vals = [f"{name}_{val}" for name, val in zip(kwargs.keys(), kwargs.values())]
    key = "-".join(str_vals)
    return key, score, save_path


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
    # search_params(3)
    params = {
    'env_type' : 'tsp',
    'num_nodes' : 100,
    'result_dir' : 'pretrained_result',
    'name_prefix' : 'torch_attn',
    'render_mode' : None,
    'epochs' : 250,
    'num_episode' : 256,
    'qkv_dim' :  64
    }
    start = time.time()
    _work(**params)
    print(f"Time taken: {time.time() - start}")