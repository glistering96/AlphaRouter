import itertools
import json
from pathlib import Path

from src.common.utils import dict_product
from src.run import parse_args, run_pretrain

import torch.multiprocessing as mp


def _work(**kwargs):
    args = parse_args()

    for k, v in kwargs.items():
        setattr(args, k, v)

    save_path = f"./{args.result_dir}/{args.name_prefix}/N_{args.num_nodes}/nn-{args.nn}-{args.embedding_dim}-" \
                f"{args.encoder_layer_num}-{args.qkv_dim}-{args.head_num}-{args.C}-one_layer_val_{args.one_layer_val}"

    score = run_pretrain(args)
    str_vals = [f"{name}_{val}" for name, val in zip(kwargs.keys(), kwargs.values())]
    key = "-".join(str_vals)
    return key, score, save_path


def search_params(num_proc):
    hyper_param_dict = {
        'num_simulations' : [15, 20, 25, 30],
        'temp_threshold' : [5, 10, 15],
        'cpuct' : [1, 1.1, 1.5, 5, 10],
        'lr' : [0.00005, 0.00001]
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
    args = parse_args()

    args.result_dir = 'pretrained_result'

    # for env_type in ['tsp', 'cvrp']:
    #     for num_nodes in [20, 50, 100]:
    #         args.env_type = 'tsp'
    #         args.num_nodes = num_nodes
    #         args.epochs = 100
    #         args.model_save_interval = 50
    #         args.log_interval = 50
    #         run_pretrain(args)


    num_nodes = 50
    args.env_type = 'tsp'
    args.num_nodes = num_nodes
    run_pretrain(args)
