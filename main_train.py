import itertools
import json
from pathlib import Path

from src.common.utils import dict_product, parse_saved_model_dir
from src.run import parse_args, run_mcts_train
import torch.multiprocessing as mp


def _work(**kwargs):
    args = parse_args()

    for k, v in kwargs.items():
        setattr(args, k, v)

    score = run_mcts_train(args)
    str_vals = [f"{name}_{val}" for name, val in zip(kwargs.keys(), kwargs.values())]
    key = "-".join(str_vals)
    return key, score


def search_params(num_proc):
    hyper_param_dict = {
        'num_simulations' : [15, 20, 25, 30],
        'temp_threshold' : [5, 10, 15],
        'cpuct' : [1, 1.1, 1.5, 5, 10],
        'lr' : [0.00005, 0.00001]
    }

    async_result = mp.Queue()
    save_path = None

    def __callback(val):
        async_result.put(val)

    # pool = mp.Pool(num_proc)
    #
    # for params in dict_product(hyper_param_dict):
    #     pool.apply_async(_work, kwds=params, callback=__callback)
    #
    # pool.close()
    # pool.join()

    result = {}

    for params in dict_product(hyper_param_dict):
        _work(**params)

    while not async_result.empty():
        k, score = async_result.get()
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
    args.result_dir = 'mcts_train_result'
    args.name_prefix = 'mcts_train_-Q-None'

    args.num_nodes = 20
    args.num_proc = 4
    args.cpuct = 1.1
    args.rollout_game = False
    args.num_simulations = int(args.num_nodes * 2.5)
    load_epoch = None
    args.epochs = 2000
    args.log_interval = 1
    args.model_save_interval = 25
    args.temp_threshold = 2
    args.one_layer_val = True
    args.mini_batch_size = 1024*8
    args.model_load = parse_saved_model_dir(args, "pretrained_result", "val_as_baseline",
                                            load_epoch, return_checkpoint=True, ignore_debug=True)
    run_mcts_train(args)

    # search_params(1)

