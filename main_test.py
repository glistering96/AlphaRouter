import itertools
import json
import math
from pathlib import Path

import torch.multiprocessing as mp

from src.common.utils import parse_saved_model_dir, dict_product, check_debug
from src.run import parse_args, run_mcts_test, run_am_test


def _work(**kwargs):
    args = parse_args()
    target_epoch = 1000
    model_result_dir = ""
    model_name_prefix = ""
    real_kwargs = {}
    run_type = 'mcts'

    for k, v in kwargs.items():
        if k in args:
            setattr(args, k, v)
            real_kwargs[k] = v

        elif k == 'epoch':
            target_epoch = v
            real_kwargs[k] = v

        elif k == 'model_result_dir':
            model_result_dir = v

        elif k == 'model_name_prefix':
            model_name_prefix = v

        elif k == 'run_result_dir':
            args.result_dir = v

        elif k == 'run_name_prefix':
            args.name_prefix = v

        elif k == 'run_type':
            run_type = v

        else:
            raise KeyError

    del real_kwargs['num_nodes']
    del real_kwargs['one_layer_val']
    del real_kwargs['encoder_layer_num']

    args.model_load = parse_saved_model_dir(args, model_result_dir, model_name_prefix, target_epoch,
                                            mcts_param=False, return_checkpoint=True, ignore_debug=True)

    save_path = parse_saved_model_dir(args, args.result_dir, args.name_prefix, mcts_param=False,
                                      return_checkpoint=False)

    score, time = run_mcts_test(args)

    str_vals = [f"{name}_{val}" for name, val in zip(real_kwargs.keys(), real_kwargs.values())]
    key = "-".join(str_vals)
    return key, score, time, save_path


def search_params(num_proc, num_nodes, run_result_dir, run_name_prefix, model_result_dir, model_name_prefix, **kwargs):
    hyper_param_dict = {
        'num_nodes': [num_nodes],
        'epoch': list(range(1000, 500000, 1000)) + ['best'],
        'cpuct': [1.1, 4.5],
        'model_result_dir': [model_result_dir],
        'model_name_prefix': [model_name_prefix],
        'run_result_dir': [run_result_dir],
        'run_name_prefix': [run_name_prefix]
    }

    for k, v in kwargs.items():
        hyper_param_dict[k] = [v]

    result = {}
    save_path = None
    runtime = 0
    i = 0

    if not check_debug():
        async_result = mp.Queue()

        def __callback(val):
            async_result.put(val)

        pool = mp.Pool(num_proc)

        for params in dict_product(hyper_param_dict):
            pool.apply_async(_work, kwds=params, callback=__callback)

        pool.close()
        pool.join()

        while not async_result.empty():
            k, score, time, save_path = async_result.get()
            result[k] = score
            runtime += time
            i += 1

    else:
        for params in dict_product(hyper_param_dict):
            _work(**params)

    assert save_path is not None, "Wrong in fetching save path"
    result['time'] = runtime / i

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


def test_with_am(num_nodes):
    args = parse_args()
    args.result_dir = f'am_test_result'
    args.num_nodes = num_nodes
    args.cpuct = 1.1
    args.rollout_game = False
    args.one_layer_val = True
    args.num_simulations = num_nodes*2

    for _encoder_layer_num in [1, 2]:
        for name_prefix in ['mean_as_baseline', 'val_as_baseline']:
            for one_layer_val in [True, False]:
                result = {}
                args.encoder_layer_num = _encoder_layer_num
                args.name_prefix = name_prefix
                args.one_layer_val = one_layer_val
                save_path = parse_saved_model_dir(args, args.result_dir, args.name_prefix, mcts_param=False,
                                                  return_checkpoint=False)

                runtime_total = 0
                i = 0

                for load_epoch in list(range(1000, 500000, 1000)) + ['best']:
                    args.model_load = parse_saved_model_dir(args, "pretrained_result", name_prefix, load_epoch, mcts_param=False,
                                                            ignore_debug=True, return_checkpoint=True)

                    try:
                        score, runtime = run_am_test(args)
                        result[load_epoch] = score
                        runtime_total += runtime
                        i += 1

                    except FileNotFoundError:
                        continue

                if i == 0:
                    continue

                result['time'] = runtime_total / i

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
    # result_dir = 'alpha_router_test_result'
    # model_result_dir = "pretrained_result"
    # N = 20
    # for nm in ['mean_as_baseline', 'val_as_baseline']:
    #     for t in [True, False]:
    #         for _encoder_layer_num in [1, 2]:
    #             try:
    #                 search_params(8, N, result_dir, nm, model_result_dir, nm, one_layer_val=t,
    #                               encoder_layer_num=_encoder_layer_num, rollout_game=False)
    #
    #             except FileNotFoundError:
    #                 print(nm, nm, t, _encoder_layer_num)
    #
    #             except AssertionError:
    #                 print(nm, nm, t, _encoder_layer_num)
    #
    # result_dir = 'am_mcts_test_result'
    # model_result_dir = "pretrained_result"
    # N = 20
    # for nm in ['mean_as_baseline', 'val_as_baseline']:
    #     for t in [True, False]:
    #         for _encoder_layer_num in [1, 2]:
    #             try:
    #                 search_params(8, N, result_dir, nm, model_result_dir, nm, one_layer_val=t,
    #                               encoder_layer_num=_encoder_layer_num, rollout_game=False)
    #
    #             except FileNotFoundError:
    #                 print(nm, nm, t, _encoder_layer_num)
    #
    #             except AssertionError:
    #                 print(nm, nm, t, _encoder_layer_num)

    test_with_am(20)