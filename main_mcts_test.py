import json
from pathlib import Path

from src.common.utils import get_param_dict, parse_saved_model_dir, dict_product, check_debug
from src.run import parse_args
from src.mcts_tester import TesterModule
import torch.multiprocessing as mp


def run_test( **kwargs):
    epochs = kwargs['epochs']
    args = parse_args()

    for k, v in kwargs.items():
        setattr(args, k, v)

    args.model_load = parse_saved_model_dir(args, "pretrained_result", args.name_prefix, epochs, mcts_param=False,
                                        ignore_debug=True, return_checkpoint=True)

    env_params, mcts_params, model_params, h_params, run_params, logger_params, optimizer_params = get_param_dict(args)

    tester = TesterModule(env_params=env_params,
                            model_params=model_params,
                            mcts_params=mcts_params,
                            logger_params=logger_params,
                            run_params=run_params)

    score, runtime = tester.run()

    save_path = parse_saved_model_dir(args, args.result_dir, args.name_prefix, mcts_param=False,
                                      return_checkpoint=False)

    # the result json folder is the same as the model folder
    return score, runtime, epochs, save_path, mcts_params


def run_parallel_test(param_ranges, num_proc=4):
    """
    Parallel test with multiprocessing
    :param param_ranges: dict of parameter ranges. e.g. {"num_nodes": [1, 2, 3]}

    """

    result = {}
    # result: {'mcts_param_set_0': {'params': mcts_params, 'scores': {epoch: score, epoch: score, ...}, 'runtime': 10},
    #          'mcts_param_set_1': {'params': mcts_params, 'scores': {epoch: score, epoch: score, ...}, 'runtime': 10},
    save_path = None
    runtime = 0
    i = 0
    async_result = mp.Queue()

    if not check_debug():
        def __callback(val):
            async_result.put(val)

        pool = mp.Pool(num_proc)

        for params in dict_product(param_ranges):
            pool.apply_async(run_test, kwds=params, callback=__callback)

        pool.close()
        pool.join()

    else:
        for params in dict_product(param_ranges):
            async_result.put(run_test(**params))

    mcts_param_idx = 0

    while not async_result.empty():
        score, time, k, save_path, mcts_params = async_result.get()

        # if mcts params are not in the result dict, then it is a new set of mcts params
        # so we need to create a new key for the result dict

        assigned = False

        # check if the mcts
        for param_set_name, param_set in result.items():
            if mcts_params in param_set.values():
                # if mcts params are in the result dict, then we just need to add the score and runtime
                param_set['score'][k] = score
                param_set['runtime'] += time
                assigned = True
                break

        if not assigned:
            # if mcts params are not in the result dict, then it is a new set of mcts params
            # so we need to create a new key for the result dict
            result[f'mcts_param_set_{mcts_param_idx}'] = {}
            result[f'mcts_param_set_{mcts_param_idx}']['params'] = mcts_params
            result[f'mcts_param_set_{mcts_param_idx}']['score'] = {k: score}
            result[f'mcts_param_set_{mcts_param_idx}']['runtime'] = time
            mcts_param_idx += 1

    assert save_path is not None, "Wrong in fetching save path"
    # save path is the same as the model path

    # calculate the average runtime for each mcts param set
    for param_set_name, param_set in result.items():
        runtime = param_set['runtime']
        i = len(param_set['score'])
        param_set['runtime'] = runtime / i

    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True, exist_ok=False)

    filename = "test_result"
    file_dir = f'{save_path}/{filename}.json'

    num_iter = 0

    while Path(file_dir).exists():
        filename = f'{filename}_{num_iter}'
        file_dir = f'{save_path}/{filename}.json'
        num_iter += 1

    with open(file_dir, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    param_dict = {
        "num_nodes": [20],
        "num_simulations": [40],
        "cpuct": [1.1, 2],
        'epochs': [495000, 500000],
        'env_type': ['tsp'],
        'encoder_layer_num': [2],
        'render_mode': [None]
    }
    run_test(**{'epochs': 40000, 'test_data_type': 'pkl' })
    # run_parallel_test(param_dict, num_proc=1)