import json
from copy import deepcopy
from pathlib import Path
from time import sleep

from src.common.utils import dict_product, save_json, cal_average_std, collect_all_checkpoints, get_result_dir
from src.run import parse_args
import torch.multiprocessing as mp
from src.run import run_am_test


def run_test(**kwargs):
    args = parse_args()

    for k, v in kwargs.items():
        setattr(args, k, v)
    score, runtime = run_am_test(args)

    print(f"Done {kwargs['result_dir']}/{kwargs['load_epoch']}")

    return score, runtime, args.test_data_idx, kwargs['load_epoch'], kwargs['result_dir']


def run_parallel_test(param_ranges, num_proc=5):
    """
    Parallel test with multiprocessing
    :param param_ranges: dict of parameter ranges. e.g. {"num_nodes": [1, 2, 3]}

    """
    result_dict = {}
    async_result = mp.Queue()

    def __callback(val):
        async_result.put(val)

    pool = mp.Pool(num_proc)

    for params in dict_product(param_ranges):
        all_files, ckpt_root = collect_all_checkpoints(params)
        all_checkpoints = [x.split('/')[-1].split('\\')[-1].split('.ckpt')[0] for x in all_files]
        result_dir = get_result_dir(params, mcts=False)

        for ckpt in all_checkpoints:
            input_params = deepcopy(params)
            input_params['load_epoch'] = ckpt
            input_params['result_dir'] = result_dir

            pool.apply_async(run_test, kwds=input_params, callback=__callback)

    pool.close()
    pool.join()

    if async_result.empty():
        return

    while not async_result.empty():
        score, runtime, test_data_idx, load_epoch, result_dir = async_result.get()

        if result_dir not in result_dict.keys():
            result_dict[result_dir] = {}

        if load_epoch not in result_dict[result_dir].keys():
            result_dict[result_dir][load_epoch] = {}

        if test_data_idx not in result_dict[result_dir][load_epoch].keys():
            result_dict[result_dir][load_epoch][test_data_idx] = {}

        result_dict[result_dir][load_epoch][test_data_idx] = {'score': score, 'runtime': runtime}

    organized_result = {}

    # average of score and runtime for load_epochs should be calculated for each result_dir and average of score and runtime
    # for each load_epoch should be calculated over all test_data_idx

    for result_dir, dir_result in result_dict.items():
        organized_result[result_dir] = {}

        for load_epoch, epoch_result in dir_result.items():
            organized_result[result_dir][load_epoch] = cal_average_std(epoch_result)

    return organized_result


def main():
    num_env = 64
    num_problems = 100

    run_param_dict = {
        'test_data_type': ['pkl'],
        'env_type': ['tsp', 'cvrp'],
        'num_nodes': [20],
        'num_parallel_env': [num_env],
        'test_data_idx': list(range(num_problems)),
        'data_path': ['./data'],
        'activation': ['relu', 'swiglu'],
        'baseline': ['val', 'mean'],
        'encoder_layer_num': [6],
        'qkv_dim': [32],
        'num_heads': [4],
        'embedding_dim': [128],
        'grad_acc': [1],
        'num_steps_in_epoch': [100 * 1000 // num_env]
    }

    result = run_parallel_test(run_param_dict, 10)

    path_format = "./result_summary/am"

    for result_dir in result.keys():
        all_result = {}

        for load_epoch in result[result_dir].keys():
            # save the result as json file
            print(f"{result_dir}/{load_epoch}: {result[result_dir][load_epoch]['average']}")
            path = f"{path_format}/{result_dir}"

            if not Path(path).exists():
                Path(path).mkdir(parents=True, exist_ok=False)

            # write the result_dict to a json file
            save_json(result[result_dir][load_epoch], f"{path}/{load_epoch}.json")

            all_result[load_epoch] = {'result_avg': result[result_dir][load_epoch]['average'],
                                      'result_std': result[result_dir][load_epoch]['std']}

        save_json(all_result, f"{path}/all_result_avg.json")


def debug():
    num_env = 64
    num_problems = 2

    run_param_dict = {
        'test_data_type': ['pkl'],
        'env_type': ['tsp', 'cvrp'],
        'num_nodes': [20],
        'num_parallel_env': [num_env],
        'test_data_idx': list(range(num_problems)),
        'data_path': ['./data'],
        'activation': ['swiglu'],
        'baseline': ['val'],
        'encoder_layer_num': [6],
        'qkv_dim': [32],
        'num_heads': [4],
        'embedding_dim': [128],
        'grad_acc': [1],
        'num_steps_in_epoch': [100 * 1000 // num_env]
    }

    result = run_parallel_test(run_param_dict, 10)

    path_format = "./result_summary/am"
    all_result = {}

    for result_dir in result.keys():
        for load_epoch in result[result_dir].keys():
            # save the result as json file
            print(f"{result_dir}/{load_epoch}: {result[result_dir][load_epoch]['average']}")
            path = f"{path_format}/{result_dir}"

            if not Path(path).exists():
                Path(path).mkdir(parents=True, exist_ok=False)

            # write the result_dict to a json file
            save_json(result[result_dir][load_epoch], f"{path}/{load_epoch}.json")

            all_result[load_epoch] = {'result_avg': result[result_dir][load_epoch]['average'],
                                      'result_std': result[result_dir][load_epoch]['std']}

        save_json(all_result, f"{path}/all_result_avg.json")



if __name__ == '__main__':
    main()
    # debug()

    # problem_size = 20
    # num_problems = 100
    # num_env = 64
    # run_param_dict = {
    #     'test_data_type': ['pkl'],
    #     'env_type': ['tsp'],
    #     'num_nodes': [problem_size],
    #     'num_parallel_env': [num_env],
    #     'test_data_idx': list(range(num_problems)),
    #     'num_simulations': [problem_size * 2],
    #     'data_path': ['./data'],
    #     'activation': ['swiglu'],
    #     'baseline': ['val'],
    #     'encoder_layer_num': [6],
    #     'qkv_dim': [32],
    #     'num_heads': [4],
    #     'embedding_dim': [128],
    #     'grad_acc': [1],
    #     'num_steps_in_epoch': [100 * 1000 // num_env],
    #     # 'load_epoch': ['epoch=1-train_score=3.79818']
    # }
    #
    # for params in dict_product(run_param_dict):
    #     run_test(**params)



    # result_dict = {}
    #
    # for i in range(num_problems):
    #     run_param_dict['test_data_idx'] = i
    #     score, runtime, i = run_test(**run_param_dict)
    #
    #     result_dict[i] = {'score': score, 'runtime': runtime}
    #
    # if not Path(f"./result_summary/").exists():
    #     Path(f"./result_summary/").mkdir(parents=True, exist_ok=False)
    #
    # # write the result_dict to a json file
    # with open(f"./result_summary/{run_param_dict['env_type']}_{problem_size}_test_result.json", 'w') as f:
    #     json.dump(result_dict, f, indent=4)
    #

    # for load_epoch in list(range(400000, 500000, 5000)) + ['best', 'best_val_loss']:
    #     run_param_dict['load_epoch'] = [load_epoch]
    #     run_parallel_test(run_param_dict, num_proc=10)

    # for param in dict_product(run_param_dict):
    #     param['load_epoch'] = 'best'
    #     run_test(**param)

    # run_cross_test()


