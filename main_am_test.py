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

    print(f"Done! Loaded from :{kwargs['result_dir']}/{kwargs['load_epoch']}. "
          f"Tested on: {kwargs['test_data_idx']}. Scored: {score:.5f} in {runtime:.2f} seconds.")

    return score, runtime, args.test_data_idx, kwargs['load_epoch'], kwargs['result_dir']


def get_ckpt_path(params, pivot=None):
    all_files, ckpt_root = collect_all_checkpoints(params)
    all_checkpoints = [x.split('/')[-1].split('\\')[-1].split('.ckpt')[0] for x in all_files]

    if pivot is not None:
        if pivot == 'train_score':
            all_checkpoints.sort(
                key=lambda x: float(x.split('-')[1].split('=')[-1])
            )
            ckpt = all_checkpoints[0]
            # minimum score ckpt
            
        elif pivot == 'epoch':
            all_checkpoints.sort(
                key=lambda x: float(x.split('-')[0].split('=')[-1])
            )
            ckpt = all_checkpoints[-1]
            
    else:
        ckpt = all_checkpoints
            
    return ckpt


def run_parallel_test(param_ranges, num_proc=5):
    """
    Parallel test with multiprocessing
    :param param_ranges: dict of parameter ranges. e.g. {"num_nodes": [1, 2, 3]}

    """
    result_dict = {}
    async_result = mp.Queue()

    def __callback(val):
        async_result.put(val)
    
    pivot = 'epoch'

    
    if num_proc > 1:
        pool = mp.Pool(num_proc)

        for params in dict_product(param_ranges):
            result_dir = get_result_dir(params, mcts=False)
            ckpt = get_ckpt_path(params, pivot=pivot)
            
            if pivot is not None:
                input_params = deepcopy(params)
                input_params['load_epoch'] = ckpt
                input_params['result_dir'] = result_dir

                pool.apply_async(run_test, kwds=input_params, callback=__callback)
                
            else:
                for _ckpt in ckpt:
                    input_params = deepcopy(params)
                    input_params['load_epoch'] = _ckpt
                    input_params['result_dir'] = result_dir

                    pool.apply_async(run_test, kwds=input_params, callback=__callback)

        pool.close()
        pool.join()

    else:
        for params in dict_product(param_ranges):
            result_dir = get_result_dir(params, mcts=False)
            ckpt = get_ckpt_path(params, pivot=pivot)
            
            if pivot is not None:
                input_params = deepcopy(params)
                input_params['load_epoch'] = ckpt
                input_params['result_dir'] = result_dir

                result = run_test(**input_params)
                print("run finished")
                async_result.put(result)

            else:
                for _ckpt in ckpt:
                    input_params = deepcopy(params)
                    input_params['load_epoch'] = _ckpt
                    input_params['result_dir'] = result_dir
                    result = run_test(**input_params)
                    async_result.put(result)

    while not async_result.empty():
        score, runtime, test_data_idx, load_epoch, result_dir = async_result.get()

        if result_dir not in result_dict.keys():
            result_dict[result_dir] = {}

        if load_epoch not in result_dict[result_dir].keys():
            result_dict[result_dir][load_epoch] = {}

        if test_data_idx not in result_dict[result_dir][load_epoch].keys():
            result_dict[result_dir][load_epoch][test_data_idx] = {}

        result_dict[result_dir][load_epoch][test_data_idx] = {'score': score, 'runtime': runtime}
    
    async_result.close()
    
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
    num_problems = 1000

    run_param_dict = {
        'test_data_type': ['pkl'],
        'env_type': ['tsp'],
        'num_nodes': [20],
        'num_parallel_env': [num_env],
        'test_data_idx': list(range(num_problems)),
        'data_path': ['./data'],
        'activation': ['swiglu'],
        'baseline': ['val', 'mean'],
        'qkv_dim': [32],
        'num_heads': [4],
        'embedding_dim': [128],
        'grad_acc': [1],
        'num_steps_in_epoch': [100 * 1000 // num_env],
    }

    print("run_param_dict: ", run_param_dict)
    for num_nodes in [100]:
        for encoder_layer_num in [6]:
            run_param_dict['num_nodes'] = [num_nodes]
            run_param_dict['encoder_layer_num'] = [encoder_layer_num]
            
            print("run_param_dict: ", run_param_dict)
            result = run_parallel_test(run_param_dict, 1)

            if 'name_prefix' in run_param_dict.keys():
                path_format = "./result_summary_5678/am"
                
            else:
                path_format = "./result_summary_5678/am"

            for result_dir in result.keys():
                all_result = {}
                path = f"{path_format}/{result_dir}"

                if not Path(path).exists():
                    Path(path).mkdir(parents=True, exist_ok=False)

                for load_epoch in result[result_dir].keys():
                    # write the result_dict to a json file
                    save_json(result[result_dir][load_epoch], f"{path}/{load_epoch}.json")

                    all_result[load_epoch] = {'result_avg': result[result_dir][load_epoch]['average'],
                                            'result_std': result[result_dir][load_epoch]['std']}

                save_json(all_result, f"{path}/all_result_avg.json")


def debug():
    num_env = 64

    run_param_dict = {
        'test_data_type': ['pkl'],
        'env_type': ['tsp'],
        'num_nodes': [20],
        'num_parallel_env': [num_env],
        'test_data_idx': [51, 71, 77, 78, 90, 98],
        'data_path': ['./data'],
        'activation': ['swiglu'],
        'baseline': ['val'],
        'encoder_layer_num': [6],
        'qkv_dim': [32],
        'num_heads': [4],
        'embedding_dim': [128],
        'grad_acc': [1],
        'num_steps_in_epoch': [100 * 1000 // num_env],
        'num_simulations': [1000],
        'cpuct': [1.1]
    }

    for num_nodes in [50]:
        run_param_dict['num_nodes'] = [num_nodes]
        result = run_parallel_test(run_param_dict, 6)

        # path_format = "./result_summary/debug/am"

        # for result_dir in result.keys():
        #     all_result = {}
        #     path = f"{path_format}/{result_dir}"

        #     if not Path(path).exists():
        #         Path(path).mkdir(parents=True, exist_ok=False)

        #     for load_epoch in result[result_dir].keys():
        #         # write the result_dict to a json file
        #         save_json(result[result_dir][load_epoch], f"{path}/{load_epoch}.json")

        #         all_result[load_epoch] = {'result_avg': result[result_dir][load_epoch]['average'],
        #                                   'result_std': result[result_dir][load_epoch]['std']}

        #     save_json(all_result, f"{path}/all_result_avg.json")


if __name__ == '__main__':
    # main()
    debug()

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


