from collections import deque
import json
import math
from copy import deepcopy
from pathlib import Path

import torch.multiprocessing as mp

from src.common.utils import (cal_average_std, collect_all_checkpoints,
                              dict_product, get_result_dir, save_json, load_json)
from src.run import parse_args, run_mcts_test
import torch

class user_queue:
    def __init__(self):
        self.queue = deque()
        
    def empty(self):
        return len(self.queue) == 0
    
    def put(self, val):
        self.queue.append(val)
        
    def get(self):
        return self.queue.popleft()
    

class user_queue:
    def __init__(self):
        self.queue = deque()
        
    def empty(self):
        return len(self.queue) == 0
    
    def put(self, val):
        self.queue.append(val)
        
    def get(self):
        return self.queue.popleft()
    

def run_test(**kwargs):
    import warnings
    warnings.simplefilter("ignore", UserWarning)
    torch.set_float32_matmul_precision('high')

    args = parse_args()

    for k, v in kwargs.items():
        setattr(args, k, v)

    # args.num_simulations = args.num_nodes * 2

    score, runtime = run_mcts_test(args)

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
            result_dir = get_result_dir(params, mcts=True)
            ckpt = get_ckpt_path(params, pivot=pivot)
            
            if pivot is not None:
                input_params = deepcopy(params)
                input_params['load_epoch'] = ckpt
                input_params['result_dir'] = result_dir

                _result = pool.apply(run_test, kwds=input_params)
                async_result.put(_result)
                
            else:
                for _ckpt in ckpt:
                    input_params = deepcopy(params)
                    input_params['load_epoch'] = _ckpt
                    input_params['result_dir'] = result_dir


                    _result = pool.apply(run_test, kwds=input_params)
                    async_result.put(_result)

        pool.close()
        pool.join()

    else:
        async_result = user_queue()
        for params in dict_product(param_ranges):
            result_dir = get_result_dir(params, mcts=True)
            ckpt = get_ckpt_path(params, pivot=pivot)
            
            if pivot is not None:
                input_params = deepcopy(params)
                input_params['load_epoch'] = ckpt
                input_params['result_dir'] = result_dir

                result = run_test(**input_params)
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
        
    organized_result = {}

    # average of score and runtime for load_epochs should be calculated for each result_dir and average of score and runtime
    # for each load_epoch should be calculated over all test_data_idx

    for result_dir, dir_result in result_dict.items():
        organized_result[result_dir] = {}

        for load_epoch, epoch_result in dir_result.items():
            organized_result[result_dir][load_epoch] = cal_average_std(epoch_result)

    return organized_result


def run_cross_test():
    num_env = 64
    num_problems = 1000
    
    run_param_dict = {
        'test_data_type': ['pkl'],

        'num_nodes': [20],
        'num_parallel_env': [num_env],
        'test_data_idx': list(range(num_problems)),
        'data_path': ['./data'],
        'activation': ['swiglu'],
        'baseline': ['mean', 'val'],
        'encoder_layer_num': [6],
        'qkv_dim': [32],
        'num_heads': [4],
        'embedding_dim': [128],
        'grad_acc': [1],
        'num_steps_in_epoch': [100 * 1000 // num_env],
        'num_simulations': [100, 500, 1000],
        'cpuct': [1.1],
        
    }
    for env_type in ['tsp']:
        for load_from in [20, 50, 100]:
            for test_num in [20, 50, 100]:
                
                if load_from == test_num:
                    continue
                
                print(f"Testing on {test_num} problems with model trained on {load_from} problems")
                run_param_dict['num_nodes'] = [load_from]
                run_param_dict['test_num'] = [test_num]
                run_param_dict['env_type'] = [env_type]
                result = run_parallel_test(run_param_dict, 6)
        
                path_format = f"./result_summary/cross_test_result/mcts/ent-0.1/trained_on-{load_from}-test_on-{test_num}"
                for result_dir in result.keys():            
                    path = f"{path_format}/{result_dir}"
                    
                    all_result = load_json(f"{path}/all_result_avg.json")

                    if not Path(path).exists():
                        Path(path).mkdir(parents=True, exist_ok=False)

                    for load_epoch in result[result_dir].keys():
                        # write the result_dict to a json file
                        save_json(result[result_dir][load_epoch], f"{path}/{load_epoch}.json")

                        all_result[load_epoch] = {'result_avg': result[result_dir][load_epoch]['average'],
                                                'result_std': result[result_dir][load_epoch]['std']}

                    save_json(all_result, f"{path}/all_result_avg.json")

    print("Done!")


def main():
    num_env = 64
    num_problems = 100
    test_seed = 1234
    
    run_param_dict = {
        'test_data_type': ['pkl'],
        'env_type': ['cvrp'],
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
        'num_steps_in_epoch': [100 * 1000 // num_env],
        'num_simulations': [100, 500, 1000],
        'test_data_seed': [test_seed]
    }

    for env_type in ['cvrp']:
        for num_nodes in [100]:        
            for selection_coef in [0.75]:
                run_param_dict['env_type'] = [env_type]
                run_param_dict['num_nodes'] = [num_nodes]
                run_param_dict['selection_coef'] = [selection_coef]

                result = run_parallel_test(run_param_dict, 1)

                path_format = f"./result_summary_{test_seed}/mcts/diff-{selection_coef}"
                
                for result_dir in result.keys():            
                    path = f"{path_format}/{result_dir}"
                    
                    all_result = load_json(f"{path}/all_result_avg.json")

                    if not Path(path).exists():
                        Path(path).mkdir(parents=True, exist_ok=False)

                    for load_epoch in result[result_dir].keys():
                        # write the result_dict to a json file
                        save_json(result[result_dir][load_epoch], f"{path}/{load_epoch}.json")

                        all_result[load_epoch] = {'result_avg': result[result_dir][load_epoch]['average'],
                                                'result_std': result[result_dir][load_epoch]['std']}

                    save_json(all_result, f"{path}/all_result_avg.json")

    print("Done!")

if __name__ == '__main__':
    main()
    
    # run_cross_test()