import json
from glob import glob
from pathlib import Path

from src.common.dir_parser import DirParser
from src.common.utils import dict_product, check_debug
from src.run import parse_args
import torch.multiprocessing as mp
from src.run import run_mcts_test


def run_test(**kwargs):
    args = parse_args()

    for k, v in kwargs.items():
        setattr(args, k, v)

    score, runtime = run_mcts_test(args)

    return score, runtime, args.test_data_idx


def run_parallel_test(param_ranges, num_proc=4):
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
        pool.apply_async(run_test, kwds=params, callback=__callback)

    pool.close()
    pool.join()

    if async_result.empty():
        return

    while not async_result.empty():
        score, runtime, test_data_idx = async_result.get()
        result_dict[test_data_idx] = {'score': score, 'runtime': runtime}

    avg_score = sum([result_dict[i]['score'] for i in range(num_problems)]) / num_problems
    avg_runtime = sum([result_dict[i]['runtime'] for i in range(num_problems)]) / num_problems

    std_score = sum([(result_dict[i]['score'] - avg_score) ** 2 for i in range(num_problems)]) / num_problems
    std_runtime = sum([(result_dict[i]['runtime'] - avg_runtime) ** 2 for i in range(num_problems)]) / num_problems

    result_dict['average'] = {'score': avg_score, 'runtime': avg_runtime}
    result_dict['std'] = {'score': std_score, 'runtime': std_runtime}

    path = f"./result_summary/mcts/{run_param_dict['env_type'][0]}_{problem_size}"
    if not Path(path).exists():
        Path(path).mkdir(parents=True, exist_ok=False)

    # write the result_dict to a json file
    with open(f"{path}/{run_param_dict['load_epoch'][0]}.json", 'w') as f:
        json.dump(result_dict, f, indent=4)


def run_cross_test():
    params = {
        'num_nodes' : 50,   # num_nodes model to load
        'result_dir' : 'pretrained_result',
        'name_prefix' : '',
        'render_mode' : None,
        'num_episode' : 1024,
        'qkv_dim' : 32,
        'env_type' : 'tsp',
        'embedding_dim': 128,
        'test_num': 20,
        'load_epoch': 'best',
        'test_data_type': 'pkl',
    }

    test_result = {}

    for load_from in [20, 50]:
        for test_num in [20, 50]:
            for test_data_idx in range(100):
                params['num_nodes'] = load_from
                params['test_data_idx'] = test_data_idx
                params['test_num'] = test_num

                score, _, test_data_idx = run_test(**params)
                key = f"Trained on {load_from} Test on {test_num}"
                if key not in test_result:
                    test_result[key] = {}

                test_result[key][test_data_idx] = score

    path = f"./result_summary/cross_test_result/mcts/{params['env_type']}"
    if not Path(path).exists():
        Path(path).mkdir(parents=True, exist_ok=False)

    # write the result_dict to a json file
    with open(f"{path}/result.json", 'w') as f:
        json.dump(test_result, f, indent=4)

    # get average score for each key
    avg_result = {}
    for key, result in test_result.items():
        avg_result[key] = sum(result.values()) / len(result)

    print(avg_result)


def collect_all_checkpoints(run_param_dict):
    all_files = []

    for params in dict_product(run_param_dict):
        args = parse_args()

        for k, v in params.items():
            setattr(args, k, v)

        ckpt_root = DirParser(args).get_model_checkpoint()

        all_files = glob(ckpt_root + '/*.ckpt')

        # get the checkpoint with minimum train_score from the all_files
        all_files = list(sorted(all_files, key=lambda x: float(x.split('train_score=')[1].split('.ckpt')[0])))

    return all_files, ckpt_root


if __name__ == '__main__':
    problem_size = 20
    num_problems = 100 // 20
    num_env = 64

    run_param_dict = {
        'test_data_type': ['pkl'],
        'env_type': ['tsp'],
        'num_nodes': [problem_size],
        'num_parallel_env': [num_env],
        'test_data_idx': list(range(num_problems)),
        'num_simulations': [problem_size*2],
        'data_path': ['./data'],
        'activation': ['swiglu'],
        'baseline': ['val'],
        'encoder_layer_num': [6],
        'qkv_dim': [32],
        'num_heads': [4],
        'embedding_dim': [128],
        'grad_acc': [1],
        'num_steps_in_epoch': [100 * 1000 // num_env],
        'load_epoch': ['epoch=1-train_score=3.79818']

    }

    all_result = {}
    all_files, ckpt_root = collect_all_checkpoints(run_param_dict)

    for k in range(len(all_files)):
        load_epoch = all_files[k]
        result = {}

        for params in dict_product(run_param_dict):
            args = parse_args()

            for k, v in params.items():
                setattr(args, k, v)

            ckpt_root = DirParser(args).get_model_checkpoint()

            params['load_epoch'] = load_epoch.split('/')[-1].split('\\')[-1].split('.ckpt')[0]

            score, runtime, args.test_data_idx = run_test(**params)
            result[args.test_data_idx] = {'score': score, 'runtime': runtime}

        # average score
        avg_score = sum([result[i]['score'] for i in range(num_problems)]) / num_problems
        avg_runtime = sum([result[i]['runtime'] for i in range(num_problems)]) / num_problems

        std_score = sum([(result[i]['score'] - avg_score) ** 2 for i in range(num_problems)]) / num_problems
        std_runtime = sum([(result[i]['runtime'] - avg_runtime) ** 2 for i in range(num_problems)]) / num_problems

        result['average'] = {'score': avg_score, 'runtime': avg_runtime}
        result['std'] = {'score': std_score, 'runtime': std_runtime}

        # save the result as json file
        print(f"{load_epoch}: {result['average']}")
        path = f"./result_summary/{ckpt_root[1:]}"

        if not Path(path).exists():
            Path(path).mkdir(parents=True, exist_ok=False)

        # write the result_dict to a json file
        file_nm = load_epoch.split('/')[-1].split('\\')[-1].split('.ckpt')[0]

        with open(f"{path}/{file_nm}.json", 'w') as f:
            json.dump(result, f, indent=4)

        all_result[file_nm] = result

    path = f"./result_summary/{ckpt_root[1:]}"
    # write the result_dict to a json file
    with open(f"{path}/all_result.json", 'w') as f:
        json.dump(all_result, f, indent=4)

    # run_param_dict['num_nodes'] = problem_size



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
    #     run_parallel_test(run_param_dict, num_proc=4)

    # run_cross_test()

