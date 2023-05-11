import json
from pathlib import Path

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



if __name__ == '__main__':
    problem_size = 20
    num_problems = 100

    run_param_dict = {
        'test_data_type': ['pkl'],
        'env_type': ['tsp'],
        'num_nodes': [problem_size],
        'num_episode': [1024],
        'test_data_idx': list(range(num_problems), ),
        'num_simulations': [problem_size*2]
    }

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

    for load_epoch in list(range(400000, 500000, 5000)) + ['best', 'best_val_loss']:
        run_param_dict['load_epoch'] = [load_epoch]
        run_parallel_test(run_param_dict, num_proc=4)

