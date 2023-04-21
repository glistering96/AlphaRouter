import warnings

import numpy as np
import torch
import torch.nn.functional as F
import time

B, T = 10000, 2000

device = torch.device('cuda', 0)
_input = torch.rand((B, T)).to(device)
target = torch.rand((B,)).to(device)

def measure_time(func, *args):
    start = time.time()
    result = func(*args)
    runtime = time.time() - start
    print(runtime)
    print(result)

def manual(_in, _target):
    # reshape_target = target.unsqueeze(-1).reshape(B, T)
    reshape_target = torch.broadcast_to(target.unsqueeze(-1), _in.shape)
    result = 0.5 * ((_input - reshape_target) ** 2).mean()
    return result


def torch_func(_in, _target):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = 0.5*F.mse_loss(_in, _target.unsqueeze(-1))
    return result


def get_result_json(path, type='mcts'):
    try:
        with open(path, "r") as f:
            result = json.load(f)
    except FileNotFoundError:
        return

    sorted_dict = dict(sorted(result.items(), key=lambda x: x[1]))

    if 'time' in sorted_dict:
        del sorted_dict['time']

    top_10_vals = list(sorted_dict.values())[:10]
    top_10_means = float(np.mean(top_10_vals))
    print(f"{type}: {top_10_means: .5f}")
    return sorted_dict

def parse_from_json(mcts_dict):
    mcst_result = {'epoch': [], 'cpuct': [], 'score': []}

    for k, v in mcts_dict.items():
        epoch, cpuct, _ = k.split('-')

        try:
            mcst_result['epoch'].append(int(epoch.split('_')[-1]))
            mcst_result['cpuct'].append(float(cpuct.split('_')[-1]))
            mcst_result['score'].append(v)
        except ValueError:
            pass

    return


if __name__ == '__main__':
    # print(manual(_input, target))
    # measure_time(manual, _input, target)
    # measure_time(torch_func, _input, target)

    # import matplotlib.pyplot as plt
    #
    # a = np.random.dirichlet([0.06925207756232687 for _ in range(100)])
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(a)
    # plt.show()


    import json
    import pandas as pd
    N = 50
    print("Two layers, val baseline:")
    path = f'./mcts_test_result/val_as_baseline/N_{N}/nn-shared_mha-128-2-32-8-10-2/param_result_0.json'
    mcts_dict = get_result_json(path, type='alpha_route')

    path = f'./am_mcts_test_result/val_as_baseline/N_{N}/nn-shared_mha-128-2-32-8-10-2/param_result.json'
    mcts_dict = get_result_json(path, type="am_mcts")

    path = f'./am_test_result/val_as_baseline/N_{N}/nn-shared_mha-128-2-32-8-10-2/param_result.json'
    mcts_dict = get_result_json(path, type="AM")

    print("One layer, mean baseline:")
    path = f'./mcts_test_result/mean_as_baseline/N_{N}/nn-shared_mha-128-1-32-8-10-1/param_result_0.json'
    mcts_dict = get_result_json(path)

    path = f'./am_mcts_test_result/mean_as_baseline/N_{N}/nn-shared_mha-128-1-32-8-10-1/param_result.json'
    mcts_dict = get_result_json(path, type="am_mcts")

    path = f'./am_test_result/mean_as_baseline/N_{N}/nn-shared_mha-128-1-32-8-10-1/param_result.json'
    mcts_dict = get_result_json(path, type="AM")



