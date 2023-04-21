import os.path

import numpy as np


def make_cord(num_rollouts, num_depots, num_nodes):
    depot_cord = np.random.rand(num_rollouts, num_depots, 2)
    node_cord = np.random.rand(num_rollouts, num_nodes, 2)
    depot_node_cord = np.concatenate([depot_cord, node_cord], axis=1)
    return depot_node_cord.astype(np.float32)


# def make_demands(num_rollouts, num_depots, num_nodes):
#     depot_demands = np.zeros((num_rollouts, num_depots))
#     node_demands = np.random.poisson(3.5, (num_rollouts, num_nodes)) / math.sqrt(110)
#     node_demands += np.random.beta(2, 8, (num_rollouts, num_nodes))
#     node_demands = np.clip(node_demands, 0.0, 0.9999)
#     depot_node_demands = np.concatenate([depot_demands, node_demands], axis=1)
#     return depot_node_demands


def make_demands(num_rollouts, num_depots, num_nodes):
    depot_demands = np.zeros((num_rollouts, num_depots))

    if num_nodes == 20 or num_nodes==10:
        demand_scaler = 30
    elif num_nodes == 50:
        demand_scaler = 40
    elif num_nodes == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError

    node_demands = np.random.randint(1, 10, size=(num_rollouts, num_nodes)) / float(demand_scaler)
    depot_node_demands = np.concatenate([depot_demands, node_demands], axis=1)
    return depot_node_demands.astype(np.float32)


if __name__ == '__main__':
    # print(make_demands(1, 1, 20).shape)
    from pathlib import Path

    num_depots = 1
    num_nodes = 20

    data_folder_path = f'../../data'
    filepath = data_folder_path + f'/D_{num_depots}-N_{num_nodes}.npz'
    print("/".join(filepath.split("/")[:-1]))
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path, exist_ok=True)

    xy = make_cord(1, num_depots, num_nodes)
    demands = make_demands(1, num_depots, num_nodes)

    np.savez_compressed(filepath, xy=xy, demands=demands)

    loaded = np.load(filepath)
    # print(loaded['xy'])
