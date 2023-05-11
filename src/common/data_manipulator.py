import os.path

import numpy as np


def make_cord(num_rollouts, num_depots, num_nodes):
    depot_cord = np.random.rand(num_rollouts, num_depots, 2).astype(np.float16)
    node_cord = np.random.rand(num_rollouts, num_nodes, 2).astype(np.float16)
    depot_node_cord = np.concatenate([depot_cord, node_cord], axis=1)
    return depot_node_cord


# def make_demands(num_rollouts, num_depots, num_nodes):
#     depot_demands = np.zeros((num_rollouts, num_depots))
#     node_demands = np.random.poisson(3.5, (num_rollouts, num_nodes)) / math.sqrt(110)
#     node_demands += np.random.beta(2, 8, (num_rollouts, num_nodes))
#     node_demands = np.clip(node_demands, 0.0, 0.9999)
#     depot_node_demands = np.concatenate([depot_demands, node_demands], axis=1)
#     return depot_node_demands


def make_demands(num_rollouts, num_depots, num_nodes):
    depot_demands = np.zeros((num_rollouts, num_depots)).astype(np.float16)

    if num_nodes == 20 or num_nodes == 10:
        demand_scaler = 30
    elif num_nodes == 50:
        demand_scaler = 40
    elif num_nodes == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError

    node_demands = np.random.randint(1, 10, size=(num_rollouts, num_nodes), dtype=np.float16) / float(demand_scaler)
    depot_node_demands = np.concatenate([depot_demands, node_demands], axis=1)
    return depot_node_demands


def create_problem(env_type, num_depots, num_nodes, num_rollouts):
    if env_type == "cvrp":
        xy = make_cord(num_rollouts, num_depots, num_nodes)
        demands = make_demands(num_rollouts, num_depots, num_nodes)
        return xy, demands

    elif env_type == "tsp":
        xy = make_cord(num_rollouts, 0, num_nodes)
        return xy, None

    else:
        raise NotImplementedError


if __name__ == '__main__':
    # print(make_demands(1, 1, 20).shape)
    for num_nodes in [20, 50, 100, 500, 1000]:
        num_depots = 1
        env_type = 'tsp'

        data_folder_path = f'../../data/{env_type}/'
        filename = f'D_{num_depots}-N_{num_nodes}.npz' if env_type == 'cvrp' else f'N_{num_nodes}.npz'

        filepath = data_folder_path + filename

        if not os.path.exists(data_folder_path):
            os.makedirs(data_folder_path, exist_ok=True)

        xy, demands = create_problem(env_type, num_depots, num_nodes, 1)
        np.savez_compressed(filepath, xy=xy, demands=demands)

        loaded = np.load(filepath)
