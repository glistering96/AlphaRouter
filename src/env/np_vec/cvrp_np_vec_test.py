import numpy as np
import torch
import traceback
import random
import multiprocessing as mp

from src.env.np_vec.cvrp_np_vec_env import CVRPNpVec


def model(avail):
    actions = []

    # get the available node from each env
    for i in range(avail.shape[0]):
        avail_nodes = list(map(lambda y: y[0], filter(lambda x: x[1] == True, enumerate(avail[i]))))
        action = random.choice(avail_nodes)
        actions.append([action])

    return np.array(actions)


def run(i):
    env = CVRPNpVec(num_nodes=100, num_env=128, step_reward=False, render_mode='human',
                   seed=None)

    obs, _ = env.reset()
    all_done = False
    a = 1

    try:
        while not all_done:
            # sample random action from only available nodes
            avail = obs['available']

            # sample one action for each env (num_env, 1)
            action = model(avail)

            obs, reward, done, _, _ = env.step(action)
            all_done = done.all()
            a += 1

        print(f"Done at step {i}")

    except:
        print(avail)
        raise ValueError(f'Error at step {a}')


if __name__ == '__main__':
    try:
        pool = mp.Pool(4)
        pool.map_async(run, range(10000))
        pool.close()
        pool.join()

    except:
        traceback.print_exc()