import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from gymnasium.wrappers import RecordVideo

from src.mcts import MCTS, Node
from src.module_base import RolloutBase

from copy import deepcopy


class MCTSTesterModule(RolloutBase):
    def __init__(self, env_params, model_params, mcts_params, logger_params, run_params, dir_parser):
        # save arguments
        super().__init__(env_params, model_params, mcts_params, logger_params, run_params, dir_parser)
        global hparam_writer

        self.env = self.env_setup.create_env(test=True)
        load_epoch = run_params['model_load']['epoch']

        self._load_model(load_epoch)

    def run(self, use_mcts):
        self.time_estimator.reset(self.epochs)
        global hparam_writer

        test_score, runtime = test_one_episode(self.env, self.model, self.mcts_params, use_mcts=use_mcts)

        # self.logger.info(f"Test score: {test_score: .5f}")
        # self.logger.info(" *** Testing Done *** ")

        # self.record_video()

        return test_score, runtime


def test_one_episode(env, agent, mcts_params, use_mcts):
    env.set_test_mode()
    obs, _ = env.reset()
    done = False
    agent.eval()

    agent.encoding = None
    num_cpu = 8
    mcts_params['num_simulations'] = mcts_params['num_simulations'] // num_cpu + 1 if use_mcts else mcts_params['num_simulations']
    pool = mp.Pool(num_cpu)
    
    start = time.time()
    save_path = Path('./debug/plot/tsp/')

    if not save_path.exists():
        save_path.mkdir(parents=True)
        
    with torch.no_grad():
        # if use_mcts:
        #     agent_type = 'mcts'
        # else:
        #     agent_type = 'am'

        while not done:
            avail = obs['available']

            if (avail == True).sum() == 1:
                action = np.where(avail == True)[2][0]

            else:
                if use_mcts:

                    
                    if num_cpu > 1:
                        result = pool.map(MCTS(env, agent, mcts_params).get_action_prob, [deepcopy(obs) for _ in range(4)])
                        visit_count_agg = dict()
                        
                        # aggregate visit counts from the result's mcts_run_info. 
                        # result is a list of (action, mcts_run_info) tuples
                        for action, mcts_run_info in result:
                            visit_counts = mcts_run_info['visit_counts_stats']
                            for a, v in visit_counts.items():
                                if a not in visit_count_agg:
                                    visit_count_agg[a] = v
                                else:
                                    visit_count_agg[a] += v
                        
                        # visit_counts_stats = {a: v for a, v in zip(actions, visit_counts)}
                        # get the action with the highest visit count
                        action = max(visit_count_agg, key=visit_count_agg.get)
                        
                    mcts = MCTS(env, agent, mcts_params)
                    action, mcts_info = mcts.get_action_prob(obs)
                    node_visit_count = mcts_info['visit_counts_stats']
                    priors = mcts_info['priors']                   
                    
                else:
                    action_probs, _ = agent(obs)
                    action_probs = action_probs.cpu().numpy().reshape(-1)
                    action = int(np.argmax(action_probs, -1))

                    priors = {a: p for a, p in enumerate(action_probs)}
                    node_visit_count = None

            next_state, reward, done, _, _ = env.step(obs, action)

            # env.plot(obs, node_visit_count=node_visit_count, priors=priors,
            #          iteration=obs['t'], agent_type=agent_type, save_path=save_path)

            obs = next_state

            if done:
                
                pool.close()
                pool.join()
                
                return reward, time.time() - start
