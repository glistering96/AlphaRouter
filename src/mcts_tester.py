from collections.abc import Callable, Iterable, Mapping
import time
from pathlib import Path
from typing import Any

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
        self.load_epoch = run_params['model_load']['epoch']
        self.model_params['ckpt'] = self.load_epoch

        self._load_model(self.load_epoch)

    def run(self, use_mcts):
        self.time_estimator.reset(self.epochs)
        global hparam_writer
        test_score, runtime, entropy = self.test_one_episode(use_mcts=use_mcts)

        # self.logger.info(f"Test score: {test_score: .5f}")
        # self.logger.info(" *** Testing Done *** ")

        # self.record_video()

        return test_score, runtime, entropy          
            
    def work(self, mcts, obs):
        mcts.model.to("cuda")
        # print(f"id of mcts model: {id(mcts.model)}")
        # print(f"sum of weights of mcts model: {mcts.model.encoder.input_embedder.weight.sum()}")
        return mcts.get_action_prob(obs)
    
    
    def test_with_mcts(self, obs, num_cpu=1, pool: mp.Pool = None, mcts_lst : list=None): 
        if num_cpu > 1:
            assert pool is not None, "pool must be provided when num_cpu > 1"
            assert mcts_lst is not None, "mcts_lst must be provided when num_cpu > 1"
            
            results = pool.starmap(self.work, zip(mcts_lst, [obs for _ in range(num_cpu)]))

            visit_count_agg = dict()

            # aggregate visit counts from the result's mcts_run_info.
            # result is a list of (action, mcts_run_info) tuples
            for _, mcts_run_info in results:
                visit_counts = mcts_run_info['visit_counts_stats']
                
                for a, v in visit_counts.items():
                    if a in visit_count_agg:
                        visit_count_agg[a] += v
                    else:
                        visit_count_agg[a] = v

            # visit_counts_stats = {a: v for a, v in zip(actions, visit_counts)}
            # get the action with the highest visit count
            action = max(visit_count_agg, key=visit_count_agg.get)

        else:
            action, mcts_info = self.mcts.get_action_prob(obs)
            node_visit_count = mcts_info['visit_counts_stats']
            priors = mcts_info['priors']
            
        return action, node_visit_count, priors

    def test_one_episode(self, use_mcts):
        self.env.set_test_mode()
        obs, _ = self.env.reset()
        done = False
        self.model.eval()
        self.model.encoding = None
        num_cpu = 1
        entropy = []
        
        if use_mcts and num_cpu > 1:
            self.model.to("cpu")
            self.model.share_memory()
            self.mcts_params["num_simulations"] = self.mcts_params['num_simulations'] // num_cpu + 1
            pool = mp.Pool(num_cpu)
            mcts_lst = [MCTS(self.env, self.model_params, self.env_params, self.mcts_params, self.dir_parser, model=self.model) for _ in range(num_cpu)]
            
        elif use_mcts and num_cpu == 1:
            self.mcts = MCTS(self.env, self.model_params, self.env_params, self.mcts_params, self.dir_parser, model=self.model)
            pool = None
            mcts_lst = None
        
        if use_mcts:
            agent_type = 'mcts'
        else:
            agent_type = 'am'
                
        start = time.time()
        
        save_path = Path(f'./debug/plot/{self.env.env_type}/{self.env.test_num}/{self.env._load_data_idx}/{agent_type}/')

        if not save_path.exists():
            save_path.mkdir(parents=True)
            
        with torch.no_grad():
            while not done:
                avail = obs['available']

                if (avail == True).sum() == 1:
                    action = np.where(avail == True)[2][0]

                else:
                    # pre_eval and get the difference of probabilities that are the highest and the fifth highest
                    self.model.to("cuda")
                    action_probs, _ = self.model(obs)
                    dist = action_probs.cpu().numpy().reshape(-1)
                    dist = np.sort(dist)
                    diff = dist[-1] - dist[-5]
                    ent = -np.sum(action_probs.cpu().numpy() * np.log(action_probs.cpu().numpy() + 1e-8))
                    entropy.append(float(ent))

                    # if the probability difference is greater than 0.75, use the action with the highest probability
                    if diff > 0.75 or use_mcts is False:
                        action_probs = action_probs.cpu().numpy().reshape(-1)
                        action = int(np.argmax(action_probs, -1))
                        priors = {a: p for a, p in enumerate(action_probs)}
                        node_visit_count = None

                    # if the probability difference is less than 0.75, use mcts to search further
                    else:
                        action, node_visit_count, priors = self.test_with_mcts(obs, num_cpu=num_cpu, pool=pool, mcts_lst=mcts_lst)

                next_state, reward, done, _, _ = self.env.step(obs, action)

                # self.env.plot(obs, agent_type, node_visit_count=node_visit_count, priors=priors,
                #          iteration=obs['t'], save_path=save_path)

                obs = next_state

                if done:
                    if num_cpu > 1:
                        pool.close()
                        pool.join()
                                            
                    # self.env.plot(obs, agent_type, node_visit_count=node_visit_count, priors=priors,
                    #      iteration=obs['t'], save_path=save_path)
                    
                    return reward, time.time() - start, entropy
