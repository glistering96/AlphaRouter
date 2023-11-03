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

class Worker(mp.Process):
    def __init__(self, input_queue, output_queue, 
                 env, model_params, env_params, mcts_params, dir_parser):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.mcts = MCTS(env, model_params, env_params, mcts_params, dir_parser)
        self.is_running = True

    def get_action_prob(self, obs):
        return self.mcts.get_action_prob(obs)
    
    def run(self):
        self.is_running = True
        
        while True and self.is_running:
            if not self.input_queue.empty():
                flag, obs = self.input_queue.get()
                self.input_queue.put((0, None))
                
                if flag == 1:
                    self.is_running = False
                    _, mcts_info = self.mcts.get_action_prob(obs)
                    self.output_queue.put((mcts_info))
                    
            else:
                self.is_running = False
                
                


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
        test_score, runtime = self.test_one_episode(use_mcts=use_mcts)

        # self.logger.info(f"Test score: {test_score: .5f}")
        # self.logger.info(" *** Testing Done *** ")

        # self.record_video()

        return test_score, runtime            
            
    
    def test_one_episode(self, use_mcts):
        with torch.no_grad():
            self.env.set_test_mode()
            obs, _ = self.env.reset()
            done = False
            self.model.eval()
            self.model.encoding = None
            

            num_cpu = 2
            self.mcts_params['num_simulations'] = self.mcts_params['num_simulations'] // num_cpu + 1
            input_queue = mp.Queue()
            output_queue = mp.Queue()
            
            for _ in range(num_cpu):
                input_queue.put((0, None))
                
            processes = []
            
            for _ in range(num_cpu):
                p = Worker(input_queue, output_queue, 
                           self.env, self.model_params, self.env_params, self.mcts_params, self.dir_parser)
                processes.append(p)
                
            for p in processes:
                p.start()
                
            mcts = MCTS(self.env, self.model_params, self.env_params, self.mcts_params, self.dir_parser, model=self.model)
            
            start = time.time()
                            
            save_path = Path('./debug/plot/tsp/')

            if not save_path.exists():
                save_path.mkdir(parents=True)

            while not done:
                avail = obs['available']

                if (avail == True).sum() == 1:
                    action = np.where(avail == True)[2][0]

                else:
                    if use_mcts and num_cpu > 1:
                        for _ in range(num_cpu):                    
                            input_queue.put((1, deepcopy(obs)))
                            
                        for p in processes:
                            p.run()
                            
                        while True:
                            synced= sum([p.is_running for p in processes])
                            
                            if synced == 0:
                                for p in processes:
                                    p.is_running = True
                                break
                        
                        visit_count_agg = {}         
                        
                        while not output_queue.empty():
                            mcts_run_info = output_queue.get()
                            visit_counts = mcts_run_info['visit_counts_stats']
                            for a, v in visit_counts.items():
                                if a in visit_count_agg:
                                    visit_count_agg[a] += v
                                else:
                                    visit_count_agg[a] = v
                        
                        action = max(visit_count_agg, key=visit_count_agg.get)
                        
                    elif use_mcts and num_cpu == 1:
                        action, mcts_info = mcts.get_action_prob(obs)
     
                        
                    else:
                        action_probs, _ = self.model(obs)
                        action_probs = action_probs.cpu().numpy().reshape(-1)
                        action = int(np.argmax(action_probs, -1))


                next_state, reward, done, _, _ = self.env.step(obs, action)
                
                obs = next_state
                print()
                print("#"*100)
                print()

                if done:
                    if num_cpu > 1:
                        for p in processes:
                            p.terminate()
                            p.close()
                            p.join()


                    return reward, time.time() - start
