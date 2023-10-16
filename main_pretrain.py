import itertools
import json
import os
import time
from pathlib import Path
from src.common.utils import check_debug
import torch

from src.common.utils import dict_product
from src.pretrain_model.pretrainer_module_pl import AMTrainer
from src.run import parse_args, run_pretrain

import torch.multiprocessing as mp
from src.common.dir_parser import DirParser


def _work(**kwargs):
    if 'load_from_the_latest' in kwargs:
        load_from_the_latest = kwargs.pop('load_from_the_latest')

    else:
        load_from_the_latest = False

    args = parse_args()

    for k, v in kwargs.items():
        setattr(args, k, v)

    if load_from_the_latest:
        saved_model_path = DirParser(args).get_model_checkpoint()

        try:
            latest_epoch = max(map(lambda x: int(x.split('-')[0].split('=')[1]), os.listdir(saved_model_path)))
            
            load_ckpt = list(filter(lambda x: f'epoch={latest_epoch}' in x, os.listdir(saved_model_path)))[0]

            # get the latest epoch full path
            load_ckpt = os.path.join(saved_model_path, load_ckpt)
                
            if latest_epoch:
                args.load_epoch = load_ckpt
                
        except:
            print(f'No saved model found in {saved_model_path}')
            args.load_epoch = None

    run_pretrain(args)


if __name__ == '__main__':    
    torch.set_float32_matmul_precision('high')
    params = {
        'num_nodes' : [100],
        'result_dir' : ['pretrained_result'],
        'name_prefix' : [""],
        'render_mode' : [None],
        'qkv_dim' : [32],
        'num_heads': [4],
        'load_from_the_latest' : [False],
        'env_type' : ['tsp', 'cvrp'],
        'embedding_dim': [128],
        'encoder_layer_num':[6],
        'nn_train_epochs': [300],
        'model_save_interval': [1],
        'num_parallel_env': [64],
        'lr': [1e-4],
        'grad_acc': [1],
        'num_steps_in_epoch': [100*1000 // 64],
        'baseline': ['mean', 'val'],
        'activation': ['relu', 'swiglu'],
        'load_from_the_latest': [True]
    }
    
    if check_debug():
        for param in dict_product(params):
            _work(**param)
    
    else:
        pool = mp.Pool(2)
        
        for param in dict_product(params):
            pool.apply_async(_work, kwds=param)
            
        pool.close()
        pool.join()

    # for param in dict_product(params):
    #     print(param)
    #     _work(**param)



