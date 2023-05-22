import itertools
import json
import os
import time
from pathlib import Path

import torch
from ray.train.lightning import LightningConfigBuilder, LightningTrainer

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

        latest_epoch = list(
            map(lambda x: int(x), list(
                filter(lambda x: x.isdigit(), list(
                    map(lambda x: x.split('-')[1].split('.')[0], os.listdir(saved_model_path)))
                       )
            )
                )
        )

        if latest_epoch:
            args.load_epoch = max(latest_epoch)

    run_pretrain(args)


def search_params(num_proc):
    hyper_param_dict = {
        'env_type':  ['tsp'],
        'num_nodes': [20],
        'result_dir' : ['pretrained_result'],
        'render_mode' : [None],
        'num_parallel_env' : [1024],
        'num_steps_in_epoch': [1],
        'grad_acc': [1],
        'model_save_interval': [100],
        'nn_train_epochs': [100000],
        'lr': [1e-3, 3e-4, 1e-4, 5e-5, 1e-5],

    }

    # hyper_param_dict = {
    #     'env_type':  ['tsp'],
    #     'num_nodes': [20],
    #     'result_dir' : ['pretrained_result'],
    #     'render_mode' : [None],
    #     'num_parallel_env' : [512, 1024],
    #     'num_steps_in_epoch': [1, 10, 100],
    #     'grad_acc': [1, 2, 4, 10],
    #     'model_save_interval': [100],
    #     'nn_train_epochs': [100000],
    #     'lr': [3e-4, 1e-4, 5e-5, 1e-5],
    #
    # }
    save_path = None

    async_result = mp.Queue()

    # def __callback(val):
    #     async_result.put(val)

    pool = mp.Pool(num_proc)

    for params in dict_product(hyper_param_dict):
        if params['num_steps_in_epoch'] < params['grad_acc']:
            continue

        pool.apply_async(_work, kwds=params,
                         # callback=__callback
                         )

    pool.close()
    pool.join()


from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining


def ray_tune_search():
    nn_train_epochs = 100000

    params = {
        'num_nodes' : 50,
        'result_dir' : 'pretrained_result',
        'name_prefix' : 'ray_tune',
        'render_mode' : None,
        'qkv_dim' : 32,
        'load_from_the_latest' : False,
        'env_type' : 'tsp',
        'embedding_dim': 128,
        'model_save_interval': 100,
        'nn_train_epochs': nn_train_epochs,
    }


    config = {
        'num_parallel_env' : tune.choice([512, 1024]),
        'num_steps_in_epoch': tune.choice([1, 10, 100]),
        'grad_acc': tune.choice([1, 2, 4, 10]),
        'lr': tune.loguniform(1e-5, 3e-4),
    }

    args = parse_args()

    for k, v in params.items():
        setattr(args, k, v)

    from ray.tune.integration.pytorch_lightning import TuneReportCallback
    callback = TuneReportCallback(
        {
            "score": "train_score",
        },
        on="train_end")

    import pytorch_lightning as pl

    def train_tune(config, epochs=10, gpus=0):
      model = AMTrainer(config)
      trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=gpus,
        progress_bar_refresh_rate=0,
        callbacks=[callback])
      trainer.fit(model)

if __name__ == '__main__':
    # params = {
    # 'num_nodes' : 50,
    # 'result_dir' : 'pretrained_result',
    # 'name_prefix' : '',
    # 'render_mode' : None,
    # 'num_episode' : 1024,
    # 'qkv_dim' : 32,
    # 'load_from_the_latest' : False,
    # 'env_type' : 'tsp',
    # 'embedding_dim': 128,
    # 'nn_train_epochs': 100000,
    # 'model_save_interval': 10,
    # 'num_parallel_env': 1024,
    # 'lr': 3e-4,
    # 'grad_acc': 2,
    # 'num_steps_in_epoch': 100
    #
    # }
    # #
    # # for grad_acc, num_steps_in_epoch in itertools.product([1, 5, 10], [1, 10, 100]):
    # #     params['grad_acc'] = grad_acc
    # #     params['num_steps_in_epoch'] = num_steps_in_epoch
    # #     _work(**params)
    #
    # torch.set_float32_matmul_precision('high')
    #
    # _work(**params)
    #
    search_params(3)
    #
    # # for qkv_dim in [32, 64]:
    # #     for embedding_dim in [128, 256]:
    # #         params['qkv_dim'] = qkv_dim
    # #         params['embedding_dim'] = embedding_dim
    # #         _work(**params)

    # ray_tune_search()


