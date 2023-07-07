from argparse import ArgumentParser

import lightning.pytorch as pl
import torch.utils.data
from lightning.pytorch import loggers as pl_loggers

from src.common.dir_parser import DirParser
from src.common.utils import get_param_dict
from src.mcts_tester import MCTSTesterModule
from src.pretrain_model.pretrain_tester import AMTesterModule
from src.pretrain_model.pretrainer_module_pl import AMTrainer


def parse_args():
    parser = ArgumentParser()

    # env params
    parser.add_argument("--env_type", type=str, help="Type of environment to use")
    parser.add_argument("--num_nodes", type=int, help="Number of nodes in the test data generation")
    parser.add_argument("--num_depots", type=int, default=1, help="Number of depots in the test data generation")
    parser.add_argument("--render_mode", type=str, default=None, help="Type of render for the environment")
    parser.add_argument("--step_reward", type=bool, default=False,
                        help="whether to have step reward. If false, only the "
                             "reward in the last transition will be returned")
    parser.add_argument("--test_data_type", type=str, default='npz', help="extension for test data file")
    parser.add_argument("--test_data_idx", type=int, default=0, help="index for loading data for pkl datasets")
    parser.add_argument("--num_parallel_env", type=int, default=512, help="number of parallel episodes to run or collect")
    parser.add_argument("--data_path", type=str, default='./data', help="Test data file locations")
    parser.add_argument("--test_num", type=int, default=None, help="Number of nodes to test on")

    # model params
    parser.add_argument("--nn", type=str, default='shared_mha', help="type of policy network to use")
    parser.add_argument("--embedding_dim", type=int, default=128, help="embedding dim of network")
    parser.add_argument("--encoder_layer_num", type=int, default=4, help="encoder layer of network.")
    parser.add_argument("--qkv_dim", type=int, default=32, help="attention dim")
    parser.add_argument("--head_num", type=int, default=4, help="attention head dim")
    parser.add_argument("--C", type=int, default=10, help="C parameter that is applied to the tanh activation on the"
                                                          " last layer output of policy network")
    parser.add_argument("--activation", type=str, default='swiglu', choices=['swiglu', 'relu'], help="activation function")

    # mcts params
    parser.add_argument("--num_simulations", type=int, default=50, help="Number of simulations")
    parser.add_argument("--temp_threshold", type=int, default=5, help="Temperature threshold")
    parser.add_argument("--noise_eta", type=float, default=0.25, help="Noise eta param")
    parser.add_argument("--cpuct", type=float, default=1.1, help="cpuct param")
    parser.add_argument("--normalize_value", type=bool, default=True, help="Normalize q values in mcts search")
    parser.add_argument("--rollout_game", type=bool, default=False, help="whether to rollout to the terminal episode")

    # run params
    parser.add_argument("--mini_batch_size", type=int, default=2048, help="mini-batch size")
    parser.add_argument("--nn_train_epochs", type=int, default=500000, help="number of training epochs")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument("--load_epoch", type=int, default=None, help="If value is not None, it will load the model")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of ADAM optimizer")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Coefficient for entropy regularizer")
    parser.add_argument("--gpu_id", type=int, default=0, help="Id of gpu to use")
    parser.add_argument("--grad_acc", type=int, default=0, help="Accumulations of gradients")
    parser.add_argument("--num_steps_in_epoch", type=int, default=4, help="num_steps_in_epoch")
    parser.add_argument("--warm_up", type=int, default=1000, help="lr scheduler warm up steps")
    parser.add_argument("--baseline", type=str, default='val', help="baseline function for policy update")

    # etc.
    parser.add_argument("--result_dir", type=str, default='result', help="Result folder directory.")
    parser.add_argument("--tb_log_dir", type=str, default='logs', help="Result log folder (tensorboard) directory.")
    parser.add_argument("--model_save_interval", type=int, default=10000, help="interval for model savings")
    parser.add_argument("--log_interval", type=int, default=10000, help="interval for model logging")
    parser.add_argument("--name_prefix", type=str, default='', help="name prefix")
    parser.add_argument("--seed", type=int, default=1, help="values smaller than 1 will not set any seeds")

    args = parser.parse_args()

    if args.test_num is None:
        args.test_num = args.num_nodes

    return args


def run_mcts_test(args):
    env_params, mcts_params, model_params, h_params, run_params, optimizer_params, logger_params = get_param_dict(args, return_logger=True)

    tester = MCTSTesterModule(env_params=env_params,
                              model_params=model_params,
                              logger_params=logger_params,
                              mcts_params=mcts_params,
                              run_params=run_params,
                              dir_parser=DirParser(args),
                              )

    return tester.run()


def run_pretrain(args):
    env_params, mcts_params, model_params, h_params, run_params, optimizer_params = get_param_dict(args)

    grad_acc = args.grad_acc if args.grad_acc > 1 else 1
    num_steps_in_epoch = args.num_steps_in_epoch

    model = AMTrainer(env_params=env_params,
                               model_params=model_params,
                               run_params=run_params,
                               optimizer_params=optimizer_params,)

    model.save_hyperparameters(h_params)
    default_root_dir = DirParser(args).get_model_root_dir()
    max_epochs = run_params['nn_train_epochs']

    logger = pl_loggers.TensorBoardLogger(save_dir=DirParser(args).get_tensorboard_logging_dir(),
                                          name='', max_queue=500)

    score_cp_callback = pl.callbacks.ModelCheckpoint(
        dirpath=default_root_dir,
        monitor="train_score",
        every_n_epochs=1,
        mode="min",
        filename="{epoch}-{train_score:.5f}",
        save_on_train_epoch_end=True,
        save_top_k=5
    )

    trainer = pl.Trainer(
        accumulate_grad_batches=grad_acc,
        logger=logger,
        log_every_n_steps=100,
        max_epochs=max_epochs,
        default_root_dir=default_root_dir,
        precision="16-mixed",
        callbacks=[score_cp_callback],
        gradient_clip_val=1.0
    )

    dummy_dl = torch.utils.data.DataLoader(torch.zeros((num_steps_in_epoch, 1, 1, 1)), batch_size=1)
    trainer.fit(model,
                train_dataloaders=dummy_dl)


def run_am_test(args):
    env_params, mcts_params, model_params, h_params, run_params, logger_params, optimizer_params = get_param_dict(args, return_logger=True)

    tester = AMTesterModule(env_params=env_params,
                            model_params=model_params,
                            logger_params=logger_params,
                            run_params=run_params,
                            dir_parser=DirParser(args)
                            )

    return tester.run()
