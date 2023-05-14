from argparse import ArgumentParser

from src.common.dir_parser import DirParser
from src.common.utils import get_param_dict
from src.mcts_tester import MCTSTesterModule
from src.pretrain_model.pretrain_tester import AMTesterModule
from src.pretrain_model.pretrainer_module import PreTrainerModule


def parse_args():
    parser = ArgumentParser()

    # env params
    parser.add_argument("--env_type", type=str, help="Type of environment to use")
    parser.add_argument("--num_nodes", type=int, help="Number of nodes in the test data generation")
    parser.add_argument("--num_depots", type=int, default=1, help="Number of depots in the test data generation")
    parser.add_argument("--render_mode", type=str, default='rgb_array', help="Type of render for the environment")
    parser.add_argument("--step_reward", type=bool, default=False,
                        help="whether to have step reward. If false, only the "
                             "reward in the last transition will be returned")
    parser.add_argument("--test_data_type", type=str, default='npz', help="extension for test data file")
    parser.add_argument("--test_data_idx", type=int, default=0, help="index for loading data for pkl datasets")

    # model params
    parser.add_argument("--nn", type=str, default='shared_mha', help="type of policy network to use")
    parser.add_argument("--embedding_dim", type=int, default=128, help="embedding dim of network")
    parser.add_argument("--encoder_layer_num", type=int, default=4, help="encoder layer of network.")
    parser.add_argument("--qkv_dim", type=int, default=32, help="attention dim")
    parser.add_argument("--head_num", type=int, default=4, help="attention head dim")
    parser.add_argument("--C", type=int, default=10, help="C parameter that is applied to the tanh activation on the"
                                                          " last layer output of policy network")

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
    parser.add_argument("--num_episode", type=int, default=128, help="number of parallel episodes to run or collect")
    parser.add_argument("--load_epoch", type=int, default=None, help="If value is not None, it will load the model")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate of ADAM optimizer")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Coefficient for entropy regularizer")
    parser.add_argument("--gpu_id", type=int, default=0, help="Id of gpu to use")
    parser.add_argument("--num_proc", type=int, default=4, help="number of episodes to run")
    parser.add_argument("--test_num", type=int, default=None, help="number of workers for data loading")

    # etc.
    parser.add_argument("--result_dir", type=str, default='result', help="Result folder directory.")
    parser.add_argument("--tb_log_dir", type=str, default='logs', help="Result log folder (tensorboard) directory.")
    parser.add_argument("--model_save_interval", type=int, default=10000, help="interval for model savings")
    parser.add_argument("--log_interval", type=int, default=10000, help="interval for model logging")
    parser.add_argument("--data_path", type=str, default='./data', help="Test data file locations")
    parser.add_argument("--name_prefix", type=str, default='', help="name prefix")
    parser.add_argument("--seed", type=int, default=1, help="values smaller than 1 will not set any seeds")

    args = parser.parse_args()

    if args.test_num is None:
        args.test_num = args.num_nodes

    return args


def run_mcts_test(args):
    env_params, mcts_params, model_params, h_params, run_params, logger_params, optimizer_params = get_param_dict(args)

    tester = MCTSTesterModule(env_params=env_params,
                              model_params=model_params,
                              logger_params=logger_params,
                              mcts_params=mcts_params,
                              run_params=run_params,
                              dir_parser=DirParser(args),
                              )

    return tester.run()


def run_pretrain(args):
    env_params, mcts_params, model_params, h_params, run_params, logger_params, optimizer_params = get_param_dict(args)

    trainer = PreTrainerModule(env_params=env_params,
                               model_params=model_params,
                               logger_params=logger_params,
                               run_params=run_params,
                               optimizer_params=optimizer_params,
                               dir_parser=DirParser(args))

    trainer.run()


def run_am_test(args):
    env_params, mcts_params, model_params, h_params, run_params, logger_params, optimizer_params = get_param_dict(args)

    tester = AMTesterModule(env_params=env_params,
                            model_params=model_params,
                            logger_params=logger_params,
                            run_params=run_params,
                            dir_parser=DirParser(args)
                            )

    return tester.run()
