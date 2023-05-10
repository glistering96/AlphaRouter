from src.common.utils import check_debug


class DirParser:
    def __init__(self, args):
        self._env_param_nm = f"{args.env_type}/N_{args.num_nodes}-B_{args.num_episode}"
        self._model_param_nm = f"{args.nn}-{args.embedding_dim}-{args.encoder_layer_num}-{args.qkv_dim}-{args.head_num}-{args.C}"
        self._mcts_param_nm = f"ns_{args.num_simulations}-temp_{args.temp_threshold}-cpuct_{args.cpuct}-" \
                              f"norm_{args.normalize_value}-rollout_{args.rollout_game}-ec_{args.ent_coef:.4f}"

        self._name_prefix = args.name_prefix
        self.tb_log_dir = args.tb_log_dir
        self._common_part = f"{self._name_prefix}/{self._env_param_nm}/{self._model_param_nm}"

    def get_tensorboard_logging_dir(self):
        model_root_dir = self.get_model_root_dir()
        model_root_dir_without_dot = model_root_dir[2:]
        return f"{self.tb_log_dir}/{model_root_dir_without_dot}"

    def get_model_root_dir(self):
        main_dir = "pretrained_result" if not check_debug() else "debug"
        return f"./{main_dir}/{self._common_part}"

    def get_model_checkpoint(self, target_epoch=None):
        main_dir = "pretrained_result"
        if target_epoch is not None:
            return f"./{main_dir}/{self._common_part}/saved_models/checkpoint-{target_epoch}.pt"

        else:
            return f"./{main_dir}/{self._common_part}/saved_models"

    def get_result_dir(self, mcts=False):
        if mcts:
            return f"{self.get_model_root_dir()}/{self._mcts_param_nm}"

        else:
            return f"{self.get_model_root_dir()}"
