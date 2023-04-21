from collections import namedtuple
from dataclasses import dataclass

import numpy as np

rollout_result = namedtuple("rollout_result", 'state action reward entropy pi val done')


@dataclass
class StepState:
    load: np.ndarray = None     # (num_rollout, 1)
    current_node: np.ndarray = None     # (num_rollout, 2)
    available: np.ndarray = None        # (num_rollout, 1)
    done: np.ndarray = None
    episode_idx: np.ndarray = None
    visited_nodes: np.ndarray = None

    t: int = 0