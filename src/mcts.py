import logging
import math
from copy import deepcopy

import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, env, model, mcts_params, training=True):

        self.env = deepcopy(env)
        self.env_type = env.env_type
        self.model = model
        self.mcts_params = mcts_params
        self.action_space = env.action_space.n
        self.expand_root = True
        self.cpuct = mcts_params['cpuct']
        self.normalize_q_value = mcts_params['normalize_value']
        self.noise_eta = mcts_params['noise_eta']
        self.rollout_game = mcts_params['rollout_game']

        self.training = training

        self.Q = {}  # stores Q values for s,a (as defined in the paper)
        self.W = {}  # sum of values
        self.Ns = {}  # sum of visit counts for state s
        self.N = {}  # stores #times edge s,a was visited
        self.P = {}  # stores initial policy (returned by neural net), dtype: numpy ndarray

        self.max_q_val = -float('inf')
        self.min_q_val = float('inf')

        if isinstance(self.env, RecordVideo):
            self.env = self.env.env

        self.target_env_address = self.env

        self._save_state_field()

    def _cal_probs(self, target_state, temp):
        s = target_state['t']
        counts = [self.N[(s, a)] if (s, a) in self.N else 0 for a in range(self.action_space)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1

        else:
            counts = [x ** (1. / temp) for x in counts]
            counts_sum = float(sum(counts)) + 1e-8
            probs = [x / counts_sum for x in counts]

        return probs

    def get_action_prob(self, target_state, temp=1):
        """
        Simulate and return the probability for the target state based on the visit counts acquired from simulations
        """

        for i in range(self.mcts_params['num_simulations']):
            self._run_simulation(target_state)

        probs = self._cal_probs(target_state, temp)
        return probs

    def _select(self, state):
        # select the action
        # argmin (Q-U), since the reward concept becomes the loss in this MCTS
        # originally, argmax (Q+U) is true but to fit into minimization, argmin (-Q+U) is selected
        s = state['t']
        avail_mask = state['available'].reshape(-1, )
        ucb_scores = {}

        # pick the action with the lowest upper confidence bound
        for a in range(self.action_space):
            # mask unavailable or non-promising actions

            if not avail_mask[a]:
                continue

            ucb = self.cpuct * self.P[(s, a)] * math.sqrt(self.Ns[s]) / (1 + self.N[(s, a)])

            Q_val = self.Q[(s, a)]

            normalizable = self.max_q_val > self.min_q_val

            if self.normalize_q_value and normalizable:
                Q_val = (Q_val - self.min_q_val) / (self.max_q_val - self.min_q_val + 1e-8)

            score = -Q_val + ucb
            ucb_scores[a] = score

        max_ucb = max(ucb_scores.values())
        a = np.random.choice([action for action, score in ucb_scores.items() if score == max_ucb])
        return int(a)

    def _expand(self, state, add_noise=False, return_action=False):
        s = state['t']

        prob_dist, val = self.model(state)
        probs = prob_dist.view(-1, ).cpu().numpy()
        avail, _ = self.env.get_avail_mask()
        avail = avail.reshape(-1, )

        self.Ns[s] = 1

        if add_noise:
            noise = np.random.dirichlet([self.noise_eta for _ in range(self.action_space)])

        for a in range(self.action_space):
            if add_noise and avail[a]:
                action_noise = float(noise[a] * avail[a])
                prob = 0.75 * probs[a] + 0.25 * action_noise

            else:
                prob = probs[a]

            self.P[(s, a)] = prob
            self.W[(s, a)] = 0
            self.Q[(s, a)] = 0
            self.N[(s, a)] = 0

        if return_action and self.training:
            action = np.random.choice(self.action_space, p=probs)
            return action, val

        elif return_action and not self.training:
            action = probs.argmax()
            return action, val

        else:
            return val

    def _cal_factor(self):
        diff = self.max_q_val - self.min_q_val
        # (분모, 분자)
        if np.isinf(self.max_q_val) or np.isinf(self.min_q_val):
            return (1, 0)

        if self.max_q_val == self.min_q_val:
            return (self.min_q_val, 0)

        if not self.expand_root:
            return diff, self.min_q_val

        else:
            return (1, 0)

    def _back_propagate(self, path, v):
        if isinstance(v, torch.Tensor):
            v = v.item()

        for s, a in reversed(path):
            self.N[(s, a)] += 1
            self.Ns[s] += 1
            self.W[(s, a)] += v

        for i, (s, a) in enumerate(reversed(path)):
            Q_val = self.W[(s, a)] / self.N[(s, a)]

            self.min_q_val = min(self.min_q_val, Q_val)
            self.max_q_val = max(self.max_q_val, Q_val)

            self.Q[(s, a)] = Q_val

    def _save_state_field(self):
        self._initial_visiting_seq = deepcopy(self.env.visiting_seq)
        self._initial_visited = deepcopy(self.env.visited)
        self._initial_pos = deepcopy(self.env.pos)
        self._initial_available = deepcopy(self.env.available)
        self._initial_t = deepcopy(self.env.t)

        if self.env_type == 'cvrp':
            self._initial_load = deepcopy(self.env.load)

    def _reset_env_field(self):
        self.env.visiting_seq = deepcopy(self._initial_visiting_seq)  # type is list
        self.env.visited = deepcopy(self._initial_visited)
        self.env.available = deepcopy(self._initial_available)
        self.env.t = deepcopy(self._initial_t)
        self.env.pos = deepcopy(self._initial_pos)

        if self.env_type == 'cvrp':
            self.env.load = self._initial_load.copy()

    def _run_simulation(self, root_state):
        obs = root_state
        self._reset_env_field()
        state_num = obs['t']

        path = []
        done = False
        v = 0

        # Initialize the first nodes
        if self.expand_root:
            _add_noise = True if self.training else False
            # _add_noise = False
            _ = self._expand(root_state, add_noise=_add_noise, return_action=False)
            # path.append((state_num, a))
            # self._back_propagate(path, v)
            self.expand_root = False

        # select child node and action
        while state_num in self.Ns and not done:
            a = self._select(obs)
            path.append((state_num, a))

            if self.training:
                obs, value, done, _ = self.env.step(a)

            else:
                obs, reward, done, _, _ = self.env.step(a)

            state_num = obs['t']

            if self.training:
                if len(path) > 30:
                    break

        if self.rollout_game:
            v = self._rollout_until_end(obs)

        else:
            if not done:
                # leaf node reached
                v = self._expand(obs)

            else:
                # terminal node reached
                v = reward

        self._back_propagate(path, v)

    def _rollout_until_end(self, obs):
        _obs = deepcopy(obs)
        _env = deepcopy(self.env)
        _env.set_test_mode()

        done = False

        while not done:
            a, _ = self.model.predict(obs)
            obs, reward, done, _, _ = _env.step(a)

        return reward
