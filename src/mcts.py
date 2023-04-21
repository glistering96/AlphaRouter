import logging
import math
from copy import deepcopy

import numpy as np
import torch

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """
    def __init__(self, env, model, mcts_params, training=True):

        self.env = deepcopy(env)
        self.model = model
        self.mcts_params = mcts_params
        self.action_space = mcts_params['action_space']
        self.expand_root = True
        self.cpuct = mcts_params['cpuct']
        self.normalize_q_value = mcts_params['normalize_value']
        self.noise_eta = mcts_params['noise_eta']
        self.rollout_game = mcts_params['rollout_game']

        self._initial_visitng_seq = deepcopy(self.env._visiting_seq)
        self._initial_visited = deepcopy(self.env._visited)
        self._initial_load = deepcopy(self.env._load)
        self._initial_pos = deepcopy(self.env._pos)
        self._initial_available = deepcopy(self.env._available)
        self._initial_t = deepcopy(self.env._t)

        self.training = training

        self.Q = {}  # stores Q values for s,a (as defined in the paper)
        self.W = {}  # sum of values
        self.Ns = {} # sum of visit counts for state s
        self.N = {}  # stores #times edge s,a was visited
        self.P = {}  # stores initial policy (returned by neural net), dtype: numpy ndarray

        self.max_q_val = float('-inf')
        self.min_q_val = float('inf')
        self.factor = (1, 0)

    def _cal_probs(self, target_state, temp):
        s = target_state['_t']
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
        s = state['_t']
        avail_mask, _ = self.env.get_avail_mask()
        ucb_scores = {}

        # pick the action with the lowest upper confidence bound
        for a in range(self.action_space):
            # mask unavailable or non-promising actions

            if not avail_mask[a]:
                continue

            ucb = self.cpuct * self.P[(s, a)] * math.sqrt(self.Ns[s]) / (1 + self.N[(s, a)])

            Q_val = self.Q[(s,a)]

            diff = self.max_q_val - self.min_q_val

            if Q_val != 0 and self.normalize_q_value :
                Q_val = (self.Q[(s,a)] - self.min_q_val) / (self.max_q_val - self.min_q_val + 1e-8)

            if diff < 1e-8:
                Q_val = 0

            score = -Q_val + ucb
            ucb_scores[a] = score

        max_ucb = max(ucb_scores.values())
        a = np.random.choice([action for action, score in ucb_scores.items() if score == max_ucb])
        return int(a)

    def _expand(self, state, add_noise=False, return_action=False):
        s = state['_t']

        prob_dist, val = self.model(state)
        probs = prob_dist.view(-1,).cpu().numpy()
        avail, _ = self.env.get_avail_mask()

        self.Ns[s] = 1

        if add_noise:
            noise = np.random.dirichlet([self.noise_eta for _ in range(self.action_space)])

        for a in range(self.action_space):
            if add_noise:
                action_noise = float(noise[a] * avail[a])
                prob = 0.75*probs[a] + 0.25*action_noise

            else:
                prob = probs[a]

            self.P[(s, a)] = prob
            self.W[(s, a)] = 0
            self.Q[(s, a)] = 0
            self.N[(s, a)] = 0

        self.env._visiting_seq = []

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

        for s, a in path:
            self.N[(s,a)] += 1
            self.Ns[s] += 1
            self.W[(s,a)] += v

        for i, (s, a) in enumerate(reversed(path)):
            Q_val = self.W[(s,a)] / self.N[(s,a)]

            if Q_val < self.min_q_val:
                self.min_q_val = Q_val

            if Q_val > self.max_q_val:
                self.max_q_val = Q_val

            self.Q[(s, a)] = Q_val

    def _reset_env_field(self):
        self.env._visiting_seq = deepcopy(self._initial_visitng_seq)    # type is list
        self.env._visited = self._initial_visited.copy()
        self.env._load = self._initial_load.copy()
        self.env._available = self._initial_available.copy()
        self.env._t = deepcopy(self._initial_t)

    def _run_simulation(self, root_state):
        obs = root_state
        self._reset_env_field()
        state_num = obs['_t']

        path = []
        done = False
        v = 0

        # Initialize the first nodes
        if self.expand_root:
            _add_noise = True if self.training else False
            # _add_noise = False
            a, v = self._expand(root_state, add_noise=_add_noise, return_action=True)
            path.append((state_num, a))
            # self._back_propagate(path, v)
            self.expand_root = False

        # select child node and action
        while state_num in self.Ns:
            a = self._select(obs)
            path.append((state_num, a))

            if self.training:
                obs, value, done, _ = self.env.step(a)

            else:
                obs, reward, done, _, _ = self.env.step(a)

            state_num = obs['_t']

            if self.training:
                if len(path) > 30:
                    break

        # v = self._rollout_until_end(obs)

        if self.rollout_game:
            v = self._rollout_until_end(obs)

        else:
            if not done:
                # leaf node reached
                v = self._expand(obs)

            else:
                # terminal node reached
                v = -self.env.get_reward()

        self._back_propagate(path, v)

    def _rollout_until_end(self, obs):
        _obs = deepcopy(obs)
        _env = deepcopy(self.env)
        _env.set_test_mode()

        done = False

        while not done:
            a, _ = self.model.predict(obs)
            obs, reward, done, _, _ = _env.step(a)

        return -reward






