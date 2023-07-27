import logging
import math
from copy import deepcopy

import numpy as np
import torch
EPS = 1e-8

log = logging.getLogger(__name__)


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    """
    Node class for MCTS
    """
    def __init__(self, state, parent=None, prior=1):
        self.state = state
        self._is_expanded = False
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_cost = 0
        self.prior = prior

    def expand(self, state, action_priors):
        """
        Expand tree by creating new children.
        """
        self._is_expanded = True

        for action, prob in enumerate(action_priors):
            if action not in self.children and prob > 1e-6:
                self.children[action] = Node(state=deepcopy(state), parent=self, prior=prob)

    def is_expanded(self):
        return self._is_expanded

    def get_cost(self):
        """
        Calculate the value of this node.
        """
        if self.visit_count == 0:
            return 0
        return self.total_cost / self.visit_count


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, env, model, mcts_params):
        self.env = env
        self.env_type = env.env_type
        self.action_space = env.action_size

        self.model = model

        self.cpuct = mcts_params['cpuct']
        self.noise_eta = mcts_params['noise_eta']
        self.rollout_game = mcts_params['rollout_game']
        self.ns = mcts_params['num_simulations']

        self.seed = 1234
        self.rand_gen = np.random.default_rng(self.seed)

        self.min_max_stats = MinMaxStats()

    def get_action_prob(self, root_state, temp=0):
        """
        Simulate and return the probability for the target state based on the visit counts acquired from simulations
        """

        root = Node(state=deepcopy(root_state), prior=0)

        action_probs, _ = self.model(root_state)
        action_probs = action_probs.cpu().numpy().reshape(-1)
        action = int(np.argmax(action_probs, -1))

        next_state = self.env.step(root_state, action)[0]

        root.expand(next_state, action_probs)

        for i in range(self.ns):
            self._run(root)

        visit_counts, actions = [], []
        est_cost = {}

        for action, child in root.children.items():
            visit_counts.append(child.visit_count)
            actions.append(action)
            est_cost[action] = child.get_cost()

        if temp == 0:
            action = actions[np.argmax(visit_counts)]

        elif temp == float('inf'):
            action = self.rand_gen.choice(actions)

        else:
            visit_counts = np.power(visit_counts, 1.0 / temp)
            visit_counts_sum = np.sum(visit_counts)
            action_probs = visit_counts / visit_counts_sum
            action = self.rand_gen.choice(actions, p=action_probs)

        min_cost_child = min(est_cost, key=est_cost.get)
        visit_counts_stats = {a: v for a, v in zip(actions, visit_counts)}
        mcts_run_info = {
            'min_cost_child': min_cost_child,
            'visit_counts_stats': visit_counts_stats,
            'priors': {a: c.prior for a, c in root.children.items()},
        }
        return action, mcts_run_info

    def _run(self, root_node):
        """
        Run one simulation in MCTS
        """
        node = root_node
        search_path = [node]

        while node.is_expanded():
            action, node = self._select(node)
            search_path.append(node)

        leaf = search_path[-1]

        next_state, env_cost, done, _, _ = self.env.step(leaf.state, action)

        predicted_cost = self._expand(leaf, next_state)

        if done:
            cost = env_cost

        else:
            # one may add rollout cost here
            cost = predicted_cost

        self._backup(search_path, cost)

    def _search(self, node):
        """
        Search from root to leaf and return the search path
        """
        search_path = [node]
        action = None

        while node.is_expanded():
            action, node = self._select(node)
            search_path.append(node)
        #
        # if action is None:
        #     action_probs_tensor, cost_tensor = self.model(node.state)
        #     action_probs = action_probs_tensor.detach().cpu().flatten().tolist()
        #     action = np.argmax(action_probs)

        return action, search_path

    def _select(self, node):
        """
        Select action among children that gives maximum action value Q plus bonus u(P).
        """
        ucb_scores = {}

        for action, child in node.children.items():
            ucb_score = self._ucb_score(node, child)
            ucb_scores[action] = ucb_score

        _, action, child = max((ucb_scores[action], action, child) for action, child in node.children.items())

        return action, child

    def _ucb_score(self, parent, child):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        u = self.cpuct * child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
        q_val = child.get_cost()

        norm_q_val = self.min_max_stats.normalize(q_val)

        return -norm_q_val + u

    def _expand(self, node, state):
        """
        Expand tree by creating new children.
        """
        is_expandable = state['available'].sum() > 0
        cost = None

        if is_expandable:
            action_probs_tensor, cost_tensor = self.model(state)
            action_probs = action_probs_tensor.detach().cpu().flatten().tolist()
            cost = cost_tensor.detach().item()
            node.expand(state, action_probs)

        return cost

    def _backup(self, search_path, value):
        """
        Update value and visit count of nodes in search path
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.total_cost += value
            q_value = node.get_cost()
            self.min_max_stats.update(q_value)
