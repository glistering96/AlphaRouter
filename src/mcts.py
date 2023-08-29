import collections
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


class DummyNode(object):
  def __init__(self):
    self.parent = None
    self.child_total_value = collections.defaultdict(float)
    self.child_number_visits = collections.defaultdict(float)


class Node:
    """
    Node class for MCTS
    """
    def __init__(self, state, action, env, min_max_stats, parent=None, cpuct=1.1):
        self.state = state
        self.env = env
        self.min_max_stats = min_max_stats
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.action = action
        self.cpuct = cpuct
        action_size = state['available'].shape[-1]

        self.child_priors = np.zeros([action_size], dtype=np.float32)
        self.child_total_value = np.zeros([action_size], dtype=np.float32)
        self.child_number_visits = np.zeros([action_size], dtype=np.float32)

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.action] = value

    def child_Q(self, normalize=False):
        denominator = np.where(self.child_number_visits == 0, 1, self.child_number_visits)
        q = self.child_total_value / denominator

        if normalize is True:
            q_norm = self.min_max_stats.normalize(q)
            q_norm = np.where(q_norm > 0, q_norm, 0)
            return q_norm

        else:
            return q

    def child_U(self):
        return math.sqrt(self.number_visits) * self.child_priors / (1 + self.child_number_visits)

    def best_child(self):
        q = self.child_Q(normalize=True)
        u = self.child_U()
        mask = (q >= 0) & (self.state['available'].reshape(-1) == True)
        ucb = -q + self.cpuct * u
        masked_children = np.where(mask, ucb, float('-inf'))
        best_child = np.argmax(masked_children)
        return best_child

    def select_leaf(self):
        current = self
        reached_terminal = self.env.is_done(self.state['visited'])

        while current.is_expanded and not reached_terminal:
            best_move = current.best_child()

            if best_move not in self.children:
                obs = self.env.step(deepcopy(self.state), best_move)[0]
                reached_terminal = self.env.is_done(obs['visited'])

                self.children[best_move] = Node(obs, best_move, parent=self, min_max_stats=self.min_max_stats,
                                           env=self.env)

            current = current.maybe_add_child(best_move)

        return current

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def maybe_add_child(self, move):
        if move not in self.children:
            obs = self.env.step(self.state, move)[0]
            self.children[move] = Node(deepcopy(obs), move, parent=self, min_max_stats=self.min_max_stats, env=self.env)
        return self.children[move]

    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value_estimate
            q_value = current.get_cost()
            self.min_max_stats.update(q_value)
            current = current.parent

    def get_cost(self):
        denominator = 1 if self.number_visits == 0 else self.number_visits
        return self.total_value / denominator


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

        root = Node(state=deepcopy(root_state), action=None, env=self.env,
                    min_max_stats=self.min_max_stats, parent=DummyNode())
        #
        # action_probs, _ = self.model(root_state)
        # action_probs = action_probs.cpu().numpy().reshape(-1)

        # next_state = self.env.step(root_state, action)[0]

        # root.expand(action_probs)

        for i in range(self.ns):
            leaf = root.select_leaf()

            if self.env.is_done(leaf.state['visited']):
                real_cost = self.env.get_reward(leaf.state['visited'], leaf.state['visiting_seq'])
                leaf.backup(real_cost)
                continue

            child_priors, value_estimate = self.model(leaf.state)
            leaf.expand(child_priors.cpu().numpy().reshape(-1))
            leaf.backup(value_estimate.item())

        visit_counts, actions = [], []
        est_cost = {}

        for action, child in root.children.items():
            visit_counts.append(child.number_visits)
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
            'priors': {a: p for a, p in enumerate(root.child_priors)},
        }
        return action, mcts_run_info

    def _run(self, root_node):
        """
        Run one simulation in MCTS
        """
        node = root_node
        search_path = [node]

        tree_depth = 0
        reached_terminal = self.env.is_done(node.state['visited'])

        while node.is_expanded() and not reached_terminal:
            tree_depth += 1
            action, node = self._select(node)
            search_path.append(node)
            reached_terminal = self.env.is_done(node.state['visited'])

        leaf = search_path[-1]

        next_state, env_cost, done, _, _ = self.env.step(leaf.state, action)

        if self.env_type == 'cvrp' and not self.env.is_done(next_state['visited']) and next_state['load'] < 0:
            print('load is negative while its traverse is not done')

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
