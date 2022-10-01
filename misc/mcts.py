import numpy as np


class State(object):
    def __init__(self, actions, prior_prob):
        self.actions = actions
        self.prior_prob = prior_prob

    def get_actions(self):
        return self.actions

    def get_prior_prob(self):
        return self.prior_prob

    def next_state(self, act):
        raise NotImplementedError()

    def value(self):
        raise NotImplementedError()

    def is_valid(self):
        raise NotImplementedError()

    def is_terminal(self):
        raise NotImplementedError()

    def rollout(self):
        raise NotImplementedError()


class TreeNode:

    def __init__(self, state):
        self.state = state
        self.actions = self.state.get_actions()
        self.prior_prob = self.state.get_prior_prob()

        n_actions = len(self.actions)
        self.action_value = [0] * n_actions
        self.visit_count = [0] * n_actions
        self.children = [None] * n_actions

    def best_child(self, c):
        score = [q + c * (p / (1 + n))
                 for q, p, n in zip(self.action_value, self.prior_prob, self.visit_count)]
        index = np.argsort(score).tolist()[::-1]

        idx = -1
        expanded = False
        for i in index:
            if self.children[i] is not None:
                idx = i
                break
            new_state = self.state.next_state(self.actions[i])
            if new_state.is_valid():
                expanded = True
                self.children[i] = TreeNode(new_state)
                idx = i
                break
        assert idx != -1
        return self.children[idx], idx, expanded

    def get_visit_count(self):
        return sum(self.visit_count)

    def most_visited_action(self):
        return self.actions[np.argmax(self.visit_count).item()]

    def update(self, action_index, leaf_value):
        self.action_value[action_index] = self.action_value[action_index] * self.visit_count[action_index] + leaf_value
        self.visit_count[action_index] += 1
        self.action_value[action_index] /= self.visit_count[action_index]

    def is_terminal(self):
        return self.state.is_terminal()

    def state_value(self):
        return self.state.value()

    def rollout(self):
        return self.state.rollout()


def search(state, search_budget, exploration_coefficient, state_value_coefficient):
    root = TreeNode(state)
    while root.get_visit_count() < search_budget:
        node = root
        trace = []
        while not node.is_terminal():
            child, action_index, expanded = node.best_child(exploration_coefficient)
            trace.append((node, action_index))
            node = child
            if expanded:
                break
        state_value = node.state_value()
        rollout_reward, success, final_state = node.rollout()
        if success:
            return None, final_state
        value = state_value_coefficient * state_value + (1 - state_value_coefficient) * rollout_reward
        for node, action_index in trace:
            node.update(action_index, value)
    action = root.most_visited_action()
    return action, None
