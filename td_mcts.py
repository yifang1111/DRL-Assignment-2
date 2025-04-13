import copy
import random
import math
import numpy as np
import pickle
from env import Game2048Env
from n_tuple_approximator import NTupleApproximator

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, env, state, score, parent=None, action=None):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        return len(self.untried_actions) == 0


class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        best_value = -float("inf")
        best_child = None
        for child in node.children.values():
            if child.visits == 0:
                uct_value = float("inf")
            else:
                uct_value = child.total_reward / child.visits + self.c * math.sqrt(math.log(node.visits) / child.visits)
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
        return best_child

    def rollout(self, sim_env, depth):
        total_reward = 0
        discount = 1
        for _ in range(depth):
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break
            action = random.choice(legal_moves)
            prev_score = sim_env.score
            _, _, done, _ = sim_env.step(action)
            reward = sim_env.score - prev_score
            total_reward += discount * reward
            discount *= self.gamma
            if done:
                return total_reward

        legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
        if legal_moves:
            values = []
            for a in legal_moves:
                sim_copy = copy.deepcopy(sim_env)
                afterstate, afterscore, _, _ = sim_copy.step(a, spawn=False)
                v = self.approximator.value(afterstate)
                values.append(v)
            total_reward += discount * max(values)
        return total_reward

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += (reward - node.total_reward) / node.visits
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            sim_env.step(node.action)

        if not node.fully_expanded():
            action = random.choice(node.untried_actions)
            sim_env.step(action)
            new_node = TD_MCTS_Node(copy.deepcopy(sim_env), sim_env.board.copy(), sim_env.score, parent=node, action=action)
            node.children[action] = new_node
            node.untried_actions.remove(action)
            node = new_node

        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution

if __name__ == '__main__':
    env = Game2048Env()

    with open("value_approximator_afterstate_weights_2_28000.pkl", "rb") as f:
        loaded_weights = pickle.load(f)
    patterns = [
        [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)],
        [(0,1), (1,1), (2,1), (0,2), (1,2), (2,2)],
        [(0,0), (1,0), (2,0), (3,0), (2,1), (3,1)],
        [(0,1), (1,1), (2,1), (3,1), (2,2), (3,2)],
        [(0,0), (1,0), (2,0), (3,0), (1,1), (2,1)],
        [(0,1), (1,1), (2,1), (3,1), (1,2), (2,2)],
        # [(0,0), (1,0), (2,0), (3,0), (3,1), (3,2)],
        # [(0,0), (1,0), (2,0), (3,0), (2,1), (2,2)],
    ]
    approximator = NTupleApproximator(board_size=4, patterns=patterns)
    approximator.weights = loaded_weights

    td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=3, gamma=1.0)

    state = env.reset()
    # env.render()

    done = False
    while not done:
        # Create the root node from the current state
        root = TD_MCTS_Node(env, state, env.score)

        # Run multiple simulations to build the MCTS tree
        for _ in range(td_mcts.iterations):
            td_mcts.run_simulation(root)

        # Select the best action (based on highest visit count)
        best_act, _ = td_mcts.best_action_distribution(root)
        # print("TD-MCTS selected action:", best_act)

        # Execute the selected action and update the state
        state, reward, done, _ = env.step(best_act)
        # env.render(action=best_act)

    print("Game over, final score:", env.score)

