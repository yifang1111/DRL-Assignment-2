import copy
import random
import math
import numpy as np
from env import Game2048Env
from n_tuple_approximator import NTupleApproximator

class DecisionNode:
    def __init__(self, env, state, score, parent=None, action=None):
        self.env = env
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


class ChanceNode:
    def __init__(self, env, state, score, parent=None):
        self.env = env
        self.state = state
        self.score = score
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_spawns = self.get_possible_spawns()

    def fully_expanded(self):
        return len(self.untried_spawns) == 0

    def get_possible_spawns(self):
        possible_spawns = []
        empty_cells = list(zip(*np.where(self.state == 0)))
        for (x, y) in empty_cells:
            possible_spawns.append((x, y, 2, 0.9))
            possible_spawns.append((x, y, 4, 0.1))
        return possible_spawns

class TD_MCTS_DCN: 
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41,
                 rollout_depth=10, gamma=0.99):
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

    def select_child_decision(self, node):
        best_value = -float("inf")
        best_child = None
        
        for action, child in node.children.items():
            if child.visits == 0:
                uct_value = float("inf")
            else:
                uct_value = (child.total_reward / child.visits) + \
                            self.c * math.sqrt(math.log(node.visits) / child.visits)
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
                
        return best_child

    def select_child_chance(self, node):
        children = list(node.children.values())
        keys = list(node.children.keys())
        probs = [k[3] for k in keys]
        sum_p = sum(probs)
        if sum_p <= 0:
            return None
        
        normalized = [p / sum_p for p in probs]
        chosen_index = np.random.choice(range(len(children)), p=normalized)
        return children[chosen_index]

    def rollout(self, sim_env, depth):
        total_reward = 0.0
        discount = 1.0

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

    def expand_decision_node(self, decision_node):
        action = random.choice(decision_node.untried_actions)
        decision_node.untried_actions.remove(action)
        sim_env = self.create_env_from_state(decision_node.state, decision_node.score)
        sim_env.step(action, spawn=False)
        
        new_chance_node = ChanceNode(
            env = sim_env,
            state = sim_env.board.copy(),
            score = sim_env.score,
            parent = decision_node
        )
        decision_node.children[action] = new_chance_node
        return new_chance_node

    def expand_chance_node(self, chance_node):
        x, y, tile, prob = chance_node.untried_spawns.pop()
        sim_env = self.create_env_from_state(chance_node.state, chance_node.score)
        sim_env.board[x, y] = tile
        
        new_decision_node = DecisionNode(
            env = sim_env,
            state = sim_env.board.copy(),
            score = sim_env.score,
            parent = chance_node
        )
        chance_node.children[(x, y, tile, prob)] = new_decision_node
        return new_decision_node

    def run_simulation(self, root):
        node = root
        
        # selection
        while True:
            if not node.fully_expanded():
                break
            
            if isinstance(node, DecisionNode):
                if len(node.children) == 0:
                    break  
                node = self.select_child_decision(node)
            else:  
                if len(node.children) == 0:
                    break
                node = self.select_child_chance(node)
                    
        # expansion
        if not node.fully_expanded():
            if isinstance(node, DecisionNode):
                node = self.expand_decision_node(node)
            else:
                node = self.expand_chance_node(node)

        sim_env = self.create_env_from_state(node.state, node.score)
        rollout_reward = self.rollout(sim_env, self.rollout_depth)

        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        
        for action, child in root.children.items():
            if total_visits > 0:
                distribution[action] = child.visits / total_visits
            else:
                distribution[action] = 0
            
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

    td_mcts = TD_MCTS_DCN(env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=0, gamma=0.99)

    state = env.reset()
    # env.render()

    done = False
    while not done:
        # Create the root node from the current state
        root = DecisionNode(env, copy.deepcopy(state), env.score)

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

