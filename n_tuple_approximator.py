import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle
from game_env import Game2048Env


# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------
def rotate90(coords, size=4):
    return [(y, size - 1 - x) for x, y in coords]

def flip_horizontal(coords, size=4):
    return [(x, size - 1 - y) for x, y in coords]


class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        self.patterns_idx = []
        for idx, pattern in enumerate(self.patterns):
            syms = self.generate_symmetries(pattern)
            for syms_ in syms:
                self.symmetry_patterns.append(syms_)
                self.patterns_idx.append(idx)  

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        syms = []
        current = pattern
        for _ in range(4):
            syms.append(current)
            syms.append(flip_horizontal(current))
            current = rotate90(current)
        return syms

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        value = 0.0
        for pattern, idx in zip(self.symmetry_patterns, self.patterns_idx):
            feature = self.get_feature(board, pattern)
            value += self.weights[idx][feature]
        return value

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for pattern, idx in zip(self.symmetry_patterns, self.patterns_idx):
            feature = self.get_feature(board, pattern)
            self.weights[idx][feature] += alpha * delta / len(self.symmetry_patterns)


def td_learning_afterstate(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    final_scores = []
    success_flags = []

    for episode in range(num_episodes):
        state = env.reset()
        trajectory = []
        done = False
        prev_score = 0
        max_tile = np.max(state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            # Evaluate all afterstates from legal actions
            candidates = []
            for a in legal_moves:
                sim_env = copy.deepcopy(env)
                score_before = sim_env.score
                afterstate, score_after, _, _ = sim_env.step(a, spawn=False)
                reward = score_after - score_before
                value = reward + gamma * approximator.value(afterstate)
                candidates.append((value, a, afterstate, reward))

            # Choose the best action and corresponding afterstate
            _, action, afterstate, reward = max(candidates)

            # Apply action in real env (this adds randomness)
            _, _, done, _ = env.step(action)
            max_tile = max(max_tile, np.max(env.board))

            trajectory.append((afterstate.copy(), reward))

        # Backward TD(0) update: V(s') ← V(s') + α [r + γV(s'') − V(s')]
        next_value = 0
        for afterstate, reward in reversed(trajectory):
            td_error = reward + gamma * next_value - approximator.value(afterstate)
            approximator.update(afterstate, td_error, alpha)
            next_value = approximator.value(afterstate)

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.mean(success_flags[-100:])
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2%}")

        if episode % 2000 == 0:
            with open(f"value_approximator_afterstate_weights_{episode}.pkl", "wb") as f:
                pickle.dump(approximator.weights, f)

    return final_scores



if __name__ == '__main__':
    patterns = [
        # 6-tuples
        [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)],
        [(0,1), (1,1), (2,1), (0,2), (1,2), (2,2)],
        [(0,0), (1,0), (2,0), (3,0), (2,1), (3,1)],
        [(0,1), (1,1), (2,1), (3,1), (2,2), (3,2)],
        [(0,0), (1,0), (2,0), (3,0), (1,1), (2,1)],
        [(0,1), (1,1), (2,1), (3,1), (1,2), (2,2)],
        # [(0,0), (1,0), (2,0), (3,0), (3,1), (3,2)],
        # [(0,0), (1,0), (2,0), (3,0), (2,1), (2,2)],
    ]   

    approximator_afterstate = NTupleApproximator(board_size=4, patterns=patterns)
    env = Game2048Env()

    # Run TD-Learning training
    # Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
    # However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.
    final_scores = td_learning_afterstate(env, approximator_afterstate, num_episodes=50000, alpha=0.1, gamma=1.0, epsilon=0.1)

    with open("value_approximator_afterstate_weights.pkl", "wb") as f:
        pickle.dump(approximator_afterstate.weights, f)
