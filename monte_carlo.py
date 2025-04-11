import copy
import numpy as np


class MCTS_Node:
    C = np.pi * 2000

    def __init__(self, parent, action, env):
        self.parent = parent
        self.action = action
        self.board = env.board.copy()
        self.score = env.score

        self.children = []
        self.N = 0
        self.sum = 0
        self.CsqrtlogN = 0
        self.inv_sqrtN = 0
        self.mean = 0

    def update(self, n, score):
        self.N += n
        self.sum += score
        self.CsqrtlogN = MCTS_Node.C * np.sqrt(np.log(self.N))
        self.inv_sqrtN = 1 / np.sqrt(self.N)
        self.mean = self.sum / self.N


class MCS:
    def __init__(self, env, value_approximator, batch_size=10):
        self.env = copy.deepcopy(env)
        self.batch_size = batch_size
        self.value_func = value_approximator

    def simulate(self, root, steps=10):
        self.set_env(root)
        legal_moves = [a for a in range(4) if self.env.is_move_legal(a)]
        if not legal_moves:
            raise NotImplementedError

        max_value = float("-inf")
        best_move = None

        for move in legal_moves:
            move_value = 0
            for _ in range(self.batch_size):
                # reset child board
                self.set_env(root)
                _, _, done, _ = self.env.step(move)
                # greedy rollout
                for _ in range(steps):
                    legal_moves = [a for a in range(4) if self.env.is_move_legal(a)]
                    if not legal_moves:
                        break
                    action = self.value_func.get_action(self.env, legal_moves)
                    _, _, done, _ = self.env.step(action)
                # compute value
                if done:
                    value = self.env.score
                else:
                    legal_moves = [a for a in range(4) if self.env.is_move_legal(a)]
                    if not legal_moves:
                        break
                    action = self.value_func.get_action(self.env, legal_moves)
                    self.env.st(action)
                    value = self.env.score + self.value_func.value(self.env.board)
                move_value += value

            if move_value > max_value:
                max_value = move_value
                best_move = move

        return best_move

    def set_env(self, node):
        self.env.board = node.board.copy()
        self.env.score = node.score
