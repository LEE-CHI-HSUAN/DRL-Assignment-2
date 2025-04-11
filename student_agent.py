# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
from game2048 import Game2048Env, NTupleApproximator
from monte_carlo import MCTS_Node, MCS


patterns = [
    ((0, 0), (0, 1), (1, 0), (1, 1)),  # corner square
    ((0, 1), (0, 2), (1, 1), (1, 2)),  # edge square
    ((0, 0), (0, 1), (0, 2), (0, 3)),  # edge line
    ((1, 0), (1, 1), (1, 2), (1, 3)),  # middle line
]
approximator = NTupleApproximator(board_size=4, patterns=patterns, gamma=0.99)
with open("value_weights.pkl", "rb") as fin:
    approximator.LUTs = pickle.load(fin)

env = Game2048Env()
mcs = MCS(env, approximator, batch_size=5)


def get_action(state, score):
    env.board = state
    env.score = score
    root = MCTS_Node(None, None, env)
    action = mcs.simulate(root, steps=5)
    return action

    # You can submit this random agent to evaluate the performance of a purely random strategy.
