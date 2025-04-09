# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
from game2048 import Game2048Env


def get_action(state, score):
    env = Game2048Env()
    return random.choice([0, 1, 2, 3])  # Choose a random action

    # You can submit this random agent to evaluate the performance of a purely random strategy.
