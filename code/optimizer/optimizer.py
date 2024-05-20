import numpy as np
from abc import abstractmethod


class Optimizer:
    def __init__(self):
        self.state_dim = None
        self.state_param_range = None
        self.energy = None
        self.answer_state = None
        self.answer_energy = None
