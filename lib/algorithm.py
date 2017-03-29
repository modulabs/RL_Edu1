import numpy as np
import sys
import os
from gym import wrappers

from abc import ABCMeta, abstractmethod

class Algorithm(metaclass=ABCMeta):
    @abstractmethod
    def initNetwork(self): pass

    @abstractmethod
    def featurize(self): pass

    @abstractmethod
    def predict(self): pass

    @abstractmethod
    def update(self): pass