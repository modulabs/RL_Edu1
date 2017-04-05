import numpy as np
import sys
import os
from gym import wrappers

from abc import ABCMeta, abstractmethod

class GamePlay(metaclass=ABCMeta):
    @abstractmethod
    def gameSetup(self): pass

    @abstractmethod
    def trainAgent(self): pass

    @abstractmethod
    def runGamePlay(self): pass
