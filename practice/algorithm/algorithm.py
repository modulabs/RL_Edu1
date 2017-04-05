import numpy as np
import sys
import os
from gym import wrappers

from abc import ABCMeta, abstractmethod

class Algorithm(metaclass=ABCMeta):
    @abstractmethod
    def initNetwork(self): pass

    @abstractmethod
    def predict(self): pass

    @abstractmethod
    def update(self): pass

    @abstractmethod
    def initTraining(self): pass

    @abstractmethod
    def initEpisode(self): pass

    @abstractmethod
    def getNextAction(self, state, mode): pass

    @abstractmethod
    def stepTrain(self): pass

    @abstractmethod
    def episodeTrain(self): pass