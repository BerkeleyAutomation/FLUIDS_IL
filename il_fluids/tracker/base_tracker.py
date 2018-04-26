import gym
import gym_urbandriving as uds
import cProfile
import time
import os
import numpy as np
import numpy.linalg as LA
import IPython
import gym_urbandriving as fluids
from il_fluids.tracker.tracker import Tracker

def overrides(super_class):
    def overrider(method):
        assert(method.__name__ in dir(super_class))
        return method
    return overrider



class BaseTracker(Tracker):

    @overrides(Tracker)
    def bug(self,state,num_agent,action):

        return True

   