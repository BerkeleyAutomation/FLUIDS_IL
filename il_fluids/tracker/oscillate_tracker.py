import gym
import gym_urbandriving as uds
import cProfile
import time
import os
import numpy as np
import numpy.linalg as LA
import IPython
import gym_urbandriving as fluids
import tracker

def overrides(super_class):
    def overrider(method):
        assert(method.__name__ in dir(super_class))
        return method
    return overrider



class OscTracker(Tracker):

    @overrides(Common)
    def bug(self,state,num_agent,action):

        key = str(num_agent)
        action = action.get_value()

        if not key in self.prev_actions.keys():
            self.prev_actions[key] = action
            self.num_changes[key] = 0

        if not self.prev_actions[key] == action:
            self.num_changes[key] += 1
            self.prev_actions[key] = action

            if self.num_changes[key] > self.thresh:
                return True

        return False


   