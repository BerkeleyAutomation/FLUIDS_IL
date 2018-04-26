import gym
import gym_urbandriving as uds
import cProfile
import time
import os
import numpy as np
import numpy.linalg as LA
import IPython
import gym_urbandriving as fluids



class Tracker:


    def __init__(self,state,il_config):

        self.prev_actions = {}

        self.num_changes = {}
        self.thresh = 7

        self.initial_state = state
        self.file_path =  il_config['file_path'] + il_config['experiment_name']
        self.file_path = self.file_path+'/tracker'

        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)


    def catch_bug(self,state,num_agent,action):

        key = str(num_agent)
        action = action.get_value()

        if not key in self.prev_actions.keys():
            self.prev_actions[key] = action
            self.num_changes[key] = 0

        if not self.prev_actions[key] == action:
            self.num_changes[key] += 1
            self.prev_actions[key] = action

            if self.num_changes[key] > self.thresh:
                IPython.embed()

    def load_initial_state(self):
        initial_state = np.load(self.file_path+'/initial_state.npy')
        return initial_state.all()

