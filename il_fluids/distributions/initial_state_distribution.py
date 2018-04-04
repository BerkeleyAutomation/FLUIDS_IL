import gym_urbandriving as fluids
from gym_urbandriving.agents import *
import numpy as np

class InitialState:

    def __init__(self,fluids_config,il_config):

        self.fluids_config = fluids_config
        self.il_config = il_config



    def sample_state(self):

    	self.num_cars = np.random.randint(self.il_config['min_cars'],high=self.il_config['max_cars'])

    	self.fluids_config['agents']['controlled_cars'] = self.num_cars

    	env = fluids.UrbanDrivingEnv(self.fluids_config)

    	return env

    def create_supervisors(self):

    	supervisors = []
    	for i in range(self.num_cars):
    		supervisor = VelocitySupervisor(agent_num = i)
    		supervisors.append(supervisor)

    	return supervisors



