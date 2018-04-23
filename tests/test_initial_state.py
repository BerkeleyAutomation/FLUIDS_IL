import gym
import gym_urbandriving as uds
import cProfile
import time
import numpy as np
import numpy.linalg as LA
import json
from il_fluids.distributions import InitialState

###A script to test behavior cloning 

#Config for FLUIDS simulator
with open('configs/fluids_config.json') as json_data_file:
    fluids_config = json.load(json_data_file)

#Config for Imitation Learning Experiment 
with open('configs/il_covariate_config.json') as json_data_file:
    il_config = json.load(json_data_file)

fluids_config['environment']['visualize'] = False

initial_state = InitialState(fluids_config,il_config)


initial_state.sample_state()
num_cars = initial_state.num_cars
assert((num_cars <= il_config['max_cars']) or (num_cars >= il_config['min_cars']))









