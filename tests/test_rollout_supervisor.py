import gym
import gym_urbandriving as uds
import cProfile
import time
import numpy as np
import numpy.linalg as LA
import json
from il_fluids.core import Trainer

from sklearn.tree import DecisionTreeClassifier

from il_fluids.data_protocols import *

###A script to test behavior cloning 

#Config for FLUIDS simulator
with open('configs/fluids_config.json') as json_data_file:
    fluids_config = json.load(json_data_file)

#Config for Imitation Learning Experiment 
with open('configs/il_covariate_config.json') as json_data_file:
    il_config = json.load(json_data_file)

fluids_config['environment']['visualize'] = False
il_config['time_horizon'] = 5
###### SELECT MODEL #################

trainer = Trainer(fluids_config,il_config)

rollout = trainer.rollout_supervisor()

assert(len(rollout) == 5)











