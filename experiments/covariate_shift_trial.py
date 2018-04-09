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

###### SELECT MODEL #################
il_config['model'] = DecisionTreeClassifier(max_depth = 4)


###RUN BEHAVIOR CLONING############
il_config['experiment_name']  = il_config['trial_name']+ '_behavior_cloning'

trainer = Trainer(fluids_config,il_config)

trainer.train_robot()


###RUN NOISE INJECTION############
il_config['experiment_name'] = il_config['trial_name'] + '_noise_injection_debug'

trainer = Trainer(fluids_config,il_config)

trainer.set_data_protocol(DART())

trainer.train_robot()















