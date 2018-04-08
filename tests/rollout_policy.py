import gym
import gym_urbandriving as uds
import cProfile
import time
import numpy as np
import numpy.linalg as LA
import json
from il_fluids.core import Trainer
from il_fluids.core import Learner
from il_fluids.core  import Plotter

###A script to test behavior cloning 

#Config for FLUIDS simulator
with open('configs/fluids_config.json') as json_data_file:
    fluids_config = json.load(json_data_file)

#Config for Imitation Learning Experiment 
with open('configs/il_velocity_config.json') as json_data_file:
    il_config = json.load(json_data_file)


file_path =  il_config['file_path'] + il_config['experiment_name']

#Trainer class
trainer = Trainer(fluids_config,il_config)


trainer.train_model()

trainer.get_stats()












