import gym
import gym_urbandriving as uds
import cProfile
import time
import numpy as np
import numpy.linalg as LA
import json
from il_fluids.core import Trainer
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

#Plotter class
plotter = Plotter(file_path)
stats = []


for i in range(il_config['num_iters']):
	#Collect demonstrations 
    trainer.collect_supervisor_rollouts()
    #update model 
    trainer.train_model()
    #Evaluate Policy 
    stats.append(trainer.get_stats())

#Save plots
plotter.save_plots(stats)












