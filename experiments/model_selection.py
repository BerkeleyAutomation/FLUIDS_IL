import gym
import gym_urbandriving as uds
import cProfile
import time
import numpy as np
import numpy.linalg as LA
import json
from il_fluids.core import Learner
from il_fluids.core  import Plotter

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

###A script to test behavior cloning 

#Config for FLUIDS simulator
with open('configs/fluids_config.json') as json_data_file:
    fluids_config = json.load(json_data_file)

#Config for Imitation Learning Experiment 
with open('configs/il_velocity_config.json') as json_data_file:
    il_config = json.load(json_data_file)


file_path =  il_config['file_path'] + il_config['experiment_name']

#Trainer class



#Params To Search Over 
params =  [1,2,4,5,10,None]
 

for param in params:
	model = DecisionTreeClassifier(max_depth = param)
	learner = Learner(file_path,model=model)

	learner.load_data()
	learner.train_model()

	stats = {}

	
	stats['train_error']= learner.get_train_error_classification()

	stats['test_error'] = learner.get_test_error_classification()
	stats['max_depth'] = param

	print(stats)














