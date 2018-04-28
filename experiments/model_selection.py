import gym
import gym_urbandriving as uds
import cProfile
import time
import numpy as np
import numpy.linalg as LA
import json
from il_fluids.core import Learner
from il_fluids.core  import Plotter
import IPython

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

###A script to test behavior cloning 

#Config for FLUIDS simulator
with open('configs/fluids_config.json') as json_data_file:
    fluids_config = json.load(json_data_file)

with open('configs/il_covariate_two_config.json') as json_data_file:
    il_config = json.load(json_data_file)


# ###RUN BEHAVIOR CLONING############
il_config['experiment_name']  = il_config['trial_name'] + "_10_noise_full"

#Params To Search Over 
params =  [5,6,7,8,9,10]
 

for param in params:
	il_config['model'] = DecisionTreeClassifier(max_depth = param)
	learner = Learner(il_config)

	learner.load_data()
	learner.train_model()

	stats = {}

	
	stats['train_error']= learner.get_train_error()

	stats['test_error'],_ = learner.get_test_error()
	stats['max_depth'] = param

	print(stats)

	













