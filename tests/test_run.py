import time
import numpy as np
import numpy.linalg as LA
import json

from sklearn.tree import DecisionTreeClassifier

from il_fluids.data_protocols import *

###A script to test behavior cloning 

#Config for FLUIDS simulator
with open('configs/fluids_config.json') as json_data_file:
    fluids_config = json.load(json_data_file)

#Config for Imitation Learning Experiment 
with open('configs/il_debug_config.json') as json_data_file:
    il_config = json.load(json_data_file)

###### SELECT MODEL #################
il_config['model'] = DecisionTreeClassifier(max_depth = 4)













