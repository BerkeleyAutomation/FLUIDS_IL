import gym
import gym_urbandriving as uds
import cProfile
import time
import numpy as np
import numpy.linalg as LA
import json
from il_fluids.core import Trainer

from sklearn.tree import DecisionTreeClassifier

from il_fluids.utils import InspectSupervisor
from il_fluids.data_protocols import DART

###A script to test behavior cloning 

#Config for FLUIDS simulator
with open('configs/fluids_config.json') as json_data_file:
    fluids_config = json.load(json_data_file)

#Config for Imitation Learning Experiment 
with open('configs/il_v_timing.json') as json_data_file:
    il_config = json.load(json_data_file)

###### SELECT MODEL #################
il_config['model'] = DecisionTreeClassifier(max_depth = 10)

fluids_config['environment']['visualize'] = False

# ###RUN BEHAVIOR CLONING############
# il_config['experiment_name']  = il_config['trial_name'] + "_30_noise_injection"

# trainer = Trainer(fluids_config,il_config)

# dcp = DART()
# dcp.noise = 0.3
# trainer.set_data_protocol(dcp)
# trainer.train_robot()


il_config['experiment_name']  = il_config['trial_name'] + "_bc_sense_test_3"

trainer = Trainer(fluids_config,il_config)
start = time.time()
#trainer.set_data_protocol(DART())
trainer.train_robot()

end = time.time()

print("TOTAL TRAIN TIME")
print(end-start)



# il_config['experiment_name']  = il_config['trial_name'] + "_bc"

# trainer = Trainer(fluids_config,il_config)

# trainer.train_robot()



















