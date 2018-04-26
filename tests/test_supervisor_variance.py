import gym
import gym_urbandriving as uds
import cProfile
import time
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
import json
import glob
from il_fluids.core import Trainer
from il_fluids.utils import InspectSupervisor

###A script to test behavior cloning 

#Config for FLUIDS simulator
with open('configs/fluids_config.json') as json_data_file:
    fluids_config = json.load(json_data_file)

#Config for Imitation Learning Experiment 
with open('configs/il_covariate_config.json') as json_data_file:
    il_config = json.load(json_data_file)

fluids_config['environment']['visualize'] = False

###### SELECT PARAMETER  #################
il_config['time_horizon'] = 5
il_config['num_sup_rollouts'] = 4
il_config['experiment_name'] = il_config['trial_name'] + 'save_unit_test_'+str(rand.uniform())


trainer = Trainer(fluids_config,il_config)

trainer.collect_supervisor_rollouts()

inspector = InspectSupervisor(il_config)

rollout = inspector.rollouts[0]

supervisors,robots = inspector.create_data_structure(rollout)

variances = inspector.compute_variances(robots)

for v in variances:
	assert (v >= 0.0)