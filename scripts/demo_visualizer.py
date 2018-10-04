import numpy as np
import os

import fluids
from fluids.obs import grid

"""
1. Scan through demonstrations in UDS/data folder
2. For each demonstration, look at observations and actions and make sure that they make sense
3. Specific criteria?
"""

#path = "../Urban_Driving_Simulator/fluids/data"
root = os.getenv("HOME") + "/../../nfs/diskstation/projects/fluids_dataset"
path = root + "/fluids_data_20180926_123838"
#print(path)

for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        n = os.path.join(root, name)
        print(n)
        npz = np.load(n)
        array = npz['arr_0']
        obs1 = array[0][fluids.OBS_GRID][:, :, :, :4]
        obs2 = array[0][fluids.OBS_GRID][:, :, :, 8:]
        obs = np.concatenate((obs1, obs2), axis=3)
        print("Observation shape: {}".format(obs.shape))
        for i in range(obs.shape[3]):
            channel = obs[:, :, :, i]
            print("Channel : {}".format(i))
            print(channel)
        acts = array[0][fluids.VelocityAction]
        print("Action shape: {}".format(acts.shape))
        print(acts)


