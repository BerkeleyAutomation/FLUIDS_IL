import numpy as np
import os

# import fluids

"""
1. Scan through demonstrations in UDS/data folder
2. For each demonstration, look at observations and actions and make sure that they make sense
3. Specific criteria?
"""

path = "../../Urban_Driving_Simulator/fluids/data"
print(path)

for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        n = os.path.join(root, name)
        print(n)
        # f = open(n, 'rb')
        # print(f)
        npz = np.load(n)
        print(npz)
        array = npz['arr_0']
        print(len(array))
