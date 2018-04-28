import gym
import gym_urbandriving as uds
import cProfile
import time
import numpy as np
import numpy.linalg as LA
from il_fluids.utils.losses import loss

###Tests the loss functions are correct

value_1 = 1
value_2 = 3

indicator_ans = loss('indicator',value_1,value_2)

assert(indicator_ans == 1)

l2_ans = loss('euclidean',value_1,value_2)

assert(l2_ans == 2.0)










