import numpy.linalg as LA
from gym_urbandriving.actions import VelocityAction
import numpy as np

#for now accept floats
def euclidean(value_1,value_2):

	return LA.norm(value_1-value_2)


def indicator(value_1,value_2):

	if np.abs(value_1-value_2) > 1e-2: 
		return 1
	else:
		return 0

def loss(loss_name,value_1,value_2):

	if loss_name == "indicator":
		return indicator(value_1,value_2)
	elif loss_name == "euclidean":
		return euclidean(value_1,value_2)
	else:
		raise Exception('Loss function not supported')