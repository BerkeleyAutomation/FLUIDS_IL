from numpy.random import uniform
import numpy as np
from numpy.random import multivariate_normal as normal
from il_fluids.data_protocols import DCP
from gym_urbandriving.actions import SteeringAction
import IPython

class DARTS(DCP):

	def __init__(self):

		self.noise = np.eye(2)
		self.use_robot_action = False

	def get_action(self,robot_action = None,supervisor_action = None):

	 	noise_action = normal(supervisor_action.get_value(),self.noise)
	 	return SteeringAction(steering=noise_action[0],acceleration=noise_action[1])

	def update(self,il_learn):

		cov = np.zeros([2,2])

		num_points= len(il_learn.X_test)
		for i in range(num_points):

			_x = il_learn.X_test[i]
			_sup_y = np.array([il_learn.Y_test[i]])
			_robot_y = il_learn.model.predict(np.array([_x]))
			
			cov_mat = np.dot((_sup_y-_robot_y).T,(_sup_y-_robot_y))

			cov += cov_mat

		if num_points > 0:
			self.noise = cov/(num_points)

		return 
