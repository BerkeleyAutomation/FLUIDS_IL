from numpy.random import uniform
from il_fluids.data_protocols import DCP
from gym_urbandriving.actions import VelocityAction

class DART(DCP):

	def __init__(self):

		self.noise = 0.1
		self.use_robot_action = False

	def get_action(self,robot_action = None,supervisor_action = None):

		if uniform() > self.noise:
			return supervisor_action
		else:
			if supervisor_action.get_value() == 0.0:
				return VelocityAction(4.0)
			elif supervisor_action.get_value() == 4.0:
				return VelocityAction(0.0)
