from il_fluids.data_protocols import DCP

class BehaviorCloning(DCP):

	def get_action(self,robot_action = None,supervisor_action = None):

		return supervisor_action

	def update(self,il_learn):
		return