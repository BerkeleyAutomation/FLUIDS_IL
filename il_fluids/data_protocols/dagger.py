
class Dagger:

	def __init__(self):

		self.use_robot_action = False
		self.iter = 0

	def get_action(self,robot_action,supervisor_action):

		if self.iter == 0:
			return supervisor_action
		else:
			return robot_action

	def update(self):
		self.use_robot_action = True
		self.iter += 1 